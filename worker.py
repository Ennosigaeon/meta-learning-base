import logging
import random
import traceback
import warnings

import pandas as pd
import pynisher2
import socket
from builtins import object, str
from datetime import datetime
from pandas.util.testing import assert_frame_equal
from sklearn.base import BaseEstimator, is_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from typing import Optional, Tuple, Dict, TYPE_CHECKING

from constants import AlgorithmStatus
from data import delete_data
from database import Database, Algorithm
from methods import ALGORITHMS
from utilities import multiclass_roc_auc_score, logloss

if TYPE_CHECKING:
    from core import Core

warnings.filterwarnings('ignore')

LOGGER = logging.getLogger('mlb')
HOSTNAME = socket.gethostname()


# Exception thrown when something goes wrong for the worker, but the worker handles the error.
class AlgorithmError(Exception):

    def __init__(self, message: str, details: str = None):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self._details = details

    @property
    def details(self) -> str:
        if self._details is not None:
            return self._details
        else:
            str(self)


class Worker(object):
    def __init__(self,
                 database: Database,
                 dataset,
                 core,
                 timeout: int = None,
                 s3_endpoint: str = None,
                 s3_access_key: str = None,
                 s3_secret_key: str = None,
                 s3_bucket: str = None,

                 max_pipeline_depth: int = 5,
                 verbose_metrics: bool = False):

        self.db = database
        self.dataset = dataset
        self.core: Core = core
        self.timeout = timeout

        self.s3_endpoint = s3_endpoint
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_bucket = s3_bucket

        self.max_pipeline_depth = max_pipeline_depth
        self.verbose_metrics = verbose_metrics

        self.subprocess_logger = logging.getLogger('mlb:worker')

        """
        Load the Dataset from the database
        """
        self.dataset = self.db.get_dataset(self.dataset.id)

    def transform_dataset(self, algorithm: BaseEstimator, n_folds: int = 5) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Given a set of fully-qualified hyperparameters, create and not working a algorithm model.
        Returns: Model object and metrics dictionary
        """

        """Load input dataset and class_column"""
        df = self.dataset.load(self.s3_endpoint, self.s3_bucket, self.s3_access_key, self.s3_secret_key)
        class_column = self.dataset.class_column

        """Split input dataset in X and y"""
        X, y = df.drop(class_column, axis=1), df[class_column]

        """
        Checks if algorithm (BaseEstimator) is a classifier. 
        
        If True, predict y_pred with the method cross_val_predict. Then calculate the evaluation metrics for the
        algorithm model and return them as a dict. Convert y_pred to pd Series and concatenate X & y_pred.
        
        If False, call fit_transform or fit and then transform on X, y and return the transformed dataset as Dataframe.
        """

        if is_classifier(algorithm):

            """Predict labels with n fold cross validation"""
            y_pred = cross_val_predict(algorithm, X, y, cv=n_folds)

            """Calculate evaluation metrics"""
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            # TODO
            log_loss = logloss(y, y_pred)
            roc_auc = multiclass_roc_auc_score(y, y_pred, average='weighted')

            """Convert np array y_pred to pd series and add it to X"""
            y_pred = pd.Series(y_pred)
            X = pd.concat([X, y_pred], axis=1)
            X.columns = range(X.shape[1])

            return X, {'accuracy': accuracy,
                       'precision': precision,
                       'recall': recall,
                       'f1': f1,
                       'neg_log_loss': log_loss,
                       'roc_auc': roc_auc
                       }
        else:
            """
            If algorithm object has method fit_transform, call fit_transform on X, y. Else, first call fit on X, y,
            then transform on X. Safe the transformed dataset in X
            """
            if hasattr(algorithm, 'fit_transform'):
                X = algorithm.fit_transform(X, y)
            else:
                # noinspection PyUnresolvedReferences
                X = algorithm.fit(X, y).transform(X)

            X = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1]))

            return X, {}

    def save_algorithm(self, algorithm_id: Optional[int], res: Tuple[pd.DataFrame, Dict[str, float]]) -> None:
        """
        Update algorithm with the calculated evaluation metrics and save it to the database. Save transformed dataset if
        it does not equal the input dataset.
        """

        """Load input dataset and class_column. Drop class_column from input dataset."""
        input_df = self.dataset.load()
        class_column = self.dataset.class_column
        input_df, dataset_class_column = input_df.drop(class_column, axis=1), input_df[class_column]

        """Call complete_algorithm to save the algorithm to the database."""
        self.db.complete_algorithm(algorithm_id=algorithm_id, **res[1])
        LOGGER.info('Saved algorithm {}.'.format(algorithm_id))

        """Check if transformed dataset res[0] equals input dataset. If False store transformed dataset to DB"""
        try:
            assert_frame_equal(res[0], input_df)
            LOGGER.info('Transformed dataset equals input dataset {} and is not stored in the DB.'
                        .format(self.dataset.id))
        except AssertionError:
            LOGGER.info('Transformed dataset will be stored in DB.')
            new_dataset = pd.concat([res[0], dataset_class_column], axis=1)
            depth = self.dataset.depth
            depth += 1
            self.core.add_dataset(new_dataset, class_column, depth=depth, budget=self.dataset.budget)
        except Exception as ex:
            # TODO check this error. Should not happen
            # Traceback (most recent call last):
            #   File "/mnt/c/local/phd/code/meta-learning-base/worker.py", line 151, in save_algorithm
            #     assert_frame_equal(res[0], input_df)
            #   File "/usr/local/lib/python3.6/dist-packages/pandas/util/testing.py", line 1458, in assert_frame_equal
            #     '{shape!r}'.format(shape=right.shape))
            #   File "/usr/local/lib/python3.6/dist-packages/pandas/util/testing.py", line 1094, in raise_assert_detail
            #     raise AssertionError(msg)
            # AssertionError: DataFrame are different
            LOGGER.fatal('This should not happen!', ex)

    def is_dataset_finished(self):
        """
        Check if dataset is finished

        First is_dataset_finished checks if there are algorithms for this dataset in the database marked as pending or
        started. If there are none it returns False.

        Then is_dataset_finished checks if a dataset has enough budget for all the algorithms in the list.
        If the dataset has run out of budget, is_dataset_finished returns True.

        Last is_dataset_finished checks if a dataset has reached max pipeline depth. If a dataset has reached
        max pipeline depth, is_dataset_finished returns True.
        """
        algorithms = self.db.get_algorithms(dataset_id=self.dataset.id, ignore_complete=False)

        # Dataset has reached max pipeline depth
        if self.dataset.depth >= self.max_pipeline_depth:
            LOGGER.info('Dataset {} has reached max pipeline depth!'.format(self.dataset))
            return True

        # No algorithms for this data set started yet
        if not algorithms:
            return False

        # No budget for dataset
        n_completed = len(algorithms)
        if n_completed >= self.dataset.budget:
            LOGGER.info('Algorithm budget for dataset {} has run out!'.format(self.dataset))
            return True

        return False

    def run_algorithm(self):
        """
        First run_algorithm checks if is_dataset_finished returns True or False. If it returns True, the dataset is
        marked as complete. If is_dataset_finished returns False, run_algorithm creates a new Algorithm instance with
        a random Algorithm method. A random set of parameter configurations then is created for the Algorithms
        hyperparameter and stored in the new Algorithm instance.

        With the create_algorithm method, the new Algorithm instance then is stored in the database.

        As a last step the methods run_algorithm calls the methods transform_dataset and save_algorithm.
        """
        if self.is_dataset_finished():
            """
            Mark the run as done successfully and remove the completed dataset from local storage.
            """
            self.db.mark_dataset_complete(self.dataset.id)
            LOGGER.info('Dataset {} has been marked as complete.'.format(self.dataset))

            delete_data(self.dataset.train_path)
            return

        """
        Choose a random algorithm to work on the dataset
        """
        try:
            algo_type = random.choice(list(ALGORITHMS.keys()))
            LOGGER.info('Starting new algorithm \'{}\'...'.format(algo_type))
            algorithm = Algorithm(algo_type,
                                  dataset_id=self.dataset.id,
                                  status=AlgorithmStatus.RUNNING,
                                  start_time=datetime.now(),
                                  host=HOSTNAME)

            """Save a random configuration of the algorithms hyperparameters in params"""
            params = algorithm.random_config()
            algorithm.hyperparameter_values = params

            param_info = 'Chose parameters for algorithm :'.format(algorithm.hyperparameter_values)
            for k in sorted(params.keys()):
                param_info += '\n\t{} = {}'.format(k, params[k])
            LOGGER.debug(param_info)
        except Exception as ex:
            if isinstance(ex, KeyboardInterrupt):
                raise ex
            LOGGER.error('Failed to select hyperparameters', ex)
            raise AlgorithmError(str(ex), traceback.format_exc())

        """
        Create the algorithm and add it to the database
        """
        algorithm = self.db.create_algorithm(dataset_id=self.dataset.id,
                                             algorithm=algorithm)

        """
        Transform the dataset and save the algorithm
        """
        try:
            wrapper = pynisher2.enforce_limits(wall_time_in_s=self.timeout, logger=self.subprocess_logger)(
                self.transform_dataset)
            instance = algorithm.instance(params)
            res = wrapper(instance)
            if wrapper.exit_status is pynisher2.TimeoutException:
                raise TimeoutError('Timeout')
            elif wrapper.exit_status is pynisher2.MemorylimitException:
                raise MemoryError('MemoryLimit')
            elif wrapper.exit_status is pynisher2.AnythingException:
                # TODO res can be None
                # 2020-03-10 22:33:35,680 - 4324 - ERROR - worker - Unexpected error testing algorithm: dataset=<5bcfaa17-0bc1-4722-986f-2f579a901cda: 2.0000000000 classes, 145.0000000000 features, 2984.0000000000 rows>
                # Traceback (most recent call last):
                #   File "/mnt/c/local/phd/code/meta-learning-base/worker.py", line 266, in run_algorithm
                #     raise pynisher2.AnythingException(res[1])
                # TypeError: 'NoneType' object is not subscriptable
                raise AlgorithmError(res[0], res[1])
            elif wrapper.exit_status == 0 and res is not None:
                LOGGER.debug('Saving algorithm...')
                self.save_algorithm(algorithm.id, res)
            else:
                # TODO not sure how we ended in this state
                # [ERROR] [22:40:01:mlb] Unexpected error testing algorithm: dataset=<47676e99-12f3-40a4-9b9c-9efaa3c7f53e: 2.0000000000 classes, 145.0000000000 features, 2984.0000000000 rows>
                # Traceback (most recent call last):
                #   File "/mnt/c/local/phd/code/meta-learning-base/worker.py", line 271, in run_algorithm
                #     raise ValueError('Something went wrong transforming data set {}; {}'.format(res, wrapper))
                # ValueError: Something went wrong transforming data set None; <pynisher2.limit_function_call.enforce_limits.__call__.<locals>.function_wrapper object at 0x7f651ca5b588>
                raise ValueError('Something went wrong transforming data set {}; {}'.format(res, wrapper.exit_status))

        except KeyboardInterrupt:
            raise
        except (TimeoutError, MemoryError) as ex:
            LOGGER.info('Algorithm violated constraints: {}'.format(str(ex)))
            self.db.mark_algorithm_errored(algorithm.id, error_message=str(ex))
        except AlgorithmError as ex:
            LOGGER.info('Algorithm raised exception: {}'.format(ex.details))
            self.db.mark_algorithm_errored(algorithm.id, error_message=ex.details)
        except Exception:
            msg = traceback.format_exc()
            LOGGER.error('Unexpected error testing algorithm: dataset={}\n{}'.format(self.dataset, msg))
            self.db.mark_algorithm_errored(algorithm.id, error_message=msg)
