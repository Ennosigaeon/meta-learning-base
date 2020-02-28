import logging
import random
import socket
import traceback
import warnings
from builtins import object, str
from datetime import datetime
from typing import Optional, Tuple, Dict, TYPE_CHECKING

import pandas as pd
from sklearn.base import BaseEstimator, is_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict

from constants import AlgorithmStatus
from database import Database, Algorithm
from methods import ALGORITHMS
from utilities import ensure_directory, multiclass_roc_auc_score, logloss

if TYPE_CHECKING:
    from core import Core

warnings.filterwarnings('ignore')

LOGGER = logging.getLogger('mlb')
HOSTNAME = socket.gethostname()


# Exception thrown when something goes wrong for the worker, but the worker handles the error.
class AlgorithmError(Exception):
    pass


class Worker(object):
    def __init__(self,
                 database: Database,
                 dataset,
                 core,
                 cloud_mode: bool = False,
                 s3_access_key: str = None,
                 s3_secret_key: str = None,
                 s3_bucket: str = None,
                 models_dir: str = 'models',
                 metrics_dir: str = 'metrics',

                 max_pipeline_depth: int = 5,
                 verbose_metrics: bool = False):

        self.db = database
        self.dataset = dataset
        self.core: Core = core
        self.cloud_mode = cloud_mode

        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_bucket = s3_bucket

        self.models_dir = models_dir
        self.metrics_dir = metrics_dir

        self.max_pipeline_depth = max_pipeline_depth
        self.verbose_metrics = verbose_metrics
        ensure_directory(self.models_dir)
        ensure_directory(self.metrics_dir)

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
        df = self.dataset.load()
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

            """Predict labels with 5 fold cross validation"""
            y_pred = cross_val_predict(algorithm, X, y, cv=n_folds)

            # TODO switch/if else ob multiclass oder nicht
            # --> multiclass
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='macro')
            recall = recall_score(y, y_pred, average='macro')
            f1 = f1_score(y, y_pred, average='macro')
            log_loss = logloss(y, y_pred)
            roc_auc = multiclass_roc_auc_score(y, y_pred, average='macro')

            # --> not multiclass

            """Convert np array y_pred to pd series and add it to X"""
            y_pred = pd.Series(y_pred)
            X = pd.concat([X, y_pred], axis=1)

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

        # TODO use is_close or pandas equivalent
        # --> gibt bei pandas nur .equals und .assert_frame_equal

        """Check if transformed dataset res[0] equals input dataset. If False store transformed dataset to DB"""
        if res[0].equals(input_df):
            LOGGER.info('Transformed dataset equals input dataset {} and is not stored in the DB.'
                        .format(self.dataset.id))

        else:
            """Load class_column and join transformed dataset with removed class_column of the input dataset.
            Add transformed dataset to DB"""
            new_dataset = pd.concat([res[0], dataset_class_column], axis=1)
            depth = self.dataset.depth
            depth += 1
            self.core.add_dataset(new_dataset, class_column, depth=depth)
            LOGGER.info('Transformed dataset will be stored in DB.')

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
            Mark the run as done successfully
            """
            self.db.mark_dataset_complete(self.dataset.id)
            LOGGER.info('Dataset {} has been marked as complete.'.format(self.dataset))
            return

        """
        Choose a random algorithm to work on the dataset
        """
        try:
            LOGGER.info('Starting new algorithm...')
            algorithm = Algorithm(random.choice(list(ALGORITHMS.keys())),
                                  dataset_id=self.dataset.id,
                                  status=AlgorithmStatus.RUNNING,
                                  start_time=datetime.now())

            """Save a random configuration of the algorithms hyperparameters in params"""
            params = algorithm.random_config()
            algorithm.hyperparameter_values = params

            param_info = 'Chose parameters for algorithm "{}":'.format(algorithm.hyperparameter_values)
            for k in sorted(params.keys()):
                param_info += '\n\t{} = {}'.format(k, params[k])
            LOGGER.debug(param_info)
        except Exception as ex:
            if isinstance(ex, KeyboardInterrupt):
                raise ex
            LOGGER.error('Error choosing algorithm with hyperparameters for dataset {}'.format(self.dataset))
            LOGGER.error(traceback.format_exc())
            raise AlgorithmError()

        """
        Create the algorithm and add it to the database
        """
        algorithm = self.db.create_algorithm(dataset_id=self.dataset.id,
                                             host=HOSTNAME,
                                             algorithm=algorithm,
                                             start_time=algorithm.start_time,
                                             status=algorithm.status)

        """
        Transform the dataset and save the algorithm
        """
        try:
            LOGGER.debug('Testing algorithm...')
            res = self.transform_dataset(algorithm.instance(params))

            LOGGER.debug('Saving algorithm...')
            self.save_algorithm(algorithm.id, res)

        except Exception as ex:
            if isinstance(ex, KeyboardInterrupt):
                raise ex
            msg = traceback.format_exc()
            LOGGER.error('Error testing algorithm: dataset={}'.format(self.dataset))
            LOGGER.error(msg)
            self.db.mark_algorithm_errored(algorithm.id, error_message=msg)
