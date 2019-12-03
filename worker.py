import logging
import random
import socket
import traceback
import warnings
from builtins import object, str
from typing import Union, Any, Optional

from sklearn.base import BaseEstimator

from database import Database
from methods import ALGORITHMS
from utilities import ensure_directory

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
                 cloud_mode: bool = False,
                 s3_access_key: str = None,
                 s3_secret_key: str = None,
                 s3_bucket: str = None,
                 models_dir: str = 'models',
                 metrics_dir: str = 'metrics',
                 verbose_metrics: bool = False):

        self.db = database
        self.dataset = dataset
        self.cloud_mode = cloud_mode

        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_bucket = s3_bucket

        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.verbose_metrics = verbose_metrics
        ensure_directory(self.models_dir)
        ensure_directory(self.metrics_dir)

        # load the Dataset from the database
        self.dataset = self.db.get_dataset(self.dataset.dataset_id)

    def transform_dataset(self, algorithm: BaseEstimator) -> Union[str, Any]:
        """
        Given a set of fully-qualified hyperparameters, create and test a
        algorithm model.
        Returns: Model object and metrics dictionary
        """

        train_path, test_path = self.dataset.load()

        # TODO Create algorithm instance from algorithm + params
        # If classifier:
        #   - Cross-Validation to fit model
        #   - Calculate averaged metrics
        # If transformer:
        #   - Transform dataset
        #   - Store transformed dataset in tmp directory

        return ''

    def save_algorithm(self, algorithm_id: Optional[int], res: Union[str, Any]) -> None:
        """
        Update a algorithm with metrics and model information and mark it as
        "complete"

        algorithm_id: ID of the algorithm to save

        model: Model object containing a serializable representation of the
            final model generated by this algorithm.

        metrics: Dictionary containing cross-validation and test metrics data
            for the model.
        """

        # TODO store algorithm with either metrics or new transformed dataset in database
        if isinstance(res, str):
            # TODO add new dataset
            pass
        else:
            # TODO store metrics
            pass

        # update the classifier in the database
        self.db.complete_algorithm(algorithm_id=algorithm_id)

        LOGGER.info('Saved algorithm %d.' % algorithm_id)

    def is_dataset_finished(self):
        algorithms = self.db.get_algorithms(dataset_id=self.dataset.id)
        if not algorithms:
            LOGGER.warning('No incomplete algorithms for dataset %d present in database.'
                           % self.dataset.id)
            return True

        n_completed = len(algorithms)
        if n_completed >= self.dataset.budget:
            LOGGER.warning('Algorithm budget has run out!')
            return True

        return False

    def run_algorithm(self):
        # check to see if our work is done
        if self.is_dataset_finished():
            # marked the run as done successfully
            self.db.mark_dataset_complete(self.dataset.id)
            LOGGER.warning('Dataset %d has ended.' % self.dataset.id)
            return

        try:
            LOGGER.debug('Choosing algorithm...')
            algorithm = random.choice(ALGORITHMS)
            params = algorithm.random_config()

        except Exception:
            LOGGER.error('Error choosing hyperparameters: datarun=%s' % str(self.dataset))
            LOGGER.error(traceback.format_exc())
            raise AlgorithmError()

        param_info = 'Chose parameters for algorithm "%s":' % algorithm.class_path
        for k in sorted(params.keys()):
            param_info += '\n\t%s = %s' % (k, params[k])
        LOGGER.info(param_info)

        LOGGER.debug('Creating algorithm...')
        algorithm = self.db.start_algorithm(dataset_id=self.dataset.id,
                                            host=HOSTNAME,
                                            algorithm=algorithm.class_path,
                                            hyperparameter_values=params)

        try:
            LOGGER.debug('Testing algorithm...')
            res = self.transform_dataset(algorithm.instance(params))

            LOGGER.debug('Saving algorithm...')
            self.save_algorithm(algorithm.id, res)

        except Exception:
            msg = traceback.format_exc()
            LOGGER.error('Error testing algorithm: datarun=%s' % str(self.dataset))
            LOGGER.error(msg)
            self.db.mark_algorithm_errored(algorithm.id, error_message=msg)
            raise AlgorithmError()
