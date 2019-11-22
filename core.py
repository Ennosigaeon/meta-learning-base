"""Core module.

This module contains the Core class, which is the one responsible for
executing and orchestrating the main Core functionalities.
"""

import logging
import random
import time
from operator import attrgetter

from data import store_data
from database import Database
from tqdm import tqdm

from constants import RunStatus
from worker import AlgorithmError, Worker

LOGGER = logging.getLogger(__name__)


class Core(object):
    _LOOP_WAIT = 5

    def __init__(
            self,

            # SQL Conf
            dialect: str = 'sqlite',
            database: str = 'assets/ml-base.db',
            username: str = None,
            password: str = None,
            host=None,
            port=None,
            query=None,

            # S3 Conf
            endpoint: str = None,
            bucket: str = None,
            access_key: str = None,
            secret_key: str = None,

            # Log Conf
            models_dir: str = 'models',
            metrics_dir: str = 'metrics',
            verbose_metrics: bool = False,
    ):
        self.db = Database(dialect, database, username, password, host, port, query)
        self.s3_endpoint: str = endpoint
        self.s3_bucket: str = bucket
        self.s3_access_key: str = access_key
        self.s3_secret_key: str = secret_key

        self.models_dir: str = models_dir
        self.metrics_dir: str = metrics_dir
        self.verbose_metrics: bool = verbose_metrics

    def add_dataset(self, train_path, test_path=None, name=None,
                    description=None, class_column=None):
        """Add a new dataset to the Database.

        Args:
            train_path (str):
                Path to the training CSV file. It can be a local filesystem path,
                absolute or relative, or an HTTP or HTTPS URL, or an S3 path in the
                format ``s3://{bucket_name}/{key}``. Required.
            test_path (str):
                Path to the testing CSV file. It can be a local filesystem path,
                absolute or relative, or an HTTP or HTTPS URL, or an S3 path in the
                format ``s3://{bucket_name}/{key}``.
                Optional. If not given, the training CSV will be split in two parts,
                train and test.
            name (str):
                Name given to this dataset. Optional. If not given, a hash will be
                generated from the training_path and used as the Dataset name.
            description (str):
                Human friendly description of the Dataset. Optional.
            class_column (str):
                Name of the column that will be used as the target variable.
                Optional. Defaults to ``'class'``.

        Returns:
            Dataset:
                The created dataset.
        """

        # TODO calculate meta features
        store_data(train_path, self.s3_endpoint, self.s3_bucket, self.s3_access_key, self.s3_secret_key)

        return self.db.create_dataset(
            train_path=train_path,
            test_path=test_path,
            name=name,
            description=description,
            class_column=class_column,
            aws_access_key=self.s3_access_key,
            aws_secret_key=self.s3_secret_key,
        )

    def work(self, choose_randomly=True, wait=True, verbose=False):
        """Get unfinished Datasets from the database and work on them.

        Args:
            choose_randomly (bool):
                If ``True``, work on all the highest-priority datasets in random order.
                Otherwise, work on them in sequential order (by ID).
                Optional. Defaults to ``True``.
            wait (bool):
                If ``True``, wait for more Dataruns to be inserted into the Database
                once all have been processed. Otherwise, exit the worker loop
                when they ds out.
                Optional. Defaults to ``False``.
            verbose (bool):
                Whether to be verbose about the process. Optional. Defaults to ``True``.
        """
        # main loop
        while True:
            # get all pending and running datasets, or all pending/running datasets
            # from the list we were given
            datasets = self.db.get_datasets(ignore_complete=True)
            if not datasets:
                if wait:
                    LOGGER.debug('No datasets found. Sleeping %d seconds and trying again.',
                                 self._LOOP_WAIT)
                    time.sleep(self._LOOP_WAIT)
                    continue

                else:
                    LOGGER.info('No datasets found. Exiting.')
                    break

            # either choose a ds randomly between priority, or take the ds with the lowest ID
            if choose_randomly:
                ds = random.choice(datasets)
            else:
                ds = sorted(datasets, key=attrgetter('id'))[0]

            # say we've started working on this dataset, if we haven't already
            self.db.mark_dataset_running(ds.id)

            LOGGER.info('Computing on datarun %d' % ds.id)
            # actual work happens here
            worker = Worker(self.db, ds, s3_access_key=self.s3_access_key,
                            s3_secret_key=self.s3_secret_key, s3_bucket=self.s3_bucket, models_dir=self.models_dir,
                            metrics_dir=self.metrics_dir, verbose_metrics=self.verbose_metrics)

            try:
                pbar = tqdm(
                    total=ds.budget,
                    ascii=True,
                    initial=ds.completed_algorithm,
                    disable=not verbose
                )

                while ds.status != RunStatus.COMPLETE:
                    worker.run_algorithm()
                    ds = self.db.get_dataset(ds.id)
                    if verbose and ds.completed_algorithm > pbar.last_print_n:
                        pbar.update(ds.completed_algorithm - pbar.last_print_n)

                pbar.close()
            except AlgorithmError:
                # the exception has already been handled; just wait a sec so we
                # don't go out of control reporting errors
                LOGGER.error('Something went wrong. Sleeping %d seconds.', self._LOOP_WAIT)
                time.sleep(self._LOOP_WAIT)
