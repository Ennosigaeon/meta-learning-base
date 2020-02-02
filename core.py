"""Core module.

This module contains the Core class, which is the one responsible for
executing and orchestrating the main Core functionalities.
"""

import logging
import random
import time
import uuid
from operator import attrgetter

import pandas as pd
from tqdm import tqdm

from constants import RunStatus
from data import store_data
from database import Database
from metafeatures import MetaFeatures
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
            host: str = None,
            port: int = None,
            query=None,

            # Generic Conf,
            work_dir: str = None,

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
        self.metafeatures = MetaFeatures()
        self.db = Database(dialect, database, username, password, host, port, query)
        self.work_dir = work_dir
        self.s3_endpoint: str = endpoint
        self.s3_bucket: str = bucket
        self.s3_access_key: str = access_key
        self.s3_secret_key: str = secret_key

        self.models_dir: str = models_dir
        self.metrics_dir: str = metrics_dir
        self.verbose_metrics: bool = verbose_metrics

    def add_dataset(self, df: pd.DataFrame, class_column: str, depth: int, name: str = None):
        """Add a new dataset to the Database.
        Args:
            df (DataFrame):

            class_column (str):

            name (str):
                Name given to this dataset. Optional. If not given, a hash will be
                generated from the training_path and used as the Dataset name.

            depth (int):

        """

        if not name or name.strip() == '':
            name = str(uuid.uuid4())

        local_file = store_data(df, self.work_dir, name)
        # TODO upload data to S3 bucket via data.upload_data(...)
        # upload_data(train_path, self.s3_endpoint, self.s3_bucket, self.s3_access_key, self.s3_secret_key)

        # 3. Metafeatures berechnen
        mf = self.metafeatures.calculate(df=df, class_column=class_column)

        # 4. Speichern in Datenbank
        return self.db.create_dataset(
            train_path=local_file,
            name=name,
            class_column=class_column,
            depth=depth,
            **mf
        )

    def add_algorithm(self, ds_id: int, algorithm: str):
        """Add a new algorithm to the Database.
        Args:
            ds_id:

            algorithm:


        Returns:
            Algorithm:
                The created algorithm.
        """
        return self.db.start_algorithm(
            dataset_id=ds_id,
            algorithm=algorithm
        )

    def work(self, choose_randomly=True, wait=True, verbose=False):
        """Get unfinished Datasets from the database and work on them.

        Args:
            choose_randomly (bool):
                If ``True``, work on all the highest-priority datasets in random order.
                Otherwise, work on them in sequential order (by ID).
                Optional. Defaults to ``True``.
            wait (bool):
                If ``True``, wait for more datasets to be inserted into the Database
                once all have been processed. Otherwise, exit the worker loop
                when they ds out.
                Optional. Defaults to ``False``.
            verbose (bool):
                Whether to be verbose about the process. Optional. Defaults to ``True``.
        """

        # ##########################################################################
        # #  Main Loop  ############################################################
        # ##########################################################################

        while True:
            """
            Get all pending and running datasets, or all pending/running datasets from the list we were given
            """
            datasets = self.db.get_datasets()
            if not datasets:
                if wait:
                    LOGGER.debug('No datasets found. Sleeping %d seconds and trying again.',
                                 self._LOOP_WAIT)
                    time.sleep(self._LOOP_WAIT)
                    continue

                else:
                    LOGGER.info('No datasets found. Exiting.')
                    break

            """
            Either choose a dataset randomly between priority, or take the dataset with the lowest ID
            """
            if choose_randomly:
                ds = random.choice(datasets)
            else:
                ds = sorted(datasets, key=attrgetter('id'))[0]

            """
            Mark dataset as RUNNING
            """
            # noinspection PyTypeChecker
            self.db.mark_dataset_running(ds.id)

            LOGGER.info('Computing on dataset {}'.format(ds.id))

            """>> Actual work happens here <<
            
            Create instance of Worker
            """
            worker = Worker(self.db, ds, self, s3_access_key=self.s3_access_key,
                            s3_secret_key=self.s3_secret_key, s3_bucket=self.s3_bucket, models_dir=self.models_dir,
                            metrics_dir=self.metrics_dir, verbose_metrics=self.verbose_metrics)

            """
            Progress bar
            """
            try:
                pbar = tqdm(
                    total=ds.budget,
                    ascii=True,
                    initial=ds.processed,
                    disable=not verbose
                )

                """
                As long as there are datasets which aren't marked as completed yet -> run_algorithm
                """
                while ds.status != RunStatus.COMPLETE:
                    worker.run_algorithm()
                    ds = self.db.get_dataset(ds.id)
                    if verbose and ds.completed_algorithm > pbar.last_print_n:
                        pbar.update(ds.completed_algorithm - pbar.last_print_n)

                pbar.close()

            except AlgorithmError:
                """ 
                The exception has already been handled; just wait a sec so we
                don't go out of control reporting errors
                """
                LOGGER.error('Something went wrong. Sleeping %d seconds.', self._LOOP_WAIT)
                time.sleep(self._LOOP_WAIT)
