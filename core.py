"""Core module.

This module contains the Core class, which is the one responsible for
executing and orchestrating the main Core functionalities.
"""

# TODO two different loggers are used. Only use one
# TODO Logging verbessern. Es ist nur aus dem log schwer zu erkennen was passiert
#   - Warning ausblenden
#   - StackTraces für Fehler in Workern ausblenden

import logging
import random

import signal
import time
import uuid
from operator import attrgetter

import pandas as pd
from tqdm import tqdm
from typing import List

from constants import RunStatus
from data import store_data, upload_data
from database import Database, Dataset
from metafeatures import MetaFeatures
from utilities import hash_file
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
            timeout: int = None,

            # S3 Conf
            endpoint: str = None,
            bucket: str = None,
            access_key: str = None,
            secret_key: str = None,

            # Log Conf
            verbose_metrics: bool = False,
    ):
        self.metafeatures = MetaFeatures()
        self.db = Database(dialect, database, username, password, host, port, query)
        self.work_dir = work_dir
        self.timeout = timeout
        self.s3_endpoint: str = endpoint
        self.s3_bucket: str = bucket
        self.s3_access_key: str = access_key
        self.s3_secret_key: str = secret_key

        self.verbose_metrics: bool = verbose_metrics
        self._abort = False

    def add_dataset(self, df: pd.DataFrame, class_column: str, depth: int, name: str = None):
        """Add a new dataset to the Database.
        Args:
            df (DataFrame):
                The input dataset.

            class_column (str):
                The class column of the input dataset which is to be predicted.

            name (str):
                Name given to this dataset. Optional. If not given, a random uuid will be
                generated from the training_path and used as the dataset name.

            depth (int):
                The max pipeline depth a dataset can reach.

        """

        """Generate name using a random uuid, if input dataset has no name"""
        if not name or name.strip() == '':
            name = str(uuid.uuid4())

        """Stores input dataset to local working directory"""
        local_file = store_data(df, self.work_dir, name)

        """Check if new dataset equals existing dataset. If False store transformed dataset to DB"""
        hashcode = hash_file(local_file)
        similar_datasets: List[Dataset] = self.db.get_dataset_by_hash(hashcode)
        for ds in similar_datasets:
            df_old = ds.load(self.s3_endpoint, self.s3_bucket, self.s3_access_key, self.s3_secret_key)
            if df.equals(df_old):
                LOGGER.info('New dataset equals dataset {} and is not stored in the DB.'.format(ds.id))
                return ds

        """Uploads input dataset to cloud"""
        upload_data(local_file, self.s3_endpoint, self.s3_bucket, self.s3_access_key, self.s3_secret_key, name)

        """Calculates metafeatures for input dataset"""
        try:
            mf = self.metafeatures.calculate(df=df, class_column=class_column)
        except ValueError as ex:
            LOGGER.exception('Failed to compute meta-features. Fallback to empty meta-features', ex)
            mf = {}
        LOGGER.info('Extracted Metafeatures')
        """Saves input dataset and calculated metafeatures to db"""
        LOGGER.info('Saving {}'.format(local_file))
        return self.db.create_dataset(
            train_path=local_file,
            name=name,
            class_column=class_column,
            depth=depth,
            hashcode=hashcode,
            **mf
        )

    def _user_abort(self):
        LOGGER.info('Received abort signal. Stopping processing after current evaluation...')
        self._abort = True

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
        signal.signal(signal.SIGUSR1, lambda signal, frame: self._user_abort())

        # ##########################################################################
        # #  Main Loop  ############################################################
        # ##########################################################################

        while True:
            if self._abort:
                LOGGER.info("Stopping processing due to user request")
                break

            """
            Get all pending and running datasets, or all pending/running datasets from the list we were given
            """
            datasets = self.db.get_datasets()
            if not datasets:
                if wait:
                    LOGGER.debug('No datasets found. Sleeping %d seconds and trying again.', self._LOOP_WAIT)
                    time.sleep(self._LOOP_WAIT)
                    continue

                else:
                    LOGGER.info('No datasets found. Exiting.')
                    break

            """Take all RUNNING datasets as first priority, then take PENDING"""
            candidates = [d for d in datasets if d.status == RunStatus.RUNNING]
            if len(candidates) == 0:
                candidates = datasets

            """Either choose a dataset randomly between priority, or take the dataset with the lowest ID"""
            if choose_randomly:
                ds = random.choice(candidates)
            else:
                ds = sorted(candidates, key=attrgetter('id'))[0]

            """
            Mark dataset as RUNNING
            """
            try:
                self.db.mark_dataset_running(ds.id)
            except UserWarning:
                LOGGER.warning('Skipping completed dataset: {}'.format(ds.id))

            LOGGER.info('Computing on dataset {}'.format(ds.id))

            """
            Progress bar
            """
            try:
                pbar = tqdm(total=ds.budget, ascii=True, initial=ds.processed, disable=not verbose)

                """Creates Worker"""
                worker = Worker(self.db, ds, self, timeout=self.timeout, s3_endpoint=self.s3_endpoint,
                                s3_access_key=self.s3_access_key, s3_secret_key=self.s3_secret_key,
                                s3_bucket=self.s3_bucket, verbose_metrics=self.verbose_metrics)

                """Call run_algorithm as long as the chosen dataset is marked as RUNNING"""
                while ds.status == RunStatus.RUNNING:
                    worker.run_algorithm()
                    ds = self.db.get_dataset(ds.id)
                    if verbose and ds.processed > pbar.last_print_n:
                        pbar.update(ds.processed - pbar.last_print_n)

                pbar.close()
            except AlgorithmError:
                """ 
                The exception has already been handled; just wait a sec so we don't go out of control reporting errors
                """
                LOGGER.error('Something went wrong. Sleeping {} seconds.'.format(self._LOOP_WAIT))
                time.sleep(self._LOOP_WAIT)
