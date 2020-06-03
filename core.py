"""Core module.

This module contains the Core class, which is the one responsible for
executing and orchestrating the main Core functionalities.
"""
import logging
import os
import random
import re
import shutil
import signal
import sys
import time
import uuid
from io import StringIO
from operator import attrgetter
from pathlib import Path
from typing import List

import math
import pandas as pd
import psutil
from pympler import muppy, summary, refbrowser
from tqdm import tqdm

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

            # Generic Conf
            work_dir: str = None,
            timeout: int = None,
            cache_percentage: float = 0.99,
            dataset_budget: int = None,
            max_pipeline_depth: int = 5,

            # Worker Conf
            complete_pipelines: bool = False,
            complete_pipeline_samples: int = 20,
            affinity: bool = False,

            # S3 Conf
            service_account: str = None,
            bucket: str = None,

            # Log Conf
            verbose_metrics: bool = False,
    ):
        self.metafeatures = MetaFeatures()
        self.db = Database(dialect, database, username, password, host, port, query)
        self.work_dir = work_dir
        self.timeout = timeout
        self.dataset_budget = dataset_budget

        self.complete_pipelines = complete_pipelines
        self.complete_pipeline_samples = complete_pipeline_samples
        self.max_pipeline_depth = max_pipeline_depth
        self.affinity = affinity

        self.s3_config: str = service_account
        self.s3_bucket: str = bucket

        self.verbose_metrics: bool = verbose_metrics
        self._abort = False

        LOGGER.info('Scanning cache dir. This may take some while...')
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.cache_total, self.cache_used, free = shutil.disk_usage(self.work_dir)
        self.cache_percentage = cache_percentage

    def add_dataset(self, df: pd.DataFrame, class_column: str, depth: int, budget: int = None, name: str = None):
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

            budget (int):
        """

        """Generate name using a random uuid, if input dataset has no name"""
        if not name or name.strip() == '':
            name = str(uuid.uuid4())
        LOGGER.info('Creating dataset {}'.format(name))

        """Stores input dataset to local working directory"""
        local_file = self._cache_locally(df, name)

        """Check if new dataset equals existing dataset. If False store transformed dataset to DB"""
        hashcode = hash_file(local_file)
        similar_datasets: List[Dataset] = self.db.get_datasets_by_hash(hashcode)
        for ds in similar_datasets:
            df_old = ds.load(self.s3_config, self.s3_bucket)
            if df.equals(df_old):
                LOGGER.info('New dataset equals dataset {} and is not stored in the DB.'.format(ds.id))
                return ds
            del df_old

        """Uploads input dataset to cloud"""
        upload_data(local_file, self.s3_config, self.s3_bucket, name)

        """Calculates metafeatures for input dataset"""
        try:
            LOGGER.info('Extracting meta-features...')
            mf, success = self.metafeatures.calculate(df=df, class_column=class_column)

            for key, value in mf.items():
                if math.isinf(value):
                    LOGGER.info(
                        'Value of Meta Feature "{}" is infinite and is replaced by constant value'.format(key))
                    if value > 0:
                        mf[key] = sys.maxsize
                    else:
                        mf[key] = -sys.maxsize
        except ValueError as ex:
            LOGGER.exception('Failed to compute meta-features. Fallback to empty meta-features', ex)
            mf, success = {}, False

        if not success:
            LOGGER.info('Meta-feature extraction failed. Marking this dataset as \'skipped\'')
            mf['status'] = RunStatus.SKIPPED

        """Saves input dataset and calculated meta-features to db"""
        if budget is None:
            budget = self.dataset_budget

        return self.db.create_dataset(
            train_path=local_file,
            name=name,
            class_column=class_column,
            depth=depth,
            budget=budget,
            hashcode=hashcode,
            **mf
        )

    def _cache_locally(self, df: pd.DataFrame, name: str) -> str:
        def clean_cache():
            LOGGER.info('Cleaning cache. This may take some while...')
            # For complete local execution do not clean cache!
            exit(-1)

            shutil.rmtree(self.work_dir)
            Path(self.work_dir).mkdir(parents=True, exist_ok=True)

            self.cache_total, self.cache_used, free = shutil.disk_usage(self.work_dir)
            LOGGER.info('Deleted local cache. Using {} of cache'.format(self.cache_used / self.cache_total))

        try:
            local_file = store_data(df, self.work_dir, name)
        except IOError:
            clean_cache()
            local_file = store_data(df, self.work_dir, name)

        self.cache_used += os.stat(local_file).st_size
        if self.cache_used / self.cache_total > self.cache_percentage:
            clean_cache()

        return local_file

    def _user_abort(self):
        LOGGER.info('Received abort signal. Stopping processing after current evaluation...')
        self._abort = True

    def work(self, use_defaults=True, choose_randomly=True, wait=True, verbose=False):
        """Get unfinished Datasets from the database and work on them.

        Args:
            use_defaults (bool):
                <MISSING>
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
        signal.signal(signal.SIGUSR1, lambda s, frame: self._user_abort())

        # ##########################################################################
        # #  Main Loop  ############################################################
        # ##########################################################################

        failure_counter = 0

        # Count number of running workers
        pids = set()
        core = None
        if self.affinity:
            for p in psutil.process_iter():
                if re.match('.*python\\d?', p.name()) and 'worker' in p.cmdline() and \
                        len([arg for arg in p.cmdline() if arg.endswith('cli.py')]) > 0:
                    if p.parent() is None or p.parent().pid not in pids:
                        pids.add(p.pid)
            core = {len(pids) - 1 % os.cpu_count()}
            LOGGER.info('Setting affinity to {}'.format(core))

        while True:
            if self._abort:
                LOGGER.info("Stopping processing due to user request")
                break

            ds = None
            if use_defaults:
                ds = self.db.select_dataset()
            else:
                # Get all pending and running datasets, or all pending/running datasets from the list we were given
                datasets = self.db.get_datasets()
                if len(datasets) > 0:
                    # Either choose a dataset randomly between priority, or take the dataset with the lowest ID"""
                    if choose_randomly:
                        ds = random.choice(datasets)
                    else:
                        ds = sorted(datasets, key=attrgetter('id'))[0]
                    del datasets
                    try:
                        self.db.mark_dataset_running(ds.id)
                    except UserWarning:
                        LOGGER.warning('Skipping completed dataset: {}'.format(ds.id))

            if not ds:
                if wait:
                    LOGGER.debug('No datasets found. Sleeping %d seconds and trying again.', self._LOOP_WAIT)
                    time.sleep(self._LOOP_WAIT)
                    continue
                else:
                    LOGGER.info('No datasets found. Exiting.')
                    break
            LOGGER.info('Computing on dataset {}'.format(ds.id))
            worker = None

            """
            Progress bar
            """
            try:
                pbar = tqdm(total=ds.budget, ascii=True, initial=ds.processed, disable=not verbose)

                """Creates Worker"""
                worker = Worker(self.db, ds, self, timeout=self.timeout,
                                s3_config=self.s3_config, s3_bucket=self.s3_bucket,
                                complete_pipelines=self.complete_pipelines,
                                complete_pipeline_samples=self.complete_pipeline_samples,
                                max_pipeline_depth=self.max_pipeline_depth,
                                affinity=core,
                                verbose_metrics=self.verbose_metrics)

                """Call run_algorithm as long as the chosen dataset is marked as RUNNING"""
                while ds.status == RunStatus.RUNNING:
                    if use_defaults:
                        worker.run_default()
                        self.db.mark_dataset_complete(ds.id)
                        break

                    success = worker.run_algorithm()
                    ds = self.db.get_dataset(ds.id)
                    if verbose and ds.processed > pbar.last_print_n:
                        pbar.update(ds.processed - pbar.last_print_n)

                    # Safety valve to abort execution if something is broken
                    if success is False:
                        LOGGER.error('Something went wrong. Sleeping {} seconds.'.format(self._LOOP_WAIT))
                        time.sleep(self._LOOP_WAIT)
                        failure_counter += 1
                        if failure_counter > 10:
                            LOGGER.fatal('Received 10 consecutive unexpected exceptions. Aborting evaluation.')

                            # We occasionally encounter OSError: [Errno 12] Cannot allocate memory. To debug the memory
                            # leak the current heap allocation is logged
                            all_objects = muppy.get_objects()
                            LOGGER.fatal('Heap Dump:\n' + '\n'.join(summary.format_(summary.summarize(all_objects))))

                            buffer = StringIO()
                            cb = refbrowser.StreamBrowser(self, maxdepth=4, str_func=lambda o: str(type(o)),
                                                          stream=buffer)
                            cb.print_tree()
                            LOGGER.fatal('References:\n' + buffer.getvalue())

                            sys.exit(1)
                    else:
                        failure_counter = 0

                pbar.close()
            except AlgorithmError:
                """ 
                The exception has already been handled; just wait a sec so we don't go out of control reporting errors
                """
                LOGGER.error('Something went wrong. Sleeping {} seconds.'.format(self._LOOP_WAIT))
                time.sleep(self._LOOP_WAIT)
            finally:
                del worker

    def export_db(self):
        return self.db.export_db()
