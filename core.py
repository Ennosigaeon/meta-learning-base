"""Core module.

This module contains the Core class, which is the one responsible for
executing and orchestrating the main Core functionalities.
"""

import logging
import random
import time
from collections import defaultdict
from operator import attrgetter

import numpy as np
from autosklearn.metalearning.metafeatures.metafeature import MetaFeature, HelperFunction
import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse
from pymfe.mfe import MFE

from data import store_data, load_data
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

    def add_dataset(self, train_path, test_path=None, reference_path=None, name=None):
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
            reference_path (str):
                Path to the referring dataset CSV file. It can be a local filesystem path,
                absolute or relative, or an HTTP or HTTPS URL, or an S3 path in the
                format ``s3://{bucket_name}/{key}``.
                Optional.
            name (str):
                Name given to this dataset. Optional. If not given, a hash will be
                generated from the training_path and used as the Dataset name.



        Returns:
            Dataset:
                The created dataset.
        """

        # store_data(train_path, self.s3_endpoint, self.s3_bucket, self.s3_access_key, self.s3_secret_key)

        # ##########################################################################
        # #  Extracting Meta Features with pymfe  ##################################
        # ##########################################################################

        # Loading train_path
        df = load_data(train_path)
        X, y = df.drop('class', axis=1), df['class']

        # Selecting Meta Features and extracting them
        mfe = MFE(
            features=(['nr_inst', 'nr_attr', 'nr_class', 'nr_outliers', 'skewness', 'kurtosis', 'cor', 'cov',
                       'attr_conc', 'sparsity', 'gravity', 'var', 'class_ent', 'attr_ent', 'mut_inf', 'eq_num_attr',
                       'ns_ratio', 'nodes', 'leaves', 'leaves_branch', 'nodes_per_attr', 'var_importance', 'one_nn',
                       'best_node', 'best_random', 'best_worst', 'linear_discr', 'naive_bayes', 'leaves_per_class']),
            random_state=42
        )
        # noinspection PyTypeChecker
        mfe.fit(X.values, y.values)
        ft = mfe.extract(cat_cols='auto', suppress_warnings=True)

        # Mapping values to Meta Feature variables
        nr_inst = ft[1][30]
        nr_attr = ft[1][28]
        nr_class = ft[1][29]
        nr_outliers = ft[1][31]

        skewness_mean = ft[1][35]
        skewness_sd = ft[1][36]
        kurtosis_mean = ft[1][13]
        kurtosis_sd = ft[1][14]
        cor_mean = ft[1][7]
        cor_sd = ft[1][8]
        cov_mean = ft[1][9]
        cov_sd = ft[1][10]
        attr_conc_mean = ft[1][0]
        attr_conc_sd = ft[1][1]
        sparsity_mean = ft[1][37]
        sparsity_sd = ft[1][38]
        gravity = ft[1][12]
        var_mean = ft[1][39]
        var_sd = ft[1][40]

        class_ent = ft[1][6]
        attr_ent_mean = ft[1][2]
        attr_ent_sd = ft[1][3]
        mut_inf_mean = ft[1][22]
        mut_inf_sd = ft[1][23]
        eq_num_attr = ft[1][11]
        ns_ratio = ft[1][32]

        nodes = ft[1][26]
        leaves = ft[1][15]
        leaves_branch_mean = ft[1][16]
        leaves_branch_sd = ft[1][17]
        nodes_per_attr = ft[1][27]
        leaves_per_class_mean = ft[1][18]
        leaves_per_class_sd = ft[1][19]
        var_importance_mean = ft[1][41]
        var_importance_sd = ft[1][42]

        one_nn_mean = ft[1][33]
        one_nn_sd = ft[1][34]
        best_node_mean = ft[1][4]
        best_node_sd = ft[1][5]
        # best_random = ft[1][0]
        # best_worst = ft[1][0]
        linear_discr_mean = ft[1][20]
        linear_discr_sd = ft[1][21]
        naive_bayes_mean = ft[1][24]
        naive_bayes_sd = ft[1][25]

        # ##########################################################################
        # #  Extracting Meta Features with Auto-Sklearn  ###########################
        # ##########################################################################

        class NumberOfMissingValues(MetaFeature):
            def _calculate(self, X, y, categorical):
                missing = ~np.isfinite(X)
                missing = missing.sum().sum()
                return missing

        class PercentageOfMissingValues(MetaFeature):
            def _calculate(self, X, y, categorical):
                missing = ~np.isfinite(X)
                missing = missing.sum().sum()
                return float(missing) / float(X.shape[0] * X.shape[1])

        class NumberOfInstancesWithMissingValues(MetaFeature):
            def _calculate(self, X, y, categorical):
                missing = ~np.isfinite(X)
                num_missing = missing.sum(axis=1)
                return int(np.sum([1 if num > 0 else 0 for num in num_missing]))

        class NumberOfFeaturesWithMissingValues(MetaFeature):
            def _calculate(self, X, y, categorical):
                missing = ~np.isfinite(X)
                num_missing = missing.sum(axis=0)
                return int(np.sum([1 if num > 0 else 0 for num in num_missing]))

        class ClassOccurrences(HelperFunction):
            def _calculate(self, X, y, categorical):
                if len(y.shape) == 2:
                    occurrences = []
                    for i in range(y.shape[1]):
                        occurrences.append(self._calculate(X, y[:, i], categorical))
                    return occurrences
                else:
                    occurrence_dict = defaultdict(float)
                    for value in y:
                        occurrence_dict[value] += 1
                    return occurrence_dict

        # noinspection PyTypeChecker
        occurrences = ClassOccurrences._calculate(self, X, y, categorical=True)
        # noinspection PyTypeChecker
        occurrence_dict = ClassOccurrences._calculate(self, X, y, categorical=True)

        class ClassProbabilityMean(MetaFeature):
            def _calculate(self, X, y, categorical):
                if len(y.shape) == 2:
                    occurrences = []
                    for i in range(y.shape[1]):
                        occurrences.extend(
                            [occurrence for occurrence in occurrence_dict[
                                i].values()])
                    occurrences = np.array(occurrences)
                else:
                    occurrences = np.array([occurrence for occurrence in occurrence_dict.values()],
                                          dtype=np.float64)
                return (occurrences / y.shape[0]).mean()

        class ClassProbabilitySTD(MetaFeature):
            def _calculate(self, X, y, categorical):
                if len(y.shape) == 2:
                    stds = []
                    for i in range(y.shape[1]):
                        std = np.array(
                            [occurrence for occurrence in occurrence_dict[
                                i].values()],
                            dtype=np.float64)
                        std = (std / y.shape[0]).std()
                        stds.append(std)
                    return np.mean(stds)
                else:
                    occurences = np.array([occurrence for occurrence in occurrence_dict.values()],
                                          dtype=np.float64)
                    return (occurences / y.shape[0]).std()

        # TODO PCA not working yet
        # class PCA(HelperFunction):
        #     def _calculate(self, X, y, categorical):
        #         import sklearn.decomposition
        #         pca = sklearn.decomposition.PCA(copy=True)
        #         rs = np.random.RandomState(42)
        #         indices = np.arange(X.shape[0])
        #         for i in range(10):
        #             try:
        #                 rs.shuffle(indices)
        #                 pca.fit(X[indices])
        #                 return pca
        #             except LinAlgError as e:
        #                 pass
        #         self.logger.warning("Failed to compute a Principle Component Analysis")
        #         return None
        #
        # # noinspection PyTypeChecker
        # pca = PCA._calculate(self, X, y, categorical=True)
        #
        # class PCAFractionOfComponentsFor95PercentVariance(MetaFeature):
        #     def _calculate(self, X, y, categorical):
        #         pca_ = pca
        #         if pca_ is None:
        #             return np.NaN
        #         sum_ = 0.
        #         idx = 0
        #         while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
        #             sum_ += pca_.explained_variance_ratio_[idx]
        #             idx += 1
        #         return float(idx) / float(X.shape[1])
        #
        # class PCASkewnessFirstPC(MetaFeature):
        #     def _calculate(self, X, y, categorical):
        #         pca_ = pca
        #         if pca_ is None:
        #             return np.NaN
        #         components = pca_.components_
        #         pca_.components_ = components[:1]
        #         transformed = pca_.transform(X)
        #         pca_.components_ = components
        #
        #         skewness = scipy.stats.skew(transformed)
        #         return skewness[0]

        # noinspection PyTypeChecker
        nr_missing_values = NumberOfMissingValues._calculate(self, X, y, categorical=True)
        # noinspection PyTypeChecker
        pct_missing_values = PercentageOfMissingValues._calculate(self, X, y, categorical=True)
        # noinspection PyTypeChecker
        nr_inst_mv = NumberOfInstancesWithMissingValues._calculate(self, X, y, categorical=True)
        pct_inst_mv = float(nr_inst_mv) / float(nr_inst)
        # noinspection PyTypeChecker
        nr_attr_mv = NumberOfFeaturesWithMissingValues._calculate(self, X, y, categorical=True)
        pct_attr_mv = float(nr_attr_mv) / float(nr_attr)
        # noinspection PyTypeChecker
        class_prob_mean = ClassProbabilityMean._calculate(self, X, y, categorical=True)
        # noinspection PyTypeChecker
        class_prob_std = ClassProbabilitySTD._calculate(self, X, y, categorical=True)

        return self.db.create_dataset(
            train_path=train_path,
            test_path='test_path',
            reference_path='reference_path',
            name=name,

            nr_inst=nr_inst,
            nr_attr=nr_attr,
            nr_class=nr_class,
            nr_missing_values=nr_missing_values,
            pct_missing_values=pct_missing_values,
            nr_inst_mv=nr_inst_mv,
            pct_inst_mv=pct_inst_mv,
            nr_attr_mv=nr_attr_mv,
            pct_attr_mv=pct_attr_mv,
            nr_outliers=nr_outliers,

            skewness_mean=skewness_mean,
            skewness_sd=skewness_sd,
            kurtosis_mean=kurtosis_mean,
            kurtosis_sd=kurtosis_sd,
            cor_mean=cor_mean,
            cor_sd=cor_sd,
            cov_mean=cov_mean,
            cov_sd=cov_sd,
            attr_conc_mean=attr_conc_mean,
            attr_conc_sd=attr_conc_sd,
            sparsity_mean=sparsity_mean,
            sparsity_sd=sparsity_sd,
            gravity=gravity,
            var_mean=var_mean,
            var_sd=var_sd,

            class_prob_mean=class_prob_mean,
            class_prob_std=class_prob_std,
            class_ent=class_ent,
            attr_ent_mean=attr_ent_mean,
            attr_ent_sd=attr_ent_sd,
            mut_inf_mean=mut_inf_mean,
            mut_inf_sd=mut_inf_sd,
            eq_num_attr=eq_num_attr,
            ns_ratio=ns_ratio,

            nodes=nodes,
            leaves=leaves,
            leaves_branch_mean=leaves_branch_mean,
            leaves_branch_sd=leaves_branch_sd,
            nodes_per_attr=nodes_per_attr,
            leaves_per_class_mean=leaves_per_class_mean,
            leaves_per_class_sd=leaves_per_class_sd,
            var_importance_mean=var_importance_mean,
            var_importance_sd=var_importance_sd,

            one_nn_mean=one_nn_mean,
            one_nn_sd=one_nn_sd,
            best_node_mean=best_node_mean,
            best_node_sd=best_node_sd,
            # best_random=best_random,
            # best_worst=best_worst,
            linear_discr_mean=linear_discr_mean,
            linear_discr_sd=linear_discr_sd,
            naive_bayes_mean=naive_bayes_mean,
            naive_bayes_sd=naive_bayes_sd
        )

    def add_algorithm(self, ds_id: int, algorithm: str):
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
            # noinspection PyTypeChecker
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
