import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import pynisher2
import scipy.sparse
import time
from pymfe.mfe import MFE
from typing import Dict, Tuple

LOGGER = logging.getLogger('mlb')


# ##########################################################################
# #  Help Functions Copied From AutoSklearn  ###############################
# ##########################################################################


def _create_logger(name):
    return logging.getLogger(name)


class PickableLoggerAdapter(object):

    def __init__(self, name):
        self.name = name
        self.logger = _create_logger(name)

    def __getstate__(self):
        """
        Method is called when pickle dumps an object.

        Returns
        -------
        Dictionary, representing the object state to be pickled. Ignores
        the self.logger field and only returns the logger name.
        """
        return {'name': self.name}

    def __setstate__(self, state):
        """
        Method is called when pickle loads an object. Retrieves the name and
        creates a logger.

        Parameters
        ----------
        state - dictionary, containing the logger name.

        """
        self.name = state['name']
        self.logger = _create_logger(self.name)


def get_logger(name):
    logger = PickableLoggerAdapter(name)
    return logger


class MetaFeatureValue(object):
    def __init__(self, name, type_, fold, repeat, value, time, comment=""):
        self.name = name
        self.type_ = type_
        self.fold = fold
        self.repeat = repeat
        self.value = value
        self.time = time
        self.comment = comment

    def to_arff_row(self):
        if self.type_ == "METAFEATURE":
            value = self.value
        else:
            value = "?"

        return [self.name, self.type_, self.fold,
                self.repeat, value, self.time, self.comment]

    def __repr__(self):
        repr = "%s (type: %s, fold: %d, repeat: %d, value: %s, time: %3.3f, " \
               "comment: %s)"
        repr = repr % tuple(self.to_arff_row()[:4] +
                            [str(self.to_arff_row()[4])] +
                            self.to_arff_row()[5:])
        return repr


class AbstractMetaFeature(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.logger = get_logger(__name__)

    @abstractmethod
    def _calculate(cls, X, y, categorical):
        pass

    def __call__(self, X, y, categorical=None):
        if categorical is None:
            categorical = [False for i in range(X.shape[1])]
        starttime = time.time()

        try:
            if scipy.sparse.issparse(X) and hasattr(self, "_calculate_sparse"):
                value = self._calculate_sparse(X, y, categorical)
            else:
                value = self._calculate(X, y, categorical)
            comment = ""
        except MemoryError as e:
            value = None
            comment = "Memory Error"

        endtime = time.time()
        return MetaFeatureValue(self.__class__.__name__, self.type_,
                                0, 0, value, endtime - starttime, comment=comment)


class MetaFeature(AbstractMetaFeature):
    def __init__(self):
        super(MetaFeature, self).__init__()
        self.type_ = "METAFEATURE"


# ##########################################################################
# #  Extracting MetaFeatures with the help of AutoSklearn  #################
# ##########################################################################

class NumberOfMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        X_numeric = X.select_dtypes(include=['number'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            missing = missing.sum().sum()

            return int(missing)
        else:
            missing_o = pd.isna(X_object)
            missing_o = missing_o.sum().sum()

            missing_n = ~np.isfinite(X_numeric)
            missing_n = missing_n.sum().sum()

            missing = missing_n + missing_o

            return int(missing)


class PercentageOfMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        X_numeric = X.select_dtypes(include=['number'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            missing = missing.sum().sum()

            return (float(missing) / float(X.shape[0] * X.shape[1])) * 100
        else:
            missing_o = pd.isna(X_object)
            missing_o = missing_o.sum().sum()

            missing_n = ~np.isfinite(X_numeric)
            missing_n = missing_n.sum().sum()

            missing = missing_n + missing_o

            return (float(missing) / float(X.shape[0] * X.shape[1])) * 100


class NumberOfInstancesWithMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        X_numeric = X.select_dtypes(include=['number'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            num_missing = missing.sum(axis=1)

            return int(np.sum([1 if num > 0 else 0 for num in num_missing]))
        else:
            missing_o = pd.isna(X_object)
            num_missing_o = missing_o.sum(axis=1)

            missing_n = ~np.isfinite(X_numeric)
            num_missing_n = missing_n.sum(axis=1)
            num_missing = num_missing_n + num_missing_o

            return int(np.sum([1 if num > 0 else 0 for num in num_missing]))


class NumberOfFeaturesWithMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        X_numeric = X.select_dtypes(include=['number'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            return int((missing.sum(axis=0) > 0).sum())
        else:
            missing_o = pd.isna(X_object)
            num_missing_o = (missing_o.sum(axis=0) > 0).sum()

            missing_n = ~np.isfinite(X_numeric)
            num_missing_n = (missing_n.sum(axis=0) > 0).sum()
            return int(num_missing_n + num_missing_o)


class ClassOccurrences(MetaFeature):
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


class ClassProbabilityMean(MetaFeature):
    def _calculate(self, X, y, categorical):
        occurrence_dict = ClassOccurrences()(X, y, categorical)

        if len(y.shape) == 2:
            occurrences = []
            for i in range(y.shape[1]):
                occurrences.extend([occurrence for occurrence in occurrence_dict[i].value.values()])
            occurrences = np.array(occurrences)
        else:
            occurrences = np.array([occurrence for occurrence in occurrence_dict.value.values()], dtype=np.float64)
        return float((occurrences / y.shape[0]).mean())


class ClassProbabilitySTD(MetaFeature):
    def _calculate(self, X, y, categorical):
        occurrence_dict = ClassOccurrences()(X, y, categorical)

        if len(y.shape) == 2:
            stds = []
            for i in range(y.shape[1]):
                std = np.array([occurrence for occurrence in occurrence_dict[i].value.values()], dtype=np.float64)
                std = (std / y.shape[0]).std()
                stds.append(std)
            return np.mean(stds)
        else:
            occurences = np.array([occurrence for occurrence in occurrence_dict.value.values()], dtype=np.float64)
            return float((occurences / y.shape[0]).std())


# class PCA(HelperFunction):
#     def _calculate(self, X, y, categorical):
#         from sklearn.decomposition.pca import PCA
#         pca = PCA(copy=True)
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
# class PCAFractionOfComponentsFor95PercentVariance(MetaFeature):
#     def _calculate(self, X, y, categorical):
#         pca_ = PCA()(X, y, categorical)
#
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
#         pca_ = PCA()(X, y, categorical)
#         if pca_ is None:
#             return np.NaN
#         components = pca_.components_
#         pca_.components_ = components[:1]
#         transformed = pca_.transform(X)
#         pca_.components_ = components
#
#         skewness = scipy.stats.skew(transformed)
#         return skewness[0]


class MetaFeatures(object):

    def __init__(self, mf_timeout: int = 600, mf_memory: int = 6144):
        self.mf_timeout = mf_timeout
        self.mf_memory = mf_memory
        self.subprocess_logger = logging.getLogger('mlb:mf')

    def calculate(self,
                  df: pd.DataFrame,
                  class_column: str,
                  max_nan_percentage: float = 0.9,
                  max_features: int = 10000,
                  random_state: int = 42) -> Tuple[Dict[str, float], bool]:
        """
        Calculates the meta-features for the given DataFrame. The actual computation is dispatched to another process
        to prevent crashes due to extensive memory usage.
        :param df:
        :param class_column:
        :param max_nan_percentage:
        :param max_features:
        :param random_state:
        :return:
        """

        wrapper = pynisher2.enforce_limits(wall_time_in_s=self.mf_timeout, logger=self.subprocess_logger,
                                           mem_in_mb=self.mf_memory)(self._calculate)
        res = wrapper(df, class_column, max_nan_percentage=max_nan_percentage, max_features=max_features,
                      random_state=random_state)
        if wrapper.exit_status is pynisher2.TimeoutException or wrapper.exit_status is pynisher2.MemorylimitException:
            LOGGER.info('MF calculation violated constraints')
            return {
                       'nr_inst': df.shape[0],
                       'nr_attr': df.shape[1]
                   }, False
        elif wrapper.exit_status is pynisher2.AnythingException:
            LOGGER.warning('Failed to extract MF due to {}'.format(res[0]))
            return {
                       'nr_inst': df.shape[0],
                       'nr_attr': df.shape[1]
                   }, False
        elif wrapper.exit_status == 0 and res is not None:
            return res

    @staticmethod
    def _calculate(df: pd.DataFrame,
                   class_column: str,
                   max_nan_percentage: float = 0.9,
                   max_features: int = 10000,
                   random_state: int = 42) -> Tuple[Dict[str, float], bool]:
        """
        Calculates the meta-features for the given DataFrame. _Attention_: Meta-feature calculation can require a lot of
        memory. This method should not be called directly to prevent the caller from crashing.
        :param df:
        :param class_column:
        :param max_nan_percentage:
        :param max_features:
        :param random_state:
        :return:
        """
        X, y = df.drop(class_column, axis=1), df[class_column]

        # Checks if number of features is bigger than max_features.
        if X.shape[1] > max_features:
            LOGGER.info(
                'Number of features is bigger then {}. Creating dataset with status skipped...'.format(max_features))
            return {
                       'nr_inst': df.shape[0],
                       'nr_attr': df.shape[1],
                   }, False

        # Extracting Missing Value Meta Features with AutoSklearn
        nr_missing_values = NumberOfMissingValues()(X, y, categorical=True).value
        pct_missing_values = PercentageOfMissingValues()(X, y, categorical=True).value
        nr_inst_mv = NumberOfInstancesWithMissingValues()(X, y, categorical=True).value
        nr_attr_mv = NumberOfFeaturesWithMissingValues()(X, y, categorical=True).value

        # Meta-Feature calculation does not work with missing data
        numeric = X.select_dtypes(include=['number']).columns
        if np.any(pd.isna(X)):
            n = X.shape[0]

            for i in X.columns:
                col = X[i]
                nan = pd.isna(col)
                if not nan.any():
                    continue
                elif nan.value_counts(normalize=True)[True] > max_nan_percentage:
                    X.drop(i, axis=1, inplace=True)
                elif i in numeric:
                    filler = np.random.normal(col.mean(), col.std(), n)
                    X[i] = col.combine_first(pd.Series(filler))
                else:
                    items = col.dropna().unique()
                    probability = col.value_counts(dropna=True, normalize=True)
                    probability = probability.where(probability > 0).dropna()
                    filler = np.random.choice(items, n, p=probability)
                    X[i] = col.combine_first(pd.Series(filler))

        for i in X.columns:
            col = X[i]
            if i in numeric:
                if not (abs(col - col.iloc[0]) > 1e-7).any() or (~np.isfinite(col)).all():
                    X.drop(i, inplace=True, axis=1)
            else:
                if not (col != col.iloc[0]).any():
                    X.drop(i, inplace=True, axis=1)

        if X.shape[0] == 0 or X.shape[1] == 0:
            LOGGER.info('X has no samples, no features or only constant values. Marking dataset as skipped.')
            return {
                       'nr_inst': X.shape[0],
                       'nr_attr': X.shape[1],
                   }, False

        """
       Selects Meta Features and extracts them
       """
        mfe = MFE(features=(['nr_inst', 'nr_attr', 'nr_class', 'nr_outliers', 'skewness', 'kurtosis', 'cor', 'cov',
                             'sparsity', 'var', 'class_ent', 'attr_ent', 'mut_inf',
                             'eq_num_attr', 'ns_ratio', 'nodes', 'leaves', 'leaves_branch', 'nodes_per_attr',
                             'var_importance', 'one_nn', 'best_node', 'linear_discr',
                             'naive_bayes', 'leaves_per_class']))
        mfe.fit(X.to_numpy(), y.to_numpy(), transform_cat=True)
        f_name, f_value = mfe.extract(cat_cols='auto', suppress_warnings=True)

        """
        Mapping values to Meta Feature variables
        """
        nr_inst = int(f_value[f_name.index('nr_inst')])
        nr_attr = int(f_value[f_name.index('nr_attr')])
        nr_class = int(f_value[f_name.index('nr_class')])
        nr_outliers = int(f_value[f_name.index('nr_outliers')])
        class_ent = float(f_value[f_name.index('class_ent')])
        eq_num_attr = float(f_value[f_name.index('eq_num_attr')])
        ns_ratio = float(f_value[f_name.index('ns_ratio')])
        nodes = float(f_value[f_name.index('nodes')])
        leaves = float(f_value[f_name.index('leaves')])
        nodes_per_attr = float(f_value[f_name.index('nodes_per_attr')])

        def get_value(key: str):
            try:
                return float(f_value[f_name.index(key)])
            except:
                return float(f_value[f_name.index(key.split('.')[0])])

        skewness_mean = get_value('skewness.mean')
        skewness_sd = get_value('skewness.sd') if nr_attr > 1 else 0

        kurtosis_mean = get_value('kurtosis.mean')
        kurtosis_sd = get_value('kurtosis.sd') if nr_attr > 1 else 0

        cor_mean = get_value('cor.mean') if nr_attr > 1 else 1
        cor_sd = get_value('cor.sd') if nr_attr > 2 else 0

        cov_mean = get_value('cov.mean') if nr_attr > 1 else 0
        cov_sd = get_value('cov.sd') if nr_attr > 2 else 0

        sparsity_mean = get_value('sparsity.mean')
        sparsity_sd = get_value('sparsity.sd') if nr_attr > 1 else 0

        var_mean = get_value('var.mean')
        var_sd = get_value('var.sd') if nr_attr > 1 else 0

        attr_ent_mean = get_value('attr_ent.mean')
        attr_ent_sd = get_value('attr_ent.sd') if nr_attr > 1 else 0

        mut_inf_mean = get_value('mut_inf.mean')
        mut_inf_sd = get_value('mut_inf.sd') if nr_attr > 1 else 0

        leaves_branch_mean = get_value('leaves_branch.mean')
        leaves_branch_sd = get_value('leaves_branch.sd')

        leaves_per_class_mean = get_value('leaves_per_class.mean')
        leaves_per_class_sd = get_value('leaves_per_class.sd')
        # not sure under which conditions this exactly happens.
        if np.isnan(leaves_per_class_sd):
            leaves_per_class_sd = 0

        var_importance_mean = get_value('var_importance.mean')
        var_importance_sd = get_value('var_importance.sd') if nr_attr > 1 else 0

        one_nn_mean = get_value('one_nn.mean')
        one_nn_sd = get_value('one_nn.sd')

        best_node_mean = get_value('best_node.mean')
        best_node_sd = get_value('best_node.sd')

        linear_discr_mean = get_value('linear_discr.mean')
        linear_discr_sd = get_value('linear_discr.sd')

        naive_bayes_mean = get_value('naive_bayes.mean')
        naive_bayes_sd = get_value('naive_bayes.sd')

        # ##########################################################################
        # #  Extracting Meta Features with AutoSklearn  ############################
        # ##########################################################################

        pct_inst_mv = (float(nr_inst_mv) / float(nr_inst)) * 100

        pct_attr_mv = (float(nr_attr_mv) / float(nr_attr)) * 100

        class_prob_mean = ClassProbabilityMean()(X, y, categorical=True).value

        class_prob_std = ClassProbabilitySTD()(X, y, categorical=True).value

        # pca_95 = PCAFractionOfComponentsFor95PercentVariance()(X, y, categorical=True).value

        # pca_skewness = PCASkewnessFirstPC())(X, y, categorical=True).value

        return {
                   'nr_inst': nr_inst,
                   'nr_attr': nr_attr,
                   'nr_class': nr_class,
                   'nr_missing_values': nr_missing_values,
                   'pct_missing_values': pct_missing_values,
                   'nr_inst_mv': nr_inst_mv,
                   'pct_inst_mv': pct_inst_mv,
                   'nr_attr_mv': nr_attr_mv,
                   'pct_attr_mv': pct_attr_mv,
                   'nr_outliers': nr_outliers,

                   'skewness_mean': skewness_mean,
                   'skewness_sd': skewness_sd,
                   'kurtosis_mean': kurtosis_mean,
                   'kurtosis_sd': kurtosis_sd,
                   'cor_mean': cor_mean,
                   'cor_sd': cor_sd,
                   'cov_mean': cov_mean,
                   'cov_sd': cov_sd,
                   'sparsity_mean': sparsity_mean,
                   'sparsity_sd': sparsity_sd,
                   'var_mean': var_mean,
                   'var_sd': var_sd,

                   'class_prob_mean': class_prob_mean,
                   'class_prob_std': class_prob_std,
                   'class_ent': class_ent,
                   'attr_ent_mean': attr_ent_mean,
                   'attr_ent_sd': attr_ent_sd,
                   'mut_inf_mean': mut_inf_mean,
                   'mut_inf_sd': mut_inf_sd,
                   'eq_num_attr': eq_num_attr,
                   'ns_ratio': ns_ratio,

                   'nodes': nodes,
                   'leaves': leaves,
                   'leaves_branch_mean': leaves_branch_mean,
                   'leaves_branch_sd': leaves_branch_sd,
                   'nodes_per_attr': nodes_per_attr,
                   'leaves_per_class_mean': leaves_per_class_mean,
                   'leaves_per_class_sd': leaves_per_class_sd,
                   'var_importance_mean': var_importance_mean,
                   'var_importance_sd': var_importance_sd,

                   'one_nn_mean': one_nn_mean,
                   'one_nn_sd': one_nn_sd,
                   'best_node_mean': best_node_mean,
                   'best_node_sd': best_node_sd,
                   'linear_discr_mean': linear_discr_mean,
                   'linear_discr_sd': linear_discr_sd,
                   'naive_bayes_mean': naive_bayes_mean,
                   'naive_bayes_sd': naive_bayes_sd
               }, True
