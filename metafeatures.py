import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse
import time
from pymfe.mfe import MFE

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
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
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
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
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
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
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
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            num_missing = missing.sum(axis=0)

            return int(np.sum([1 if num > 0 else 0 for num in num_missing]))
        else:
            missing_o = pd.isna(X_object)
            num_missing_o = missing_o.sum(axis=0)

            missing_n = ~np.isfinite(X_numeric)
            num_missing_n = missing_n.sum(axis=0)
            num_missing = num_missing_n + num_missing_o

            return int(np.sum([1 if num > 0 else 0 for num in num_missing]))


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

    @staticmethod
    def calculate(df: pd.DataFrame, class_column: str, random_state: int = 42):
        # ##########################################################################
        # #  Extracting Meta Features with pymfe  ##################################
        # ##########################################################################

        """
        Loads dataframe and splits it in X, y
        """
        X, y = df.drop(class_column, axis=1), df[class_column]

        if X.shape[0] == 0 or X.shape[1] == 0:
            LOGGER.info('X has no samples or features. Setting meta-features to default.')
            return {
                'nr_inst': 0,
                'nr_attr': 0,
                'nr_class': 0,
                'nr_missing_values': np.nan,
                'pct_missing_values': np.nan,
                'nr_inst_mv': np.nan,
                'pct_inst_mv': np.nan,
                'nr_attr_mv': np.nan,
                'pct_attr_mv': np.nan,
                'nr_outliers': 0,

                'skewness_mean': np.nan,
                'skewness_sd': np.nan,
                'kurtosis_mean': np.nan,
                'kurtosis_sd': np.nan,
                'cor_mean': np.nan,
                'cor_sd': np.nan,
                'cov_mean': np.nan,
                'cov_sd': np.nan,
                'sparsity_mean': np.nan,
                'sparsity_sd': np.nan,
                'var_mean': np.nan,
                'var_sd': np.nan,

                'class_prob_mean': np.nan,
                'class_prob_std': np.nan,
                'class_ent': np.nan,
                'attr_ent_mean': np.nan,
                'attr_ent_sd': np.nan,
                'mut_inf_mean': np.nan,
                'mut_inf_sd': np.nan,
                'eq_num_attr': np.nan,
                'ns_ratio': np.nan,

                'nodes': np.nan,
                'leaves': np.nan,
                'leaves_branch_mean': np.nan,
                'leaves_branch_sd': np.nan,
                'nodes_per_attr': np.nan,
                'leaves_per_class_mean': np.nan,
                'leaves_per_class_sd': np.nan,
                'var_importance_mean': np.nan,
                'var_importance_sd': np.nan,

                'one_nn_mean': np.nan,
                'one_nn_sd': np.nan,
                'best_node_mean': np.nan,
                'best_node_sd': np.nan,
                'linear_discr_mean': np.nan,
                'linear_discr_sd': np.nan,
                'naive_bayes_mean': np.nan,
                'naive_bayes_sd': np.nan
            }

        """
        Selects Meta Features and extracts them
        """
        mfe = MFE(features=(['nr_inst', 'nr_attr', 'nr_class', 'nr_outliers', 'skewness', 'kurtosis', 'cor', 'cov',
                             'sparsity', 'var', 'class_ent', 'attr_ent', 'mut_inf',
                             'eq_num_attr', 'ns_ratio', 'nodes', 'leaves', 'leaves_branch', 'nodes_per_attr',
                             'var_importance', 'one_nn', 'best_node', 'linear_discr',
                             'naive_bayes', 'leaves_per_class']),
                  random_state=random_state)

        transform_cat = True
        if np.any(pd.isna(X)):
            transform_cat = False

        mfe.fit(X.to_numpy(), y.to_numpy(), transform_cat=transform_cat)
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

        try:
            skewness_mean = float(f_value[f_name.index('skewness.mean')])
        except:
            skewness_mean = float(f_value[f_name.index('skewness')])
        try:
            skewness_sd = float(f_value[f_name.index('skewness.sd')])
        except:
            skewness_sd = float(f_value[f_name.index('skewness')])

        try:
            kurtosis_mean = float(f_value[f_name.index('kurtosis.mean')])
        except:
            kurtosis_mean = float(f_value[f_name.index('kurtosis')])
        try:
            kurtosis_sd = float(f_value[f_name.index('kurtosis.sd')])
        except:
            kurtosis_sd = float(f_value[f_name.index('kurtosis')])

        try:
            cor_mean = float(f_value[f_name.index('cor.mean')])
        except:
            cor_mean = float(f_value[f_name.index('cor')])
        try:
            cor_sd = float(f_value[f_name.index('cor.sd')])
        except:
            cor_sd = float(f_value[f_name.index('cor')])

        try:
            cov_mean = float(f_value[f_name.index('cov.mean')])
        except:
            cov_mean = float(f_value[f_name.index('cov')])
        try:
            cov_sd = float(f_value[f_name.index('cov.sd')])
        except:
            cov_sd = float(f_value[f_name.index('cov')])

        try:
            sparsity_mean = float(f_value[f_name.index('sparsity.mean')])
        except:
            sparsity_mean = float(f_value[f_name.index('sparsity')])
        try:
            sparsity_sd = float(f_value[f_name.index('sparsity.sd')])
        except:
            sparsity_sd = float(f_value[f_name.index('sparsity')])

        try:
            var_mean = float(f_value[f_name.index('var.mean')])
        except:
            var_mean = float(f_value[f_name.index('var')])
        try:
            var_sd = float(f_value[f_name.index('var.sd')])
        except:
            var_sd = float(f_value[f_name.index('var')])

        try:
            attr_ent_mean = float(f_value[f_name.index('attr_ent.mean')])
        except:
            attr_ent_mean = float(f_value[f_name.index('attr_ent')])
        try:
            attr_ent_sd = float(f_value[f_name.index('attr_ent.sd')])
        except:
            attr_ent_sd = float(f_value[f_name.index('attr_ent')])

        try:
            mut_inf_mean = float(f_value[f_name.index('mut_inf.mean')])
        except:
            mut_inf_mean = float(f_value[f_name.index('mut_inf')])
        try:
            mut_inf_sd = float(f_value[f_name.index('mut_inf.sd')])
        except:
            mut_inf_sd = float(f_value[f_name.index('mut_inf')])

        try:
            leaves_branch_mean = float(f_value[f_name.index('leaves_branch.mean')])
        except:
            leaves_branch_mean = float(f_value[f_name.index('leaves_branch')])
        try:
            leaves_branch_sd = float(f_value[f_name.index('leaves_branch.sd')])
        except:
            leaves_branch_sd = float(f_value[f_name.index('leaves_branch')])

        try:
            leaves_per_class_mean = float(f_value[f_name.index('leaves_per_class.mean')])
        except:
            leaves_per_class_mean = float(f_value[f_name.index('leaves_per_class')])
        try:
            leaves_per_class_sd = float(f_value[f_name.index('leaves_per_class.sd')])
        except:
            leaves_per_class_sd = float(f_value[f_name.index('leaves_per_class')])

        try:
            var_importance_mean = float(f_value[f_name.index('var_importance.mean')])
        except:
            var_importance_mean = float(f_value[f_name.index('var_importance')])
        try:
            var_importance_sd = float(f_value[f_name.index('var_importance.sd')])
        except:
            var_importance_sd = float(f_value[f_name.index('var_importance')])

        try:
            one_nn_mean = float(f_value[f_name.index('one_nn.mean')])
        except:
            one_nn_mean = float(f_value[f_name.index('one_nn')])
        try:
            one_nn_sd = float(f_value[f_name.index('one_nn.sd')])
        except:
            one_nn_sd = float(f_value[f_name.index('one_nn')])

        try:
            best_node_mean = float(f_value[f_name.index('best_node.mean')])
        except:
            best_node_mean = float(f_value[f_name.index('best_node')])
        try:
            best_node_sd = float(f_value[f_name.index('best_node.sd')])
        except:
            best_node_sd = float(f_value[f_name.index('best_node')])

        try:
            linear_discr_mean = float(f_value[f_name.index('linear_discr.mean')])
        except:
            linear_discr_mean = float(f_value[f_name.index('linear_discr')])
        try:
            linear_discr_sd = float(f_value[f_name.index('linear_discr.sd')])
        except:
            linear_discr_sd = float(f_value[f_name.index('linear_discr')])

        try:
            naive_bayes_mean = float(f_value[f_name.index('naive_bayes.mean')])
        except:
            naive_bayes_mean = float(f_value[f_name.index('naive_bayes')])
        try:
            naive_bayes_sd = float(f_value[f_name.index('naive_bayes.sd')])
        except:
            naive_bayes_sd = float(f_value[f_name.index('naive_bayes')])

        # ##########################################################################
        # #  Extracting Meta Features with AutoSklearn  ############################
        # ##########################################################################

        nr_missing_values = NumberOfMissingValues()(X, y, categorical=True).value

        pct_missing_values = PercentageOfMissingValues()(X, y, categorical=True).value

        nr_inst_mv = NumberOfInstancesWithMissingValues()(X, y, categorical=True).value

        pct_inst_mv = (float(nr_inst_mv) / float(nr_inst)) * 100

        nr_attr_mv = NumberOfFeaturesWithMissingValues()(X, y, categorical=True).value

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
        }
