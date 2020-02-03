from collections import defaultdict

import numpy as np
import pandas as pd
from autosklearn.metalearning.metafeatures.metafeature import MetaFeature
from pymfe.mfe import MFE


class NumberOfMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
        X_object = X.select_dtypes(include=['object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            missing = missing.sum().sum()

            return missing
        else:
            missing_o = pd.isna(X_object)
            missing_o = missing_o.sum().sum()

            missing_n = ~np.isfinite(X_numeric)
            missing_n = missing_n.sum().sum()

            missing = missing_n + missing_o

            return missing


class PercentageOfMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
        X_object = X.select_dtypes(include=['object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            missing = missing.sum().sum()

            return float(missing) / float(X.shape[0] * X.shape[1])
        else:
            missing_o = pd.isna(X_object)
            missing_o = missing_o.sum().sum()

            missing_n = ~np.isfinite(X_numeric)
            missing_n = missing_n.sum().sum()

            missing = missing_n + missing_o

            return float(missing) / float(X.shape[0] * X.shape[1])


class NumberOfInstancesWithMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
        X_object = X.select_dtypes(include=['object'])

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
        X_object = X.select_dtypes(include=['object'])

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
        return (occurrences / y.shape[0]).mean()


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
            return (occurences / y.shape[0]).std()


# # TODO PCA not working yet
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

    def calculate(self, df: pd.DataFrame, class_column: str, random_state: int = 42):
        # ##########################################################################
        # #  Extracting Meta Features with pymfe  ##################################
        # ##########################################################################

        """
        Loading train_path
        """
        X, y = df.drop(class_column, axis=1), df[class_column]

        """
        Selecting Meta Features and extracting them
        """
        mfe = MFE(features=(['nr_inst', 'nr_attr', 'nr_class', 'nr_outliers', 'skewness', 'kurtosis', 'cor', 'cov',
                             'attr_conc', 'sparsity', 'gravity', 'var', 'class_ent', 'attr_ent', 'mut_inf',
                             'eq_num_attr', 'ns_ratio', 'nodes', 'leaves', 'leaves_branch', 'nodes_per_attr',
                             'var_importance', 'one_nn', 'best_node', 'linear_discr',
                             'naive_bayes', 'leaves_per_class']),
                  random_state=random_state)

        # noinspection PyTypeChecker
        mfe.fit(X.values, y.values)
        f_name, f_value = mfe.extract(cat_cols='auto', suppress_warnings=True)

        """
        Mapping values to Meta Feature variables
        """
        nr_inst = f_value[f_name.index('nr_inst')]
        nr_attr = f_value[f_name.index('nr_attr')]
        nr_class = f_value[f_name.index('nr_class')]
        nr_outliers = f_value[f_name.index('nr_outliers')]

        skewness_mean = f_value[f_name.index('skewness.mean')]
        skewness_sd = f_value[f_name.index('skewness.sd')]
        kurtosis_mean = f_value[f_name.index('kurtosis.mean')]
        kurtosis_sd = f_value[f_name.index('kurtosis.sd')]
        cor_mean = f_value[f_name.index('cor.mean')]
        cor_sd = f_value[f_name.index('cor.sd')]
        cov_mean = f_value[f_name.index('cov.mean')]
        cov_sd = f_value[f_name.index('cov.sd')]
        attr_conc_mean = f_value[f_name.index('attr_conc.mean')]
        attr_conc_sd = f_value[f_name.index('attr_conc.sd')]
        sparsity_mean = f_value[f_name.index('sparsity.mean')]
        sparsity_sd = f_value[f_name.index('sparsity.sd')]
        gravity = f_value[f_name.index('gravity')]
        var_mean = f_value[f_name.index('var.mean')]
        var_sd = f_value[f_name.index('var.sd')]

        class_ent = f_value[f_name.index('class_ent')]
        attr_ent_mean = f_value[f_name.index('attr_ent.mean')]
        attr_ent_sd = f_value[f_name.index('attr_ent.sd')]
        mut_inf_mean = f_value[f_name.index('mut_inf.mean')]
        mut_inf_sd = f_value[f_name.index('mut_inf.sd')]
        eq_num_attr = f_value[f_name.index('eq_num_attr')]
        ns_ratio = f_value[f_name.index('ns_ratio')]

        nodes = f_value[f_name.index('nodes')]
        leaves = f_value[f_name.index('leaves')]
        leaves_branch_mean = f_value[f_name.index('leaves_branch.mean')]
        leaves_branch_sd = f_value[f_name.index('leaves_branch.sd')]
        nodes_per_attr = f_value[f_name.index('nodes_per_attr')]
        leaves_per_class_mean = f_value[f_name.index('leaves_per_class.mean')]
        leaves_per_class_sd = f_value[f_name.index('leaves_per_class.sd')]
        var_importance_mean = f_value[f_name.index('var_importance.mean')]
        var_importance_sd = f_value[f_name.index('var_importance.sd')]

        one_nn_mean = f_value[f_name.index('one_nn.mean')]
        one_nn_sd = f_value[f_name.index('one_nn.sd')]
        best_node_mean = f_value[f_name.index('best_node.mean')]
        best_node_sd = f_value[f_name.index('best_node.sd')]
        linear_discr_mean = f_value[f_name.index('linear_discr.mean')]
        linear_discr_sd = f_value[f_name.index('linear_discr.sd')]
        naive_bayes_mean = f_value[f_name.index('naive_bayes.mean')]
        naive_bayes_sd = f_value[f_name.index('naive_bayes.sd')]

        # ##########################################################################
        # #  Extracting Meta Features with Auto-Sklearn  ###########################
        # ##########################################################################

        nr_missing_values = NumberOfMissingValues()(X, y, categorical=True).value

        pct_missing_values = PercentageOfMissingValues()(X, y, categorical=True).value

        nr_inst_mv = NumberOfInstancesWithMissingValues()(X, y, categorical=True).value
        pct_inst_mv = float(nr_inst_mv) / float(nr_inst)

        nr_attr_mv = NumberOfFeaturesWithMissingValues()(X, y, categorical=True).value
        pct_attr_mv = float(nr_attr_mv) / float(nr_attr)

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
            'attr_conc_mean': attr_conc_mean,
            'attr_conc_sd': attr_conc_sd,
            'sparsity_mean': sparsity_mean,
            'sparsity_sd': sparsity_sd,
            'gravity': gravity,
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
