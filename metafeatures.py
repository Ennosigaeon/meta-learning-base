from collections import defaultdict

import numpy as np
import pandas as pd
from autosklearn.metalearning.metafeatures.metafeature import MetaFeature
from pymfe.mfe import MFE


class Metafeatures(object):

    def calculate_metafeatures(self, df: pd.DataFrame, class_column: str = None):

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

        """
        Mapping values to Meta Feature variables
        """
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
                        occurrences.extend(
                            [occurrence for occurrence in occurrence_dict[i].value.values()])
                    occurrences = np.array(occurrences)
                else:
                    occurrences = np.array([occurrence for occurrence in occurrence_dict.value.values()],
                                           dtype=np.float64)
                return (occurrences / y.shape[0]).mean()

        class ClassProbabilitySTD(MetaFeature):
            def _calculate(self, X, y, categorical):
                occurrence_dict = ClassOccurrences()(X, y, categorical)

                if len(y.shape) == 2:
                    stds = []
                    for i in range(y.shape[1]):
                        std = np.array(
                            [occurrence for occurrence in occurrence_dict[
                                i].value.values()],
                            dtype=np.float64)
                        std = (std / y.shape[0]).std()
                        stds.append(std)
                    return np.mean(stds)
                else:
                    occurences = np.array([occurrence for occurrence in occurrence_dict.value.values()],
                                          dtype=np.float64)
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
        # noinspection PyTypeChecker
        # pca_95 = PCAFractionOfComponentsFor95PercentVariance._calculate(self, X, y, categorical=True)
        # noinspection PyTypeChecker
        # pca_skewness = PCASkewnessFirstPC._calculate(self, X, y, categorical=True)

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
            # best_random:best_random,
            # best_worst:best_worst,
            'linear_discr_mean': linear_discr_mean,
            'linear_discr_sd': linear_discr_sd,
            'naive_bayes_mean': naive_bayes_mean,
            'naive_bayes_sd': naive_bayes_sd
        }
