import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pymfe import _internal
from pymfe.general import MFEGeneral
from pymfe.info_theory import MFEInfoTheory
from pymfe.model_based import MFEModelBased
from pymfe.statistical import MFEStatistical

from dswizard import pynisher

LOGGER = logging.getLogger('mlb')


class MetaFeatures(object):

    def __init__(self, mf_timeout: int = 30, mf_memory: int = 2048):
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

        wrapper = pynisher.enforce_limits(wall_time_in_s=self.mf_timeout, logger=self.subprocess_logger,
                                          mem_in_mb=self.mf_memory)(self._calculate)
        res = wrapper(df, class_column, max_nan_percentage=max_nan_percentage, max_features=max_features,
                      random_state=random_state)
        if wrapper.exit_status is pynisher.TimeoutException or wrapper.exit_status is pynisher.MemorylimitException:
            LOGGER.info('MF calculation violated constraints')
            return {
                       'nr_inst': df.shape[0],
                       'nr_attr': df.shape[1]
                   }, False
        elif wrapper.exit_status is pynisher.AnythingException:
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

        missing = pd.isna(X)
        nr_missing_values = MetaFeatures.ft_nr_missing_val(missing)
        nr_inst_mv = MetaFeatures.ft_nr_inst_missing_values(missing)
        nr_attr_mv = MetaFeatures.ft_nr_attr_missing_values(missing)
        pct_missing_values = (float(nr_missing_values) / float(X.shape[0] * X.shape[1])) * 100
        pct_inst_mv = (float(nr_inst_mv) / float(X.shape[0])) * 100
        pct_attr_mv = (float(nr_attr_mv) / float(X.shape[1])) * 100
        class_prob = MetaFeatures.ft_class_prob(y)

        # Meta-Feature calculation does not work with missing data.
        numeric = X.select_dtypes(include=['number']).columns
        if np.any(missing):
            n = X.shape[0]

            for i in X.columns:
                col = X[i]
                nan = missing[i]
                if not nan.any():
                    continue
                # Drop column if it contains of more nans then the max_nan_percentage.
                elif nan.value_counts(normalize=True)[True] > max_nan_percentage:
                    X.drop(i, axis=1, inplace=True)
                # Impute column if values are numeric.
                elif i in numeric:
                    filler = np.random.normal(col.mean(), col.std(), n)
                    X[i] = col.combine_first(pd.Series(filler))
                # Impute column if values are categorical.
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

        # Mark dataset as skipped if X has no samples, no features or only constant values.
        if X.shape[0] == 0 or X.shape[1] == 0:
            LOGGER.info('X has no samples, no features or only constant values. Marking dataset as skipped.')
            return {
                       'nr_inst': X.shape[0],
                       'nr_attr': X.shape[1],
                   }, False

        C_tmp = X.select_dtypes(exclude=['number'])
        N_tmp = X.select_dtypes(include=['number'])

        C = MetaFeatures._set_data_categoric(N_tmp, C_tmp, True)
        N = MetaFeatures._set_data_numeric(N_tmp, C_tmp, True)

        nr_inst = MFEGeneral.ft_nr_inst(X)
        nr_attr = MFEGeneral.ft_nr_attr(X)

        precomp_statistical = MFEStatistical.precompute_statistical_cor_cov(N)
        skewness = MFEStatistical.ft_skewness(N)
        kurtosis = MFEStatistical.ft_kurtosis(N)

        if N.shape[1] > 1:
            cor = MFEStatistical.ft_cor(N, abs_corr_mat=precomp_statistical['abs_corr_mat'])
            cor = cor[~np.isnan(cor)]
            cov = MFEStatistical.ft_cov(N, cov_mat=precomp_statistical['cov_mat'])
        else:
            cor = np.ones(1)
            cov = np.zeros(1)
        sparsity = MFEStatistical.ft_sparsity(X.to_numpy())
        var = MFEStatistical.ft_var(N)

        precomp_info = MFEInfoTheory.precompute_class_freq(y)
        precomp_info.update(MFEInfoTheory.precompute_entropy(y, C))
        if C.size > 0:
            attr_ent = MFEInfoTheory.ft_attr_ent(C, precomp_info['attr_ent'])
            mut_inf = MFEInfoTheory.ft_mut_inf(C, y, precomp_info['mut_inf'])
            eq_num_attr = MFEInfoTheory.ft_eq_num_attr(C, y, class_ent=precomp_info['class_ent'],
                                                       mut_inf=precomp_info['mut_inf'])
            ns_ratio = MFEInfoTheory.ft_ns_ratio(C, y, attr_ent=precomp_info['attr_ent'],
                                                 mut_inf=precomp_info['mut_inf'])
        else:
            attr_ent = np.zeros(2)
            mut_inf = np.zeros(2)
            eq_num_attr = 0
            ns_ratio = 0

        precomp_model = MFEModelBased.precompute_model_based_class(N, y, None)
        leaves_branch = MFEModelBased.ft_leaves_branch(precomp_model['table'], precomp_model['tree_depth'])
        leaves_per_class = MFEModelBased.ft_leaves_per_class(precomp_model['table'])
        var_importance = MFEModelBased.ft_var_importance(precomp_model['model'])

        return {
                   'nr_inst': int(nr_inst),
                   'nr_attr': int(nr_attr),
                   'nr_num': int(X.shape[1] - C_tmp.shape[1]),
                   'nr_cat': int(C_tmp.shape[1]),
                   'nr_class': int(MFEGeneral.ft_nr_class(y)),
                   'nr_missing_values': int(nr_missing_values),
                   'pct_missing_values': float(pct_missing_values),
                   'nr_inst_mv': int(nr_inst_mv),
                   'pct_inst_mv': float(pct_inst_mv),
                   'nr_attr_mv': int(nr_attr_mv),
                   'pct_attr_mv': float(pct_attr_mv),

                   'nr_outliers': int(MFEStatistical.ft_nr_outliers(N)),
                   'skewness_mean': float(skewness.mean()),
                   'skewness_sd': float(skewness.std(ddof=1)) if nr_attr > 1 else 0,
                   'kurtosis_mean': float(kurtosis.mean()),
                   'kurtosis_sd': float(kurtosis.std(ddof=1)) if nr_attr > 1 else 0,
                   'cor_mean': float(cor.mean()) if nr_attr > 1 else 1,
                   'cor_sd': float(cor.std(ddof=1)) if nr_attr > 2 else 0,
                   'cov_mean': float(cov.mean()) if nr_attr > 1 else 0,
                   'cov_sd': float(cov.std(ddof=1)) if nr_attr > 2 else 0,
                   'sparsity_mean': float(sparsity.mean()),
                   'sparsity_sd': float(sparsity.std(ddof=1)) if nr_attr > 1 else 0,
                   'var_mean': float(var.mean()),
                   'var_sd': float(var.std(ddof=1)) if nr_attr > 1 else 0,
                   'class_prob_mean': float(class_prob.mean()),
                   'class_prob_std': float(class_prob.std(ddof=0)),

                   'class_ent': float(MFEInfoTheory.ft_class_ent(y, precomp_info['class_ent'], precomp_info['class_freqs'])),
                   'attr_ent_mean': float(attr_ent.mean()),
                   'attr_ent_sd': float(attr_ent.std(ddof=1)) if nr_attr > 1 else 0,
                   'mut_inf_mean': float(mut_inf.mean()),
                   'mut_inf_sd': float(mut_inf.std(ddof=1)) if nr_attr > 1 else 0,
                   'eq_num_attr': float(eq_num_attr),
                   'ns_ratio': float(ns_ratio),

                   'nodes': float(MFEModelBased.ft_nodes(precomp_model['table'])),
                   'leaves': float(MFEModelBased.ft_leaves(precomp_model['table'])),
                   'leaves_branch_mean': float(leaves_branch.mean()),
                   'leaves_branch_sd': float(leaves_branch.std(ddof=1)),
                   'nodes_per_attr': float(MFEModelBased.ft_nodes_per_attr(N, precomp_model['table'])),
                   'leaves_per_class_mean': float(leaves_per_class.mean()),
                   'leaves_per_class_sd': float(leaves_per_class.std(ddof=1)) if not np.isnan(leaves_per_class).any() else 0,
                   'var_importance_mean': float(var_importance.mean()),
                   'var_importance_sd': float(var_importance.std(ddof=1)) if nr_attr > 1 else 0,

                   'one_nn_mean': 0, 'one_nn_sd': 0,
                   'best_node_mean': 0, 'best_node_sd': 0,
                   'linear_discr_mean': 0, 'linear_discr_sd': 0,
                   'naive_bayes_mean': 0, 'naive_bayes_sd': 0
               }, True

    @classmethod
    def ft_nr_missing_val(cls, M):
        return int(M.sum().sum())

    @classmethod
    def ft_nr_inst_missing_values(cls, M):
        return int((M.sum(axis=1) > 0).sum())

    @classmethod
    def ft_nr_attr_missing_values(cls, M):
        return int((M.sum(axis=0) > 0).sum())

    @classmethod
    def ft_class_prob(cls, y):
        return y.value_counts(normalize=True)

    @classmethod
    def _set_data_categoric(cls, N, C, transform_num: bool,
                            num_bins: bool = None) -> np.ndarray:
        data_cat = C.to_numpy()

        if transform_num and not N.empty:
            data_num_discretized = _internal.transform_num(N, num_bins=num_bins)

            if data_num_discretized is not None:
                data_cat = np.concatenate((data_cat, data_num_discretized), axis=1)
        return data_cat

    @classmethod
    def _set_data_numeric(
            cls,
            N, C,
            transform_cat: bool) -> np.ndarray:
        data_num = N

        if transform_cat and not C.empty:
            categorical_dummies = pd.get_dummies(C)

            if categorical_dummies is not None:
                data_num = pd.concat([data_num, categorical_dummies],
                                     axis=1).astype(float)

        return data_num.to_numpy()
