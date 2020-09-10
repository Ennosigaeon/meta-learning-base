import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abydos.distance import Jaccard
from abydos.tokenizer import _Tokenizer
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

# TODO exclude datasets with all algorithm failed
# TODO pipelines with recurrent algorithms -> d1063_6485
# select * from pipelines p where d0_id = 1 and d1_id = 11 and d2_id = 6485
from nltk import ngrams


class NGramsTokenizer(_Tokenizer):

    def tokenize(self, string=None):
        self._string = string
        self._ordered_tokens = []
        for i in range(len(string)):
            self._ordered_tokens += ngrams(self._string, i)
        super(NGramsTokenizer, self).tokenize()
        return self


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


# def load_perf(series, metric: str = 'f1_score'):
#     return performances[series['schema']][performances[series['schema']]['ds'] == series['id']].sort_values(metric)

def calculate_input():
    with open('assets/exports/export_datasets_0.pkl', 'rb') as f:
        df = pickle.load(f)

    to_log = ['nr_inst', 'nr_attr', 'nr_class']
    df.dropna(inplace=True)
    df[to_log] = np.log(df[to_log])

    scaler = RobustScaler()
    to_scale = ['nr_inst', 'nr_attr', 'nr_class', 'skewness_mean', 'skewness_sd', 'kurtosis_mean', 'kurtosis_sd',
                'cov_mean', 'cov_sd', 'var_mean', 'var_sd', 'class_ent', 'attr_ent_mean', 'attr_ent_mean', 'ns_ratio',
                'nodes', 'leaves', 'leaves_branch_mean', 'leaves_branch_sd', 'nodes_per_attr']
    df[to_scale] = scaler.fit_transform(df[to_scale])

    print(scaler.center_)
    print(scaler.scale_)

    performances = {}
    schemas = pd.unique(df['schema'])
    for schema in schemas:
        with open('assets/exports/performance_{}.pkl'.format(schema), 'rb') as f:
            performances[schema] = pickle.load(f)
    performances = pd.concat(performances.values())

    original_datasets = df
    df = df.rename(columns={'id': 'ds'})
    df['index'] = df[['schema', 'ds']].astype(str).agg('_'.join, axis=1)
    df = df.set_index('index')
    performances = performances.rename(columns={'depth': 'steps'})

    merged = pd.merge(df, performances, on=['schema', 'ds'])
    merged['index'] = merged[['schema', 'ds']].astype(str).agg('_'.join, axis=1)
    merged = merged.set_index('index')

    df['order'] = ''
    for idx in merged.index:
        try:
            df.at[idx, 'order'] = extract_algo_order(merged.loc[idx])[1]
        except TypeError as ex:
            print(ex)

    with open('datasets_scaled.pkl', 'wb') as f:
        pickle.dump(df, f)
    with open('datasets_original.pkl', 'wb') as f:
        pickle.dump(original_datasets, f)
    with open('performances.pkl', 'wb') as f:
        pickle.dump(performances, f)
    with open('merged.pkl', 'wb') as f:
        pickle.dump(merged, f)

    return original_datasets, df, performances, merged


def load_input():
    with open('datasets_scaled.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('datasets_original.pkl', 'rb') as f:
        original_datasets = pickle.load(f)
    with open('performances.pkl', 'rb') as f:
        performances = pickle.load(f)
    with open('merged.pkl', 'rb') as f:
        merged = pickle.load(f)
    return original_datasets, df, performances, merged


def extract_algo_order(df: pd.DataFrame, weights=None):
    # TODO df has only one algorithm
    copy = df.copy()
    copy['metric'] = copy[metric]
    copy['metric_var'] = copy[metric + '_var'] * copy['steps']

    if weights is not None and weights.sum() > 0:
        for i, idx in enumerate(copy.index.unique()):
            copy.at[idx, 'metric'] = copy.at[idx, 'metric'] * weights[i]

    score = copy[['algo', 'metric', 'metric_var']] \
        .groupby('algo') \
        .agg('mean' if weights is None else 'sum') \
        .sort_values('metric', ascending=False)
    order = ''.join(score.index.map(mapping))
    return score, order


def ohe_encoding(df: pd.DataFrame, ohe: OneHotEncoder):
    try:
        check_is_fitted(ohe)
    except NotFittedError:
        ohe.fit(df['algo'].values.reshape(-1, 1))

    num_df = df.drop(
        ['schema', 'ds', 'depth', 'accuracy', 'accuracy_var', 'f1_score', 'f1_score_var', 'precision', 'precision_var',
         'recall', 'recall_var', 'log_loss', 'log_loss_var', 'roc_auc', 'roc_auc_var'], axis=1)
    ohe_algos = pd.DataFrame(ohe.transform(df['algo'].values.reshape(-1, 1)), columns=ohe.get_feature_names())
    num_df = pd.concat([num_df.reset_index(), ohe_algos], axis=1).drop(['algo'], axis=1).set_index('index')
    return num_df


mapping = {'ada_boosting': 'a', 'bernoulli_nb': 'b', 'bernoulli_rbm': 'c', 'binarizer': 'd', 'decision_tree': 'e',
           'factor_analysis': 'f', 'fast_ica': 'g', 'feature_agglomeration': 'h', 'generic_univariate_select': 'i',
           'gradient_boosting': 'j', 'imputation': 'k', 'kbinsdiscretizer': 'l', 'knn_imputer': 'm', 'kpca': 'n',
           'libsvm_svc': 'o', 'linear_discriminant_analysis': 'p', 'max_abs_scaler': 'q', 'minmax': 'r',
           'missing_indicator': 's', 'multi_column_label_encoder': 't', 'multinomial_nb': 'u', 'normalize': 'v',
           'one_hot_encoding': 'w', 'pca': 'x', 'polynomial': 'y', 'quantile_transformer': 'z', 'random_forest': '0',
           'random_trees_embedding': '1', 'robust_scaler': '2', 'select_k_best': '3', 'select_percentile': '4',
           'sgd': '5', 'standard_scaler': '6', 'truncated_svd': '7', 'variance_threshold': '8'}

metric = 'f1_score'
dist_measure = Jaccard(NGramsTokenizer())
k = 2

original_datasets, df, performances, merged = load_input()
schemas = pd.unique(df['schema'])

X_train = merged[merged['depth'] < 3]
rf = RandomForestRegressor(n_jobs=7)
ohe = OneHotEncoder(sparse=False)
num_X_train = ohe_encoding(X_train, ohe)
rf.fit(num_X_train, X_train['f1_score'])

with open('fitted_pipeline.pkl', 'wb') as f:
    pickle.dump((ohe, rf), f)

sys.exit()

# Remove datasets without any algorithms
df = df.loc[pd.unique(merged.index)]
for schema in schemas:
    distance = {'random': [], 'hill_climbing': [], 'empty': [], 'kdtree_2': [], 'wkdtree_2': [], 'kdtree_3': [],
                'wkdtree_3': [], 'kdtree_5': [], 'wkdtree_5': [], 'kdtree_10': [], 'wkdtree_10': [], 'rf': []}

    X_train = merged[(merged['depth'] < 3) & (merged['schema'] != schema)]
    X_val = merged[(merged['depth'] < 3) & (merged['schema'] == schema)]

    rf = RandomForestRegressor(n_jobs=7)
    ohe = OneHotEncoder(sparse=False)
    num_X_train = ohe_encoding(X_train, ohe)
    num_X_val = ohe_encoding(X_val, ohe)

    rf.fit(num_X_train, X_train['f1_score'])
    perf_predict = rf.predict(num_X_val)
    perf_true = X_val['f1_score']
    print(mean_squared_error(perf_true, perf_predict))
    a = 0

    X_train = df[(df['depth'] < 3) & (df['schema'] != schema)]
    X_val = df[(df['depth'] < 3) & (df['schema'] == schema)]
    # print(X_train.shape)
    #
    tree = KDTree(X_train.drop(['schema', 'ds', 'depth', 'order'], axis=1))
    #
    #
    # hc_score, hc_order = extract_algo_order(merged.loc[X_train.index])
    # for tup in X_val.iterrows():
    #     idx, row = tup
    #     try:
    #         true_perf = merged.loc[idx].copy()
    #         true_score, true_order = extract_algo_order(true_perf)
    #
    #         all_options = ''.join(mapping.values())
    #         distance['random'].append(dist_measure.dist(true_order,
    #                                                     ''.join(random.sample(all_options, len(all_options)))))
    #         distance['hill_climbing'].append(dist_measure.dist(true_order, hc_order))
    #         distance['empty'].append(dist_measure.dist(true_order, ''))
    #
    #         for k in [2, 3, 5, 10]:
    #             dist, ref_idx = tree.query(row.drop(['schema', 'ds', 'depth', 'order']), k=k)
    #             reference = X_train.iloc[ref_idx, :]
    #             weight = 1 - dist / dist.sum()
    #
    #             perf_ref: pd.DataFrame = merged.loc[reference.index]
    #             kd_score, kd_order = extract_algo_order(perf_ref)
    #             wkd_score, wkd_order = extract_algo_order(perf_ref, weight)
    #
    #             distance['kdtree_{}'.format(k)].append(dist_measure.dist(true_order, kd_order))
    #             distance['wkdtree_{}'.format(k)].append(dist_measure.dist(true_order, wkd_order))
    #
    #     except KeyError as ex:
    #         print(ex)
    #     except TypeError as ex:
    #         print(ex)

    # Plot distances
    d, best_match = tree.query(X_val.drop(['schema', 'ds', 'depth', 'order'], axis=1), k=k)
    if d.ndim == 1:
        d = np.atleast_2d(d).T
    best_match = np.atleast_2d(best_match).T

    d_sanitized = reject_outliers(d.mean(axis=1))
    print(schema, distance.keys())
    for key, value in distance.items():
        print(pd.Series(value).mean())
    print('\n\n\n')

    # noinspection PyTypeChecker
    kde: KernelDensity = KernelDensity().fit(d_sanitized.reshape(-1, 1))
    X_plot = np.linspace(d_sanitized.min(), d_sanitized.max(), 1000)
    y_plot = kde.score_samples(X_plot.reshape(-1, 1))
    plt.figure()
    plt.plot(X_plot, np.exp(y_plot))
    # plt.hist(d)
    plt.title(schema)
    plt.savefig('assets/eval/{}.png'.format(schema))
    plt.show()

    # for i in range(2, 20):
    #     ms = KMeans(n_clusters=i)
    #     ms.fit(data)
    #
    #     n_clusters = len(ms.cluster_centers_)
    #     print("n Cluster: {}".format(n_clusters), ms.inertia_ / data.shape[0])
    #     # print("Silhouette Coefficient: {0.3f}".format(metrics.silhouette_score(data, ms.labels_, metric='sqeuclidean')))
