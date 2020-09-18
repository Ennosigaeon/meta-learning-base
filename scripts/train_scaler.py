import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.simplefilter("ignore", category=FutureWarning)


# TODO exclude datasets with all algorithm failed
# TODO pipelines with recurrent algorithms -> d1063_6485
# select * from pipelines p where d0_id = 1 and d1_id = 11 and d2_id = 6485


def calculate_input():
    with open('assets/exports/export_datasets_0.pkl', 'rb') as f:
        df = pickle.load(f)

    df.dropna(inplace=True)
    performances = {}
    schemas = pd.unique(df['schema'])
    for schema in schemas:
        with open('assets/exports/performance_{}.pkl'.format(schema), 'rb') as f:
            performances[schema] = pickle.load(f)
    performances = pd.concat(performances.values())

    df = df.rename(columns={'id': 'ds'})
    df['index'] = df[['schema', 'ds']].astype(str).agg('_'.join, axis=1)
    df = df.set_index('index')
    performances = performances.rename(columns={'depth': 'steps'})

    merged = pd.merge(df, performances, on=['schema', 'ds'])
    merged['index'] = merged[['schema', 'ds']].astype(str).agg('_'.join, axis=1)
    merged = merged.set_index('index')

    # Remove datasets without any algorithms
    df = df.loc[pd.unique(merged.index)]

    with open('extracted_data.pkl', 'wb') as f:
        pickle.dump((df, performances, merged), f)

    return df, performances, merged


def load_input():
    with open('extracted_data.pkl', 'rb') as f:
        df, performances, merged = pickle.load(f)
    return df, performances, merged


if __name__ == '__main__':

    metric = 'f1_score'

    df, performances, merged = load_input()

    schemas = pd.unique(df['schema'])
    X_train = merged[merged['depth'] < 3]

    from sklearn_pandas import DataFrameMapper

    ohe = DataFrameMapper([(['algo'], OneHotEncoder(), {'prefix': ''})], df_out=True, default=None)
    log = DataFrameMapper([(['nr_inst'], FunctionTransformer(np.log)),
                           (['nr_attr'], FunctionTransformer(np.log)),
                           (['nr_class'], FunctionTransformer(np.log)), ], df_out=True, input_df=True, default=None)
    scale = DataFrameMapper([(['nr_inst'], RobustScaler()),
                             (['nr_attr'], RobustScaler()),
                             (['nr_class'], RobustScaler()),
                             (['skewness_mean'], RobustScaler()),
                             (['skewness_sd'], RobustScaler()),
                             (['kurtosis_mean'], RobustScaler()),
                             (['kurtosis_sd'], RobustScaler()),
                             (['cov_mean'], RobustScaler()),
                             (['cov_sd'], RobustScaler()),
                             (['var_mean'], RobustScaler()),
                             (['var_sd'], RobustScaler()),
                             (['class_ent'], RobustScaler()),
                             (['attr_ent_mean'], RobustScaler()),
                             (['attr_ent_sd'], RobustScaler()),
                             (['ns_ratio'], RobustScaler()),
                             (['nodes'], RobustScaler()),
                             (['leaves'], RobustScaler()),
                             (['leaves_branch_mean'], RobustScaler()),
                             (['leaves_branch_sd'], RobustScaler()),
                             (['nodes_per_attr'], RobustScaler())], df_out=True, input_df=True, default=None)

    pipelines = {
        'rf': Pipeline([
            ('ohe', ohe),
            ('log', log),
            ('scale', scale),
            ('regression', RandomForestRegressor(n_jobs=6, n_estimators=20))
        ]),
        # 'svr': Pipeline([
        #     ('ohe', ohe),
        #     ('log', log),
        #     ('scale', scale),
        #     ('regression', SVR())
        # ]),
        'sgd': Pipeline([
            ('ohe', ohe),
            ('log', log),
            ('scale', scale),
            ('regression', SGDRegressor())
        ])
    }

    # Xt = pipeline.fit_transform(X_train, X_train['f1_score'])

    for schema in sorted(schemas):
        X_train = merged[(merged['depth'] < 3) & (merged['schema'] != schema)]
        X_val = merged[(merged['depth'] < 3) & (merged['schema'] == schema)]
        y_train = X_train['f1_score'], X_train['f1_score_var']
        y_val = X_val['f1_score'], X_val['f1_score_var']

        to_drop = ['schema', 'ds', 'depth', 'accuracy', 'accuracy_var', 'f1_score', 'f1_score_var', 'precision',
                   'precision_var', 'recall', 'recall_var', 'log_loss', 'log_loss_var', 'roc_auc', 'roc_auc_var']
        X_train = X_train.drop(to_drop, axis=1)
        X_val = X_val.drop(to_drop, axis=1)

        for name, pipeline in pipelines.items():
            mean_pipeline = clone(pipeline)
            mean_pipeline.fit(X_train, y_train[0])

            var_pipeline = clone(pipeline)
            var_pipeline.fit(X_train, y_train[1])

            mean_predict = mean_pipeline.predict(X_val)
            var_predict = var_pipeline.predict(X_val)
            print(name, schema, mean_squared_error(y_val[0], mean_predict),
                  mean_squared_error(y_val[1], var_predict))
            with open('assets/eval/{}_{}.pkl'.format(name, schema), 'wb') as f:
                joblib.dump((mean_pipeline, var_pipeline), f)
