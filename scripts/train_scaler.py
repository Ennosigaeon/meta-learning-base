import pickle
import warnings

import joblib
import pandas as pd
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler

from automl.util import util

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

    log = ColumnTransformer([('log', FunctionTransformer(util.object_log), [0, 1, 4])], remainder='passthrough')
    scale = ColumnTransformer([('scale', RobustScaler(),
                                [0, 1, 2, 5, 7, 9, 11, 12, 13, 14, 15, 18, 19, 22, 23] + list(range(26, 38)))],
                              remainder='passthrough')
    ohe = ColumnTransformer([('ohe', OneHotEncoder(), [42])], remainder='passthrough')

    pipelines = {
        'rf': Pipeline([
            ('log', log),
            ('scale', scale),
            ('ohe', ohe),
            ('regression', RandomForestRegressor(n_jobs=6, n_estimators=20))
        ]),
        # 'svr': Pipeline([
        #     ('ohe', ohe),
        #     ('log', log),
        #     ('scale', scale),
        #     ('regression', SVR())
        # ]),
        'sgd': Pipeline([
            ('log', log),
            ('scale', scale),
            ('ohe', ohe),
            ('regression', SGDRegressor())
        ])
    }

    # Xt = pipeline.fit_transform(X_train, X_train['f1_score'])

    for schema in sorted(schemas):
        schema = 'complete'
        X_train = merged[merged['depth'] < 3]
        X_val = merged[merged['depth'] < 3]
        y_train = X_train['f1_score'], X_train['f1_score_var']
        y_val = X_val['f1_score'], X_val['f1_score_var']

        to_drop = ['schema', 'ds', 'depth', 'accuracy', 'accuracy_var', 'f1_score', 'f1_score_var', 'precision',
                   'precision_var', 'recall', 'recall_var', 'log_loss', 'log_loss_var', 'roc_auc', 'roc_auc_var']
        X_train = X_train.drop(to_drop, axis=1)
        X_val = X_val.drop(to_drop, axis=1)

        for name, pipeline in pipelines.items():
            mean_pipeline = clone(pipeline)
            mean_pipeline.fit(X_train.to_numpy(), y_train[0].to_numpy())

            var_pipeline = clone(pipeline)
            var_pipeline.fit(X_train.to_numpy(), y_train[1].to_numpy())

            mean_predict = mean_pipeline.predict(X_val)
            var_predict = var_pipeline.predict(X_val)
            print(name, schema, mean_squared_error(y_val[0], mean_predict),
                  mean_squared_error(y_val[1], var_predict))
            with open('assets/eval/{}_{}.pkl'.format(name, schema), 'wb') as f:
                joblib.dump((mean_pipeline, var_pipeline), f)
