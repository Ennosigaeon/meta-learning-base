import logging
from datetime import datetime

import sklearn
# import tpot
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

# from arff2pandas import a2p
import pandas as pd
import numpy as np

from config import DatasetConfig
from data import load_openml
from database import Database
from metafeatures import MetaFeatures
from utilities import logloss, multiclass_roc_auc_score

logging.basicConfig(level=logging.DEBUG)

"""Database Credentials"""
# db = Database('sqlite', 'ml-base.db')
db = Database('postgres', 'april_dump', 'postgres', 'postgres', '192.168.50.4', 5432)

"""Load new dataset from OpenML, CSV or ARFF"""
config = DatasetConfig({'openml': 1510, 'train_path': None})
df = load_openml(config)
# df = pd.read_csv('')
# with open('amldata.arff') as f:
#     df = a2p.load(f)

"""Set class column to be predicted"""
class_column = 'Class'

"""Calculate Meta-Features for new Dataset"""
mf, success = MetaFeatures().calculate(df=df, class_column=class_column)

"""Transform Meta-Features into a Vector"""
new_mf_vec = pd.DataFrame(mf, index=['i']).to_numpy()

"""Save all Datasets from DB in a DataFrame"""
engine = db.engine
with engine.connect() as conn:
    datasets = pd.read_sql_query(
        '''SELECT 
        id, nr_inst, nr_attr, nr_class, nr_missing_values, pct_missing_values, nr_inst_mv, pct_inst_mv, nr_attr_mv, 
        pct_attr_mv, nr_outliers, skewness_mean, skewness_sd, kurtosis_mean, kurtosis_sd, cor_mean, cor_sd, cov_mean,
        cov_sd, sparsity_mean, sparsity_sd, var_mean, var_sd, class_prob_mean, class_prob_std, class_ent, attr_ent_mean,
        attr_ent_sd, mut_inf_mean, mut_inf_sd, eq_num_attr, ns_ratio, nodes, leaves, leaves_branch_mean,
        leaves_branch_sd, nodes_per_attr, leaves_per_class_mean, leaves_per_class_sd, var_importance_mean,
        var_importance_sd, one_nn_mean, one_nn_sd, best_node_mean, best_node_sd, linear_discr_mean, linear_discr_sd,
        naive_bayes_mean, naive_bayes_sd
        FROM datasets ORDER BY id desc LIMIT 1000''', conn)  # LIMIT just for testing

id_vec = datasets['id']
datasets = datasets.drop(columns=['id'])

"""Calculate euclidean distance to search for most similar Dataset"""
dist_list = []

for index, row in datasets.iterrows():
    curr_row = row.to_numpy(dtype="float")
    dist = np.linalg.norm(new_mf_vec - curr_row)
    dist_list.append(dist)

distances = pd.Series(dist_list, name='distance')
top_df = pd.concat([id_vec, datasets, distances], axis=1).sort_values(by='distance').head(10)

"""Get Dataset with most similar Meta-Features"""
most_sim = top_df.iloc[0]['id']

"""Get Pipelines for Dataset from Database"""
engine = db.engine
with engine.connect() as conn:
    pipelines = pd.read_sql_query( '''
        with recursive pipelines (input_dataset, id, algorithm, output_dataset, accuracy, depth, treepath) as 
        (select  input_dataset, id, algorithm, output_dataset, accuracy, 0 as depth,
        CAST (id AS VARCHAR(255)) AS treepath
        from algorithms
        where input_dataset = '215060' and status = 'complete'
        
        union all 
        
        select  a2.input_dataset, a2.id, a2.algorithm, a2.output_dataset, a2.accuracy, b2.depth + 1 as depth,
        CAST (b2.treepath || '->' || CAST (a2.id as VARCHAR(255)) as VARCHAR(255)) as treepath
        from algorithms a2
        
        inner join pipelines b2 on a2.input_dataset = b2.output_dataset)
        
        select input_dataset, id, algorithm, output_dataset, accuracy, treepath, depth from pipelines 
        '''.format(most_sim), conn)

pipelines = pipelines.sort_values(by='accurancy', ascending=False).head(5)
"""Run Pipelines on new Dataset"""
# pipelines = pd.DataFrame(np.array([['p1', 2, '18214840000182150100001821463'],
#                                    ['p2', 5, '1821484.1821501.1821507'],
#                                    ['p3', 8, '1821484.1821501.1822276']]),
#                          columns=['bla1', 'bla2', 'path'])

X_out, y_out = df.drop(class_column, axis=1), df[class_column]
pipeline_res = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1_score', 'log_loss', 'roc_auc'])

for index, row in pipelines.iterrows():
    path = row['path'].split('->')

    for i in path:
        algorithm = db.get_algorithm(i)
        algorithm = algorithm.instance()

        if is_classifier(algorithm):
            resultlist = []

            """Predict labels with n fold cross validation"""
            y_pred = cross_val_predict(algorithm, X_out, y_out, cv=5)

            """Calculate evaluation metrics"""
            accuracy = accuracy_score(y_out, y_pred)
            resultlist.append(accuracy)
            precision = precision_score(y_out, y_pred, average='weighted')
            resultlist.append(precision)
            recall = recall_score(y_out, y_pred, average='weighted')
            resultlist.append(recall)
            f1 = f1_score(y_out, y_pred, average='weighted')
            resultlist.append(f1)
            log_loss = logloss(y_out, y_pred)
            resultlist.append(log_loss)
            roc_auc = multiclass_roc_auc_score(y_out, y_pred, average='weighted')
            resultlist.append(roc_auc)

            pipeline_res.loc[len(pipeline_res.index)] = resultlist

        else:
            """
            If algorithm object has method fit_transform, call fit_transform on X, y. Else, first call fit on X, y,
            then transform on X. Safe the transformed dataset in X
            """
            if hasattr(algorithm, 'fit_transform'):
                X_out = algorithm.fit_transform(X_out, y_out)
            else:
                # noinspection PyUnresolvedReferences
                X_out = algorithm.fit(X_out, y_out).transform(X_out)

            X_out = pd.DataFrame(data=X_out, index=range(X_out.shape[0]), columns=range(X_out.shape[1]))


"""Find Pipelines for new Dataset with AutoML, TPOT, Random Search"""
X, y = df.drop(class_column, axis=1), df[class_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

"""TPOT"""
# pipeline_optimizer = tpot.TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
# pipeline_optimizer.fit(X_train, y_train)
# print("Accuracy score", pipeline_optimizer.score(X_test, y_test))
# tpot.export('tpot_pipeline.py')

"""Auto-Sklearn"""
# from autosklearn.classification import AutoSklearnClassifier
#
# automl = AutoSklearnClassifier()
# automl.fit(X_train, y_train)
# y_hat = automl.predict(X_test)
# print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

"""Random Search / Random Pipeline from Database"""

"""Evaluate and compare Pipelines"""
