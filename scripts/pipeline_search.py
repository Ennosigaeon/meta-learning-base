import logging

import sklearn
from arff2pandas import a2p
import pandas as pd
import numpy as np

from config import DatasetConfig
from data import load_openml
from database import Database
from metafeatures import MetaFeatures

logging.basicConfig(level=logging.DEBUG)

"""Database Credentials"""
# db = Database('sqlite', 'ml-base.db')
db = Database('postgres', 'postgres', 'postgres', 'usu4867!', '35.242.255.138', 5432)

"""Load new dataset from OpenML, CSV or ARFF"""
config = DatasetConfig({'openml': 15, 'train_path': None})
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
    select_statement = '''
        WITH recursive pipelines
                (AlgoID,
                Algorithm,
                InputData,
                OutputData,
                Depth)
        AS (SELECT
                id,
                algorithm,
                input_dataset,
                output_dataset,
                0
                
        FROM algorithms
        WHERE input_dataset = {}
        
        UNION ALL
        
        SELECT
                algorithms.id,
                algorithms.algorithm,
                algorithms.input_dataset,
                algorithms.output_dataset,
                pipelines.Depth + 1
                
        FROM algorithms
            JOIN pipelines ON algorithms.input_dataset = pipelines.OutputData
        )   
        SELECT * 
        FROM pipelines
        ORDER BY Depth
        '''.format(most_sim)

"""Run Pipelines on new Dataset"""

"""Find Pipelines for new Dataset with AutoML, TPOT, Random Search"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, class_column, train_size=0.75, test_size=0.25)

"""TPOT"""
import tpot

pipeline_optimizer = tpot.TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print("Accuracy score", pipeline_optimizer.score(X_test, y_test))
tpot.export('tpot_pipeline.py')

"""Auto-Sklearn"""
from autosklearn import classification

automl = classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

"""Random Search / Random Pipeline from Database"""

"""Evaluate and compare Pipelines"""
