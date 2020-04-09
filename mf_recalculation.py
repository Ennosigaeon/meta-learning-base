import os
import sys
import warnings

import math
import numpy as np

from data import load_data
from database import Database
from metafeatures import MetaFeatures

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

# db = Database('sqlite', 'ml-base.db')
db = Database('postgres', 'postgres', 'postgres', 'usu4867!', '35.242.255.138', 5432)

engine = db.engine
with engine.connect() as conn:
    select_statement = '''
        select id, "name", class_column from datasets
        WHERE nr_inst = 'NaN' or nr_attr = 'NaN' or nr_class = 'NaN' or nr_missing_values = 'NaN' or pct_missing_values = 'NaN' or nr_inst_mv = 'NaN' or pct_inst_mv = 'NaN' or nr_attr_mv = 'NaN' or pct_attr_mv = 'NaN' or nr_outliers = 'NaN' or skewness_mean = 'NaN' or skewness_sd = 'NaN' or kurtosis_mean = 'NaN' or kurtosis_sd = 'NaN' or cor_mean = 'NaN' or cor_sd = 'NaN' or cov_mean = 'NaN' or cov_sd = 'NaN' or sparsity_mean = 'NaN' or sparsity_sd = 'NaN' or var_mean = 'NaN' or var_sd = 'NaN' or class_prob_mean = 'NaN' or class_prob_std = 'NaN' or class_ent = 'NaN' or attr_ent_mean = 'NaN' or attr_ent_sd = 'NaN' or mut_inf_mean = 'NaN' or mut_inf_sd = 'NaN' or eq_num_attr = 'NaN' or ns_ratio = 'NaN' or nodes = 'NaN' or leaves = 'NaN' or leaves_branch_mean = 'NaN' or leaves_branch_sd = 'NaN' or nodes_per_attr = 'NaN' or leaves_per_class_mean = 'NaN' or leaves_per_class_sd = 'NaN' or var_importance_mean = 'NaN' or var_importance_sd = 'NaN' or one_nn_mean = 'NaN' or one_nn_sd = 'NaN' or best_node_mean = 'NaN' or best_node_sd = 'NaN' or linear_discr_mean = 'NaN' or linear_discr_sd = 'NaN' or naive_bayes_mean = 'NaN' or naive_bayes_sd = 'NaN' or
              nr_inst IS NULL or nr_attr IS NULL or nr_class IS NULL or nr_missing_values IS NULL or pct_missing_values IS NULL or nr_inst_mv IS NULL or pct_inst_mv IS NULL or nr_attr_mv IS NULL or pct_attr_mv IS NULL or nr_outliers IS NULL or skewness_mean IS NULL or skewness_sd IS NULL or kurtosis_mean IS NULL or kurtosis_sd IS NULL or cor_mean IS NULL or cor_sd IS NULL or cov_mean IS NULL or cov_sd IS NULL or sparsity_mean IS NULL or sparsity_sd IS NULL or var_mean IS NULL or var_sd IS NULL or class_prob_mean IS NULL or class_prob_std IS NULL or class_ent IS NULL or attr_ent_mean IS NULL or attr_ent_sd IS NULL or mut_inf_mean IS NULL or mut_inf_sd IS NULL or eq_num_attr IS NULL or ns_ratio IS NULL or nodes IS NULL or leaves IS NULL or leaves_branch_mean IS NULL or leaves_branch_sd IS NULL or nodes_per_attr IS NULL or leaves_per_class_mean IS NULL or leaves_per_class_sd IS NULL or var_importance_mean IS NULL or var_importance_sd IS NULL or one_nn_mean IS NULL or one_nn_sd IS NULL or best_node_mean IS NULL or best_node_sd IS NULL or linear_discr_mean IS NULL or linear_discr_sd IS NULL or naive_bayes_mean IS NULL or naive_bayes_sd IS null
        order by id;
        '''

    rs = conn.execute(select_statement)
    for row in rs:
        id = row['id']
        name = row['name']
        class_column = row['class_column']
        store = True
        print(id)

        # df = load_data('data/' + name + '.parquet')
        local_file = 'data/' + name + '.parquet'
        df = load_data(local_file, s3_config='assets/limbo-233520-a283e9f868c1.json',
                       s3_bucket='usu-mlb', name=name)
        if df.shape[1] > 10000:
            print('Skipping {} due to many features'.format(id))
            continue

        mf = MetaFeatures().calculate(df=df, class_column=class_column)
        if mf['nr_inst'] == 0 and mf['nr_attr'] == 0:
            print('Empty dataframe')
            continue

        for key, value in mf.items():
            if math.isinf(value):
                if value > 0:
                    mf[key] = sys.maxsize
                else:
                    mf[key] = -sys.maxsize
            if np.isnan(value):
                store = False
                break

        if not store:
            print('Skipping {} due to missing value in {}'.format(id, key))
            continue

        update_statement = '''
            UPDATE datasets SET 
                nr_inst={nr_inst},
                nr_attr={nr_attr},
                nr_class={nr_class},
                nr_missing_values={nr_missing_values},
                pct_missing_values={pct_missing_values},
                nr_inst_mv={nr_inst_mv},
                pct_inst_mv={pct_inst_mv},
                nr_attr_mv={nr_attr_mv},
                pct_attr_mv={pct_attr_mv},
                nr_outliers={nr_outliers},
                skewness_mean={skewness_mean},
                skewness_sd={skewness_sd},
                kurtosis_mean={kurtosis_mean},
                kurtosis_sd={kurtosis_sd},
                cor_mean={cor_mean},
                cor_sd={cor_sd},
                cov_mean={cov_mean},
                cov_sd={cov_sd},
                sparsity_mean={sparsity_mean},
                sparsity_sd={sparsity_sd},
                var_mean={var_mean},
                var_sd={var_sd},
                class_prob_mean={class_prob_mean},
                class_prob_std={class_prob_std},
                class_ent={class_ent},
                attr_ent_mean={attr_ent_mean},
                attr_ent_sd={attr_ent_sd},
                mut_inf_mean={mut_inf_mean},
                mut_inf_sd={mut_inf_sd},
                eq_num_attr={eq_num_attr},
                ns_ratio={ns_ratio},
                nodes={nodes},
                leaves={leaves},
                leaves_branch_mean={leaves_branch_mean},
                leaves_branch_sd={leaves_branch_sd},
                nodes_per_attr={nodes_per_attr},
                leaves_per_class_mean={leaves_per_class_mean},
                leaves_per_class_sd={leaves_per_class_sd},
                var_importance_mean={var_importance_mean},
                var_importance_sd={var_importance_sd},
                one_nn_mean={one_nn_mean},
                one_nn_sd={one_nn_sd},
                best_node_mean={best_node_mean},
                best_node_sd={best_node_sd},
                linear_discr_mean={linear_discr_mean},
                linear_discr_sd={linear_discr_sd},
                naive_bayes_mean={naive_bayes_mean},
                naive_bayes_sd={naive_bayes_sd}
            WHERE id={id};
            '''.format(**mf, id=id)
        # print(update_statement)

        print('Updating')
        conn.execute(update_statement)

        os.remove(local_file)
