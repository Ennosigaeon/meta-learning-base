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
db = Database('postgres', 'default', 'postgres', 'postgres', 'localhost', 5432)

engine = db.engine
with engine.connect() as conn:
    select_statement = '''
        select * from datasets
        WHERE nr_cat is null
        order by id desc;
        '''

    rs = conn.execute(select_statement)
    for row in rs:
        id = row['id']
        name = row['name']
        class_column = row['class_column']
        store = True
        print(id)

        try:
            # df = load_data('data/' + name + '.parquet')
            local_file = '../data/' + name + '.parquet'
            df = load_data(local_file, name=name)

            mf, success = MetaFeatures().calculate(df=df, class_column=class_column)
            if mf['nr_inst'] == 0 or mf['nr_attr'] == 0:
                print('Empty dataframe')
                update_statement = '''
                    UPDATE datasets SET 
                        nr_inst=0,
                        nr_attr=0,
                        nr_num=0,
                        nr_cat=0,
                        nr_class='NaN',
                        nr_missing_values='NaN',
                        pct_missing_values='NaN',
                        nr_inst_mv='NaN',
                        pct_inst_mv='NaN',
                        nr_attr_mv='NaN',
                        pct_attr_mv='NaN',
                        nr_outliers='NaN',
                        skewness_mean='NaN',
                        skewness_sd='NaN',
                        kurtosis_mean='NaN',
                        kurtosis_sd='NaN',
                        cor_mean='NaN',
                        cor_sd='NaN',
                        cov_mean='NaN',
                        cov_sd='NaN',
                        sparsity_mean='NaN',
                        sparsity_sd='NaN',
                        var_mean='NaN',
                        var_sd='NaN',
                        class_prob_mean='NaN',
                        class_prob_std='NaN',
                        class_ent='NaN',
                        attr_ent_mean='NaN',
                        attr_ent_sd='NaN',
                        mut_inf_mean='NaN',
                        mut_inf_sd='NaN',
                        eq_num_attr='NaN',
                        ns_ratio='NaN',
                        nodes='NaN',
                        leaves='NaN',
                        leaves_branch_mean='NaN',
                        leaves_branch_sd='NaN',
                        nodes_per_attr='NaN',
                        leaves_per_class_mean='NaN',
                        leaves_per_class_sd='NaN',
                        var_importance_mean='NaN',
                        var_importance_sd='NaN',
                        one_nn_mean='NaN',
                        one_nn_sd='NaN',
                        best_node_mean='NaN',
                        best_node_sd='NaN',
                        linear_discr_mean='NaN',
                        linear_discr_sd='NaN',
                        naive_bayes_mean='NaN',
                        naive_bayes_sd='NaN'
                    WHERE id={};
                    '''.format(id)
                conn.execute(update_statement)
                continue
            if 'nr_class' not in mf:
                print('Calculation failed.')
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
                    nr_num={nr_num},
                    nr_cat={nr_cat},
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

            print('Updating')
            conn.execute(update_statement)
        except OSError as ex:
            print(ex)
