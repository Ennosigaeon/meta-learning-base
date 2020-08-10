from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

db_url = URL(drivername='postgres', database='default', username='postgres',
             password='postgres', host='localhost', port=5432)
engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)

with engine.connect() as con:
    con.execute('SET work_mem TO "2000MB";')
    for schema in ['d1461', 'd1464', 'd1486', 'd1489', 'd1590', 'd23512', 'd23517', 'd3', 'd31', 'd40668', 'd40685',
                   'd40975', 'd40981', 'd40984', 'd41027', 'd41143', 'd41146', 'd54']:
        print(schema)
        con.execute('SET search_path TO {};'.format(schema))

        # Remove obsolete algorithms
        con.execute('''
                delete from algorithms where id not in (
                    select a.id from algorithms a
                    join datasets d on a.input_dataset = d.id
                    where a.start_time >= d.start_time and a.end_time <= d.end_time
                );''')
        # Mark algorithms as errored that did not modify dataset
        con.execute('''
                update algorithms
                set status = 'errored', accuracy = 0, f1_score = 0, "precision" = 0, recall = 0, roc_auc_score = 0, neg_log_loss = 100
                where output_dataset = input_dataset and status != 'errored';''')
        # Remove dataset duplicates
        con.execute('''
                update algorithms a
                set output_dataset = sub.id
                from (
                    select id, unnest(replacement) as replacement from (
                        select min(id) as id, count(*) as count, array_agg(id) as replacement
                        from datasets d
                        where status != 'skipped' and nr_inst != 0
                        group by nr_inst, nr_attr, nr_class, nr_missing_values, pct_missing_values, nr_inst_mv, pct_inst_mv, nr_attr_mv, pct_attr_mv, nr_outliers, skewness_mean, skewness_sd, kurtosis_mean, kurtosis_sd, cor_mean, cor_sd, cov_mean, cov_sd, sparsity_mean, sparsity_sd, var_mean, var_sd, class_prob_mean, class_prob_std, class_ent, attr_ent_mean, attr_ent_sd, mut_inf_mean, mut_inf_sd, eq_num_attr, ns_ratio, nodes, leaves, leaves_branch_mean, leaves_branch_sd, nodes_per_attr, leaves_per_class_mean, leaves_per_class_sd, var_importance_mean, var_importance_sd, one_nn_mean, one_nn_sd, best_node_mean, best_node_sd, linear_discr_mean, linear_discr_sd, naive_bayes_mean, naive_bayes_sd
                    ) s where count > 1
                ) as sub
                where a.output_dataset = sub.replacement and sub.id != sub.replacement;
                update algorithms a
                set input_dataset = sub.id
                from (
                    select id, unnest(replacement) as replacement from (
                        select min(id) as id, count(*) as count, array_agg(id) as replacement
                        from datasets d
                        where status != 'skipped' and nr_inst != 0
                        group by nr_inst, nr_attr, nr_class, nr_missing_values, pct_missing_values, nr_inst_mv, pct_inst_mv, nr_attr_mv, pct_attr_mv, nr_outliers, skewness_mean, skewness_sd, kurtosis_mean, kurtosis_sd, cor_mean, cor_sd, cov_mean, cov_sd, sparsity_mean, sparsity_sd, var_mean, var_sd, class_prob_mean, class_prob_std, class_ent, attr_ent_mean, attr_ent_sd, mut_inf_mean, mut_inf_sd, eq_num_attr, ns_ratio, nodes, leaves, leaves_branch_mean, leaves_branch_sd, nodes_per_attr, leaves_per_class_mean, leaves_per_class_sd, var_importance_mean, var_importance_sd, one_nn_mean, one_nn_sd, best_node_mean, best_node_sd, linear_discr_mean, linear_discr_sd, naive_bayes_mean, naive_bayes_sd
                    ) s where count > 1
                ) as sub
                where a.input_dataset = sub.replacement and sub.id != sub.replacement;
                delete from datasets where id in (
                  select replacement from (
                    select id, unnest(replacement) as replacement from (
                      select min(id) as id, count(*) as count, array_agg(id) as replacement
                        from datasets d
                        where status != 'skipped' and nr_inst != 0
                        group by nr_inst, nr_attr, nr_class, nr_missing_values, pct_missing_values, nr_inst_mv, pct_inst_mv, nr_attr_mv, pct_attr_mv, nr_outliers, skewness_mean, skewness_sd, kurtosis_mean, kurtosis_sd, cor_mean, cor_sd, cov_mean, cov_sd, sparsity_mean, sparsity_sd, var_mean, var_sd, class_prob_mean, class_prob_std, class_ent, attr_ent_mean, attr_ent_sd, mut_inf_mean, mut_inf_sd, eq_num_attr, ns_ratio, nodes, leaves, leaves_branch_mean, leaves_branch_sd, nodes_per_attr, leaves_per_class_mean, leaves_per_class_sd, var_importance_mean, var_importance_sd, one_nn_mean, one_nn_sd, best_node_mean, best_node_sd, linear_discr_mean, linear_discr_sd, naive_bayes_mean, naive_bayes_sd
                      ) s where count > 1
                    ) s where replacement != id
                );''')
