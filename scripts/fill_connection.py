from typing import List

from data import store_data
from database import Database, Dataset
from utilities import hash_file
from worker import Worker
import pandas as pd

db = Database('postgres', 'postgres', 'postgres', 'usu4867!', '35.242.255.138', 5432)


engine = db.engine
with engine.connect() as conn:
    res = conn.execute('''
        select a.id as id, d.id as dataset from algorithms a
        join datasets d on a.input_dataset = d.id
        where input_dataset in (
            select id from datasets where "depth" < 1
        ) and output_dataset is null and a.status = \'complete\'
        order by a.id''')
    for row in res:
        dataset = db.get_dataset(row['dataset'])
        algorithm = db.get_algorithm(row['id'])
        instance = algorithm.instance()

        worker = Worker(db, dataset, None, s3_config='../assets/limbo-233520-a283e9f868c1.json', s3_bucket='usu-mlb',
                        timeout=60)
        X, d = worker.transform_dataset(instance)

        input_df = dataset.load()
        df = pd.concat([X, input_df[dataset.class_column]], axis=1)

        local_file = store_data(df, './data/', 'tmp.parquet')
        hashcode = hash_file(local_file)
        similar_datasets: List[Dataset] = db.get_datasets_by_hash(hashcode)
        if len(similar_datasets) != 1:
            print('Unable to determine similar datasets for {}'.format(algorithm.id))
            continue

        ds = similar_datasets[0]
        if ds.depth != 1:
            print('Dataset has wrong depth: {}'.format(ds.id))
            continue

        df_old = ds.load('../assets/limbo-233520-a283e9f868c1.json', 'usu-mlb')
        if df.equals(df_old):
            print('Setting {} to output {}'.format(algorithm.id, ds.id))
            conn.execute('''
                    UPDATE algorithms SET
                    output_dataset = {}
                    WHERE id = {};
                    '''.format(ds.id, algorithm.id))
        else:
            print('Failed to find identical dataset for {}'.format(algorithm.id))




