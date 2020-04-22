import os
import warnings

from data import load_data
from database import Database
from metafeatures import NumberOfMissingValues, PercentageOfMissingValues, \
    NumberOfInstancesWithMissingValues, NumberOfFeaturesWithMissingValues

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

# db = Database('sqlite', 'ml-base.db')
db = Database('postgres', 'postgres', 'postgres', 'usu4867!', '35.242.255.138', 5432)

engine = db.engine
with engine.connect() as conn:
    select_statement = '''
        select id, "name", class_column from datasets
        WHERE nr_missing_values = 0
        order by id;
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
            df = load_data(local_file, s3_config='../assets/limbo-233520-a283e9f868c1.json',
                           s3_bucket='usu-mlb', name=name)

            if df.shape[1] > 10000:
                print('Skipping {} due to many features'.format(id))
                continue

            X, y = df.drop(class_column, axis=1), df[class_column]
            nr_missing_values = NumberOfMissingValues()(X, y, categorical=True).value

            if nr_missing_values == 0:
                continue

            nr_inst = X.shape[0]
            nr_attr = X.shape[1]
            pct_missing_values = PercentageOfMissingValues()(X, y, categorical=True).value
            nr_inst_mv = NumberOfInstancesWithMissingValues()(X, y, categorical=True).value
            nr_attr_mv = NumberOfFeaturesWithMissingValues()(X, y, categorical=True).value
            pct_inst_mv = (float(nr_inst_mv) / float(nr_inst)) * 100
            pct_attr_mv = (float(nr_attr_mv) / float(nr_attr)) * 100

            update_statement = '''
                UPDATE datasets SET 
                    nr_missing_values={nr_missing_values},
                    pct_missing_values={pct_missing_values},
                    nr_inst_mv={nr_inst_mv},
                    pct_inst_mv={pct_inst_mv},
                    nr_attr_mv={nr_attr_mv},
                    pct_attr_mv={pct_attr_mv}
                WHERE id={id};
                '''.format(nr_missing_values=nr_missing_values, pct_missing_values=pct_missing_values,
                           nr_inst_mv=nr_inst_mv, pct_inst_mv=pct_inst_mv,
                           nr_attr_mv=nr_attr_mv, pct_attr_mv=pct_attr_mv,
                           id=id)
            # print(update_statement)

            print('Updating')
            conn.execute(update_statement)

            os.remove(local_file)
        except OSError as ex:
            print(ex)
