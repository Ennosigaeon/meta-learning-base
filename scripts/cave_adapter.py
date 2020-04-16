import base64
import pickle
from sqlalchemy import create_engine
from automl.components.classification import _classifiers
import pandas as pd
import os


def base_64_to_object(b64str: str):
    """
    Inverse of object_to_base_64.
    Decode base64-encoded string and then unpickle it.
    """
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)


# Specify the algorithm to load
alg = 'gradient_boosting'
output_path = '/mnt/c/users/usitgrams/cave_configs/'

# Load data from an algorithm
engine = create_engine('postgresql://admin:admin123@127.0.0.1:5432/openml_data')
daten = pd.read_sql(sql="select * from algorithms where algorithm = '" + alg + "' and status != 'errored';",con=engine)

# Create X and y for exhaustive feature selection (different format than the files for CAVE)
X = pd.DataFrame()
y = pd.Series()

config_csv = pd.DataFrame()
runhistory_csv = pd.DataFrame()
trajectory_csv = pd.DataFrame()
for index, algorithm in daten.iterrows():
    # Create config_csv
    configuration = base_64_to_object(algorithm['hyperparameter_values_64'])
    X = X.append(configuration, ignore_index=True)
    configuration['CONFIG_ID'] = algorithm['id']
    for key in configuration:
        if isinstance(configuration[key], bool):
            configuration[key] = str(configuration[key])
    config_csv = config_csv.append(pd.DataFrame.from_dict(configuration, orient='index').T, ignore_index=True)
    # Create runhistory_csv
    run = {'cost': 1 - algorithm['accuracy'],
           'time': (algorithm['end_time'] - algorithm['start_time']).total_seconds(),
           'status': 'StatusType.SUCCESS',
           'seed': 42,
           'config_id': algorithm['id']}
    runhistory_csv = runhistory_csv.append(run,ignore_index=True)
    # Create trajectory_csv
    traject = {'cpu_time': 42,
               'wallclock_time': 42,
               'evaluations': 1,
               'cost': 1 - algorithm['accuracy'],
               'config_id': algorithm['id']}
    trajectory_csv = trajectory_csv.append(pd.DataFrame.from_dict(traject, orient='index').T, ignore_index=True)
    y = traject['cost']
trajectory_csv['evaluations'] = trajectory_csv['evaluations'].astype(int)

# Create folder for CAVE configuration files
if not os.path.exists(output_path + alg):
    os.makedirs(output_path + alg)

# Save them as csv
config_csv.to_csv(output_path + alg + '/configurations.csv', index=False)
runhistory_csv.to_csv(output_path + alg + '/runhistory.csv', index=False)
trajectory_csv.to_csv(output_path + alg + '/trajectory.csv', index=False)

# Create Configuration Space
cs = _classifiers.get(alg).get_hyperparameter_search_space()

# Write pcs
from ConfigSpace.read_and_write import pcs
with open(output_path + alg + '/configspace.pcs_new', 'w') as fh:
    fh.write(pcs.write(cs))

# Create scenario
with open(output_path + alg + '/scenario.txt', 'w') as sc:
    sc.write('initial_incumbent = DEFAULT\n')
    sc.write('runcount_limit = inf\n')
    sc.write('execdir .\n')
    sc.write('paramfile = configspace.pcs_new\n')
    sc.write('run_obj = quality\n')

# Encode data
from sklearn.preprocessing import LabelEncoder
categorical = X.select_dtypes(include=['category', 'object'])
labelEncoder = LabelEncoder()
for colname in categorical:
    X[colname] = labelEncoder.fit_transform(X[colname])

# Exhaustive Feature Selection
from sklearn.ensemble import RandomForestRegressor
randomForest = RandomForestRegressor()
randomForest.fit(X, y)
from itertools import combinations
features = X.columns
for L in range(0, len(features) + 1):
    for subset in combinations(X.columns, L):
        print(subset)

# Compute fANOVA externally (plots look better)
X = config_csv.drop(columns=['CONFIG_ID'], axis=1).to_numpy()
y = runhistory_csv['cost'].to_numpy()
from fanova import fANOVA
f = fANOVA(X,y)
import fanova.visualizer
vis = fanova.visualizer.Visualizer(f, cs, "./fANOVA_plots/")
vis.create_all_plots()
