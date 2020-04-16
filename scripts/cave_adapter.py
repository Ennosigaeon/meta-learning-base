import base64
import pickle
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from automl.components.classification import _classifiers
import pandas as pd
import numpy as np
import os


def base_64_to_object(b64str: str):
    """
    Inverse of object_to_base_64.
    Decode base64-encoded string and then unpickle it.
    """
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)


# Specify the algorithm to load
alg = 'libsvm_svc'
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
trajectory_csv['evaluations'] = trajectory_csv['evaluations'].astype(int)
y = trajectory_csv['cost']

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

# Impute inactive hyperparameter
from sklearn.impute import SimpleImputer
simpleImputer = SimpleImputer(strategy='constant', fill_value=-1)
X = pd.DataFrame(simpleImputer.fit_transform(X), columns=X.columns)

# Random Forest Feature Importance
from sklearn.ensemble import RandomForestRegressor
randomForest = RandomForestRegressor()
randomForest.fit(X, y)
importances = randomForest.feature_importances_
print('Random Forest Feature Importance Results')
for i,v in enumerate(importances):
    print(X.columns.tolist()[i] + ': ' + str(v))
plt.barh(X.columns.tolist(), importances, align='center', edgecolor='black', color='grey')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Mittlere Varianzabnahme', fontsize=15)
plt.show()

# Exhaustive Feature Selection
from itertools import combinations
from sklearn.model_selection import cross_validate
features = X.columns
results = []
best_score = {'mean_score': -100.0}
for L in range(0, len(features) + 1):
    for subset in combinations(features, L):
        cv_scores = cross_validate(RandomForestRegressor(), X[list(subset)], y, cv=5, scoring='neg_root_mean_squared_error')
        result = {'count': len(subset),
                        'hyperparameter': list(subset),
                        'mean_score': np.mean(cv_scores['test_score']),
                        'cv_scores': cv_scores['test_score']}
        results.append(result)
        if result['mean_score'] > best_score['mean_score']:
            best_score = result

# Plot Exhaustive Feature Selection Importance Values
results = pd.DataFrame(results)
mean_scores = pd.DataFrame()
for k in features:
    mean_scores = pd.concat([mean_scores, results[results.hyperparameters.apply(lambda x: k in x)]['mean_score']], axis=1)
mean_scores.columns = features
mean_scores.boxplot()
plt.show()

# Compute fANOVA externally (plots look better than from CAVE)
X = config_csv.drop(columns=['CONFIG_ID'], axis=1).to_numpy()
y = runhistory_csv['cost'].to_numpy()
from fanova import fANOVA
f = fANOVA(X,y)
import fanova.visualizer
vis = fanova.visualizer.Visualizer(f, cs, "./fANOVA_plots/")
vis.create_all_plots()
