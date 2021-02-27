import base64
import pickle
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sqlalchemy import create_engine
from dswizard.components.classification import _classifiers
from itertools import combinations
from fanova import fANOVA
from sklearn.model_selection import cross_validate
from ConfigSpace.read_and_write import pcs
import fanova.visualizer
from random import randint
import pandas as pd
import numpy as np
import os

from methods import ALGORITHMS


def base_64_to_object(b64str: str):
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)


def check_conditions(parameter):
    if 'degree' in parameter and parameter['degree'] != -1 and 'kernel' in parameter and labelEncoder.inverse_transform([parameter['kernel'].astype(int)])[0] != 'poly':
        parameter['degree'] = -1
    if 'coef0' in parameter and parameter['coef0'] != -1 and 'kernel' in parameter and labelEncoder.inverse_transform([parameter['kernel'].astype(int)])[0] not in ["poly", "sigmoid"]:
        parameter['coef0'] = -1
    if 'gamma' in parameter and parameter['gamma'] != -1 and 'kernel' in parameter and labelEncoder.inverse_transform([parameter['kernel'].astype(int)])[0] not in ["rbf", "poly", "sigmoid"]:
        parameter['gamma'] = -1
    if 'degree' in parameter and parameter['degree'] == -1 and 'kernel' in parameter and labelEncoder.inverse_transform([parameter['kernel'].astype(int)])[0] == 'poly':
        parameter['degree'] = 2
    if 'coef0' in parameter and parameter['coef0'] == -1 and 'kernel' in parameter and labelEncoder.inverse_transform([parameter['kernel'].astype(int)])[0] in ["poly", "sigmoid"]:
        parameter['coef0'] = 0.
    if 'gamma' in parameter and parameter['gamma'] == -1 and 'kernel' in parameter and labelEncoder.inverse_transform([parameter['kernel'].astype(int)])[0] in ["rbf", "poly", "sigmoid"]:
        parameter['gamma'] = 0.1
    return parameter


def evaluate_grenzen(grenzen, labels, grenzwert):
    test = pd.concat([grenzen,labels], axis=1)
    test.columns = ['fehler', 'name']
    print("Worth to optimize:")
    for ind, row in test.iterrows():
        if row['fehler'] < grenzwert:
            print(row['name'])


# Specify the algorithm to load
alg = 'random_forest'
output_path = '/mnt/c/users/usitgrams/cave_configs/'

# Load data from the algorithm
engine = create_engine('postgresql://admin:admin123@127.0.0.1:5432/openml_data')
framework = pd.read_sql(sql="select * from algorithms where algorithm = '" + alg + "' and status != 'errored';", con=engine)
openML_1 = pd.read_sql(sql="select * from algorithms_2 where algorithm = '" + alg + "' and status != 'errored';", con=engine)
openML_2 = pd.read_sql(sql="select * from algorithms_3 where algorithm = '" + alg + "' and status != 'errored';", con=engine)
daten = pd.concat([openML_1, openML_2], axis=0)

# Create X and y for exhaustive feature selection (different format than the files for CAVE)
X = pd.DataFrame()
y = pd.Series()

config_csv = pd.DataFrame()
runhistory_csv = pd.DataFrame()
trajectory_csv = pd.DataFrame()
daten = daten.sort_values('accuracy')
time = 1
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
    runhistory_csv = runhistory_csv.append(run, ignore_index=True)
    # Create trajectory_csv
    traject = {'cpu_time': 42,
               'wallclock_time': time,
               'evaluations': 1,
               'cost': 1 - algorithm['accuracy'],
               'config_id': algorithm['id']}
    trajectory_csv = trajectory_csv.append(pd.DataFrame.from_dict(traject, orient='index').T, ignore_index=True)
    time += 1
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
categorical = X.select_dtypes(include=['category', 'object', 'bool'])
labelEncoder = LabelEncoder()
for colname in categorical:
    X[colname] = labelEncoder.fit_transform(X[colname])

# Impute inactive hyperparameter
simpleImputer = SimpleImputer(strategy='constant', fill_value=-1)
X = pd.DataFrame(simpleImputer.fit_transform(X), columns=X.columns)

# Random Forest Feature Importance
randomForest = RandomForestRegressor()
randomForest.fit(X, y)
importances = randomForest.feature_importances_
print('Random Forest Feature Importance Results')
for i, v in enumerate(importances):
    print(X.columns.tolist()[i] + ': ' + str(v))
plt.barh(X.columns.tolist(), importances, align='center', edgecolor='black', color='lightblue')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Mittlere Varianzabnahme', fontsize=15)
plt.show()

# Exhaustive Feature Selection
features = X.columns
results = []
best_score = {'mean_score': -100.0}
for L in range(0, len(features) + 1):
    for subset in combinations(features, L):
        cv_scores = cross_validate(RandomForestRegressor(), X[list(subset)], y, cv=2,
                                   scoring='neg_root_mean_squared_error')
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
    mean_scores = pd.concat([mean_scores, results[results.hyperparameter.apply(lambda x: k in x)]['mean_score']],
                            axis=1)
mean_scores.columns = features
plt.figure()
color = {
    'boxes': 'y',
    'whiskers': 'grey',
    'medians': 'white',
    'caps': 'grey'
}
box = mean_scores.boxplot(vert=False, notch=True, grid=False, color=color, patch_artist=True, sym='+')
plt.xlabel('Fehler', fontsize=15)
plt.xticks(fontsize=9)
plt.yticks(fontsize=12)
plt.show()

# Compute fANOVA externally (plots look better than from CAVE)
X = config_csv.drop(columns=['CONFIG_ID'], axis=1).to_numpy()
y = runhistory_csv['cost'].to_numpy()
f = fANOVA(X, y)
vis = fanova.visualizer.Visualizer(f, cs, "./fANOVA_plots/")
vis.create_all_plots()

# Search Space Evaluation
choice = 'criterion'
# Preprocessing
XandY = pd.concat([X,y],axis=1)
XandY = XandY[XandY[choice].notna()]
y = XandY['cost']
X = XandY.drop(columns=['cost'], axis=[1])
X_numeric = X.select_dtypes(include=['number'])
X_object = X.select_dtypes(include=['category', 'object', 'bool'])
choiceLabelEncoder = LabelEncoder()
labelEncoder = LabelEncoder()
for colname in X_object:
    if colname != choice:
        X[colname] = labelEncoder.fit_transform(X[colname])
if choice in X_object.columns:
    X[choice] = choiceLabelEncoder.fit_transform(X[choice])
simpleImputer = SimpleImputer(strategy='constant', fill_value=-1)
X = pd.DataFrame(simpleImputer.fit_transform(X), columns=X.columns)

# Train Random Forest
full_emp_randomForest = RandomForestRegressor()
full_emp_randomForest.fit(X, y)

# Create a grid
alle = {}
for feature in X.columns:
    if feature in X_numeric.columns and not all(x in [0., 1., -1] for x in X[feature].unique()):
        minimum = min(X[feature])
        maximum = max(X[feature])
        if feature == choice:
            if minimum == maximum:
                alle[feature] = [minimum] * 1000
            else:
                diff = (maximum - minimum) / 1000
                alle[feature] = np.arange(minimum, maximum, diff)
        else:
            if minimum == maximum:
                alle[feature] = [minimum] * 25
            else:
                diff = (maximum - minimum) / 25
                alle[feature] = np.arange(minimum, maximum, diff)
    else:
        alle[feature] = X[feature].unique()
from sklearn.model_selection import ParameterGrid
grid = ParameterGrid(alle)

# Evaluate the Grid
grid_results = pd.DataFrame()
n_samples = 50000
iteration = 0
while iteration < n_samples:
    params = grid[randint(0, grid.__len__())]
    params = check_conditions(params)
    test = np.array(pd.Series(params)).reshape(1, -1)
    params['prediction'] = full_emp_randomForest.predict(test)[0]
    grid_results = grid_results.append(params,ignore_index=True)
    iteration += 1

# Train Gaussian Process with results
hyperparameter = grid_results[choice]
prediction = grid_results['prediction']
standardScaler = StandardScaler()
hyperparameter = standardScaler.fit_transform(hyperparameter.values.reshape(-1,1))
rbf = 1.0 ** 2 * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 20.0))
gaussianProcessRegressor = GaussianProcessRegressor(kernel=rbf)
batch_offset = 0
batch_size = 1000
while batch_offset < len(grid_results):
    start = batch_offset
    end = batch_offset + batch_size
    print('Training batch [' + str(start) + ':' + str(end) + ']')
    gaussianProcessRegressor.fit(hyperparameter[start:end], prediction[start:end])
    batch_offset += batch_size

# Plot
if choice in X_object.columns or all(x in [0., 1., -1] for x in standardScaler.inverse_transform(np.unique(hyperparameter))):
    X_ = np.unique(hyperparameter).reshape(-1,1)
else:
    X_ = np.linspace(min(hyperparameter), max(hyperparameter), 1000)
y_mean, y_cov = gaussianProcessRegressor.predict(X_, return_std=True)
optimum = min(y_mean)
std = np.std(y_mean)
grenzwert = optimum + std
plt.figure()
if choice in X_object.columns or all(x in [0., 1., -1] for x in standardScaler.inverse_transform(np.unique(hyperparameter))):
    if choice in X_object.columns:
        plt.scatter(y=y_mean, x=choiceLabelEncoder.inverse_transform(pd.DataFrame(standardScaler.inverse_transform(np.unique(hyperparameter))).astype(int)), color='black', zorder=10, s=200, label='Mittelwert')
        plt.vlines(x=choiceLabelEncoder.inverse_transform(pd.DataFrame(standardScaler.inverse_transform(np.unique(hyperparameter))).astype(int)), ymin=y_mean-y_cov, ymax=y_mean+y_cov, color='tomato', linewidths=3, label='Standardabweichung')
        plt.hlines(grenzwert, xmin=choiceLabelEncoder.inverse_transform(pd.DataFrame(standardScaler.inverse_transform(np.unique(hyperparameter))).astype(int))[0], linewidths=3,
                   xmax=choiceLabelEncoder.inverse_transform(pd.DataFrame(standardScaler.inverse_transform(np.unique(hyperparameter))).astype(int))[len(y_mean) - 1], color='red', linestyles='dotted')
        evaluate_grenzen(pd.DataFrame(y_mean), pd.DataFrame(X_object[choice].unique()), grenzwert)
    elif all(x in [0., 1., -1] for x in standardScaler.inverse_transform(np.unique(hyperparameter))):
        plt.scatter(y=y_mean, x=['False', 'True'], color='black', zorder=10, s=200, label='Mittelwert')
        plt.vlines(x=['False', 'True'], ymin=y_mean-y_cov, ymax=y_mean+y_cov, color='lightsalmon', linewidths=3, label='Standardabweichung')
        evaluate_grenzen(pd.DataFrame(y_mean), pd.DataFrame(['False', 'True']), grenzwert)
    else:
        plt.scatter(y=y_mean, x=range(0, len(y_mean)), color='black', zorder=10, s=200, label='Mittelwert')
        plt.vlines(x=range(0, len(y_mean)), ymin=y_mean-y_cov, ymax=y_mean+y_cov, color='tomato', linewidths=3, label='Standardabweichung')
        evaluate_grenzen(pd.DataFrame(y_mean),pd.DataFrame(range(0, len(y_mean) - 1)), grenzwert)
else:
    plt.plot(standardScaler.inverse_transform(X_[:,0]), y_mean, color='black', linewidth=3, label='Mittelwert')
    plt.hlines(grenzwert, xmin=standardScaler.inverse_transform(min(hyperparameter)), xmax=standardScaler.inverse_transform(max(hyperparameter)), color='red', linestyles='dotted', linewidths=3, label='Grenzwert')
    plt.fill_between(standardScaler.inverse_transform(X_[:,0]), y_mean - y_cov, y_mean + y_cov,
                     alpha=0.2, color='k', label='Standardabweichung')
    test1 = pd.DataFrame(standardScaler.inverse_transform(X_[:, 0]))
    test2 = pd.DataFrame(y_mean)
    test3 = pd.concat([test1, test2], axis=1)
    test3.columns = ['value', 'cost']
    help_var = grenzwert
    print('Found intersections: ')
    for ind, row in test3.iterrows():
        if row['cost'] > grenzwert > help_var or row['cost'] < grenzwert < help_var:
            print(row['value'])
        help_var = row['cost']
plt.ylabel('Fehler', fontsize=15)
plt.xlabel(choice, fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(frameon=False, loc='lower center', ncol=2)
plt.show()

# Ablation Analysis
standard = ALGORITHMS[alg]().get_hyperparameter_search_space().get_default_configuration()._values
standard['default'] = 1.0
X['default'] = 0.0
beide = X.append(standard,ignore_index=True)
X_numeric = beide.select_dtypes(include=['number'])
X_object = beide.select_dtypes(include=['category', 'object', 'bool'])
labelEncoder = LabelEncoder()
for colname in X_object:
    beide[colname] = labelEncoder.fit_transform(beide[colname])
X = beide[beide['default'] == 0.0]
standard = beide[beide['default'] == 1.0]
X = X.drop(columns=['default'], axis=1)
standard = standard.drop(columns=['default'], axis=1)
simpleImputer = SimpleImputer(strategy='constant', fill_value=-1)
X = pd.DataFrame(simpleImputer.fit_transform(X), columns=X.columns)

ablation_randomForest = RandomForestRegressor()
ablation_randomForest.fit(X, y)

# Create a grid
alle = {}
for feature in X.columns:
    if feature in X_numeric.columns and not all(x in [0., 1., -1] for x in X[feature].unique()):
        minimum = min(X[feature])
        maximum = max(X[feature])
        if minimum == maximum:
            alle[feature] = [minimum] * 1000
        else:
            diff = (maximum - minimum) / 40
            alle[feature] = np.arange(minimum, maximum, diff)
    else:
        alle[feature] = X[feature].unique()
from sklearn.model_selection import ParameterGrid
grid = ParameterGrid(alle)

# Evaluate the Grid
grid_results = pd.DataFrame()
n_samples = 10000
iteration = 0
while iteration < n_samples:
    params = grid[randint(0, grid.__len__())]
    params = check_conditions(params)
    test = np.array(pd.Series(params)).reshape(1,-1)
    params['prediction'] = ablation_randomForest.predict(test)[0]
    grid_results = grid_results.append(params,ignore_index=True)
    iteration += 1

worst = grid_results['prediction'].min()
best = grid_results['prediction'].max()
worst_config = grid_results[grid_results['prediction'] == max(grid_results['prediction'])].iloc[0].to_dict()
del worst_config['prediction']
best_config = grid_results[grid_results['prediction'] == min(grid_results['prediction'])].iloc[0].to_dict()
del best_config['prediction']


def check(param, standard_copy, best_config):
    if param == 'kernel' and best_config['kernel'] == 'sigmoid':
        standard_copy = standard.copy().iloc[0].to_dict()
        standard_copy['coef0'] = best_config['coef0']
        standard_copy['gamma'] = best_config['gamma']
    elif param == 'kernel' and best_config['kernel'] == 'rbf':
        standard_copy['gamma'] = best_config['gamma']
    elif param == 'kernel' and best_config['kernel'] == 'poly':
        standard_copy['coef0'] = best_config['coef0']
        standard_copy['gamma'] = best_config['gamma']
        standard_copy['degree'] = best_config['degree']
    elif param == 'kernel' and best_config['kernel'] == 'linear':
        standard_copy['gamma'] = -1
    return standard_copy


S = {}
last = ablation_randomForest.predict(np.array(pd.Series(worst_config)).reshape(1, -1))[0]
improvements = []
improvements_names = []
scoring = [last]
scoring_names = ['Maximum']
for i in range(0, len(best_config) - 1):
    results = {}
    for param in best_config:
        if param not in S.keys():
            if param not in ['gamma', 'coef0', 'degree']:
                standard_copy = worst_config.copy()
                standard_copy[param] = best_config[param]
                standard_copy = check(param,standard_copy,best_config)
                for key in S:
                    standard_copy[key] = best_config[key]
                    standard_copy = check(param, standard_copy, best_config)
                results[param] = ablation_randomForest.predict(np.array(pd.Series(standard_copy)).reshape(1, -1))[0]
    print(results)
    if len(results) != 0:
        maximum = min(zip(results.values(), results.keys()))
        S[maximum[1]] = maximum[0]
        print(maximum[1] + ' with improvement: '+ str((maximum[0] - last)))
        improvements.append(-1*(maximum[0] - last))
        improvements_names.append(maximum[1])
        scoring_names.append(maximum[1])
        scoring.append(maximum[0])
        last = maximum[0]

plt.figure()
plt.bar(improvements_names, improvements, align='center', edgecolor='black', color='yellowgreen')
plt.xticks(fontsize=12, rotation=75)
plt.yticks(fontsize=12)
plt.ylabel('Verbessserung', fontsize=15)
plt.show()

plt.figure()
plt.plot(scoring_names, scoring, '-o', linewidth=3, markersize=10, color='peru', markerfacecolor='black')
plt.xticks(fontsize=12, rotation=75)
plt.yticks(fontsize=12)
plt.ylabel('Fehler', fontsize=15)
plt.show()
