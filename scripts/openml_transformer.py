import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Callable

from numpy import nan
import numpy as np
import openml
import pandas as pd
from ConfigSpace.configuration_space import Configuration
from sklearn.base import is_classifier
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

from config import DatasetConfig
from constants import AlgorithmStatus
from core import Core
from database import Database, Algorithm
from methods import ALGORITHMS

mapping = {
    'sklearn.ensemble.forest.RandomForestClassifier': 'random_forest',
    'sklearn.feature_selection.variance_threshold.VarianceThreshold': 'variance_threshold',
    'sklearn.ensemble._forest.RandomForestClassifier': 'random_forest',
    'sklearn.decomposition.pca.PCA': 'pca',
    'sklearn.ensemble.forest.ExtraTreesClassifier': 'extra_trees',
    'sklearn.tree._classes.ExtraTreesClassifier': 'extra_trees',
    # 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier': None,
    'sklearn.preprocessing.data.Binarizer': 'binarizer',
    'sklearn.tree.tree.DecisionTreeClassifier': 'decision_tree',
    'sklearn.tree._classes.DecisionTreeClassifier': 'decision_tree',
    'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': 'linear_discriminant_analysis',
    'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier': 'gradient_boosting',
    'sklearn.ensemble._weight_boosting.AdaBoostClassifier': 'ada_boosting',
    'sklearn.ensemble.weight_boosting.AdaBoostClassifier': 'ada_boosting',
    'sklearn.feature_selection.univariate_selection.SelectKBest': 'select_k_best',
    'sklearn.preprocessing.data.StandardScaler': 'standard_scaler',
    'sklearn.preprocessing._data.StandardScaler': 'standard_scaler',
    'sklearn.naive_bayes.GaussianNB': 'gaussian_nb',
    'sklearn.svm.classes.SVC': 'libsvm_svc',
    'sklearn.svm._classes.SVC': 'libsvm_svc',
    'sklearn.impute.SimpleImputer': 'imputation',
    'sklearn.impute._base.SimpleImputer': 'imputation',
    'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis': 'qda',
    'sklearn.linear_model.logistic.LogisticRegression': 'logistic_regression',
    'sklearn.linear_model._logistic.LogisticRegression': 'logistic_regression',
    'sklearn.naive_bayes.MultinomialNB': 'multinomial_nb',
    'sklearn.naive_bayes.BernoulliNB': 'bernoulli_nb',
    'sklearn.neighbors.classification.KNeighborsClassifier': 'k_neighbors',
    'sklearn.neighbors._classification.KNeighborsClassifier': 'k_neighbors',
    'sklearn.preprocessing.data.OneHotEncoder': 'one_hot_encoding',
    'sklearn.preprocessing._encoders.OneHotEncoder': 'one_hot_encoding',
    'sklearn.preprocessing.data.MinMaxScaler': 'minmax',
    'sklearn.cluster.hierarchical.FeatureAgglomeration': 'feature_agglomeration',
    'sklearn.linear_model._stochastic_gradient.SGDClassifier': 'sgd',
    'sklearn.neural_network.multilayer_perceptron.MLPClassifier': 'mlp_classifier',
    'sklearn.preprocessing.data.RobustScaler': 'robust_scaler',
    'sklearn.preprocessing.data.PolynomialFeatures': 'polynomial'
}


def get_dataset(id):
    ds = openml.datasets.get_dataset(int(eval(id)))
    X, y, categorical_indicator, attribute_names = ds.get_data(
        dataset_format='dataframe',
        target=ds.default_target_attribute
    )
    if ds.qualities['NumberOfMissingValues'] > 100000 or X.shape[1] > 500000:
        return None, None, None
    dataset_conf = DatasetConfig({'openml': id, 'train_path': None})
    dataset_conf.format = ds.format
    dataset_conf.class_column = ds.default_target_attribute
    dataset_conf.name = '{}_{}_{}'.format(ds.name, int(eval(id)), time.time())
    return X, y, dataset_conf


def delete_hp(names: List[str], hp: Dict[str, Any]):
    for name in names:
        if name in hp:
            del hp[name]
    return hp


def convert_hp(types: List[Tuple[str, Callable]], hp: Dict[str, Any]):
    for name, func in types:
        if name in hp:
            hp[name] = func(hp[name])
    return hp


def init_component(hp):
    hp = hp.replace('true', True)
    hp = hp.replace('false', False)
    for key in hp.keys():
        try:
            if isinstance(hp[key], str):
                hp[key] = float(hyperparameter[key])
            else:
                hp[key] = hyperparameter[key]
            if hp[key] == 'True':
                hp[key] = True
            elif hp[key] == 'False':
                hp[key] = False
        except:
            hp[key] = hyperparameter[key]
    return hp


def init_preprocessor(X, y, component, hp):
    hp = init_component(hp)
    hp = delete_hp(['copy', 'verbose', 'random_state', 'memory', 'cache_size'], hp)

    if component.name().endswith('ImputationComponent'):
        hp = delete_hp(['fill_value', 'missing_values'], hp)

    if component.name().endswith('PCAComponent'):
        if 'svd_solver' in hp and hp['svd_solver'] == 'auto':
            if max(*X.shape) > 500000:
                hp['svd_solver'] = 'full'
                del hp['iterated_power']
            else:
                hp['svd_solver'] = 'randomized'
        elif ('n_components' in hp and hp['n_components'] is None) or (
                'n_components' in hp and hp['n_components'] == 'null') or (
                'n_components' in hp and hp['n_components'] == 'None'):
            hp['keep_variance'] = 1.0
        if 'n_components' in hp and isinstance(hp['n_components'], float):
            hp['keep_variance'] = hp['n_components'] / min(*X.shape)
        if 'iterated_power' in hp and hp['iterated_power'] == 'auto':
            if 'keep_variance' in hp and hp['keep_variance'] < (0.1 * min(*X.shape)):
                hp['iterated_power'] = 4
            else:
                hp['iterated_power'] = 7
        hp = delete_hp(['n_components'], hp)

    if component.name().endswith('BinarizerComponent'):
        if 'threshold' in hp:
            hp['threshold_factor'] = hp['threshold'] / np.mean(X.var())
            del hp['threshold']

    if component.name().endswith('SelectKBestComponent'):
        if 'k' in hp:
            hp['k_factor'] = hp['k'] / X.shape[1]
            del hp['k']
        if 'score_func' in hp and 'f_regression' in hp['score_func']:
            hp['score_func'] = 'f_regression'
        if 'score_func' in hp and 'f_classif' in hp['score_func']:
            hp['score_func'] = 'f_classif'
        if 'score_func' in hp and 'mutual_info' in hp['score_func']:
            hp['score_func'] = 'mutual_info'
        if 'score_func' in hp and 'chi2' in hp['score_func']:
            hp['score_func'] = 'chi2'

    if component.name().endswith('FeatureAgglomerationComponent'):
        if 'compute_full_tree' in hp and hp['compute_full_tree'] == 'auto':
            hp['compute_full_tree'] = False
        if 'pooling_func' in hp and 'mean' in hp['pooling_func']:
            hp['pooling_func'] = 'mean'
        if 'pooling_func' in hp and 'median' in hp['pooling_func']:
            hp['pooling_func'] = 'median'
        if 'pooling_func' in hp and 'max' in hp['pooling_func']:
            hp['pooling_func'] = 'max'
        if 'n_clusters' in hp:
            hp['n_clusters_factor'] = hp['n_clusters'] / X.shape[1]
        hp = delete_hp(['connectivity', 'n_clusters'], hp)

    if component.name().endswith('OneHotEncoderComponent'):
        return algorithm.get_hyperparameter_search_space().get_default_configuration()

    cs = algorithm.get_hyperparameter_search_space()
    default = cs.get_default_configuration()
    default = default.get_dictionary()
    for key in hp.keys():
        try:
            if isinstance(hp[key], str):
                default[key] = float(hp[key])
            else:
                default[key] = hp[key]
        except:
            default[key] = hp[key]

    default = convert_hp([
        ('max_bins', int),
        ('with_mean', bool),
        ('with_std', bool),
        ('whiten', bool),
        ('add_indicator', bool),
        ('probability', bool),
        ('degree', int),
        ('iterated_power', int)
    ], default)

    if 'iterated_power' in default and 'solver' in default and default['solver'] != 'randomized':
        del default['iterated_power']
    if 'distance_threshold' in default and 'distance_threshold' not in hp:
        del default['distance_threshold']
    config = None
    try:
        config = Configuration(cs, default)
    except ValueError as v:
        print(v)

    return config


def init_classifier(X, y, component, hp):
    hp = init_component(hp)

    if component.name().endswith('RandomForest') or component.name().endswith(
            'ExtraTreesClassifier') or component.name().endswith('DecisionTree'):
        if 'n_estimators' in hp and hp['n_estimators'] == 'warn':
            hp['n_estimators'] = 10
        elif 'max_depth' in hp and hp['max_depth'] is not None and not isinstance(hp['max_depth'], str):
            hp['max_depth_factor'] = (hp['max_depth'] / X.shape[1])
        if 'max_leaf_nodes' in hp and hp['max_leaf_nodes'] is not None and not isinstance(hp['max_leaf_nodes'], str):
            hp['max_leaf_nodes_factor'] = (hp['max_leaf_nodes'] / X.shape[0])
        if 'min_samples_leaf' in hp and hp['min_samples_leaf'] is not None and not isinstance(hp['min_samples_leaf'],str):
            hp['min_samples_leaf'] = (hp['min_samples_leaf'] / X.shape[0])
        if 'max_features' in hp and hp['max_features'] == 'null' or np.isnan(hp['max_features']):
            hp['max_features'] = 1.0
        if 'max_features' in hp and hp['max_features'] is not None and not isinstance(hp['max_features'], str):
            hp['max_features'] = (hp['max_features'] / X.shape[1])
        if component.name().endswith('ExtraTreesClassifier') and 'oob_score' in hp:
            del hp['oob_score']
        elif 'max_features' in hp and hp['max_features'] is not None and isinstance(hp['max_features'], str):
            if hp['max_features'] == 'auto' or hp['max_features'] == 'sqrt':
                hp['max_features'] = np.sqrt(X.shape[1]) / X.shape[1]
            elif hp['max_features'] == 'log2':
                hp['max_features'] = np.log2(X.shape[1]) / X.shape[1]
        if 'min_samples_split' in hp and hp['min_samples_split'] is not None and not isinstance(hp['min_samples_split'],
                                                                                                str):
            hp['min_samples_split'] = (hp['min_samples_split'] / X.shape[0])
        if 'min_impurity_split' in hp:
            del hp['min_impurity_split']
        if 'presort' in hp:
            del hp['presort']
        if 'min_impurity_split' in hp:
            del hp['min_impurity_split']

    if component.name().endswith('GradientBoostingClassifier'):
        if 'max_leaf_nodes' in hp and hp['max_leaf_nodes'] is not None and not isinstance(hp['max_leaf_nodes'], str):
            hp['max_leaf_nodes_factor'] = (hp['max_leaf_nodes'] / X.shape[0])
        if 'max_depth' in hp and hp['max_depth'] is not None and not isinstance(hp['max_depth'], str):
            hp['max_depth_factor'] = (hp['max_depth'] / X.shape[1])
            del hp['max_depth']
        if 'min_samples_leaf' in hp and hp['min_samples_leaf'] is not None and not isinstance(hp['min_samples_leaf'], str):
            hp['min_samples_leaf'] = (hp['min_samples_leaf'] / X.shape[0])
        if 'max_bins' in hp and hp['max_bins'] == 256:
            hp['max_bins'] = 255
        if 'scoring' in hp and hp['scoring'] == 'null':
            hp['scoring'] = 'accuracy'
        if 'n_iter_no_change' in hp and hp['n_iter_no_change'] == 'null':
            hp['n_iter_no_change'] = 0

    if component.name().endswith('LibSVM_SVC'):
        if 'max_iter' in hp:
            del hp['max_iter']
        if 'gamma' in hp and isinstance(hp['gamma'], str) and hp['gamma'] == 'auto':
            hp['gamma'] = 1 / X.shape[1]
        if 'gamma' in hp and isinstance(hp['gamma'], str) and hp['gamma'] == 'scale':
            hp['gamma'] = 1 / (X.shape[1] * np.mean(X.var()))
        if 'decision_function_shape' in hp and hp['decision_function_shape'] is None:
            hp['decision_function_shape'] = 'ovo'
        if 'decision_function_shape' in hp and hp['decision_function_shape'] is 'None':
            hp['decision_function_shape'] = 'ovo'
        if 'kernel' in hp and hp['kernel'] == 'poly':
            if 'coef0' not in hp:
                hp['coef0'] = 0.
            if 'degree' not in hp:
                hp['degree'] = 2
            if 'gamma' not in hp:
                hp['gamma'] = 0.1
        if 'kernel' in hp and hp['kernel'] == 'sigmoid':
            if 'coef0' not in hp:
                hp['coef0'] = 0.
            if 'gamma' not in hp:
                hp['gamma'] = 0.1
        if 'kernel' in hp and hp['kernel'] == 'rbf':
            if 'gamma' not in hp:
                hp['gamma'] = 0.1

    if component.name().endswith('MultinomialNB') or component.name().endswith('BernoulliNB'):
        if 'class_prior' in hp:
            del hp['class_prior']
        if 'binarize' in hp:
            del hp['binarize']

    if component.name().endswith('SGDClassifier'):
        if 'n_iter_no_change' in hp:
            del hp['n_iter_no_change']

    if component.name().endswith('LinearDiscriminantAnalysis'):
        if 'n_components' in hp and np.isnan(hp['n_components']):
            hp['n_components'] = min(len(y.unique()) - 1, X.shape[1])
        if 'shrinkage' in hp and np.isnan(hp['shrinkage']):
            hp['shrinkage'] = 0.0
        if 'priors' in hp:
            del hp['priors']
        if 'store_covariance' in hp:
            del hp['store_covariance']

    if component.name().endswith('GaussianNB'):
        if 'priors' in hp:
            del hp['priors']

    if component.name().endswith('LogisticRegression'):
        if 'multi_class' in hp and hp['multi_class'] == 'auto':
            if len(y.unique()) == 2 or hp['solver'] == 'liblinear':
                hp['multi_class'] = 'ovr'
            else:
                hp['multi_class'] = 'multinomial'
        elif 'multi_class' in hp and hp['multi_class'] == 'warn':
            hp['multi_class'] = 'ovr'
        if 'solver' in hp and hp['solver'] == 'warn':
            hp['solver'] = 'liblinear'
        if 'l1_ratio' in hp and 'penalty' in hp and hp['penalty'] != 'elasticnet':
            del hp['l1_ratio']

    if component.name().endswith('MLPClassifier'):
        if 'batch_size' in hp:
            del hp['batch_size']
        if 'hidden_layer_sizes' in hp:
            hp['layer_1_size'] = hp['hidden_layer_sizes'][1]
            hp['layer_2_size'] = hp['hidden_layer_sizes'][1]
            del hp['hidden_layer_sizes']

    if component.name().endswith('QuadraticDiscriminantAnalysis'):
        if 'store_covariances' in hp:
            del hp['store_covariances']
        if 'priors' in hp:
            del hp['priors']

    if component.name().endswith('KNeighborsClassifier'):
        if 'metric_params' in hp:
            del hp['metric_params']

    hp = delete_hp(['copy', 'verbose', 'random_state', 'memory', 'cache_size', 'max_depth', 'max_leaf_nodes', 'n_jobs',
                    'warm_start', 'class_weight'], hp)

    cs = algorithm.get_hyperparameter_search_space()
    default = cs.get_default_configuration()
    default = default.get_dictionary()
    for key in hp.keys():
        try:
            if isinstance(hp[key], str):
                default[key] = float(hp[key])
            else:
                default[key] = hp[key]
        except:
            default[key] = hp[key]
    if 'bootstrap' in default and 'oob_score' in default and default['bootstrap'] is False:
        del default['oob_score']
    if 'degree' in default and 'kernel' in default and default['kernel'] != 'poly':
        del default['degree']
    if 'coef0' in default and 'kernel' in default and default['kernel'] not in ['poly', 'sigmoid']:
        del default['coef0']
    if 'gamma' in default and 'kernel' in default and default['kernel'] not in ['poly', 'sigmoid', 'rbf']:
        del default['gamma']
    if 'shrinkage' in default and 'solver' in default and default['solver'] not in ["lsqr", "eigen"]:
        del default['shrinkage']
    if 'dual' in default and 'penalty' in default and default['penalty'] != 'l2' and 'solver' in default and default['solver'] != 'liblinear':
        del default['dual']
    if 'dual' in default and 'solver' in default and default['solver'] != 'liblinear':
        del default['dual']
    if 'learning_rate_init' in default and 'solver' in default and default['solver'] not in ["sgd", "adam"]:
        del default['learning_rate_init']
    if 'power_t' in default and 'solver' in default and default['solver'] not in ["sgd"]:
        del default['power_t']
    if 'power_t' in default and 'learning_rate' in default and default['learning_rate'] not in ["invscaling"]:
        del default['power_t']
    if 'momentum' in default and 'solver' in default and default['solver'] not in ["sgd"]:
        del default['momentum']
    if 'nesterovs_momentum' in default and 'momentum' in default and default['momentum'] != 0.0:
        del default['nesterovs_momentum']
    if 'nesterovs_momentum' in default and 'solver' in default and default['solver'] != 'sgd':
        del default['nesterovs_momentum']
    if 'early_stopping' in default and 'solver' in default and default['solver'] not in ["sgd", "adam"]:
        del default['early_stopping']
    if 'validation_fraction' in default and 'early_stopping' in default and default['early_stopping'] is True:
        del default['validation_fraction']
    if 'beta_1' in default and 'solver' in default and default['solver'] not in ["adam"]:
        del default['beta_1']
    if 'beta_2' in default and 'solver' in default and default['solver'] not in ["adam"]:
        del default['beta_2']
    if 'epsilon' in default and 'solver' in default and default['solver'] not in ["adam"]:
        del default['epsilon']
    if 'n_iter_no_change' in default and 'solver' in default and default['solver'] not in ["sgd", "adam"]:
        del default['n_iter_no_change']

    default = convert_hp([('max_iter', int),
                          ('n_estimators', int),
                          ('leaf_size', int),
                          ('layer_1_size', int),
                          ('layer_2_size', int),
                          ('max_bins', int),
                          ('n_neighbors', int),
                          ('probability', bool),
                          ('fit_intercept', bool),
                          ('shrinking', bool),
                          ('shuffle', bool),
                          ('dual', bool),
                          ('add_indicator', bool),
                          ('bootstrap', bool),
                          ('early_stopping', bool),
                          ('oob_score', bool),
                          ('degree', int),
                          ('p', int)
                          ], default)
    return Configuration(cs, default)


core = Core(work_dir='data/')
database = Database('sqlite', 'assets/ml-base.db', None, None, None, None, None)
engine = create_engine('postgresql://admin:admin123@127.0.0.1:5432/openml_data')
daten = pd.read_sql(sql='''select * from data where "0.name" = 'sklearn.impute._base.SimpleImputer';''', con=engine)
for index, pipeline in daten.iterrows():
    length = 0
    step = '{}.name'.format(length)
    abort = False
    while step in pipeline and pipeline[step] is not None:
        if pipeline[step] not in mapping.keys():
            abort = True
            break
        length += 1
        step = '{}.name'.format(length)

    if abort:
        print('Unknown algorithm {}'.format(pipeline[step]))
        continue

    try:
        dataset = pipeline['dataset']
        X, y, dataset_conf = get_dataset(dataset)
        if dataset_conf is None:
            continue
        df = pd.concat([X, y], axis=1)
        dataset_id = core.add_dataset(df, dataset_conf.class_column, depth=0, name=dataset_conf.name).id
    except Exception as e:
        print(e)
        dataset = -1
        continue

    created_ids = []
    for step in range(length):
        try:
            algorithm_name = mapping[pipeline[str(step) + '.name']]
            algorithm = ALGORITHMS[algorithm_name]()
            hyperparameter = pd.Series(eval(pipeline[str(step) + '.hyperparameter']))
            if is_classifier(algorithm):
                config = init_classifier(X, y, algorithm, hyperparameter)
            else:
                config = init_preprocessor(X, y, algorithm, hyperparameter)

            algorithm_id = database.create_algorithm(dataset_id,
                                                     Algorithm(algorithm_name, input_dataset=dataset_id,
                                                               status=AlgorithmStatus.COMPLETE,
                                                               output_dataset=None,
                                                               start_time=datetime.now(), end_time=datetime.now(),
                                                               hyperparameter_values=config, host='openML')).id
            created_ids.append(str(algorithm_id))

            if is_classifier(algorithm):
                database.complete_algorithm(algorithm_id, None,
                                            accuracy=pipeline['predictive_accuracy'],
                                            f1=pipeline['f_measure'],
                                            precision=pipeline['precision'],
                                            roc_auc=pipeline['area_under_roc_curve'],
                                            recall=pipeline['weighted_recall'])
                # TODO check if classifier can be in middle of pipeline
                if step != length - 1:
                    raise ValueError('Detected classifier in the middle of the pipeline')

            else:
                if hasattr(algorithm, 'fit_transform'):
                    X = algorithm.fit_transform(pd.DataFrame(X), pd.Series(y))
                else:
                    X = algorithm.fit(pd.DataFrame(X), pd.Series(y)).transform(pd.DataFrame(X))
                X = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1]))
                df = pd.concat([X, y], axis=1)
                dataset_id = core.add_dataset(df, dataset_conf.class_column, depth=0, budget=0).id
                database.complete_algorithm(algorithm_id, dataset_id)
        except Exception as e:
            print('{}: {}'.format(pipeline['index'], e))
            print('deleting ids: ' + str(created_ids))
            db_url = URL(drivername='sqlite', database='assets/ml-base.db', username=None, password=None, host=None,
                         port=None, query=None)
            engine_to_delete = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600,
                                             connect_args={"check_same_thread": False})
            engine_to_delete.execute(
                'delete from main.algorithms where main.algorithms.id in ({})'.format(', '.join(created_ids)))
            engine_to_delete.dispose()
            break
