from __future__ import absolute_import, unicode_literals

import hashlib
import importlib
import json
import os
from builtins import object
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import pymysql
from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from constants import ALGORITHM_STATUS, AlgorithmStatus, RunStatus
from data import load_data
from utilities import base_64_to_object, object_to_base_64


class DBSession(object):
    def __init__(self, db, commit=False):
        self.db = db
        self.commit = commit

    def __enter__(self):
        self.db.session = self.db.get_session()

    def __exit__(self, type, error, traceback):
        if error is not None:
            self.db.session.rollback()
        elif self.commit:
            self.db.session.commit()

        self.db.session.close()
        self.db.session = None


def try_with_session(commit: bool = False):
    """
    Decorator for instance methods on Database that need a sqlalchemy session.

    This wrapping function checks if the Database has an active session yet. If
    not, it wraps the function call in a ``with DBSession():`` block.
    """

    def wrap(func):
        def call(db, *args, **kwargs):
            # if the Database has an active session, don't create a new one
            if db.session is not None:
                result = func(db, *args, **kwargs)
                if commit:
                    db.session.commit()
            else:
                # otherwise, use the session generator
                with DBSession(db, commit=commit):
                    result = func(db, *args, **kwargs)

            return result

        return call

    return wrap


Base = declarative_base()


class Dataset(Base):
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)

    # columns necessary for loading/processing data
    train_path = Column(String, nullable=False)
    test_path = Column(String)
    reference_path = Column(String)
    processed = Column(Integer)

    # metadata columns
    # Type: Continuous
    numberOfNumericAttributes = Column(Integer)

    # Type: Categorical
    numberOfCategoricalAttributes = Column(Integer)
    numberOfBinaryAttributes = Column(Integer)

    # Type: Generic
    numberOfInstances = Column(Integer)
    numberOfAttributes = Column(Integer)
    dimensionality = Column(Integer)
    numberOfMissingValues = Column(Integer)
    percentageOfMissingValues = Column(Integer)
    numberOfInstancesWithMissingValues = Column(Integer)
    percentageOfInstancesWithMissingValues = Column(Integer)
    numberOfClasses = Column(Integer)
    classEntropy = Column(Integer)

    def load(self, test_size=0.3, random_state=0,
             aws_access_key=None, aws_secret_key=None):
        data = load_data(self.name, self.train_path, aws_access_key, aws_secret_key)

        if self.test_path:
            if self.name.endswith('.csv'):
                test_name = self.name.replace('.csv', '_test.csv')
            else:
                test_name = self.name + '_test'

            test_data = load_data(test_name, self.test_path,
                                  aws_access_key, aws_secret_key)
            return data, test_data

        else:
            return train_test_split(data, test_size=test_size, random_state=random_state)

    @staticmethod
    def _make_name(path):
        md5 = hashlib.md5(path.encode('utf-8'))
        return md5.hexdigest()

    def __init__(self,
                 train_path,
                 test_path=None,
                 reference_path=None,
                 status=None,
                 name=None,
                 id=None):

        self.train_path = train_path
        self.test_path = test_path
        self.reference_path = reference_path
        self.name = name or self._make_name(train_path)
        self.status = status

        self.id = id

    def __repr__(self):
        base = "<%s: %s, %d classes, %d features, %d rows>"
        return base % (self.name, self.numberOfClasses, self.numberOfAttributes, self.numberOfInstances)


class Algorithm(Base):
    __tablename__ = 'algorithms'

    # relational columns
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)

    #
    algorithm = Column(String(300), nullable=False)

    # hyperparameter
    accuracy = Column(Integer)
    average_precision = Column(Integer)
    f1_score = Column(Integer)
    precision = Column(Integer)
    recall = Column(Integer)
    neg_log_loss = Column(Integer)
    # base 64 encoding of the hyperparameter names and values
    hyperparameter_values_64 = Column(Text)

    # performance metrics
    cv_judgment_metric = Column(Numeric(precision=20, scale=10))
    cv_judgment_metric_stdev = Column(Numeric(precision=20, scale=10))
    test_judgment_metric = Column(Numeric(precision=20, scale=10))

    start_time = Column(DateTime)
    end_time = Column(DateTime)
    status = Column(Enum(*ALGORITHM_STATUS))
    error_message = Column(Text)

    @property
    def hyperparameter_values(self):
        return base_64_to_object(self.hyperparameter_values_64)

    @hyperparameter_values.setter
    def hyperparameter_values(self, value):
        self.hyperparameter_values_64 = object_to_base_64(value)

    @property
    def mu_sigma_judgment_metric(self):
        # compute the lower confidence bound on the cross-validated
        # judgment metric
        if self.cv_judgment_metric is None:
            return None
        return self.cv_judgment_metric - 2 * self.cv_judgment_metric_stdev

    def __init__(self,
                 algorithm: str,
                 dataset_id: int,
                 id: int = None,
                 status=None,
                 start_time: datetime = None,
                 end_time: datetime = None,
                 error_message: str = None):
        self.algorithm: str = algorithm
        self.status = status
        self.id: Optional[int] = id
        self.dataset_id: int = dataset_id
        self.start_time: Optional[datetime] = start_time
        self.end_time: Optional[datetime] = end_time
        self.error_message: Optional[str] = error_message

    def _load_cs(self):
        """
        method: method code or path to JSON file containing all the information
            needed to specify this enumerator.
        """
        config_path = os.path.join(os.path.dirname(__file__), 'methods', self.algorithm)
        with open(config_path) as f:
            config = json.load(f)

        self.root_params = config['root_hyperparameters']
        self.conditions = config['conditional_hyperparameters']
        self.class_path = config['class']

        # create hyperparameters from the parameter config
        cs = ConfigurationSpace()

        self.parameters = {}
        for k, v in list(config['hyperparameters'].items()):
            param_type = v['type']
            if param_type == 'int':
                self.parameters[k] = UniformIntegerHyperparameter(k, lower=v['range'][0], upper=v['range'][1],
                                                                  default_value=v.get('default', None))
            elif param_type == 'int_exp':
                self.parameters[k] = UniformIntegerHyperparameter(k, lower=v['range'][0], upper=v['range'][1],
                                                                  default_value=v.get('default', None), log=True)
            elif param_type == 'float':
                self.parameters[k] = UniformFloatHyperparameter(k, lower=v['range'][0], upper=v['range'][1],
                                                                default_value=v.get('default', None))
            elif param_type == 'float_exp':
                self.parameters[k] = UniformFloatHyperparameter(k, lower=v['range'][0], upper=v['range'][1],
                                                                default_value=v.get('default', None), log=True)
            elif param_type == 'string':
                # noinspection PyArgumentList
                self.parameters[k] = CategoricalHyperparameter(k, choices=v['values'],
                                                               default_value=v.get('default', None))
            elif param_type == 'bool':
                # noinspection PyArgumentList
                self.parameters[k] = CategoricalHyperparameter(k, choices=[True, False],
                                                               default_value=v.get('default', None))
            else:
                raise ValueError(f'Unknown hyperparameter type {param_type}')

        cs.add_hyperparameters(self.parameters.values())

        for condition, dic in self.conditions.items():
            for k, v in dic.items():
                cs.add_condition(InCondition(self.parameters[k], self.parameters[condition], v))

        self.cs = cs

    def random_config(self):
        return self.cs.sample_configuration()

    def default_config(self):
        return self.cs.get_default_configuration()

    def instance(self, params: Dict[str, Any] = None) -> BaseEstimator:
        if params is None:
            params = self.random_config()

        module_name = self.class_path.rpartition(".")[0]
        class_name = self.class_path.split(".")[-1]

        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(**params)

    def __repr__(self):
        params = '\n'.join(
            [
                '\t{}: {}'.format(name, value)
                for name, value in self.hyperparameter_values.items()
            ]
        )

        to_print = [
            'Algorithm id: {}'.format(self.id),
            'Params chosen: \n{}'.format(params),
            'Cross Validation Score: {:.3f} +- {:.3f}'.format(
                self.cv_judgment_metric, self.cv_judgment_metric_stdev),
            'Test Score: {:.3f}'.format(self.test_judgment_metric),
        ]

        return '\n'.join(to_print)


class Database(object):
    def __init__(self, dialect: str, database: str, username: str = None, password: str = None,
                 host: str = None, port: int = None, query: str = None):
        """
        Accepts configuration for a database connection, and defines SQLAlchemy
        ORM objects for all the tables in the database.
        """

        # Prepare environment for pymysql
        pymysql.install_as_MySQLdb()
        pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
        pymysql.converters.conversions = pymysql.converters.encoders.copy()
        pymysql.converters.conversions.update(pymysql.converters.decoders)

        db_url = URL(drivername=dialect, database=database, username=username,
                     password=password, host=host, port=port, query=query)
        self.engine = create_engine(db_url)
        self.session = None
        self.get_session = sessionmaker(bind=self.engine,
                                        expire_on_commit=False)

        # create ORM objects for the tables
        self._define_tables()

    def _define_tables(self) -> None:
        """
        Define the SQLAlchemy ORM class for each table in the ModelHub database.

        These must be defined after the Database class is initialized so that
        the database metadata is available (at runtime).
        If the database does not already exist, it will be created. If it does
        exist, it will not be updated with new schema -- after schema changes,
        the database must be destroyed and reinitialized.
        """

        Base.metadata.create_all(bind=self.engine)

    # ##########################################################################
    # #  Standard query methods  ###############################################
    # ##########################################################################

    @try_with_session()
    def get_dataset(self, dataset_id):
        """ Get a specific datarun. """
        return self.session.query(Dataset).get(dataset_id)

    @try_with_session()
    def get_datasets(self, ignore_pending: bool = False, ignore_running: bool = False,
                     ignore_complete: bool = True) -> Optional[List[Dataset]]:
        # TODO adapt

        query = self.session.query(Dataset)
        datasets = query.all()

        if not len(datasets):
            return None
        return datasets

    @try_with_session()
    def get_algorithms(self, ignore_pending=False, ignore_running=False,
                       ignore_complete=True) -> Optional[List[Algorithm]]:
        """
        Get a list of all datasets matching the chosen filters.

        Args:
            ignore_pending: if True, ignore algorithms that have not been started
            ignore_running: if True, ignore algorithms that are already running
            ignore_complete: if True, ignore completed algorithms
        """
        # TODO adapt
        query = self.session.query(Algorithm)
        if ignore_pending:
            query = query.filter(Algorithm.status != RunStatus.PENDING)
        if ignore_running:
            query = query.filter(Algorithm.status != RunStatus.RUNNING)
        if ignore_complete:
            query = query.filter(Algorithm.status != RunStatus.COMPLETE)

        algorithms = query.all()

        if not len(algorithms):
            return None
        return algorithms

    @try_with_session()
    def get_algorithm(self, algorithm_id):
        """ Get a specific algorithm. """
        return self.session.query(Algorithm).get(algorithm_id)

    # ##########################################################################
    # #  Methods to update the database  #######################################
    # ##########################################################################

    @try_with_session(commit=True)
    def create_dataset(self, **kwargs):
        dataset = Dataset(**kwargs)
        self.session.add(dataset)
        return dataset

    # @try_with_session(commit=True)
    # def create_algorithm(self, dataset_id: int, name: str, algorithm: str) -> 'Database.Algorithm':
    #     algorithm_obj = self.Algorithm(dataset_id=dataset_id, name=name, algorithm=algorithm)
    #     self.session.add(algorithm_obj)
    #     return algorithm_obj

    @try_with_session(commit=True)
    def create_algorithm(self, **kwargs):
        algorithm = Algorithm(**kwargs)
        self.session.add(algorithm)
        return algorithm

    @try_with_session(commit=True)
    def start_algorithm(self,
                        dataset_id: int,
                        algorithm: str,
                        hyperparameter_values: Dict = None) -> Algorithm:
        """
        Save a new, fully qualified algorithm object to the database.
        Returns: the ID of the newly-created algorithm
        """
        # TODO adapt
        algorithm = Algorithm(dataset_id=dataset_id,
                              algorithm=algorithm,
                              start_time=datetime.now(),
                              status=AlgorithmStatus.RUNNING)
        self.session.add(algorithm)
        return algorithm

    @try_with_session(commit=True)
    def complete_algorithm(self, algorithm_id, accuracy, average_precision, f1_score, precision, recall, neg_log_loss):
        """
        Set all the parameters on a algorithm that haven't yet been set, and mark
        it as complete.
        """
        # TODO adapt
        algorithm = self.session.query(Algorithm).get(algorithm_id)

        algorithm.accuracy = accuracy
        algorithm.average_precision = average_precision
        algorithm.f1_score = f1_score
        algorithm.precision = precision
        algorithm.recall = recall
        algorithm.neg_log_loss = neg_log_loss
        algorithm.end_time = datetime.now()
        algorithm.status = AlgorithmStatus.COMPLETE

    @try_with_session(commit=True)
    def mark_algorithm_errored(self, algorithm_id, error_message):
        """
        Mark an existing algorithm as having errored and set the error message. If
        the algorithm's hyperpartiton has produced too many erring algorithms, mark it
        as errored as well.
        """
        algorithm = self.session.query(Algorithm).get(algorithm_id)
        algorithm.error_message = error_message
        algorithm.status = AlgorithmStatus.ERRORED
        algorithm.end_time = datetime.now()

    @try_with_session(commit=True)
    def mark_dataset_complete(self, dataset_id: int) -> None:
        """
        Set the status of the Datarun to COMPLETE and set the 'end_time' field
        to the current datetime.
        """
        dataset = self.get_dataset(dataset_id)
        dataset.status = RunStatus.COMPLETE
        dataset.end_time = datetime.now()

    @try_with_session(commit=True)
    def mark_dataset_running(self, dataset_id: int) -> None:
        """
        Set the status of the Datarun to COMPLETE and set the 'end_time' field
        to the current datetime.
        """
        dataset = self.get_dataset(dataset_id)
        dataset.status = RunStatus.RUNNING
        dataset.end_time = datetime.now()
