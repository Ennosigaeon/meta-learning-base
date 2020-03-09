from __future__ import absolute_import, unicode_literals

import hashlib
from builtins import object
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import pymysql
from ConfigSpace.configuration_space import Configuration
from sklearn.base import BaseEstimator
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text, create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from constants import AlgorithmStatus, RunStatus
from data import load_data, delete_data
from methods import ALGORITHMS
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

    # TODO set index on appropriate columns

    """
    Columns necessary for loading/processing data
    """
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    class_column = Column(String(100))

    train_path = Column(String, nullable=False)
    status = Column(String(50), nullable=False)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    processed = Column(Integer)
    budget = Column(Integer)
    depth = Column(Integer)
    hashcode = Column(String(40), index=True)

    """
    Metadata columns
    """
    nr_inst = Column(Numeric)
    nr_attr = Column(Numeric)
    nr_class = Column(Numeric)
    nr_missing_values = Column(Numeric)
    pct_missing_values = Column(Numeric)
    nr_inst_mv = Column(Numeric)
    pct_inst_mv = Column(Numeric)
    nr_attr_mv = Column(Numeric)
    pct_attr_mv = Column(Numeric)
    nr_outliers = Column(Numeric)

    skewness_mean = Column(Numeric)
    skewness_sd = Column(Numeric)
    kurtosis_mean = Column(Numeric)
    kurtosis_sd = Column(Numeric)
    cor_mean = Column(Numeric)
    cor_sd = Column(Numeric)
    cov_mean = Column(Numeric)
    cov_sd = Column(Numeric)
    attr_conc_mean = Column(Numeric)
    attr_conc_sd = Column(Numeric)
    sparsity_mean = Column(Numeric)
    sparsity_sd = Column(Numeric)
    gravity = Column(Numeric)
    var_mean = Column(Numeric)
    var_sd = Column(Numeric)

    class_prob_mean = Column(Numeric)
    class_prob_std = Column(Numeric)
    class_ent = Column(Numeric)
    attr_ent_mean = Column(Numeric)
    attr_ent_sd = Column(Numeric)
    mut_inf_mean = Column(Numeric)
    mut_inf_sd = Column(Numeric)
    eq_num_attr = Column(Numeric)
    ns_ratio = Column(Numeric)

    nodes = Column(Numeric)
    leaves = Column(Numeric)
    leaves_branch_mean = Column(Numeric)
    leaves_branch_sd = Column(Numeric)
    nodes_per_attr = Column(Numeric)
    leaves_per_class_mean = Column(Numeric)
    leaves_per_class_sd = Column(Numeric)
    var_importance_mean = Column(Numeric)
    var_importance_sd = Column(Numeric)

    one_nn_mean = Column(Numeric)
    one_nn_sd = Column(Numeric)
    best_node_mean = Column(Numeric)
    best_node_sd = Column(Numeric)
    linear_discr_mean = Column(Numeric)
    linear_discr_sd = Column(Numeric)
    naive_bayes_mean = Column(Numeric)
    naive_bayes_sd = Column(Numeric)

    def load(self, s3_endpoint: str = None, s3_bucket: str = None, s3_access_key=None, s3_secret_key=None):
        return load_data(self.train_path, s3_endpoint, s3_bucket, s3_access_key, s3_secret_key, self.name)

    @staticmethod
    def _make_name(path):
        md5 = hashlib.md5(path.encode('utf-8'))
        return md5.hexdigest()

    def __init__(self, train_path, name=None, id=None, status=RunStatus.PENDING,
                 start_time: datetime = None, end_time: datetime = None, processed: int = 0, budget: int = 5, depth: int = 0,
                 hashcode: str = None,
                 nr_inst=None, nr_attr=None, nr_class=None, nr_outliers=None, skewness_mean=None, skewness_sd=None,
                 kurtosis_mean=None, kurtosis_sd=None, cor_mean=None, cor_sd=None, cov_mean=None, cov_sd=None,
                 attr_conc_mean=None, attr_conc_sd=None, sparsity_mean=None, sparsity_sd=None, gravity=None,
                 var_mean=None, var_sd=None, class_ent=None, attr_ent_mean=None, attr_ent_sd=None, mut_inf_mean=None,
                 mut_inf_sd=None, eq_num_attr=None, ns_ratio=None, nodes=None, leaves=None,
                 leaves_branch_mean=None, leaves_branch_sd=None, nodes_per_attr=None, leaves_per_class_mean=None,
                 leaves_per_class_sd=None, var_importance_mean=None, var_importance_sd=None, one_nn_mean=None,
                 one_nn_sd=None, best_node_mean=None, best_node_sd=None,
                 linear_discr_mean=None, linear_discr_sd=None, naive_bayes_mean=None, naive_bayes_sd=None,
                 nr_missing_values=None, pct_missing_values=None, nr_inst_mv=None, nr_attr_mv=None, pct_inst_mv=None,
                 pct_attr_mv=None, class_prob_mean=None, class_prob_std=None, class_column=None,):
        self.train_path = train_path
        self.name = name or self._make_name(train_path)
        self.status = status
        self.id: Optional[int] = id
        self.start_time: Optional[datetime] = start_time
        self.end_time: Optional[datetime] = end_time
        self.processed: Optional[int] = processed
        self.budget: Optional[int] = budget
        self.class_column = class_column
        self.depth: int = depth
        self.hashcode: str = hashcode

        self.nr_inst = nr_inst
        self.nr_attr = nr_attr
        self.nr_class = nr_class
        self.nr_missing_values = nr_missing_values
        self.pct_missing_values = pct_missing_values
        self.nr_inst_mv = nr_inst_mv
        self.pct_inst_mv = pct_inst_mv
        self.nr_attr_mv = nr_attr_mv
        self.pct_attr_mv = pct_attr_mv
        self.nr_outliers = nr_outliers

        self.skewness_mean = skewness_mean
        self.skewness_sd = skewness_sd
        self.kurtosis_mean = kurtosis_mean
        self.kurtosis_sd = kurtosis_sd
        self.cor_mean = cor_mean
        self.cor_sd = cor_sd
        self.cov_mean = cov_mean
        self.cov_sd = cov_sd
        self.attr_conc_mean = attr_conc_mean
        self.attr_conc_sd = attr_conc_sd
        self.sparsity_mean = sparsity_mean
        self.sparsity_sd = sparsity_sd
        self.gravity = gravity
        self.var_mean = var_mean
        self.var_sd = var_sd

        self.class_prob_mean = class_prob_mean
        self.class_prob_std = class_prob_std
        self.class_ent = class_ent
        self.attr_ent_mean = attr_ent_mean
        self.attr_ent_sd = attr_ent_sd
        self.mut_inf_mean = mut_inf_mean
        self.mut_inf_sd = mut_inf_sd
        self.eq_num_attr = eq_num_attr
        self.ns_ratio = ns_ratio

        self.nodes = nodes
        self.leaves = leaves
        self.leaves_branch_mean = leaves_branch_mean
        self.leaves_branch_sd = leaves_branch_sd
        self.nodes_per_attr = nodes_per_attr
        self.leaves_per_class_mean = leaves_per_class_mean
        self.leaves_per_class_sd = leaves_per_class_sd
        self.var_importance_mean = var_importance_mean
        self.var_importance_sd = var_importance_sd

        self.one_nn_mean = one_nn_mean
        self.one_nn_sd = one_nn_sd
        self.best_node_mean = best_node_mean
        self.best_node_sd = best_node_sd
        self.linear_discr_mean = linear_discr_mean
        self.linear_discr_sd = linear_discr_sd
        self.naive_bayes_mean = naive_bayes_mean
        self.naive_bayes_sd = naive_bayes_sd

    def __repr__(self):
        return "<{}: {} classes, {} features, {} rows>".format(self.name, self.nr_class, self.nr_attr, self.nr_inst)


class Algorithm(Base):
    __tablename__ = 'algorithms'

    """
    Relational columns
    """
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)

    """
    Algorithm columns
    """
    algorithm = Column(String(300), nullable=False)
    status = Column(String(50), nullable=False)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    error_message = Column(Text)

    """
    Hyperparameter
    """
    # base 64 encoding of the hyperparameter names and values
    hyperparameter_values_64 = Column(Text)

    """
    Performance metrics
    """
    accuracy = Column(Numeric(precision=20, scale=10))
    f1_score = Column(Numeric(precision=20, scale=10))
    precision = Column(Numeric(precision=20, scale=10))
    recall = Column(Numeric(precision=20, scale=10))
    neg_log_loss = Column(Numeric(precision=20, scale=10))
    roc_auc_score = Column(Numeric(precision=20, scale=10))

    cv_judgment_metric = Column(Numeric(precision=20, scale=10))
    cv_judgment_metric_stdev = Column(Numeric(precision=20, scale=10))
    test_judgment_metric = Column(Numeric(precision=20, scale=10))

    """
    Hostname
    """
    host = Column(String)

    """Decode hyperparameters from base64 to object"""
    @property
    def hyperparameter_values(self) -> dict:
        return base_64_to_object(self.hyperparameter_values_64)

    """Encode hyperparameters from object to base64"""
    @hyperparameter_values.setter
    def hyperparameter_values(self, value: Configuration):
        if value is None:
            d = {}
        else:
            d = value.get_dictionary()

        self.hyperparameter_values_64 = object_to_base_64(d)

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
                 status: AlgorithmStatus = None,
                 start_time: datetime = None,
                 end_time: datetime = None,
                 error_message: str = None,
                 hyperparameter_values: Configuration = None,
                 host=None):

        self.algorithm: str = algorithm
        self.status = status
        self.id: Optional[int] = id
        self.dataset_id: int = dataset_id
        self.start_time: Optional[datetime] = start_time
        self.end_time: Optional[datetime] = end_time
        self.error_message: Optional[str] = error_message
        self.hyperparameter_values = hyperparameter_values
        self.host = host
        self._load_cs()

    def _load_cs(self):
        """
        method: method code or path to JSON file containing all the information
            needed to specify this enumerator.
        """
        cs = ALGORITHMS[self.algorithm].get_hyperparameter_search_space()
        self.cs = cs

    def random_config(self):
        return self.cs.sample_configuration()

    def default_config(self):
        return self.cs.get_default_configuration()

    def instance(self, params: Dict[str, Any] = None) -> BaseEstimator:
        if params is None:
            if self.hyperparameter_values is None:
                params = self.random_config()
            else:
                params = self.hyperparameter_values

        return ALGORITHMS[self.algorithm](**params)

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
    def get_dataset(self, dataset_id) -> Dataset:
        """ Get a specific dataset. """
        return self.session.query(Dataset).get(dataset_id)

    @try_with_session()
    def get_datasets(self, ignore_pending: bool = False, ignore_running: bool = False,
                     ignore_complete: bool = True) -> Optional[List[Dataset]]:

        """
        Get a list of all datasets matching the chosen filters.

        Args:
            ignore_pending: if True, ignore datasets that have not been started
            ignore_running: if True, ignore datasets that are already running
            ignore_complete: if True, ignore completed datasets
        """

        query = self.session.query(Dataset)
        if ignore_pending:
            query = query.filter(Dataset.status != RunStatus.PENDING)
        if ignore_running:
            query = query.filter(Dataset.status != RunStatus.RUNNING)
        if ignore_complete:
            query = query.filter(Dataset.status != RunStatus.COMPLETE)

        datasets = query.all()

        if len(datasets) == 0:
            return None
        return datasets

    @try_with_session()
    def get_dataset_by_hash(self, hashcode: str) -> List[Dataset]:
        """ Get a specific dataset. """
        return self.session.query(Dataset).filter(Dataset.hashcode == hashcode).all()

    @try_with_session()
    def get_algorithm(self, algorithm_id) -> Algorithm:
        """ Get a specific algorithm. """
        return self.session.query(Algorithm).get(algorithm_id)

    @try_with_session()
    def get_algorithms(self, dataset_id: int = None, ignore_errored: bool = False, ignore_running: bool = False,
                       ignore_complete: bool = True) -> Optional[List[Algorithm]]:

        """
        Get a list of all algorithms matching the chosen filters.

        Args:
            ignore_errored: if True, ignore algorithms that are errored
            ignore_running: if True, ignore algorithms that are already running
            ignore_complete: if True, ignore completed algorithms
        """

        query = self.session.query(Algorithm)
        if dataset_id is not None:
            query = query.filter(Algorithm.dataset_id == dataset_id)
        if ignore_errored:
            query = query.filter(Algorithm.status != AlgorithmStatus.ERRORED)
        if ignore_running:
            query = query.filter(Algorithm.status != AlgorithmStatus.RUNNING)
        if ignore_complete:
            query = query.filter(Algorithm.status != AlgorithmStatus.COMPLETE)

        algorithms = query.all()

        if not len(algorithms):
            return None
        return algorithms

    # ##########################################################################
    # #  Methods to update the database  #######################################
    # ##########################################################################

    @try_with_session(commit=True)
    def create_dataset(self, **kwargs) -> Dataset:
        dataset = Dataset(**kwargs)
        self.session.add(dataset)
        return dataset

    @try_with_session(commit=True)
    def create_algorithm(self,
                         dataset_id: int,
                         algorithm: Algorithm, start_time, status, host,
                         hyperparameter_values: Dict = None) -> Algorithm:

        """Update dataset values in the db"""
        self.session.query(Dataset).filter(Dataset.id == dataset_id).update({Dataset.processed: Dataset.processed + 1},
                                                                            synchronize_session=False)
        self.session.commit()

        """Add algorithm to the db"""
        self.session.add(algorithm)
        return algorithm

    @try_with_session(commit=True)
    def complete_algorithm(self, algorithm_id, accuracy: float = None, f1: float = None, precision: float = None,
                           recall: float = None, neg_log_loss: float = None, roc_auc: float = None):

        """Set all the parameters on a algorithm that haven't yet been set, and mark it as complete."""

        algorithm = self.session.query(Algorithm).get(algorithm_id)

        algorithm.accuracy = accuracy
        algorithm.f1_score = f1
        algorithm.precision = precision
        algorithm.recall = recall
        algorithm.neg_log_loss = neg_log_loss
        algorithm.roc_auc_score = roc_auc
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
    def mark_dataset_running(self, dataset_id: int) -> None:
        """
        Set the status of the dataset to RUNNING and set the 'start_time' field to the current datetime.
        """
        dataset = self.get_dataset(dataset_id)
        if dataset.status == RunStatus.RUNNING:
            # Already running
            return
        elif dataset.status == RunStatus.COMPLETE:
            # Already completed. Can happen due to race conditions
            raise UserWarning('Cannot mark the completed dataset {} as running'.format(dataset_id))
        elif dataset.status == RunStatus.PENDING:
            dataset.status = RunStatus.RUNNING
            dataset.start_time = datetime.now()
        else:
            raise ValueError('Dataset {} in unknown status {}'.format(dataset_id, dataset.status))

    @try_with_session(commit=True)
    def mark_dataset_complete(self, dataset_id: int) -> None:
        """
        Set the status of the dataset to COMPLETE and set the 'end_time' field to the current datetime.
        """
        dataset = self.get_dataset(dataset_id)
        if dataset.status == RunStatus.COMPLETE:
            # Already completed
            return
        elif dataset.status == RunStatus.PENDING:
            raise UserWarning('Cannot mark the pending dataset {} as completed'.format(dataset_id))
        elif dataset.status == RunStatus.RUNNING:
            dataset.status = RunStatus.COMPLETE
            dataset.end_time = datetime.now()
        else:
            raise ValueError('Dataset {} in unknown status {}'.format(dataset_id, dataset.status))
