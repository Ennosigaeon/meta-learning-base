from __future__ import absolute_import, unicode_literals

import hashlib
import pickle
from builtins import object
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import Configuration
from sklearn.base import BaseEstimator
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text, create_engine, BigInteger
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from components.base import EstimatorComponent
from constants import AlgorithmStatus, RunStatus
from data import load_data
from methods import ALGORITHMS
from utilities import base_64_to_object, object_to_base_64


class DBSession(object):
    def __init__(self, db: 'Database', commit=False):
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
    # __table_args__ = {'schema': 'mlb'}

    HybridType = Integer()
    HybridType = HybridType.with_variant(BigInteger(), 'postgresql')

    """
    Columns necessary for loading/processing data
    """
    id = Column(HybridType, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    class_column = Column(String(100))

    train_path = Column(String, nullable=False)
    status = Column(String(50), nullable=False, index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    processed = Column(Integer)
    budget = Column(Integer)
    depth = Column(Integer, nullable=False, index=True)
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
    sparsity_mean = Column(Numeric)
    sparsity_sd = Column(Numeric)
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

    def load(self, s3_config: str = None, s3_bucket: str = None):
        return load_data(self.train_path, s3_config, s3_bucket, self.name)

    @staticmethod
    def _make_name(path):
        md5 = hashlib.md5(path.encode('utf-8'))
        return md5.hexdigest()

    def __init__(self, train_path, name=None, id=None, status=RunStatus.PENDING,
                 start_time: datetime = None, end_time: datetime = None, processed: int = 0, budget: int = 5,
                 depth: int = 0, hashcode: str = None,
                 nr_inst=None, nr_attr=None, nr_class=None, nr_outliers=None, skewness_mean=None, skewness_sd=None,
                 kurtosis_mean=None, kurtosis_sd=None, cor_mean=None, cor_sd=None, cov_mean=None, cov_sd=None,
                 sparsity_mean=None, sparsity_sd=None,
                 var_mean=None, var_sd=None, class_ent=None, attr_ent_mean=None, attr_ent_sd=None, mut_inf_mean=None,
                 mut_inf_sd=None, eq_num_attr=None, ns_ratio=None, nodes=None, leaves=None,
                 leaves_branch_mean=None, leaves_branch_sd=None, nodes_per_attr=None, leaves_per_class_mean=None,
                 leaves_per_class_sd=None, var_importance_mean=None, var_importance_sd=None, one_nn_mean=None,
                 one_nn_sd=None, best_node_mean=None, best_node_sd=None,
                 linear_discr_mean=None, linear_discr_sd=None, naive_bayes_mean=None, naive_bayes_sd=None,
                 nr_missing_values=None, pct_missing_values=None, nr_inst_mv=None, nr_attr_mv=None, pct_inst_mv=None,
                 pct_attr_mv=None, class_prob_mean=None, class_prob_std=None, class_column=None):
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
        self.sparsity_mean = sparsity_mean
        self.sparsity_sd = sparsity_sd
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

    def get_mf(self):
        return [self.nr_inst, self.nr_attr, self.nr_class, self.nr_missing_values, self.pct_missing_values,
                self.nr_inst_mv, self.pct_inst_mv, self.nr_attr_mv, self.pct_attr_mv, self.nr_outliers,
                self.skewness_mean, self.skewness_sd, self.kurtosis_mean, self.kurtosis_sd, self.cor_mean, self.cor_sd,
                self.cov_mean, self.cov_sd, self.sparsity_mean, self.sparsity_sd, self.var_mean, self.var_sd,
                self.class_prob_mean, self.class_prob_std, self.class_ent, self.attr_ent_mean, self.attr_ent_sd,
                self.mut_inf_mean, self.mut_inf_sd, self.eq_num_attr, self.ns_ratio, self.nodes, self.leaves,
                self.leaves_branch_mean, self.leaves_branch_sd, self.nodes_per_attr, self.leaves_per_class_mean,
                self.leaves_per_class_sd, self.var_importance_mean, self.var_importance_sd, self.one_nn_mean,
                self.one_nn_sd, self.best_node_mean, self.best_node_sd, self.linear_discr_mean, self.linear_discr_sd,
                self.naive_bayes_mean, self.naive_bayes_sd]


class Algorithm(Base):
    __tablename__ = 'algorithms'
    # __table_args__ = {'schema': 'mlb'}

    HybridType = Integer()
    HybridType = HybridType.with_variant(BigInteger(), 'postgresql')

    """
    Relational columns
    """
    id = Column(HybridType, primary_key=True, autoincrement=True)
    input_dataset = Column(HybridType, ForeignKey('datasets.id'), nullable=False)
    output_dataset = Column(HybridType, ForeignKey('datasets.id'), nullable=True)

    """
    Algorithm columns
    """
    algorithm = Column(String(300), nullable=False)
    status = Column(String(50), nullable=False, index=True)
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

    """
    Hostname
    """
    host = Column(String)

    @property
    def hyperparameter_values(self) -> dict:
        """Decode hyperparameters from base64 to object"""
        return base_64_to_object(self.hyperparameter_values_64)

    @hyperparameter_values.setter
    def hyperparameter_values(self, value: Configuration):
        """Encode hyperparameters from object to base64"""
        if value is None:
            d = {}
        else:
            d = value.get_dictionary()

        self.hyperparameter_values_64 = object_to_base_64(d)

    # @property
    # def mu_sigma_judgment_metric(self):
    #     # compute the lower confidence bound on the cross-validated
    #     # judgment metric
    #     if self.cv_judgment_metric is None:
    #         return None
    #     return self.cv_judgment_metric - 2 * self.cv_judgment_metric_stdev

    def __init__(self,
                 algorithm: str,
                 input_dataset: int,
                 output_dataset: Optional[int],
                 id: int = None,
                 status: AlgorithmStatus = None,
                 start_time: datetime = None,
                 end_time: datetime = None,
                 error_message: str = None,
                 hyperparameter_values: Configuration = None,
                 hyperparameter_values_64: str = None,
                 accuracy: float = None,
                 f1_score: float = None,
                 precision: float = None,
                 recall: float = None,
                 neg_log_loss: float = None,
                 roc_auc_score: float = None,
                 host=None):

        self.algorithm: str = algorithm
        self.status = status
        self.id: Optional[int] = id
        self.input_dataset: int = input_dataset
        self.output_dataset: Optional[int] = output_dataset
        self.start_time: Optional[datetime] = start_time
        self.end_time: Optional[datetime] = end_time
        self.error_message: Optional[str] = error_message
        if hyperparameter_values is not None:
            self.hyperparameter_values = hyperparameter_values
        else:
            self.hyperparameter_values_64 = hyperparameter_values_64
        self.host = host

        self.accuracy = accuracy
        self.f1_score = f1_score
        self.precision = precision
        self.recall = recall
        self.neg_log_loss = neg_log_loss
        self.roc_auc_score = roc_auc_score

        self._load_cs()

    def _load_cs(self):
        """
        method: method code or path to JSON file containing all the information
            needed to specify this enumerator.
        """
        cs = ALGORITHMS[self.algorithm].get_hyperparameter_search_space()
        self.cs = cs

    def random_config(self) -> Configuration:
        return self.cs.sample_configuration()

    def default_config(self) -> Configuration:
        return self.cs.get_default_configuration()

    def instance(self, params: dict = None) -> EstimatorComponent:
        if params is None:
            if self.hyperparameter_values is None:
                params = self.random_config().get_dictionary()
            else:
                params = self.hyperparameter_values

        instance = ALGORITHMS[self.algorithm](**params)
        # noinspection PyTypeChecker
        # EstimatorComponent inherits from BaseEstimator
        return instance

    def get_performance(self):
        return [self.accuracy, self.f1_score, self.precision, self.recall, self.neg_log_loss, self.roc_auc_score]

    def get_config_array(self):
        return Configuration(self.cs, self.hyperparameter_values).get_array()

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
            # 'Cross Validation Score: {:.3f} +- {:.3f}'.format(
            #     self.cv_judgment_metric, self.cv_judgment_metric_stdev),
            # 'Test Score: {:.3f}'.format(self.test_judgment_metric),
        ]

        return '\n'.join(to_print)


class Database(object):
    def __init__(self, dialect: str, database: str, username: str = None, password: str = None,
                 host: str = None, port: int = None, query: str = None):
        """
        Accepts configuration for a database connection, and defines SQLAlchemy
        ORM objects for all the tables in the database.
        """

        db_url = URL(drivername=dialect, database=database, username=username,
                     password=password, host=host, port=port, query=query)
        self.engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
        self.session: Optional[Session] = None
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
    def get_datasets(self) -> Optional[List[Dataset]]:
        """
        Get a list of datasets that can be worked on.
        """

        # Load running datasets first
        datasets = self.session.query(Dataset) \
            .filter(Dataset.status == RunStatus.RUNNING) \
            .limit(100) \
            .all()

        # Fallback to pending datasets
        if len(datasets) == 0:
            datasets = self.session.query(Dataset) \
                .filter(Dataset.status == RunStatus.PENDING) \
                .order_by(Dataset.depth) \
                .limit(100) \
                .all()

        if len(datasets) == 0:
            return None
        return datasets

    @try_with_session(commit=True)
    def select_dataset(self) -> Optional[Dataset]:
        try:
            rs = self.session.execute('''
            UPDATE {table}
            SET    status = '{running}' 
            WHERE  id = (
                SELECT id  FROM {table} WHERE  status = '{pending}'
                ORDER BY depth LIMIT 1 FOR UPDATE SKIP LOCKED
             )
            RETURNING id;
            '''.format(table=Dataset.__tablename__, running=RunStatus.RUNNING, pending=RunStatus.PENDING))

        except OperationalError:
            rs = self.session.execute('''
                SELECT id  FROM {table} WHERE  status = '{pending}' ORDER BY depth LIMIT 1
            '''.format(table=Dataset.__tablename__, pending=RunStatus.PENDING))

        try:
            id = next(rs)['id']
            ds = self.get_dataset(id)
            ds.status = RunStatus.RUNNING
            ds.start_time = datetime.now()
            return ds
        except StopIteration:
            return None

    @try_with_session()
    def get_datasets_by_hash(self, hashcode: str) -> List[Dataset]:
        """ Get a specific dataset. """
        return self.session.query(Dataset).filter(Dataset.hashcode == hashcode).all()

    @try_with_session()
    def get_algorithm(self, algorithm_id) -> Algorithm:
        """ Get a specific algorithm. """
        return self.session.query(Algorithm).get(algorithm_id)

    @try_with_session()
    def get_algorithm_count(self, dataset_id: int) -> int:
        """
        Get a list of all algorithms matching the give dataset.

        Args:
            dataset_id: id of the corresponding dataset
        """
        n = self.session.query(Algorithm).filter(Algorithm.input_dataset == dataset_id).count()
        return n

    # ##########################################################################
    # #  Methods to update the database  #######################################
    # ##########################################################################

    @try_with_session(commit=True)
    def create_dataset(self, **kwargs) -> Dataset:
        dataset = Dataset(**kwargs)
        self.session.add(dataset)
        return dataset

    @try_with_session(commit=True)
    def create_algorithm(self, dataset_id: int, algorithm: Algorithm) -> Algorithm:
        """Update dataset values in the db"""
        self.session.query(Dataset).filter(Dataset.id == dataset_id).update({Dataset.processed: Dataset.processed + 1},
                                                                            synchronize_session=False)
        self.session.commit()

        """Add algorithm to the db"""
        self.session.add(algorithm)
        return algorithm

    @try_with_session(commit=True)
    def complete_algorithm(self, algorithm_id, dataset_id: Optional[int], accuracy: float = None, f1: float = None,
                           precision: float = None,
                           recall: float = None, neg_log_loss: float = None, roc_auc: float = None):
        """Set all the parameters on a algorithm that haven't yet been set, and mark it as complete."""
        algorithm = self.session.query(Algorithm).get(algorithm_id)

        algorithm.output_dataset = dataset_id
        algorithm.end_time = datetime.now()
        algorithm.status = AlgorithmStatus.COMPLETE

        algorithm.accuracy = accuracy
        algorithm.f1_score = f1
        algorithm.precision = precision
        algorithm.recall = recall
        algorithm.neg_log_loss = neg_log_loss
        algorithm.roc_auc_score = roc_auc

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

        algorithm.accuracy = 0
        algorithm.f1_score = 0
        algorithm.precision = 0
        algorithm.recall = 0
        # Log loss is not limited. Value of 100 may be too small
        algorithm.neg_log_loss = 100
        algorithm.roc_auc_score = 0

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

    def export_db(self, max_depth: int = 4, algorithm: bool = True):
        with self.engine.connect() as con:
            con.execute('SET search_path TO mlb;')
            con.execute('SET work_mem TO "2000MB";')

            con.execute('''
            create materialized view if not exists unique_algorithms as
            select MIN(id) as id, COUNT(id) as count, algorithm, hyperparameter_values_64, input_dataset, output_dataset,
            AVG(accuracy) as accuracy, AVG(f1_score) as f1_score, AVG("precision") as "precision", AVG(recall) as recall, AVG(neg_log_loss) as neg_log_loss, AVG(roc_auc_score) as roc_auc_score
            from mlb.algorithms a
            group by algorithm, hyperparameter_values_64, input_dataset, output_dataset
            ''')

            for depth in range(2, max_depth):
                con.execute('drop MATERIALIZED view if exists pipelines;')

                query = '''create materialized view pipelines as select'''
                query += ','.join(['''
                    d{d}.nr_inst as d{d}_nr_inst,
                    d{d}.nr_attr as d{d}_nr_attr,
                    d{d}.nr_class as d{d}_nr_class,
                    d{d}.nr_missing_values as d{d}_nr_missing_values,
                    d{d}.pct_missing_values as d{d}_pct_missing_values,
                    d{d}.nr_inst_mv as d{d}_nr_inst_mv,
                    d{d}.pct_inst_mv as d{d}_pct_inst_mv,
                    d{d}.nr_attr_mv as d{d}_nr_attr_mv,
                    d{d}.pct_attr_mv as d{d}_pct_attr_mv,
                    d{d}.nr_outliers as d{d}_nr_outliers,
                    d{d}.skewness_mean as d{d}_skewness_mean,
                    d{d}.skewness_sd as d{d}_skewness_sd,
                    d{d}.kurtosis_mean as d{d}_kurtosis_mean,
                    d{d}.kurtosis_sd as d{d}_kurtosis_sd,
                    d{d}.cor_mean as d{d}_cor_mean,
                    d{d}.cor_sd as d{d}_cor_sd,
                    d{d}.cov_mean as d{d}_cov_mean,
                    d{d}.cov_sd as d{d}_cov_sd,
                    d{d}.sparsity_mean as d{d}_sparsity_mean,
                    d{d}.sparsity_sd as d{d}_sparsity_sd,
                    d{d}.var_mean as d{d}_var_mean,
                    d{d}.var_sd as d{d}_var_sd,
                    d{d}.class_prob_mean as d{d}_class_prob_mean,
                    d{d}.class_prob_std as d{d}_class_prob_std,
                    d{d}.class_ent as d{d}_class_ent,
                    d{d}.attr_ent_mean as d{d}_attr_ent_mean,
                    d{d}.attr_ent_sd as d{d}_attr_ent_sd,
                    d{d}.mut_inf_mean as d{d}_mut_inf_mean,
                    d{d}.mut_inf_sd as d{d}_mut_inf_sd,
                    d{d}.eq_num_attr as d{d}_eq_num_attr,
                    d{d}.ns_ratio as d{d}_ns_ratio,
                    d{d}.nodes as d{d}_nodes,
                    d{d}.leaves as d{d}_leaves,
                    d{d}.leaves_branch_mean as d{d}_leaves_branch_mean,
                    d{d}.leaves_branch_sd as d{d}_leaves_branch_sd,
                    d{d}.nodes_per_attr as d{d}_nodes_per_attr,
                    d{d}.leaves_per_class_mean as d{d}_leaves_per_class_mean,
                    d{d}.leaves_per_class_sd as d{d}_leaves_per_class_sd,
                    d{d}.var_importance_mean as d{d}_var_importance_mean,
                    d{d}.var_importance_sd as d{d}_var_importance_sd,
                    d{d}.one_nn_mean as d{d}_one_nn_mean,
                    d{d}.one_nn_sd as d{d}_one_nn_sd,
                    d{d}.best_node_mean as d{d}_best_node_mean,
                    d{d}.best_node_sd as d{d}_best_node_sd,
                    d{d}.linear_discr_mean as d{d}_linear_discr_mean,
                    d{d}.linear_discr_sd as d{d}_linear_discr_sd,
                    d{d}.naive_bayes_mean as d{d}_naive_bayes_mean,
                    d{d}.naive_bayes_sd as d{d}_naive_bayes_sd,
                    a{a}.algorithm as a{a}_algorithm,
                    a{a}.hyperparameter_values_64 as a{a}_hyperparameter_values_64,
                    a{a}.accuracy as a{a}_accuracy,
                    a{a}.f1_score as a{a}_f1_score,
                    a{a}.precision as a{a}_precision,
                    a{a}.recall as a{a}_recall,
                    a{a}.neg_log_loss as a{a}_neg_log_loss,
                    a{a}.roc_auc_score as a{a}_roc_auc_score'''.format(d=j, a=j + 1) for j in range(0, depth + 1)])
                query += '''\nfrom mlb.datasets d0\njoin unique_algorithms a1 on d0.id = a1.input_dataset\n'''
                for j in range(1, depth + 1):
                    query += '''join mlb.datasets d{idx} on d{idx}.id = a{idx}.output_dataset\n'''.format(idx=j)
                    query += '''join unique_algorithms a{} on d{}.id = a{}.input_dataset\n'''.format(j + 1, j, j + 1)
                query += 'where d0."depth" = 0 and a{}.accuracy is not null;'.format(depth + 1)
                print(query, end='\n\n\n')

                con.execute(query)
                con.execute('declare pip_curs CURSOR FOR SELECT * FROM pipelines')
                chunk = 0
                while True:
                    dataset_samples = None
                    hyperparam_samples = {}

                    df = pd.read_sql('FETCH 100000 FROM pip_curs', con)
                    if df.shape[0] == 0:
                        break
                    print('Chunk {}'.format(chunk))

                    dataset_candidates = []
                    hyperparam_candidates = []

                    for d in range(depth + 1):
                        algorithm = df.filter(regex='a{}_*'.format(d + 1)).rename(columns=lambda n: n[3:]) \
                            .apply(lambda row: Algorithm(**row.to_dict(), input_dataset=0, output_dataset=None), axis=1)

                        mf = df.filter(regex='d{}_*'.format(d))
                        mf.insert(0, 'algorithm', algorithm.apply(lambda a: a.algorithm))
                        mf.insert(1, 'depth', d)
                        dataset_candidates.append(mf)

                        hp = pd.DataFrame(data={'algorithm': algorithm.apply(lambda a: a.algorithm),
                                                'depth': d,
                                                'hp': algorithm.apply(lambda a: a.get_config_array())
                                                })
                        hyperparam_candidates.append(hp)

                        classifier = algorithm.apply(lambda a: not (a.accuracy is None or np.isnan(a.accuracy)))
                        if classifier.sum() == 0:
                            continue

                        performance = np.stack(algorithm[classifier].apply(lambda a: a.get_performance()).values)
                        for ds_candidate in dataset_candidates:
                            ds_sample = ds_candidate[classifier].copy()
                            ds_sample['depth'] = d - ds_sample['depth']
                            ds_sample = np.concatenate((ds_sample.to_numpy(), performance), axis=1)

                            if dataset_samples is None:
                                dataset_samples = ds_sample
                            else:
                                dataset_samples = np.concatenate((dataset_samples, ds_sample))

                        for hp_candidate in hyperparam_candidates:
                            hp_sample = hp_candidate[classifier].copy()
                            hp_sample['depth'] = d - hp_sample['depth']

                            for algo in hp_sample['algorithm'].unique():
                                idx = hp_sample['algorithm'] == algo
                                hp = np.concatenate((np.stack(
                                    hp_sample[idx].apply(lambda row: np.insert(row['hp'], 0, row['depth']), axis=1)),
                                                     performance[idx]), axis=1)

                                if algo not in hyperparam_samples:
                                    hyperparam_samples[algo] = hp
                                else:
                                    hyperparam_samples[algo] = np.concatenate((hyperparam_samples[algo], hp))

                    with open('export_depth_{}.{}.pkl'.format(depth, chunk), 'wb') as f:
                        pickle.dump((dataset_samples, hyperparam_samples), f)
                    chunk += 1

                con.execute('CLOSE  pip_curs;')
