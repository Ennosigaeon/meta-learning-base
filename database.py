from __future__ import absolute_import, unicode_literals

import hashlib
import pickle
import textwrap
from builtins import object
from datetime import datetime
from typing import Optional, List

import pandas as pd
from ConfigSpace.configuration_space import Configuration
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text, create_engine, BigInteger
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from automl.components.base import EstimatorComponent
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
    nr_num = Column(Numeric)
    nr_cat = Column(Numeric)
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
                 nr_inst=None, nr_attr=None, nr_num=None, nr_cat=None, nr_class=None, nr_outliers=None,
                 skewness_mean=None, skewness_sd=None, kurtosis_mean=None, kurtosis_sd=None, cor_mean=None, cor_sd=None,
                 cov_mean=None, cov_sd=None, sparsity_mean=None, sparsity_sd=None,
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
        self.nr_num = nr_num
        self.nr_cat = nr_cat
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
        return [self.nr_inst, self.nr_attr, self.nr_num, self.nr_cat, self.nr_class, self.nr_missing_values,
                self.pct_missing_values,
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

        # TODO 'd1498' very slow
        self.schemas = ['d1043', 'd1063', 'd12', 'd1461', 'd1464', 'd1471', 'd1486', 'd1489', 'd1590', 'd18',
                        'd23512', 'd23517', 'd3', 'd31', 'd40668', 'd40685', 'd40975', 'd40981', 'd40984', 'd41027',
                        'd41143', 'd41146', 'd41168', 'd458', 'd469', 'd478', 'd54']

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
                ORDER BY depth DESC LIMIT 1 FOR UPDATE SKIP LOCKED
             )
            RETURNING id;
            '''.format(table=Dataset.__tablename__, running=RunStatus.RUNNING, pending=RunStatus.PENDING))

        except OperationalError:
            rs = self.session.execute('''
                SELECT id  FROM {table} WHERE  status = '{pending}' ORDER BY depth DESC LIMIT 1
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

    def _cleanup(self, con, schema: str):
        if schema == 'd3':
            con.execute('''delete from d3.algorithms where id >= 10735;
                        delete from d3.datasets where id >= 727;''')
        if schema == 'd12':
            con.execute('''delete from d12.algorithms where id < 16648;
                        delete from d12.datasets where id < 1725;''')

        # Remove obsolete algorithms
        con.execute('''
            delete from algorithms where id in (
                select a.id from algorithms a
                join datasets d on a.input_dataset = d.id
                where a.start_time < d.start_time or a.end_time > d.end_time
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

    def export_pipelines(self, max_depth: int = None, base_dir: str = 'assets/exports', schema: str = None,
                         cleanup: bool = False):
        with self.engine.connect() as con:
            for s in self.schemas:
                if schema is not None and s != schema:
                    continue

                con.execute('SET search_path TO {};'.format(s))
                con.execute('SET work_mem TO "2000MB";')

                if max_depth is None:
                    rs = con.execute('select max(depth) from datasets d')
                    for row in rs:
                        max_depth = row[0] + 1
                    if max_depth is None:
                        raise ValueError('max_depth is None')

                # Legacy clean-up
                if cleanup:
                    self._cleanup(con, s)

                con.execute('''
                create materialized view if not exists unique_algorithms as
                select MIN(id) as id, COUNT(id) as count, status, algorithm, hyperparameter_values_64, input_dataset, output_dataset,
                AVG(accuracy) as accuracy, AVG(f1_score) as f1_score, AVG("precision") as "precision", AVG(recall) as recall, AVG(neg_log_loss) as neg_log_loss, AVG(roc_auc_score) as roc_auc_score
                from algorithms a
                group by algorithm, status, hyperparameter_values_64, input_dataset, output_dataset;
                ''')
                con.execute('create materialized view if not exists unique_datasets as select * from datasets d;')

                con.execute('drop MATERIALIZED view if exists pipelines;')

                query = 'create materialized view pipelines as select'
                query += ','.join([textwrap.dedent('''
                    d{d}.id as d{d}_id,
                    a{a}.algorithm as a{a}_algorithm,
                    a{a}.accuracy as a{a}_accuracy,
                    a{a}.f1_score as a{a}_f1_score,
                    a{a}.precision as a{a}_precision,
                    a{a}.recall as a{a}_recall,
                    a{a}.neg_log_loss as a{a}_neg_log_loss,
                    a{a}.roc_auc_score as a{a}_roc_auc_score '''.format(d=j, a=j + 1)) for j in range(0, max_depth)])
                for j in range(0, max_depth):
                    if j == 0:
                        query += 'from datasets d{idx}\n'.format(idx=j)
                    else:
                        query += '''join datasets d{idx} on d{idx}.id = a{idx}.output_dataset\n'''.format(idx=j)

                    cond = '''.status = 'complete' ''' if j < max_depth else '.accuracy is not null'
                    query += '''join algorithms a{a} on d{d}.id = a{a}.input_dataset and a{a}{cond}\n'''.format(
                        a=j + 1, d=j, cond=cond)
                query += 'where d0."depth" = 0;'.format(max_depth)
                print(query, end='\n\n\n')

                con.execute(query)

                dataset_performances = None
                for d in range(max_depth):
                    query = '''select '{}' as schema, ds, algo, depth,
                                AVG(accuracy) as accuracy, COALESCE(variance(accuracy), 0) as accuracy_var,
                                AVG(f1_score) as f1_score, COALESCE(variance(f1_score), 0) as f1_score_var,
                                AVG(precision) as precision, COALESCE(variance(precision), 0) as precision_var,
                                AVG(recall) as recall, COALESCE(variance(recall), 0) as recall_var,
                                AVG(log_loss) as log_loss, COALESCE(variance(log_loss), 0) as log_loss_var,
                                AVG(roc_auc) as roc_auc, COALESCE(variance(roc_auc), 0) as roc_auc_var
                                from (\n'''.format(
                        s)

                    unions = []
                    for i in range(d + 1, max_depth + 1):
                        # cond = ' and '.join(['d{}_id < d{}_id'.format(j, j+1) for j in range(0, i - 1)]) if i > 1 else 'true'

                        unions.append(textwrap.dedent(
                            '''select distinct d{d}_id as ds, a{al}_algorithm as algo, {dep} as depth,
                              a{a}_accuracy as accuracy, a{a}_f1_score as f1_score, a{a}_precision as precision,
                              a{a}_recall as recall, a{a}_neg_log_loss as log_loss, a{a}_roc_auc_score as roc_auc from pipelines where a{a}_accuracy is not null'''.format(
                                d=d, al=d + 1, a=i, dep=i - d)))
                    query += '\nunion '.join(unions)

                    query += '\n) as raw group by ds, algo, depth order by ds, algo, depth;'
                    print(query)

                    df = pd.read_sql(query, con)
                    if df.shape[0] == 0:
                        break

                    if dataset_performances is None:
                        dataset_performances = df
                    else:
                        dataset_performances = pd.concat([dataset_performances, df], ignore_index=True)

                with open('{}/performance_{}.pkl'.format(base_dir, s), 'wb') as f:
                    pickle.dump(dataset_performances, f)

    def export_datasets(self, base_dir: str = 'assets/exports'):
        with self.engine.connect() as con:
            con.execute('SET search_path TO public;')
            con.execute('SET work_mem TO "2000MB";')

            query = '''create temp table all_datasets as  select
                schema as schema,
                id as id,
                depth,
                nr_inst as nr_inst,
                nr_attr as nr_attr,
                nr_num / nr_attr as nr_num,
                nr_cat / nr_attr as nr_cat,
                nr_class as nr_class,
                nr_missing_values as nr_missing_values,
                pct_missing_values as pct_missing_values,
                nr_inst_mv as nr_inst_mv,
                pct_inst_mv / 100 as pct_inst_mv,
                nr_attr_mv as nr_attr_mv,
                pct_attr_mv / 100 as pct_attr_mv,
                nr_outliers as nr_outliers,
                skewness_mean as skewness_mean,
                skewness_sd as skewness_sd,
                kurtosis_mean as kurtosis_mean,
                kurtosis_sd as kurtosis_sd,
                cor_mean as cor_mean,
                cor_sd as cor_sd,
                cov_mean as cov_mean,
                cov_sd as cov_sd,
                sparsity_mean as sparsity_mean,
                sparsity_sd as sparsity_sd,
                var_mean as var_mean,
                var_sd as var_sd,
                class_prob_mean as class_prob_mean,
                class_prob_std as class_prob_std,
                class_ent as class_ent,
                attr_ent_mean as attr_ent_mean,
                attr_ent_sd as attr_ent_sd,
                mut_inf_mean as mut_inf_mean,
                mut_inf_sd as mut_inf_sd,
                eq_num_attr as eq_num_attr,
                ns_ratio as ns_ratio,
                nodes as nodes,
                leaves as leaves,
                leaves_branch_mean as leaves_branch_mean,
                leaves_branch_sd as leaves_branch_sd,
                nodes_per_attr as nodes_per_attr,
                leaves_per_class_mean as leaves_per_class_mean,
                leaves_per_class_sd as leaves_per_class_sd,
                var_importance_mean as var_importance_mean,
                var_importance_sd as var_importance_sd
                FROM ('''

            query += 'union '.join([
                '''select '{s}' as schema, id, depth, nr_inst, nr_attr, nr_num, nr_cat, nr_class, nr_missing_values, pct_missing_values, nr_inst_mv, pct_inst_mv, nr_attr_mv, pct_attr_mv, nr_outliers, skewness_mean, skewness_sd, kurtosis_mean, kurtosis_sd, cor_mean, cor_sd, cov_mean, cov_sd, sparsity_mean, sparsity_sd, var_mean, var_sd, class_prob_mean, class_prob_std, class_ent, attr_ent_mean, attr_ent_sd, mut_inf_mean, mut_inf_sd, eq_num_attr, ns_ratio, nodes, leaves, leaves_branch_mean, leaves_branch_sd, nodes_per_attr, leaves_per_class_mean, leaves_per_class_sd, var_importance_mean, var_importance_sd from {s}.datasets d where status = 'complete'\n'''.format(
                    s=s) for s in self.schemas])
            query += ') as sub'
            print(query, end='\n\n\n')

            con.execute(query)
            con.execute('declare pip_curs CURSOR FOR SELECT * FROM all_datasets')
            chunk = 0
            while True:

                df = pd.read_sql('FETCH 500000 FROM pip_curs', con)
                if df.shape[0] == 0:
                    break
                print('Chunk {}'.format(chunk))

                with open('{}/export_datasets_{}.pkl'.format(base_dir, chunk), 'wb') as f:
                    pickle.dump(df, f)
                chunk += 1

            con.execute('CLOSE  pip_curs;')
