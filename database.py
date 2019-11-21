from __future__ import absolute_import, unicode_literals

import hashlib
import json
import os
import pickle
from builtins import object
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, MetaData, Numeric, String, Text, create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from constants import (BUDGET_TYPES, CLASSIFIER_STATUS, DATARUN_STATUS, METRICS, SCORE_TARGETS, ClassifierStatus,
                       RunStatus)
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
        the database must be destroyed and reinialized.
        """
        metadata = MetaData(bind=self.engine)
        Base = declarative_base(metadata=metadata)
        db = self

        # TODO adapt Datarun, Dataset and Classifier tables
        class Dataset(Base):
            __tablename__ = 'datasets'

            id = Column(Integer, primary_key=True, autoincrement=True)
            name = Column(String(100), nullable=False)

            # columns necessary for loading/processing data
            class_column = Column(String(100), nullable=False)
            train_path = Column(String(200), nullable=False)
            test_path = Column(String(200))
            description = Column(String(1000))

            # metadata columns, for convenience
            n_examples = Column(Integer, nullable=False)
            k_classes = Column(Integer, nullable=False)
            d_features = Column(Integer, nullable=False)
            majority = Column(Numeric(precision=10, scale=9), nullable=False)
            size_kb = Column(Integer, nullable=False)

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

            def _add_extra_fields(self, aws_access_key=None, aws_secret_key=None):

                data = load_data(self.name, self.train_path, aws_access_key, aws_secret_key)

                if self.n_examples is None:
                    self.n_examples = len(data)

                if self.k_classes is None:
                    self.k_classes = len(np.unique(data[self.class_column]))

                if self.d_features is None:
                    total_features = data.shape[1] - 1
                    for column in data.columns:
                        if data[column].dtype == 'object':
                            total_features += len(np.unique(data[column])) - 1

                    self.d_features = total_features

                if self.majority is None:
                    counts = data[self.class_column].value_counts()
                    self.majority = float(max(counts)) / float(sum(counts))

                if self.size_kb is None:
                    self.size_kb = int(np.array(data).nbytes / 1000)

            @staticmethod
            def _make_name(path):
                md5 = hashlib.md5(path.encode('utf-8'))
                return md5.hexdigest()

            def __init__(self, train_path, test_path=None, name=None, description=None,
                         class_column=None, n_examples=None, majority=None, k_classes=None,
                         size_kb=None, d_features=None, id=None, aws_access_key=None,
                         aws_secret_key=None):

                self.train_path = train_path
                self.test_path = test_path
                self.name = name or self._make_name(train_path)
                self.description = description or self.name
                self.class_column = class_column
                self.id = id

                self.n_examples = n_examples
                self.d_features = d_features
                self.majority = majority
                self.k_classes = k_classes
                self.size_kb = size_kb

                self._add_extra_fields(aws_access_key, aws_secret_key)

            def __repr__(self):
                base = "<%s: %s, %d classes, %d features, %d rows>"
                return base % (self.name, self.description, self.k_classes,
                               self.d_features, self.n_examples)

        class Datarun(Base):
            __tablename__ = 'dataruns'

            # relational columns
            id = Column(Integer, primary_key=True, autoincrement=True)
            dataset_id = Column(Integer, ForeignKey('datasets.id'))
            dataset = relationship('Dataset', back_populates='dataruns')

            description = Column(String(200), nullable=False)
            priority = Column(Integer)

            # hyperparameter selection and tuning settings
            selector = Column(String(200), nullable=False)
            k_window = Column(Integer)
            tuner = Column(String(200), nullable=False)
            gridding = Column(Integer, nullable=False)
            r_minimum = Column(Integer)

            # budget settings
            budget_type = Column(Enum(*BUDGET_TYPES))
            budget = Column(Integer)
            deadline = Column(DateTime)

            # which metric to use for judgment, and how to compute it
            metric = Column(Enum(*METRICS))
            score_target = Column(Enum(*[s + '_judgment_metric' for s in
                                         SCORE_TARGETS]))

            # variables that store the status of the datarun
            start_time = Column(DateTime)
            end_time = Column(DateTime)
            status = Column(Enum(*DATARUN_STATUS), default=RunStatus.PENDING)

            def __repr__(self):
                base = "<ID = %d, dataset ID = %s, strategy = %s, budget = %s (%s), status: %s>"
                return base % (self.id, self.dataset_id, self.description,
                               self.budget_type, self.budget, self.status)

            @property
            def completed_classifiers(self):
                return len(self.get_complete_classifiers())

            def get_scores(self):
                columns = [
                    'id',
                    'cv_judgment_metric',
                    'cv_judgment_metric_stdev',
                    'test_judgment_metric',
                ]

                classifiers = db.get_classifiers(datarun_id=self.id)
                scores = [
                    {
                        key: value
                        for key, value in vars(classifier).items()
                        if key in columns
                    }
                    for classifier in classifiers
                ]

                scores = pd.DataFrame(scores)
                scores.sort_values(by='cv_judgment_metric', ascending=False, inplace=True)
                scores['rank'] = scores['cv_judgment_metric'].rank(ascending=0)

                return scores.reset_index(drop=True)

            def get_complete_classifiers(self):
                return db.get_classifiers(datarun_id=self.id, status=ClassifierStatus.COMPLETE)

            def export_best_classifier(self, path, force=False):
                if os.path.exists(path) and not force:
                    print('The indicated path already exists. Use `force=True` to overwrite.')

                base_path = os.path.dirname(path)
                if base_path and not os.path.exists(base_path):
                    os.makedirs(base_path)

                classifier = self.get_best_classifier()
                model = classifier.load_model()
                with open(path, 'wb') as pickle_file:
                    pickle.dump(model, pickle_file)

                print("Classifier {} saved as {}".format(classifier.id, path))

            def describe(self):
                dataset = db.get_dataset(self.dataset_id)

                elapsed = self.end_time - self.start_time if self.end_time else 'Not finished yet.'

                to_print = [
                    'Datarun {} summary:'.format(self.id),
                    "\tDataset: '{}'".format(dataset.train_path),
                    "\tColumn Name: '{}'".format(dataset.class_column),
                    "\tJudgment Metric: '{}'".format(self.metric),
                    '\tClassifiers Tested: {}'.format(len(db.get_classifiers(datarun_id=self.id))),
                    '\tElapsed Time: {}'.format(elapsed),
                ]

                print('\n'.join(to_print))

        Dataset.dataruns = relationship('Datarun', order_by='Datarun.id',
                                        back_populates='dataset')

        class Classifier(Base):
            __tablename__ = 'classifiers'

            # relational columns
            id = Column(Integer, primary_key=True, autoincrement=True)
            datarun_id = Column(Integer, ForeignKey('dataruns.id'))
            datarun = relationship('Datarun', back_populates='classifiers')
            hyperpartition_id = Column(Integer, ForeignKey('hyperpartitions.id'))
            hyperpartition = relationship('Hyperpartition',
                                          back_populates='classifiers')

            # name of the host where the model was trained
            host = Column(String(50))

            # these columns point to where the output is stored
            model_location = Column(String(300))
            metrics_location = Column(String(300))

            # base 64 encoding of the hyperparameter names and values
            hyperparameter_values_64 = Column(Text, nullable=False)

            # performance metrics
            cv_judgment_metric = Column(Numeric(precision=20, scale=10))
            cv_judgment_metric_stdev = Column(Numeric(precision=20, scale=10))
            test_judgment_metric = Column(Numeric(precision=20, scale=10))

            start_time = Column(DateTime)
            end_time = Column(DateTime)
            status = Column(Enum(*CLASSIFIER_STATUS), nullable=False)
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
                return (self.cv_judgment_metric - 2 * self.cv_judgment_metric_stdev)

            def __repr__(self):

                params = '\n'.join(
                    [
                        '\t{}: {}'.format(name, value)
                        for name, value in self.hyperparameter_values.items()
                    ]
                )

                to_print = [
                    'Classifier id: {}'.format(self.id),
                    'Classifier type: {}'.format(
                        db.get_hyperpartition(self.hyperpartition_id).method),
                    'Params chosen: \n{}'.format(params),
                    'Cross Validation Score: {:.3f} +- {:.3f}'.format(
                        self.cv_judgment_metric, self.cv_judgment_metric_stdev),
                    'Test Score: {:.3f}'.format(self.test_judgment_metric),
                ]

                return '\n'.join(to_print)

            def load_metrics(self):
                """Return the metrics"""
                if self.metrics_location.startswith('s3'):
                    pickled = self.load_s3_data(
                        self.metrics_location,
                        self.aws_access_key,
                        self.aws_secret_key,
                    )
                    return json.load(pickled)

                else:
                    with open(self.metrics_location, 'rb') as f:
                        return json.load(f)

        Datarun.classifiers = relationship('Classifier',
                                           order_by='Classifier.id',
                                           back_populates='datarun')

        self.Dataset = Dataset
        self.Classifier = Classifier

        Base.metadata.create_all(bind=self.engine)

    # ##########################################################################
    # #  Standard query methods  ###############################################
    # ##########################################################################

    @try_with_session()
    def get_dataset(self, dataset_id):
        """ Get a specific datarun. """
        return self.session.query(self.Dataset).get(dataset_id)

    @try_with_session()
    def get_datasets(self, ignore_pending: bool = False, ignore_running: bool = False,
                     ignore_complete: bool = True) -> Optional[List['Database.Dataset']]:
        # TODO adapt

        query = self.session.query(self.Dataset)
        if ignore_pending:
            query = query.filter(self.Dataset.status != RunStatus.PENDING)
        if ignore_running:
            query = query.filter(self.Dataset.status != RunStatus.RUNNING)
        if ignore_complete:
            query = query.filter(self.Dataset.status != RunStatus.COMPLETE)

        datasets = query.all()

        if not len(datasets):
            return None
        return datasets

    @try_with_session()
    def get_classifiers(self, ignore_pending=False, ignore_running=False,
                        ignore_complete=True) -> Optional[List['Database.Classifier']]:
        """
        Get a list of all datasets matching the chosen filters.

        Args:
            ignore_pending: if True, ignore classifiers that have not been started
            ignore_running: if True, ignore classifiers that are already running
            ignore_complete: if True, ignore completed classifiers
        """
        # TODO adapt
        query = self.session.query(self.Classifier)
        if ignore_pending:
            query = query.filter(self.Classifier.status != RunStatus.PENDING)
        if ignore_running:
            query = query.filter(self.Classifier.status != RunStatus.RUNNING)
        if ignore_complete:
            query = query.filter(self.Classifier.status != RunStatus.COMPLETE)

        classifiers = query.all()

        if not len(classifiers):
            return None
        return classifiers

    @try_with_session()
    def get_classifier(self, classifier_id):
        """ Get a specific classifier. """
        return self.session.query(self.Classifier).get(classifier_id)

    # ##########################################################################
    # #  Methods to update the database  #######################################
    # ##########################################################################

    @try_with_session(commit=True)
    def create_dataset(self, **kwargs):
        dataset = self.Dataset(**kwargs)
        self.session.add(dataset)
        return dataset

    @try_with_session(commit=True)
    def start_classifier(self, dataset_id: int, host: str, algorithm: str,
                         hyperparameter_values: Dict) -> 'Database.Classifier':
        """
        Save a new, fully qualified classifier object to the database.
        Returns: the ID of the newly-created classifier
        """
        # TODO adapt
        classifier = self.Classifier(dataset_id=dataset_id,
                                     host=host,
                                     algorithm=algorithm,
                                     hyperparameter_values=hyperparameter_values,
                                     start_time=datetime.now(),
                                     status=ClassifierStatus.RUNNING)
        self.session.add(classifier)
        return classifier

    @try_with_session(commit=True)
    def complete_classifier(self, classifier_id, model_location,
                            metrics_location, cv_score, cv_stdev, test_score):
        """
        Set all the parameters on a classifier that haven't yet been set, and mark
        it as complete.
        """
        # TODO adapt
        classifier = self.session.query(self.Classifier).get(classifier_id)

        classifier.model_location = model_location
        classifier.metrics_location = metrics_location
        classifier.cv_judgment_metric = cv_score
        classifier.cv_judgment_metric_stdev = cv_stdev
        classifier.test_judgment_metric = test_score
        classifier.end_time = datetime.now()
        classifier.status = ClassifierStatus.COMPLETE

    @try_with_session(commit=True)
    def mark_classifier_errored(self, classifier_id, error_message):
        """
        Mark an existing classifier as having errored and set the error message. If
        the classifier's hyperpartiton has produced too many erring classifiers, mark it
        as errored as well.
        """
        classifier = self.session.query(self.Classifier).get(classifier_id)
        classifier.error_message = error_message
        classifier.status = ClassifierStatus.ERRORED
        classifier.end_time = datetime.now()

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
