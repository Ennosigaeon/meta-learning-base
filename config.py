"""Configuration Module."""

from __future__ import absolute_import, unicode_literals

import argparse
from builtins import object

import yaml

from constants import SQL_DIALECTS


class Config(object):
    """
    Class which stores configuration for one aspect of Core. Subclasses of
    Config should define the list of all configurable parameters and any
    default values for those parameters other than None (in PARAMETERS and
    DEFAULTS, respectively). The object can be initialized with any number of
    keyword arguments; only kwargs that are in PARAMETERS will be used. This
    means you can (relatively) safely do things like ``args = parser.parse_args()``
    ``conf = Config(**vars(args))`` and only relevant parameters will be set.

    Subclasses do not need to define __init__ or any other methods.
    """
    _PREFIX = None
    _CONFIG = None

    @classmethod
    def _add_prefix(cls, name):
        if cls._PREFIX:
            return '{}_{}'.format(cls._PREFIX, name)
        else:
            return name

    @classmethod
    def _get_arg(cls, args, name, use_prefix):
        class_value = getattr(cls, name)

        if use_prefix:
            name = cls._add_prefix(name)

        if isinstance(class_value, dict):
            required = 'default' not in class_value
            default = class_value.get('default')
        elif isinstance(class_value, tuple):
            required = False
            default = class_value[1]
        else:
            required = False
            default = None

        if required and name not in args:
            raise KeyError(name)

        return args.get(name, default)

    def __init__(self, args, path=None):
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        config_arg = self._CONFIG or self._PREFIX
        if not path and config_arg:
            path = args.get(config_arg + '_config')

        if path:
            with open(path, 'r') as f:
                args = yaml.load(f)
                use_prefix = False
        else:
            use_prefix = True

        for name, value in vars(self.__class__).items():
            if not name.startswith('_') and not callable(value):
                setattr(self, name, self._get_arg(args, name, use_prefix))

    @classmethod
    def get_parser(cls):
        """Get an ArgumentParser for this config."""
        parser = argparse.ArgumentParser(add_help=False)

        # make sure the text for these arguments is formatted correctly
        # this allows newlines in the help strings
        parser.formatter_class = argparse.RawTextHelpFormatter

        if cls._PREFIX:
            parser.add_argument('--{}-config'.format(cls._PREFIX),
                                help='path to yaml {} config file'.format(cls._PREFIX))

        for name, description in vars(cls).items():
            if not name.startswith('_') and not callable(description):
                arg_name = '--' + cls._add_prefix(name).replace('_', '-')

                if isinstance(description, dict):
                    parser.add_argument(arg_name, **description)

                elif isinstance(description, tuple):
                    description, default = description
                    parser.add_argument(arg_name, help=description, default=default)

                else:
                    parser.add_argument(arg_name, help=description)

        return parser

    def to_dict(self):
        """Get a dict representation of this configuraiton."""
        return {
            name: value
            for name, value in vars(self).items()
            if not name.startswith('_') and not callable(value)
        }

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.to_dict())


class S3Config(Config):
    """ Stores configuration for AWS S3 connections """
    _PREFIX = 's3'

    endpoint = 'S3 endpoint'
    access_key = 'S3 access key'
    secret_key = 'S3 secret key'
    bucket = 'S3 bucket to store data'


class DatasetConfig(Config):
    """ Stores configuration of a Dataset """
    _CONFIG = 'run'

    name = 'Given name for this dataset.'
    train_path = {
        'help': 'Path to raw training data',
    }
    openml = {
        'help': 'Flag whether this dataset is stored in OpenML',
    }
    class_column = {
        'help': 'Column containing class labels',
        'default': 'Class'
    }


class SQLConfig(Config):
    """ Stores configuration for SQL database setup & connection """
    _PREFIX = 'sql'

    dialect = {
        'help': 'Dialect of SQL to use',
        'default': 'sqlite',
        'choices': SQL_DIALECTS
    }
    database = ('Name of, or path to, SQL database', 'ml-base.db')
    username = 'Username for SQL database'
    password = 'Password for SQL database'
    host = 'Hostname for database machine'
    port = 'Port used to connect to database'
    query = 'Specify extra login details'


class LogConfig(Config):
    models_dir = ('Directory where computed models will be saved', 'models')
    metrics_dir = ('Directory where model metrics will be saved', 'metrics')
    verbose_metrics = {
        'help': (
            'If set, compute full ROC and PR curves and '
            'per-label metrics for each algorithm'
        ),
        'action': 'store_true',
        'default': False
    }


class GenericConfig(Config):
    """ Generic configurations """
    _CONFIG = 'run'

    work_dir = {
        'help': 'Path to data location',
        # 'default': '/vagrant/data/',  # local
        'default': '/data/disk1/',  # gcloud
        'required': False
    }
    timeout = {
        'help': 'Time in seconds to execute a single algorithm',
        'default': 7200,
        'required': False
    }
