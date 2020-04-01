"""Constants module."""

import logging
from builtins import object

# A bunch of constants which are used throughout the project, mostly for config.
SQL_DIALECTS = ['sqlite', 'mysql', 'postgres']
ALGORITHM_STATUS = ['running', 'errored', 'complete']
RUN_STATUS = ['pending', 'running', 'complete']

S3_PREFIX = '^s3://'
HTTP_PREFIX = '^https?://'

LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NONE': logging.NOTSET
}


class AlgorithmStatus(object):
    RUNNING = 'running'
    ERRORED = 'errored'
    COMPLETE = 'complete'


class RunStatus(object):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETE = 'complete'
    SKIPPED = 'skipped'


# these are the strings that are used to index into results dictionaries
class Metrics(object):
    ACCURACY = 'accuracy'
    RANK_ACCURACY = 'rank_accuracy'
    COHEN_KAPPA = 'cohen_kappa'
    F1 = 'f1'
    F1_MICRO = 'f1_micro'
    F1_MACRO = 'f1_macro'
    ROC_AUC = 'roc_auc'  # receiver operating characteristic
    ROC_AUC_MICRO = 'roc_auc_micro'
    ROC_AUC_MACRO = 'roc_auc_macro'
    AP = 'ap'  # average precision
    MCC = 'mcc'  # matthews correlation coefficient
    PR_CURVE = 'pr_curve'
    ROC_CURVE = 'roc_curve'


METRICS_BINARY = [
    Metrics.ACCURACY,
    Metrics.COHEN_KAPPA,
    Metrics.F1,
    Metrics.ROC_AUC,
    Metrics.AP,
    Metrics.MCC,
]

METRICS_MULTICLASS = [
    Metrics.ACCURACY,
    Metrics.RANK_ACCURACY,
    Metrics.COHEN_KAPPA,
    Metrics.F1_MICRO,
    Metrics.F1_MACRO,
    Metrics.ROC_AUC_MICRO,
    Metrics.ROC_AUC_MACRO,
]

METRICS = list(set(METRICS_BINARY + METRICS_MULTICLASS))
