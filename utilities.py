from __future__ import absolute_import, unicode_literals

import base64
import hashlib
import logging
import pickle

from builtins import str
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelBinarizer

logger = logging.getLogger('mlb')


def hash_file(fname: str, buf_size: int = 1048576):
    sha1 = hashlib.sha1()

    with open(fname, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def object_to_base_64(obj):
    """ Pickle and base64-encode an object. """
    pickled = pickle.dumps(obj)
    return base64.b64encode(pickled)


def base_64_to_object(b64str):
    """
    Inverse of object_to_base_64.
    Decode base64-encoded string and then unpickle it.
    """
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    """
    from https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
    """
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)


def logloss(y_test, y_pred):
    """
    from https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
    """
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return log_loss(y_test, y_pred)
