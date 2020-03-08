import logging
import os
import time

import boto3
import botocore
import openml
import pandas as pd
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from typing import Tuple

from config import DatasetConfig

LOGGER = logging.getLogger('mlb')

WORK_DIR = 'data'


def _get_local_path(path: str, name: str = None) -> str:
    file_type = '.parquet'

    if name is None:
        name = path.split('/')[-1]
    if not name.endswith(file_type):
        name = name + file_type

    cwd = os.getcwd()
    data_path = os.path.join(cwd, WORK_DIR)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, name)


def load_openml(dataset_conf: DatasetConfig) -> pd.DataFrame:
    ds = openml.datasets.get_dataset(dataset_conf.openml)
    X, y, categorical_indicator, attribute_names = ds.get_data(
        dataset_format='dataframe',
        target=ds.default_target_attribute
    )
    df = pd.concat([X, y], axis=1)

    # Fix configuration
    dataset_conf.format = ds.format
    dataset_conf.class_column = ds.default_target_attribute
    dataset_conf.name = '{}_{}_{}'.format(ds.name, dataset_conf.openml, time.time())

    return df


def load_data(path: str, s3_endpoint: str = None, s3_access_key: str = None,
              s3_secret_key: str = None, name: str = None) -> pd.DataFrame:
    if not os.path.isfile(path):
        local_path = _get_local_path(path, name)
        if os.path.isfile(local_path):
            path = local_path
        elif s3_access_key is None:
            raise FileNotFoundError(f'{path} does not exist')
        else:
            client = boto3.client(
                's3',
                endpoint_url=s3_endpoint,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
            )

            bucket = path.split('/')[2]
            file_to_download = path.replace('s3://{}/'.format(bucket), '')

            try:
                LOGGER.info('Downloading {}'.format(path))
                client.download_file(bucket, file_to_download, local_path)
                path = local_path
            except ClientError as e:
                LOGGER.error('An error occurred trying to download from AWS3.'
                             'The following error has been returned: {}'.format(e))

    return pd.read_parquet(path)


def store_data(df: pd.DataFrame, work_dir: str, name: str) -> str:
    # TODO what if name already exists?
    # TODO what if disk is full? python-diskcache
    if not os.path.isdir(work_dir):
        LOGGER.info('Creating work directory \'{}\''.format(work_dir))
        os.mkdir(work_dir)

    path = os.path.join(work_dir, name) + '.parquet'
    df.to_parquet(path)
    return path


def delete_data(train_path: str):
    if os.path.exists(train_path):
        os.remove(train_path)
    else:
        LOGGER.info('Dataset does not exist in local storage.')


def upload_data(input_file: str, s3_endpoint: str, s3_bucket: str, s3_access_key: str, s3_secret_key: str,
                name: str = None) -> Tuple[str, str]:
    local_path = _get_local_path(input_file, name)

    # S3 Storage is disabled
    if s3_endpoint is None or s3_bucket is None or s3_secret_key is None or s3_access_key is None:
        return local_path, local_path

    client: BaseClient = boto3.client(
        's3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    """Checks if s3_bucket already exists. If not call create_bucket"""
    try:
        client.head_bucket(Bucket=s3_bucket)
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = e.response['Error']['Code']
        if error_code == '404':
            client.create_bucket(Bucket=s3_bucket)

    if name is None:
        name = input_file.split('/')[-1]
    try:
        LOGGER.info('Uploading {}'.format(input_file))
        client.upload_file(input_file, s3_bucket, name)
        LOGGER.debug('1')
        remote_path = "{0}/{1}/{2}".format(s3_endpoint, s3_bucket, name)
        LOGGER.debug('2')
        local_path = _get_local_path(input_file, name)
        # TODO move file from local_file to local_path
        return local_path, remote_path
    except ClientError as e:
        LOGGER.error('An error occurred trying to upload to AWS3.'
                     'The following error has been returned: {}'.format(e))
