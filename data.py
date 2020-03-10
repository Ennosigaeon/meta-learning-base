import logging
import os
import time
from typing import Tuple

import boto3
import botocore
import openml
import pandas as pd
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from config import DatasetConfig

LOGGER = logging.getLogger('mlb')


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


def load_data(path: str, s3_endpoint: str = None, s3_bucket: str = None, s3_access_key: str = None,
              s3_secret_key: str = None, name: str = None) -> pd.DataFrame:
    if not os.path.isfile(path):
        if s3_access_key is None:
            raise FileNotFoundError(f'{path} does not exist')
        else:
            client = boto3.client(
                's3',
                endpoint_url=s3_endpoint,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
            )

            try:
                LOGGER.info('Downloading {}'.format(path))
                client.download_file(s3_bucket, '{}.parquet'.format(name), path)
            except ClientError as e:
                LOGGER.error('An error occurred trying to download from AWS3.'
                             'The following error has been returned: {}'.format(e))

    return pd.read_parquet(path)


def store_data(df: pd.DataFrame, work_dir: str, name: str) -> str:
    LOGGER.info('Saving file to working directory.')

    # TODO what if disk is full? python-diskcache

    # Checks if working directory already exists --> If not create working directory
    if not os.path.isdir(work_dir):
        LOGGER.info('Creating work directory \'{}\''.format(work_dir))
        os.mkdir(work_dir)

    # TODO what if filename already exists?
    # Checks if filename already exists in directory. As long as isfile(name) = True --> create uuid as new name
    # while os.path.isfile(name + '.parquet') is True:
    #     LOGGER.info('Filename {} already exists.'.format(name))
    #     name = str(uuid.uuid4())
    #     LOGGER.info('Set generated uuid {} as new filename.'.format(name))

    # Change dtype of column names to string --> parquet must have string column names
    df.columns = df.columns.astype(str)

    # Create path and save dataframe as parquet-file to that path
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
    # S3 Storage is disabled
    if s3_endpoint is None or s3_bucket is None or s3_secret_key is None or s3_access_key is None:
        return input_file, input_file

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
        client.upload_file(input_file, s3_bucket, '{}.parquet'.format(name))
        remote_path = "{0}/{1}/{2}".format(s3_endpoint, s3_bucket, name)
        return input_file, remote_path
    except ClientError as e:
        LOGGER.error('An error occurred trying to upload to AWS3.'
                     'The following error has been returned: {}'.format(e))
