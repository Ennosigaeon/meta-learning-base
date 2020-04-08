import logging
import os

import openml
import pandas as pd
import time
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.cloud.storage import Client
from typing import Tuple

from config import DatasetConfig

LOGGER = logging.getLogger('mlb')


def load_openml(dataset_conf: DatasetConfig) -> pd.DataFrame:
    LOGGER.info("Loading openml dataset {}".format(dataset_conf.openml))
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


def load_data(path: str, s3_config: str = None, s3_bucket: str = None, name: str = None) -> pd.DataFrame:
    if not os.path.isfile(path):
        if s3_config is None:
            raise FileNotFoundError(f'{path} does not exist')
        else:
            LOGGER.info('Downloading {}'.format(path))
            try:
                client: Client = storage.Client.from_service_account_json(s3_config)
                bucket = client.get_bucket(s3_bucket)
                blob = bucket.get_blob('{}.parquet'.format(name))
                if blob is None:
                    raise FileNotFoundError('File {} does not exist'.format(path))

                with open(path, 'wb') as f:
                    blob.download_to_file(f)

            except GoogleAPICallError as e:
                LOGGER.error('An error occurred trying to download from Google Storage.'
                             'The following error has been returned: {}'.format(e))

    return pd.read_parquet(path)


def store_data(df: pd.DataFrame, work_dir: str, name: str) -> str:
    path = os.path.join(work_dir, name) + '.parquet'
    LOGGER.debug('Saving dataframe locally in {}'.format(path))

    # Checks if working directory already exists --> If not create working directory
    if not os.path.isdir(work_dir):
        LOGGER.info('Creating work directory \'{}\''.format(work_dir))
        os.mkdir(work_dir)

    # Change dtype of column names to string --> parquet must have string column names
    df.columns = df.columns.astype(str)

    # Save dataframe as parquet-file to that path
    df.to_parquet(path)

    return path


def delete_data(train_path: str):
    try:
        if os.path.exists(train_path):
            os.remove(train_path)
        else:
            LOGGER.info('Dataset does not exist in local storage.')
    except OSError:
        LOGGER.warning('Unable to delete {}'.format(train_path))


def upload_data(input_file: str, s3_config: str, s3_bucket: str, name: str = None) -> Tuple[str, str]:
    # S3 Storage is disabled
    if s3_config is None or s3_bucket is None:
        return input_file, input_file

    client: Client = storage.Client.from_service_account_json(s3_config)

    """Checks if s3_bucket already exists. If not call create_bucket"""
    try:
        client.get_bucket(s3_bucket)
    except NotFound:
        client.create_bucket(s3_bucket)

    if name is None:
        name = input_file.split('/')[-1]
    try:
        LOGGER.debug('Uploading {} to S3'.format(input_file))

        bucket = client.get_bucket(s3_bucket)
        blob = bucket.blob('{}.parquet'.format(name))
        blob.upload_from_filename(input_file)

        remote_path = "{0}/{1}".format(s3_bucket, name)
        return input_file, remote_path
    except GoogleAPICallError as e:
        LOGGER.error('An error occurred trying to upload to Google Storage.'
                     'The following error has been returned: {}'.format(e))
