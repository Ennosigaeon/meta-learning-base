import logging
import os

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError

LOGGER = logging.getLogger('mlb')


def _download_from_s3(path: str, local_path: str, aws_access_key: str = None, aws_secret_key: str = None,
                      **kwargs) -> str:
    client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
    )

    bucket = path.split('/')[2]
    file_to_download = path.replace('s3://{}/'.format(bucket), '')

    try:
        LOGGER.info('Downloading {}'.format(path))
        client.download_file(bucket, file_to_download, local_path)

        return local_path

    except ClientError as e:
        LOGGER.error('An error occurred trying to download from AWS3.'
                     'The following error has been returned: {}'.format(e))


def _download_from_url(url: str, local_path: str, **kwargs) -> str:
    data = requests.get(url).text
    with open(local_path, 'wb') as outfile:
        outfile.write(data.encode())

    LOGGER.info('File saved at {}'.format(local_path))

    return local_path


DOWNLOADERS = {
    's3': _download_from_s3,
    'http': _download_from_url,
    'https': _download_from_url,
}


def _download(path: str, local_path: str, **kwargs) -> str:
    protocol = path.split(':', 1)[0]
    downloader = DOWNLOADERS.get(protocol)

    if not downloader:
        raise ValueError('Unknown protocol: {}'.format(protocol))

    return downloader(path, local_path, **kwargs)


def _get_local_path(name: str, path: str, aws_access_key: str = None, aws_secret_key: str = None) -> str:
    if os.path.isfile(path):
        return path

    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')

    if not name.endswith('csv'):
        name = name + '.csv'

    local_path = os.path.join(data_path, name)

    if os.path.isfile(local_path):
        return local_path

    if not os.path.isfile(local_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        _download(path, local_path, aws_access_key=aws_access_key, aws_secret_key=aws_secret_key)
        return local_path


def load_data(name: str, path: str, aws_access_key: str = None, aws_secret_key: str = None) -> pd.DataFrame:
    """Load data from the given path.

    If the path is an URL or an S3 path, download it and make a local copy
    of it to avoid having to dowload it later again.

    Args:
        name (str):
            Name of the dataset. Used to cache the data locally.
        path (str):
            Local path or S3 path or URL.
        aws_access_key (str):
            AWS access key. Optional.
        aws_secret_key (str):
            AWS secret key. Optional.

    Returns:
        pandas.DataFrame:
            The loaded data.
    """
    local_path = _get_local_path(name, path, aws_access_key=aws_access_key, aws_secret_key=aws_secret_key)

    return pd.read_csv(local_path)
