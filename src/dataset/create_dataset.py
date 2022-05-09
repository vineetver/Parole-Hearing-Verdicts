"""
Create_dataset.py
Contains functions for importing and exporting dataset.
"""
import os
from datetime import datetime
import pandas as pd
import s3fs

AWS_BUCKET_NAME = 'texas-dataset-bucket'
AWS_ACCESS_KEY = os.environ['ACCESS_KEY_S3']
AWS_SECRET_KEY = os.environ['ACCESS_KEY_SECRET_S3']


def read_data() -> pd.DataFrame:
    """
    Reads the dataset for a given year and month.
    Args:
    Returns:
        The raw parole dataset in the S3 bucket.
    """

    file_path = get_raw_data_path()

    df = pd.read_csv(file_path, parse_dates=['projected_release', 'maximum_sentence_date',
                                             'parole_eligibility_date', 'sentence_date', 'offense_date', 'next_parole_review_date'],

                     infer_datetime_format=True, low_memory=False,
                     storage_options={'key': AWS_ACCESS_KEY, 'secret': AWS_SECRET_KEY})
    return df


def get_raw_data_path() -> str:
    """
    Gets the file path for the given year and month.
    Args:
    Returns:
        The file path for raw parole dataset.
    """

    url = f's3://{AWS_BUCKET_NAME}'
    file_path = f'{url}/paroledata/texas_data.csv'

    return file_path


def write_data(df: pd.DataFrame, suffix: str, scratch: bool = True) -> str:
    """
    This function writes the dataframe to a csv file in the bucket
    Args:
        df: The dataset to write.
        suffix: The suffix to add to the file name.
        scratch: Write the file with the suffix
    Returns:
        str: The file path for the written file.
    """

    assert suffix.endswith('.csv'), 'Suffix must end with .csv'

    path = f's3://{AWS_BUCKET_NAME}'

    if scratch:
        path = os.path.join(path, 'scratch')
    path = os.path.join(path, suffix).replace('\\', '/')

    df.to_csv(path, index=False, storage_options={'key': AWS_ACCESS_KEY, 'secret': AWS_SECRET_KEY})
    return path


def get_output_path(stage: str, version: str = None) -> str:
    """
    This function returns the output file path for the given stage and version.
    Args:
        stage: The stage to write to.
        version: The version of the stage to write to.
    Returns:
        str: The file path for the given stage and version.
    """

    output_path = f'{stage}'

    if version is not None:
        version = get_time_stamp()
    else:
        version = get_latest_time_stamp(list_files(output_path))

    path = os.path.join(output_path, version).replace('\\', '/')

    return path


def write_output_data(df: pd.DataFrame, stage: str, version: str = None, overwrite: bool = False) -> str:
    """
    This function writes the dataframe to a csv file in the bucket
    Args:
        df: The dataset to write.
        stage: The stage to write to (e.g. 'clean', 'preprocess').
        version: The version of the stage to write to.
        overwrite: Check if the file already exists
    Returns:
        str: The file path for the written file.
    """

    assert len(stage) > 0, 'Please provide a stage name to write to. (e.g. "clean", "preprocess")'

    output_path = f's3://{AWS_BUCKET_NAME}/dataset/{get_output_path(stage, version)}'
    path = os.path.join(output_path + '.csv').replace('\\', '/')

    if overwrite is False:
        assert path not in list_files('dataset'), 'File already exists'

    df.to_csv(path, index=False, storage_options={'key': AWS_ACCESS_KEY, 'secret': AWS_SECRET_KEY})

    return path


def get_output_data(stage: str, version: str = None) -> pd.DataFrame:
    """
    This function returns the dataframe for the given stage and version.
    Args:
        stage: The stage to read from.
        version: The version of the stage to read from.
    Returns:
        pd.Dataframe: The dataframe for the given stage and version.
    """

    assert len(stage) > 0, 'Please provide a stage name to read from. (e.g. "clean", "preprocess")'

    if version is None:
        version = get_latest_time_stamp(list_files(stage))

    output_path = f's3://{AWS_BUCKET_NAME}/{stage}/{version}'
    path = os.path.join(output_path + '.csv').replace('\\', '/')

    print(f'Current version: {version}')

    df = pd.read_csv(path, parse_dates=['projected_release', 'maximum_sentence_date',
                                        'parole_eligibility_date', 'sentence_date', 'offense_date', 'next_parole_review_date'],
                     infer_datetime_format=True, low_memory=False,
                     storage_options={'key': AWS_ACCESS_KEY, 'secret': AWS_SECRET_KEY})

    return df


def get_time_stamp() -> str:
    """
    This function returns the time stamp for the current time.
    Returns:
        str: time stamp (current).
    """

    time_stamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')

    return time_stamp


def get_latest_time_stamp(files: list) -> datetime:
    """
    This function returns the time stamp for the latest file.
    Returns:
        datetime: The time stamp for the latest file.
    """

    format = '%Y%m%d-%H%M%S'
    timestamps = []

    for file in files:
        if file.endswith('.csv'):
            suffix = file.split('/')[-1].split('.')[0]
            datetime.strptime(suffix, format)
            timestamps.append(suffix)

    return max(timestamps)


def list_files(dictionary: str) -> list:
    """
    This function returns a list of files in the given path.
    Paths are relative to the root of the bucket. e.g. 'dataset/clean/'

    Args:
        dictionary: The dictionary to search
    Returns:
        list: The list of files in the AWS S3 Filesystem.
    """

    fs = s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)
    path = f's3://{AWS_BUCKET_NAME}/{dictionary}'

    return fs.ls(path)
