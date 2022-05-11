import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def drop_next_parole_review_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops the next parole review date column.

    Args:
        df (pd.DataFrame): Dataframe to be processed.

    Returns:
        pd.DataFrame: Processed dataframe.
    """

    return df.drop(columns='next_parole_review_date')


def drop_none_last_parole_decision(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops rows with none last parole decision.

    Args:
        df (pd.DataFrame): Dataframe to be processed.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    return df[df['last_parole_decision'] != 'None']


def drop_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops rows with null values.

    Args:
        df (pd.DataFrame): Dataframe to be processed.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    return df.dropna()


def drop_unnamed_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops the unnamed column if it exists
    """

    if 'Unnamed: 0' in df.columns:
        return df.drop(columns='Unnamed: 0')


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops duplicate rows.
    """
    return df.drop_duplicates()


def drop_name_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops the name column.
    e.g. name, race, county etc

    Args:
        df (pd.DataFrame): Dataframe to be processed.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    return df.drop(columns=['name', 'county', 'tdcj_offense', 'race', 'current_facility'])


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops the id columns.

    Args:
        df (pd.DataFrame): Dataframe to be processed.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    return df.drop(columns=['sid_number', 'case_number'])


def binary_label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function converts the binary categorical columns to numerical.
    e.g. 'Yes' -> 1, 'No' -> 0.

    Args:
        df (pd.DataFrame): Dataframe to be processed.

    Returns:
        pd.DataFrame: Processed dataframe.
    """

    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['last_parole_decision'] = np.where(df['last_parole_decision'].str.startswith('Approve'), 1, 0)
    df['parole_review_status'] = np.where(df['parole_review_status'] == 'IN PAROLE REVIEW PROCESS', 1, 0)

    return df


def remove_capital_life(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function removes the capital life from Sentence (Years) column.
    Assumes that life and capital life is 999 years
    e.g. 'Capital Life' -> '999 years'
    """
    column = 'sentence_years'
    df[column] = np.where((df[column] == 'Capital Life') | (df[column] == 'Life'), 999, df[column])
    df[column] = df[column].astype(float)
    return df


