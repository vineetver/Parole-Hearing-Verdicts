"""
This file is used to clean the data.
e.g. remove the duplicates, nan, and feature encoding.
"""

from src.dataset.create_dataset import read_data, write_output_data
from src.feature.preprocessing import *


def main() -> None:
    """
    This function is used to clean the data.
    e.g. remove the duplicates, nan, and feature encoding.
    """

    # Read the data from AWS S3 Bucket
    df = read_data()

    # Drop Unnamed: 0 column
    df = drop_unnamed_column(df)

    # drop duplicates
    df = drop_duplicate_rows(df)

    # drop nan values
    df = drop_nan_values(df)

    # drop useless columns
    df = drop_next_parole_review_date(df)
    df = drop_none_last_parole_decision(df)
    df = drop_name_column(df)
    df = drop_id_columns(df)
    df = binary_label_encoding(df)
    df = remove_capital_life(df)

    write_output_data(df, 'clean', version='yes')


if __name__ == "__main__":
    main()
