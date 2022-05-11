"""
This file contains the main function for feature engineering
"""

import pandas as pd
from src.dataset.create_dataset import get_output_data, write_output_data
from src.feature import feature_selection


def main():
    """
    Main function for feature engineering
    """

    # Read the cleaned data from AWS S3 Bucket (path: /data/clean/ver=latest
    df = get_output_data('data/clean')

    # Feature engineering
    date_features = feature_selection.DateFeature().generate_feature(df)

    # Concatenate the date features to the main dataframe
    df = pd.concat([df, date_features], axis=1)

    # Write the output data to AWS S3 Bucket (path: /data/feature/ver=latest
    write_output_data(df, 'data/feature', version='yes')


if __name__ == '__main__':
    main()
