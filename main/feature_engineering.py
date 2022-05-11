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
    df = get_output_data('clean')
