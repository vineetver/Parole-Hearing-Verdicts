"""
This file scrapes High Value Dataset from socrata API, and saves it to a AWS S3 bucket.
The dataset contains five months of data with duplicates.
"""

from sodapy import Socrata
from src.dataset.create_dataset import write_data, get_raw_data_path
import pandas as pd
import os

TOKEN = os.environ['SOCRATA_TOKEN']
AWS_BUCKET_NAME = 'texas-data-bucket'
AWS_ACCESS_KEY = os.environ['ACCESS_KEY_S3']
AWS_SECRET_KEY = os.environ['ACCESS_KEY_SECRET_S3']


def main() -> None:
    # Datasets endpoint to scrape
    datasets = ['tyje-q8w4', 'skhe-r982', 'q4fk-2ptv', 'cg6t-bapy', 'y6p5-x3b9']

    # Initialize the client
    client = Socrata('dataset.texas.gov', TOKEN)

    # Create a dataframe to store the dataset
    df = pd.DataFrame()

    # Loop through the datasets
    for dataset in datasets:
        results = client.get(str(dataset), limit=100000)

        # Create a dataframe from the results and append it to the main dataframe
        df = pd.concat([df, pd.DataFrame.from_records(results)])

    try:
        # Write the dataframe to AWS S3 Bucket
        write_data(df, get_raw_data_path())
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
