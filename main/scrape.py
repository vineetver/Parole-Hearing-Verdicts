from sodapy import Socrata
from src.dataset.create_dataset import write_data, get_raw_data_path
import pandas as pd
import os

TOKEN = os.environ['SOCRATA_TOKEN']
AWS_BUCKET_NAME = os.environ['AWS_BUCKET_NAME']
AWS_BUCKET_KEY = os.environ['AWS_BUCKET_KEY']
AWS_BUCKET_SECRET = os.environ['AWS_BUCKET_SECRET']


def main() -> None:
    """
    This function scrapes the dataset from the Socrata API and stores it in a csv file.

    Args:
    Returns:
        None
    """

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
