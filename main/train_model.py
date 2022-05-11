from src.model import classifiers
from src.dataset.create_dataset import get_output_data


def main():
    """
    Main function to train the model
    """
    # Get final dataframe from AWS S3 bucket (path: data/feature/ver=latest
    df = get_output_data('data/feature')

    # Define label and features
    label = 'parole_review_status'
    features = list(df.columns.drop(
        [label, 'years_from_sentence.1', 'years_from_parole_eligibility.1', 'sentence_date', 'offense_date', 'parole_eligibility_date',
         'maximum_sentence_date', 'projected_release']))

    # Initialize the model
    model = classifiers.RandomForestModel(features, label)

    X, Y = model.preprocess(df)

    x_train, x_test, y_train, y_test = model.split(X, Y, test_size=0.3)

    model.fit(x_train, y_train)

if __name__ == "__main__":
    main()
