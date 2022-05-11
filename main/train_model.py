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
        [label, 'years_from_sentence.1', 'years_from_parole_eligibility.1', 'sentence_date', 'offense_date',
         'parole_eligibility_date', 'maximum_sentence_date', 'projected_release', 'last_parole_decision'
         ]))

    # Initialize the model
    model = classifiers.RandomForestModel(features, label)

    # Get X (features) and Y (label)
    X, Y = model.preprocess(df)

    #  Stratified train/test split to handle imbalanced data
    x_train, x_test, y_train, y_test = model.split(X, Y, test_size=0.3)

    # Train the model
    model.fit(x_train, y_train)

    # Evaluate the model (MCC)
    model.evaluate(x_test, y_test)

    # 10-fold cross validation
    model.cross_validate(X, Y, 10)

    # Confusion matrix
    model.plot_confusion_matrix(x_test, y_test)

    # plot feature importance
    feature_importance = model.feature_importance()
    model.plot_feature_importance(feature_importance)


if __name__ == "__main__":
    main()
