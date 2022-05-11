from abc import ABC, abstractmethod
from typing import List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay


class Model(ABC):
    """Abstract class for models."""

    def __init__(self, features: List[str] = None, label: str = None, params: dict = None):
        """
        Initialize the model with the given features, label and the given parameters.

        Args:
            features: The features to use.
            label:  The label to use.
            params: The parameters of the model to use.
        """
        if features is None:
            features = []
        if label is None:
            label = []
        if params is None:
            params = {}
        self.label = label
        self.features = features
        self.params = params
        self.model = None

    def preprocess(self, df: pd.DataFrame):
        """ Any model specific preprocessing that needs to be done before training the model."""
        pass

    def split(self, X, Y, test_size: float):
        """Split the data into training and tests sets."""
        pass

    def normalize(self, X):
        """Normalize the data."""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict the labels for the given data."""
        pass

    @abstractmethod
    def evaluate(self, X, Y):
        """Evaluate the model."""
        pass

    @abstractmethod
    def cross_validate(self, X, Y, n_splits: int = 10):
        """Cross validate the model."""
        pass

    def feature_importance(self, X, Y):
        """Get the feature importance."""
        pass


class RandomForestModel(Model, ABC):
    """Random Forest Model."""

    def __init__(self, features: List[str] = None, label: str = None, params: dict = {}):
        model_params = {'n_jobs': -1, 'random_state': 0}
        model_params.update(params)
        super().__init__(features=features, label=label, params=model_params)

    def preprocess(self, df: pd.DataFrame) -> tuple[Any, Any]:
        """ Model specific preprocessing.

        Args:
            df (pd.DataFrame): The dataframe to preprocess.
        Returns:
            X: features
            Y: labels
        """
        df.dropna(inplace=True)
        X = df[self.features].values
        Y = df[self.label].values
        return X, Y

    def split(self, X, Y, test_size: float = 0.2):
        """Split the data into training and tests sets.

        Args:
            X: features
            Y: labels
            test_size (float): The size of the test set.

        Returns:
            X_train (nd.array): The training data.
            Y_train (nd.array): The training labels.
            X_test (nd.array): The test data.
            Y_test (nd.array): The test labels.
        """
        assert 0 < test_size < 1, "test_size must be between 0 and 1"
        sample = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        for train, test in sample.split(X, Y):
            x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        return x_train, x_test, y_train, y_test

    def fit(self, X, Y) -> RandomForestClassifier:
        """Train the model.

        Args:
            X (pd.DataFrame): The data to train on.
            Y (pd.Series): The labels to train on.
        Returns:
            self.model (sklearn.ensemble.RandomForestClassifier): The trained model.
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of rows."
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, Y)
        return self.model

    def predict(self, X) -> np.ndarray:
        """Predict the labels for the given data.

        Args:
            X (pd.DataFrame): The data to predict on.
        Returns:
            Y_pred (np.ndarray): The predicted labels.
        """
        assert self.model is not None, "Model has not been trained yet."
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, Y) -> float:
        """Evaluate the model.

        Args:
            X:  training features
            Y:  training labels
        Returns:
            MCC (dict): The metrics MCC.
        """
        assert self.model is not None, "Model has not been trained yet."
        y_pred = self.predict(X)
        mcc = matthews_corrcoef(Y, y_pred)
        return mcc

    def cross_validate(self, X, Y, n_splits: int = 10) -> List[float]:
        """Cross validate the model.

        Args:
            X: testing features
            Y: testing labels
            n_splits: number of splits
        Returns:
            MCC (list): The metrics MCC.
        """
        assert self.model is not None, "Model has not been trained yet."
        mcc_scores = []
        for fold, (train, test) in enumerate(StratifiedKFold(n_splits=n_splits, random_state=0).split(X, Y)):
            print('=============================')
            print(f'Fold: {fold}')
            x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
            self.fit(x_train, y_train)
            score = self.evaluate(x_test, y_test)
            print(f'F1 score: {score}')
            mcc_scores.append(score)
        return mcc_scores

    def plot_confusion_matrix(self, X, Y, normalize: str = None) -> None:
        """
        Plot the confusion matrix.
        Args:
            X: The data to predict on.
            Y: The labels to compare against.
            normalize: Whether to normalize the matrix e.g. 'true', 'pred', 'all' etc
        """
        y_pred = self.predict(X)
        cm = confusion_matrix(Y, y_pred, normalize=normalize, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(include_values=True, cmap='Blues')
        plt.show()

    def feature_importance(self) -> pd.DataFrame:
        """
        Plot the feature importance.
        Args:
            X: The data to predict on.
            Y: The labels to compare against.
        """
        assert self.model is not None, "Model has not been trained yet."
        feature_df = pd.DataFrame({'feature'   : self.features,
                                   'importance': self.model.feature_importances_})
        return feature_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

    def plot_feature_importance(self, feature_importance: pd.DataFrame) -> None:
        """
        Plot the feature importance.
        """
        assert self.model is not None, "Model has not been trained yet."

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance, palette='rocket')
        plt.title('Feature Importance')
        plt.show()