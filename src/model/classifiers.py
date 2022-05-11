from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class Model(ABC):
    """Abstract class for models."""

    def __init__(self, features: List[str] = None, label: str = None, params: dict = None):
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
