"""
This file contains the code for feature generation.
"""
from typing import List
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class FeatureEngineer(ABC):
    """
    Abstract class for feature engineering.
    """

    def __init__(self, feature_name: str, column_name: List[str]):
        """
        Initialize the feature engineering class with the feature name and the column(s) name used to create the feature.
        Args:
            feature_name (str): The name of the feature.
            column_name (List[str]): The name of the column(s) used to create the feature.
        """
        self.feature_name = feature_name
        self.column_name = column_name

    @abstractmethod
    def generate_feature(self, *args, **kwargs):
        """
        Generate feature.
        """
        pass

    @abstractmethod
    def feature_dtype(self):
        """
        indicates the dtype of the feature
        """
        pass


class DateFeature(FeatureEngineer):
    """
    This class generates the features for the distance.
    """

    def __init__(self):
        """
        Initialize the class.
        """
        super().__init__('sentence', ['sentence_date', 'offence_date', 'parole_eligibility_date', 'maximum_sentence_date'])

    def generate_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the feature.
        """

        # get the latest offence or sentence date as reference for other features
        if max(df['sentence_date']) > max(df['offense_date']):
            current_date = max(df['sentence_date'])
        else:
            current_date = max(df['offense_date'])

        # get the difference between the current date and parole eligibility date to get number of years from parole eligibility
        years = []
        for i in df['parole_eligibility_date']:
            year = (current_date - pd.to_datetime(i, format='%Y-%m-%d')).days / 365.25
            years.append(year)
        df['years_from_parole_eligibility'] = years

        # get the difference between the current date and max sentence date to get number of years for sentence
        sentence_years = []
        for i in df['maximum_sentence_date']:
            if int(i.split('-')[0]) > current_date.year + 150:
                sentence_years.append(9999)
            else:
                sentence_years.append(int(i.split('-')[0]) - current_date.year)
        df['years_from_sentence'] = sentence_years
        return df[self.feature_dtype().keys()]

    def feature_dtype(self):
        """
        indicates the dtype of the feature
        """
        return {
            'years_from_parole_eligibility': np.int32,
            'years_from_sentence'          : np.int32
        }
