"""
preprocessing.py - Scikit-learn Preprocessing and Feature Engineering Pipelines.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


CATEGORICAL_FEATURES = [
    'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month', 'poutcome'
]

NUMERIC_FEATURES = [
    'age', 'balance', 'day_of_week', 'campaign', 'pdays', 'previous'
]

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ['previously_contacted']
ALL_NUMERIC_WITH_DURATION = NUMERIC_FEATURES + ['duration', 'previously_contacted']


class PreContactFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering step that:
    1. Ensures target leakage feature 'duration' is excluded if exclude_duration=True.
    2. Derives binary indicator 'previously_contacted' (pdays != -1).
    3. Imputes missing categorical values explicitly with 'Unknown' and converts to string.
    """
    def __init__(self, exclude_duration=True):
        self.exclude_duration = exclude_duration

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Guard against duration leakage in main pre-contact model
        if self.exclude_duration and 'duration' in X_out.columns:
            X_out = X_out.drop(columns=['duration'])

        # Indicator for previous campaign contact
        if 'pdays' in X_out.columns:
            X_out['previously_contacted'] = (X_out['pdays'] != -1).astype(int)
        else:
            X_out['previously_contacted'] = 0

        # Handle missing categorical values by filling with 'Unknown' and casting to string
        for col in CATEGORICAL_FEATURES:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna('Unknown').astype(str)

        return X_out


def build_preprocessor(include_scaling=True, include_duration=False):
    """
    Builds the ColumnTransformer for features.
    - Categorical features: Encoded with OneHotEncoder(handle_unknown='ignore').
    - Numeric features: Missing imputed with median, scaled if include_scaling=True.
    """
    num_features = ALL_NUMERIC_WITH_DURATION if include_duration else ALL_NUMERIC_FEATURES

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    if include_scaling:
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, num_features),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    return preprocessor


def build_full_pipeline(classifier, include_scaling=True, exclude_duration=True):
    """
    Assembles complete end-to-end Pipeline:
    Feature Engineering -> ColumnTransformer -> Classifier.
    """
    pipeline = Pipeline([
        ('engineer', PreContactFeatureEngineer(exclude_duration=exclude_duration)),
        ('preprocessor', build_preprocessor(include_scaling=include_scaling, include_duration=(not exclude_duration))),
        ('classifier', classifier)
    ])
    return pipeline
