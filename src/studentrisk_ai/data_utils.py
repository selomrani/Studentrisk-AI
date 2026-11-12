import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


# here we define all functions that are used to process the data

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns
        q1 = X[self.numeric_columns_].quantile(0.25)
        q3 = X[self.numeric_columns_].quantile(0.75)
        iqr = q3 - q1
        self.lower_limits_ = q1 - self.factor * iqr
        self.upper_limits_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_columns_] = X[self.numeric_columns_].clip(
            self.lower_limits_,
            self.upper_limits_,
            axis=1,
        )
        return X

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def validate_data(df, target_column='Mental_Health_Risk'):
    if target_column not in df.columns:
        raise ValueError(f'Missing target column: {target_column}')
    if df.empty:
        raise ValueError('Input data is empty')
    return df


# split 20% of the data for testing, while keeping the same target proportions
def split_data(df, test_size=0.2, target_column='Mental_Health_Risk'):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[target_column],
    )
    return train_df.copy(), test_df.copy()


# fill values using statistics learned from the training data
def fill_missing_values(train_df, test_df=None):
    train_df = train_df.copy()
    test_df = train_df.copy() if test_df is None else test_df.copy()

    for column in train_df.columns:
        if train_df[column].dtype == 'object':
            values = train_df[column].mode()
            value = values.iloc[0] if not values.empty else 'Unknown'
        else:
            value = train_df[column].median()
        train_df[column] = train_df[column].fillna(value)
        test_df[column] = test_df[column].fillna(value)
    return train_df, test_df


# social-media time cannot be greater than total screen time
def fix_social_media(train_df, test_df=None):
    train_df = train_df.copy()
    test_df = train_df.copy() if test_df is None else test_df.copy()

    for df in (train_df, test_df):
        invalid = df['Social_Media_Hours'] > df['Screen_Time_Hours']
        df.loc[invalid, 'Social_Media_Hours'] = np.nan
        df['Social_Media_Hours'] = df['Social_Media_Hours'].fillna(
            train_df['Social_Media_Hours'].median()
        )
        df['Social_Media_Hours'] = df['Social_Media_Hours'].clip(
            upper=df['Screen_Time_Hours']
        )
    return train_df, test_df


# cap numeric outliers using limits calculated from the training data
def cap_outliers(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    numeric_columns = train_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        q1 = train_df[column].quantile(0.25)
        q3 = train_df[column].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        train_df[column] = train_df[column].clip(lower_limit, upper_limit)
        test_df[column] = test_df[column].clip(lower_limit, upper_limit)
    return train_df, test_df


# these columns are identifiers and are not useful for prediction
def drop_unutilized_columns(train_df, test_df=None):
    columns_to_drop = ['Student_ID']
    train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
    if test_df is None:
        return train_df
    test_df = test_df.drop(columns=columns_to_drop, errors='ignore')
    return train_df, test_df


# encode categories using the same columns in train and test
def encode_categorical_variables(
    train_df, test_df, target_column='Mental_Health_Risk'
):
    categorical_columns = train_df.select_dtypes(include=['object']).columns.drop(
        target_column, errors='ignore'
    )
    train_df = pd.get_dummies(train_df, columns=categorical_columns, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=categorical_columns, drop_first=True)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=False)
    return train_df, test_df


# normalize numeric variables using only training-set minimums and maximums
def normalize_numerical_variables(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    numerical_columns = train_df.select_dtypes(include=[np.number]).columns
    for column in numerical_columns:
        min_value = train_df[column].min()
        max_value = train_df[column].max()
        if max_value != min_value:
            train_df[column] = (train_df[column] - min_value) / (max_value - min_value)
            test_df[column] = (test_df[column] - min_value) / (max_value - min_value)
    return train_df, test_df


def split_features_and_target(df, target_column='Mental_Health_Risk'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# save model as pkl
def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


# run all data preparation steps in one go
def process_data(file_path):
    df = validate_data(load_data(file_path))
    train_df, test_df = split_data(df)
    train_df, test_df = fill_missing_values(train_df, test_df)
    train_df, test_df = fix_social_media(train_df, test_df)
    train_df, test_df = drop_unutilized_columns(train_df, test_df)
    outlier_capper = OutlierCapper()
    train_df = outlier_capper.fit_transform(train_df)
    test_df = outlier_capper.transform(test_df)
    train_df, test_df = encode_categorical_variables(train_df, test_df)
    train_df, test_df = normalize_numerical_variables(train_df, test_df)
    X_train, y_train = split_features_and_target(train_df)
    X_test, y_test = split_features_and_target(test_df)
    return X_train, y_train, X_test, y_test
