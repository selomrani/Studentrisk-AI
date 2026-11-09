import numpy as np
import pandas as pd
# here we define all functions that are used to process the data and prepare it for training and evaluation
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# we fill missing values with the median for numerical columns and the mode for categorical columns
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    return df

# fix the social media hours column by replacing any values that are greater than the screen time hours with NaN, then fill those NaN values with the median of the column, and finally clip the values to be no greater than the screen time hours

def fix_social_media(df):
    invalid = df['Social_Media_Hours'] > df['Screen_Time_Hours']
    df.loc[invalid, 'Social_Media_Hours'] = np.nan
    df['Social_Media_Hours'] = df['Social_Media_Hours'].fillna(df['Social_Media_Hours'].median())
    df['Social_Media_Hours'] = df['Social_Media_Hours'].clip(upper=df['Screen_Time_Hours'])
    return df

# we drop unutilized columns that are not useful for training or evaluation
def drop_unutilized_columns(df):
    columns_to_drop = ['Student_ID']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

# we encode categorical variables using one-hot encoding
def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df

# we normalize numerical variables using min-max scaling
def normalize_numerical_variables(df):
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    for column in numerical_columns:
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)
    return df

# split 20% of the data for testing and 80% for training
def split_data(df, test_size=0.2):
    test_df = df.sample(frac=test_size, random_state=42)
    train_df = df.drop(test_df.index)
    return train_df, test_df

def split_features_and_target(df, target_column='Mental_Health_Risk'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# save model as pkl
def save_model(model, file_path):
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

