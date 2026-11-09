# here we define all functions that are used to process the data and prepare it for training and evaluation

# we fill missing values with the median for numerical columns and the mode for categorical columns
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    return df


