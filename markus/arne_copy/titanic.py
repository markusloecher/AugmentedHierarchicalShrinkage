from sklearn.preprocessing import OrdinalEncoder


def preprocess(df):
    # Extract relevant columns
    categorical_columns = ["Sex", "Embarked"]
    numerical_columns = ["Age", "Fare", "Pclass", "Parch", "PassengerId"]
    feat_columns = categorical_columns + numerical_columns

    # Ordinal encode categorical columns
    df[categorical_columns] = OrdinalEncoder().fit_transform(df[categorical_columns])
    df = df.loc[:, feat_columns + ["Survived"]]

    # Impute/drop missing values
    df.loc[:, "Age"].fillna(df["Age"].mean(), inplace=True)
    df = df.dropna()
    
    X_df = df.loc[:, feat_columns]
    y_df = df.loc[:, "Survived"]

    return X_df, y_df