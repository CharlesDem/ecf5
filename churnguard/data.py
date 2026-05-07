import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """load csv data from given string path"""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """clean data and return features dataframe as X, target serie as y"""

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" in df.columns:
        y = (df["Churn"] == "Yes").astype(int)
        X = df.drop(columns=["Churn"])
    else:
        X = df
        y = None

    return (X, y)
