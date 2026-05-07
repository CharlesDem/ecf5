from pathlib import Path

import pandas as pd

from churnguard.data import load_data, preprocess


TEST_DIR = Path(__file__).parent
TEST_DATA_PATH = TEST_DIR / "data" / "telco_churn_test.csv"


def test_test_data_file_exists():
    print(TEST_DATA_PATH)
    assert TEST_DATA_PATH.exists()


def test_load_data_returns_dataframe():
    df_to_test = load_data(TEST_DATA_PATH)
    assert df_to_test.shape == (101, 21)


def test_load_data_has_expected_columns():
    df_to_test = load_data(TEST_DATA_PATH)
    assert len(df_to_test.columns) == 21


def test_preprocess_returns_features_and_target():
    df_to_test = load_data(TEST_DATA_PATH)

    X, y = preprocess(df_to_test)

    assert len(X.columns) == 19
    assert __is_binary_int_series(y)


def test_preprocess_handles_missing_total_charges():
    df_to_test = load_data(TEST_DATA_PATH)

    assert len(df_to_test) == 101

    X, _ = preprocess(df_to_test)

    assert len(X) == 100  # la serie avec TotalCharges vide a été drop
    assert X["TotalCharges"].isna().sum() == 0


def __is_binary_int_series(s: pd.Series) -> bool:
    return s.dtype == "int64" and s.isin([0, 1]).all()
