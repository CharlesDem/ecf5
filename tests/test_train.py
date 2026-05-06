import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churnguard.data import load_data, preprocess
from churnguard.evaluate import compute_metrics
from churnguard.train import train_model


def test_train_model_returns_fitted_pipeline():
    df = load_data("data/telco_churn.csv")

    X, y = preprocess(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = train_model(X_train, y_train, 'random_forest', {'n_estimators': 200, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1})

    assert isinstance(model, Pipeline)

    predictions = model.predict(X)
    assert set(predictions).issubset({0, 1})

    testcase_2_rows_df = __get_two_rows_df() # sans doute pas pertinent de tester l'inférence du model ?
    X_test_case, y_test_case = preprocess(testcase_2_rows_df)

    test_case_predict = model.predict(X_test_case)
    assert test_case_predict.tolist() == [0, 1]

def test_compute_metrics_returns_expected_keys():

    df = load_data("data/telco_churn_test.csv")

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = train_model(X_train, y_train, 'random_forest', {'n_estimators': 200, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1})

    metrics = compute_metrics(model, X_test, y_test)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert set(metrics.keys()) == expected_keys

    for value in metrics.values():
        assert 0 <= value <= 1



def __get_two_rows_df():

    return pd.DataFrame([
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 72, 
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",  
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 6840.0,  
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,  #nouveau client
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",  # plus risqué
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0,
        },
    ])
