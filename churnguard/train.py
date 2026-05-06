from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator

def train_model(X_train, y_train, model_name: str, params: dict) -> Pipeline:

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    cat_cols = [ # pas repris la boucle car je ne veux pas de X dans les params pour ne pas avoir à faire le split deux fois
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])

    model = Pipeline([
        ('prep', preprocess),
        ('model', __set_model(model_name, params)),
    ])

    model.fit(X_train, y_train)
    return model

def __set_model(model_name: str, params: dict) -> BaseEstimator:

    match model_name:
        case "logistic_regression":
            return LogisticRegression(**params)

        case "random_forest":
            return RandomForestClassifier(**params)

        case "gradient_boosting":
            return GradientBoostingClassifier(**params)

        case _:
            raise ValueError(f"model non géré, choisissez [logistic_regression, random_forest, gradient_boosting]: {model_name}")

