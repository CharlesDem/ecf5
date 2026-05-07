import os

from dotenv import load_dotenv
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
from mlflow.models.signature import infer_signature
from churnguard.evaluate import compute_metrics

load_dotenv()



PROD = "production"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "churnguard"
MODEL_NAME = "churnguard-model"

METRIC = "roc_auc"
ARTIFACT_PATH = "model"

def train_model(X, y, model_name: str, params: dict) -> Pipeline:

    """ Entraine le modele selon le type demandé et les params donnés, et log le tout dans mlflow"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])

    model = Pipeline([
        ('prep', preprocess),
        ('model', __set_model(model_name, params)),
    ])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))

    with mlflow.start_run(run_name=model_name):
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("split_random_state", 42)
        mlflow.log_params(params)

        model.fit(X_train, y_train)

        metrics = compute_metrics(model, X_test, y_test)

        mlflow.log_metrics(metrics)

        input_example = X_train.head(1)

        signature = infer_signature(
            input_example,
            model.predict(input_example),
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    return model

def promotion_to_laprod():

    """ 
    Promotion du meilleur model vers la prod :
            - on récupère l'experiment
            - on itère le modèle pour trouver le meilleur roc_aux
            - on enregistre le modèle sur le registry
            - on le promeut en @production avec un alias
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"experiment pas trouved : {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("pas de  runs")

    best_run = runs[0]
    run_id = best_run.info.run_id
    score = best_run.data.metrics[METRIC]

    print(f"meilleur run: {run_id}")
    print(f"{METRIC}: {score}")

    model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=PROD,
        version=registered_model.version,
    )

    print(f"{MODEL_NAME}@{PROD} -> version {registered_model.version}")


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

