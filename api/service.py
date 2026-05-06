import os

import mlflow
from dotenv import load_dotenv

from exceptions import ModelNotLoadedError
from models import CustomerFeatures, PredictionResponse


class PredictionService:
    def __init__(self) -> None:
        self.model = None
        self.model_version = None

    def load_model(self) -> None:
        load_dotenv()
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

        try:
            self.model = mlflow.pyfunc.load_model("models:/churnguard@production")
            client = mlflow.MlflowClient()
            model_version = client.get_model_version_by_alias("churnguard", "production")
            self.model_version = str(model_version.version)
        except Exception as e:
            print(f"Erreur chargement modèle MLflow: {e}")
            self.model = None
            self.model_version = None

    def predict_customer(self, customer: CustomerFeatures) -> PredictionResponse:
        return self.predict_batch([customer])[0]

    def predict_batch(self, customers: list[CustomerFeatures]) -> list[PredictionResponse]:
        if self.model is None:
            raise ModelNotLoadedError()

        import pandas as pd

        df = pd.DataFrame([customer.model_dump() for customer in customers])

        model_impl = getattr(self.model, "_model_impl", None)
        sklearn_model = getattr(model_impl, "sklearn_model", self.model)

        if hasattr(sklearn_model, "predict_proba"):
            probabilities = [float(row[1]) for row in sklearn_model.predict_proba(df)]
        else:
            probabilities = [float(prediction) for prediction in self.model.predict(df)]

        return [
            PredictionResponse(churn=probability >= 0.5, probability=probability)
            for probability in probabilities
        ]
