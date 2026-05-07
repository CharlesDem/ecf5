import os
import time

import mlflow
from dotenv import load_dotenv

from exceptions import ModelNotLoadedError
from models import CustomerFeatures, PredictionResponse


class PredictionService:
    
    def __init__(self) -> None:
        load_dotenv()

        self.model = None
        self.model_version = None

        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        self.model_name = os.getenv("MODEL_NAME", "churnguard-model")
        self.model_alias = os.getenv("MODEL_ALIAS", "production")
        self.retry_delay = int(os.getenv("MODEL_RETRY_DELAY", "5"))

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.MlflowClient()

    def load_model_once(self) -> bool:
        try:
            model_uri = f"models:/{self.model_name}@{self.model_alias}"

            print(f"Chargement modèle : {model_uri}", flush=True)

            self.model = mlflow.pyfunc.load_model(model_uri)

            model_version = self.client.get_model_version_by_alias(
                self.model_name,
                self.model_alias,
            )

            self.model_version = str(model_version.version)

            print(
                f"Modèle chargé : {self.model_name}@{self.model_alias} "
                f"version {self.model_version}",
                flush=True,
            )

            return True

        except Exception as e:
            print(
                f"Modèle indisponible, retry dans {self.retry_delay}s : {e}",
                flush=True,
            )
            self.model = None
            self.model_version = None
            return False

    def load_model(self) -> None:
        while not self.load_model_once():
            time.sleep(self.retry_delay)


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
