from typing import Annotated

from fastapi import APIRouter, Depends, Request

from exceptions import ModelNotLoadedError
from models import CustomerBatch, CustomerFeatures, PredictionResponse
from service import PredictionService


router = APIRouter()


def get_prediction_service(request: Request) -> PredictionService:
    prediction_service = getattr(request.app.state, "prediction_service", None)

    if prediction_service is None:
        raise ModelNotLoadedError()

    return prediction_service


@router.get("/health")
def health(request: Request) -> dict[str, str | None]:
    prediction_service = getattr(request.app.state, "prediction_service", None)

    if prediction_service is None or prediction_service.model is None:
        return {
            "status": "L'API n'est pas encore prête : aucun modèle avec l'alias production n'a été trouvé.",  # TODO ajout url pour dire où promote un model
            "model": "churnguard",
            "version": None,
        }

    return {
        "status": "ok",
        "model": "churnguard",
        "version": prediction_service.model_version,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerFeatures,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictionResponse:
    return prediction_service.predict_customer(customer)


@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(
    customers: CustomerBatch,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> list[PredictionResponse]:
    return prediction_service.predict_batch(customers.root)
