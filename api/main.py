from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from controller import router
from error_handler import model_not_loaded_exception_handler, validation_exception_handler
from exceptions import ModelNotLoadedError
from service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    prediction_service = PredictionService()
    prediction_service.load_model()
    app.state.prediction_service = prediction_service
    yield


app = FastAPI(title="ChurnGuard API", lifespan=lifespan)
app.include_router(router)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_exception_handler)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
