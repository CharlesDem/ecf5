from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from exceptions import ModelNotLoadedError


async def validation_exception_handler(_request: Request, rve: RequestValidationError):
    batch_errors = {"batch_empty", "batch_too_large"}
    status_code = (
        400 if any(error["type"] in batch_errors for error in rve.errors()) else 422
    )
    return JSONResponse(status_code=status_code, content={"detail": rve.errors()})


async def model_not_loaded_exception_handler(
    _request: Request,
    _exception: ModelNotLoadedError,
):
    return JSONResponse(status_code=503, content={"detail": "Modèle non chargé"})
