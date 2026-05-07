from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic_core import PydanticCustomError


MAX_BATCH_SIZE = 100


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=10000)
    TotalCharges: float = Field(ge=0, le=1000000)


class CustomerBatch(RootModel[list[CustomerFeatures]]):
    @model_validator(mode="after")
    def validate_size(self):
        size = len(self.root)

        if size == 0:
            raise PydanticCustomError("batch_empty", "batch must not be empty")

        if size > MAX_BATCH_SIZE:
            raise PydanticCustomError(
                "batch_too_large",
                f"batch must have at most {MAX_BATCH_SIZE} customers",
            )

        return self


class PredictionResponse(BaseModel):
    churn: bool
    probability: float = Field(ge=0, le=1)
