from pydantic import BaseModel
from typing import Optional, List


class RawChurnFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class ShapReason(BaseModel):
    feature: str
    impact: float


class PredictionRequest(BaseModel):
    customer_id: str
    features: RawChurnFeatures


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_risk_score: int
    will_churn: bool
    model_version: str
    top_reasons: Optional[List[ShapReason]] = None