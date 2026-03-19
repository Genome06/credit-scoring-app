from pydantic import BaseModel, Field
from typing import Optional

class CreditInput(BaseModel):
    # Basic Features
    EXT_SOURCE_1: Optional[float] = Field(None, example=0.5)
    EXT_SOURCE_2: Optional[float] = Field(None, example=0.5)
    EXT_SOURCE_3: Optional[float] = Field(None, example=0.5)
    
    # Financial
    AMT_ANNUITY: float = Field(..., example=24700.5)
    AMT_CREDIT: float = Field(..., example=406597.5)
    AMT_INCOME_TOTAL: float = Field(..., example=202500.0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, example=351000.0)
    
    # Demographics & Employment
    DAYS_BIRTH: int = Field(..., example=-15000)
    DAYS_EMPLOYED: int = Field(..., example=-1500)
    REGION_RATING_CLIENT_W_CITY: int = Field(..., example=2)
    
    # Categorical (Text)
    NAME_EDUCATION_TYPE: str = Field(..., example="Higher education")
    NAME_INCOME_TYPE: str = Field(..., example="Working")

class PredictionResponse(BaseModel):
    probability: float
    risk_rating: str  # Low, Medium, High
    message: str