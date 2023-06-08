from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class CreateUpdateChurn(SQLModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 800,
                "Geography": "France",
                "Gender": "Female",
                "Age": 34,
                "Tenure": 8,
                "Balance": 135791.51,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50250.0
            }
        }
