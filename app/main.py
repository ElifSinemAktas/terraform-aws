from fastapi import FastAPI, Depends, Request
import os
from sqlmodels import CreateUpdateChurn
from sqlalchemy.orm import Session
from mlflow.sklearn import load_model
import joblib
import numpy as np
import pandas as pd

model = joblib.load("randomforest.pkl")

app = FastAPI()

# prediction function
def make_churn_prediction(model, request):
    # parse input from request
    CreditScore=request["CreditScore"],
    Geography=request["Geography"],
    Gender=request["Gender"],
    Age=request['Age'],
    Tenure=request['Tenure'],
    Balance=request['Balance'],
    NumOfProducts=request['NumOfProducts'],
    HasCrCard=request['HasCrCard'],
    IsActiveMember=request['IsActiveMember'],
    EstimatedSalary=request['EstimatedSalary'],

    print(CreditScore)

    # Make an input vector
    info = [[CreditScore[0], Geography[0], Gender[0], Age[0], Tenure[0], 
             Balance[0], NumOfProducts[0], HasCrCard[0], IsActiveMember[0], EstimatedSalary[0]]]
    
    print(info)

    df = pd.DataFrame(data=info , columns = model.feature_names_in_)
    # Predict
    prediction = model.predict(df)
    # prediction = model.predict(np.array(info, dtype=float))

    return prediction[0].item()



# Churn Prediction endpoint
@app.post("/prediction/churn")
def predict_churn(request: CreateUpdateChurn):
    prediction = make_churn_prediction(model=model, request=request.dict())
    return {"prediction": prediction}

# Welcome page
@app.get("/")
async def root():
    return {"data":"Welcome to MLOps API"}