import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from urllib.parse import urlparse
import mlflow.sklearn 
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import joblib
import pathlib

# Read data
df = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Churn_Modelling.csv")
print(df.head())

print(df.info())

# Feature matrix
X = df.iloc[:, 3:13]
print(X.shape)
print(X[:3])

# Output variable
y = df.iloc[:, 13]
print(y.shape)
print(y[:6])

# split test train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

n_estimators=50
pipeline = Pipeline([
        ('ct-ohe', ColumnTransformer([('ct', OneHotEncoder(handle_unknown='ignore', categories='auto'), [1,2])], remainder='passthrough')),
        ('scaler', StandardScaler(with_mean=False)),
        ('estimator', RandomForestClassifier(n_estimators=n_estimators))
])
    
pipeline.fit(X_train, y_train)


joblib.dump(pipeline, "randomforest.pkl")