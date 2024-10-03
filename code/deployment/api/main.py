from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load('../../../models/titanic_model.pkl')

# Define the expected input schema
class TitanicPassenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int

# Prediction endpoint
@app.post("/predict/")
def predict(passenger: TitanicPassenger):
    features = np.array([[passenger.Pclass, passenger.Sex, passenger.Age, 
                          passenger.SibSp, passenger.Parch, 
                          passenger.Fare, passenger.Embarked]])
    
    prediction = model.predict(features)
    return {"survived": int(prediction[0])}
