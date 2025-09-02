from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Load model and scaler
#model = joblib.load('random_forest_model.joblib')
#scaler = joblib.load('scaler.joblib')

model = tf.keras.models.load_model('deep-model1.keras')
scaler = joblib.load('deep-scaler.joblib')

app = FastAPI()
# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with your frontend's origin (e.g. "http://localhost:8100")
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)
# Define the input data model
class PatientData(BaseModel):
    Sex: int
    PhysicalHealthDays: float
    MentalHealthDays: float
    LastCheckupTime: int
    PhysicalActivities: int
    SleepHours: float
    RemovedTeeth: int
    HadAngina: int
    HadStroke: int
    DeafOrHardOfHearing: int
    BlindOrVisionDifficulty: int
    DifficultyConcentrating: int
    DifficultyWalking: int
    DifficultyDressingBathing: int
    DifficultyErrands: int
    SmokerStatus: int
    ECigaretteUsage: int
    ChestScan: int
    RaceEthnicityCategory: int
    AgeCategory: int
    HeightInMeters: float
    WeightInKilograms: float
    BMI: float
    AlcoholDrinkers: int
    HIVTesting: int
    FluVaxLast12: int
    PneumoVaxEver: int
    TetanusLast10Tdap: int
    HighRiskLastYear: int
    CovidPos: int

  

'''
# ... add all other needed features here
 '''
@app.post("/predict")
def predict_asthma(data: PatientData):
    # Convert incoming data to DataFrame (or array) matching model input
    input_df = pd.DataFrame([data.dict()])

    # You might need to do the same encoding/mapping/scaling you used in training:
    # For example, map string columns to ints (you can write a helper function)

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict probability
    #proba = model.predict_proba(input_scaled)[:, 1][0]
    proba = model.predict(input_scaled).flatten()[0]

    # Predict class
    pred = model.predict(input_scaled)[0]

    return {
        "asthma_probability": float(proba),
        "asthma_prediction": int(pred)
    }
