from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import re



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetClassifier


# Load model and scaler
#model = joblib.load('random_forest_model.joblib')
#scaler = joblib.load('scaler.joblib')

model = tf.keras.models.load_model('ncd_models/hypertension/deep-model1.keras')
scaler = joblib.load('ncd_models/hypertension/deep-scaler.joblib')

model2 = tf.keras.models.load_model('ncd_models/arthritis/deep-model1.keras')
scaler2 = joblib.load('ncd_models/arthritis/deep-scaler.joblib')

model3 = tf.keras.models.load_model('ncd_models/lung_cancer/deep-model1.keras')
scaler3 = joblib.load('ncd_models/lung_cancer/deep-scaler.joblib')

model4 = TabNetClassifier()
model4.load_model('ncd_models/asthma/tabnet_asthma_model.zip')
scaler4 = joblib.load('ncd_models/asthma/tabnet_scaler.joblib')

model5 = tf.keras.models.load_model('ncd_models/diabetes/deep-model1.keras')
scaler5 = joblib.load('ncd_models/diabetes/deep-scaler.joblib')



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Classifier --------------------
classifier = pipeline(
    "zero-shot-classification",
    model="./models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/snapshots/6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
    tokenizer="./models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/snapshots/6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
    framework="pt",
)

#----------------- Map definitions----------------------------------------
# -------------------- Helpers --------------------
def extract_number(text):
    m = re.findall(r"\d+(?:\.\d+)?", text)
    return float(m[0]) if m else None

def extract_boolean(text):  # yes → 1, no → 0
    text = text.lower()
    if "yes" in text:
        return 1
    if "no" in text:
        return 0
    return None

def extract_gender(text):  # female → 1, male → 0
    text = text.lower()
    if "female" in text:
        return 1
    if "male" in text:
        return 0
    return None

def classify(text, labels):
    if not text or not labels:
        return None
    result = classifier(text, candidate_labels=labels)
    return labels[result["scores"].index(max(result["scores"]))]
#----------------------Helpers-----------------------------

#genderMap = { 'Female': 1, 'Male': 0 }
#yesNoMap = { 'No': 0, 'Yes': 1 }
removedTeethMap = {
    'All': 0, '6 or more, but not all': 1, '1 to 5': 2, 'None of them': 3
}
lastCheckupTimeMap = {
    '5 or more years ago': 0,
    'Within past 5 years (2 years but less than 5 years ago)': 1,
    'Within past 2 years (1 year but less than 2 years ago)': 2,
    'Within past year (anytime less than 12 months ago)': 3
}
smokerStatusMap = {
    'Never smoked': 0,
    'Former smoker': 1,
    'Current smoker - now smokes some days': 2,
    'Current smoker - now smokes every day': 3
}
eCigaretteUsageMap = {
    'Never used e-cigarettes in my entire life': 0,
    'Not at all (right now)': 1,
    'Use them some days': 2,
    'Use them every day': 3
}
raceEthnicityMap = {
    'Multiracial, Non-Hispanic': 0,
    'Other race only, Non-Hispanic': 1,
    'Black only, Non-Hispanic': 2,
    'Hispanic': 3,
    'White only, Non-Hispanic': 4
}
ageCategoryMap = {
    'Age 18 to 24': 0, 'Age 25 to 29': 1, 'Age 30 to 34': 2, 'Age 35 to 39': 3,
    'Age 40 to 44': 4, 'Age 45 to 49': 5, 'Age 50 to 54': 6, 'Age 55 to 59': 7,
    'Age 60 to 64': 8, 'Age 65 to 69': 9, 'Age 70 to 74': 10,
    'Age 75 to 79': 11, 'Age 80 or older': 12
}
tetanusMap = {
    'No, did not receive any tetanus shot in the past 10 years': 0,
    'Yes, received tetanus shot, but not Tdap': 1,
    'Yes, received tetanus shot but not sure what type': 2,
    'Yes, received Tdap': 3
}
covidPosMap = {
    'No': 0,
    'Tested positive using home test without a health professional': 1,
    'Yes': 2
}

# Fields by type
numeric_fields = {
    "PhysicalHealthDays", "MentalHealthDays", "SleepHours", "PhysicalActivities",
    "HeightInMeters", "WeightInKilograms", "BMI"
}
yes_no_fields = {
    "HadAngina", "HadStroke", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
    "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
    "DifficultyErrands", "ChestScan", "AlcoholDrinkers", "HIVTesting",
    "FluVaxLast12", "PneumoVaxEver", "HighRiskLastYear"
}
#----------------- Map definitions----------------------------------------
# -------------------- API Route --------------------
@app.post("/parse_speech")
async def parse_speech(request: Request):
    data = await request.json()
    spoken_text = data.get("text", "")
    question_key = data.get("questionKey", "")

    value = None
    # -------------------- API Route --------------------
    #----------------------------Mapping----------------------------------
    # Mapping for if using the voice form---one-by-one
    # *Mapping for if using the fill-in form-is done in the Angular

    if question_key in numeric_fields:
        value = extract_number(spoken_text)

    elif question_key in yes_no_fields:
        value = extract_boolean(spoken_text)
        

    elif question_key == "Sex":
        #label = classify(spoken_text, list(genderMap.keys()))
        #value = genderMap.get(label)
        value = extract_gender(spoken_text)
        

    elif question_key == "LastCheckupTime":
        label = classify(spoken_text, list(lastCheckupTimeMap.keys()))
        value = lastCheckupTimeMap.get(label)

    elif question_key == "RemovedTeeth":
        label = classify(spoken_text, list(removedTeethMap.keys()))
        value = removedTeethMap.get(label)

    elif question_key == "SmokerStatus":
        label = classify(spoken_text, list(smokerStatusMap.keys()))
        value = smokerStatusMap.get(label)

    elif question_key == "ECigaretteUsage":
        label = classify(spoken_text, list(eCigaretteUsageMap.keys()))
        value = eCigaretteUsageMap.get(label)

    elif question_key == "RaceEthnicityCategory":
        label = classify(spoken_text, list(raceEthnicityMap.keys()))
        value = raceEthnicityMap.get(label)

    elif question_key == "AgeCategory":
        label = classify(spoken_text, list(ageCategoryMap.keys()))
        value = ageCategoryMap.get(label)

    elif question_key == "TetanusLast10Tdap":
        label = classify(spoken_text, list(tetanusMap.keys()))
        value = tetanusMap.get(label)

    elif question_key == "CovidPos":
        label = classify(spoken_text, list(covidPosMap.keys()))
        value = covidPosMap.get(label)

    return {"extractedValue": value}
#----------------------------Mapping-----------------------------------

#---------------------------------Model--------------------------------------------------

class PatientData(BaseModel):# Define the input data model
    age: int
    education: float
    sex: int
    is_smoking: int
    cigsPerDay: float
    BPMeds: float
    #prevalentStroke: int
    #prevalentHyp: int
    #diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float
    

@app.post("/predict/hypertension")
def predict_asthma(data: PatientData):
    
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    input_scaled = scaler.transform(input_df)    # Scale features

    # proba = model.predict_proba(input_scaled)[:, 1][0] # Predict probability
    proba = model.predict(input_scaled).flatten()[0]
    
    pred = model.predict(input_scaled)[0]    # Predict class

    return {
        "ncd_probability": float(proba),
        "ncd_prediction": int(pred)
    }









#---------------------------------Model2--------------------------------------------------

class PatientData2(BaseModel):# Defining the input data model
    Checkup: int
    Exercise: int
    #Heart_Disease: int
    Skin_Cancer: int
    Other_Cancer: int
   #Depression: int
    Diabetes: int
    #Arthritis: int
    Sex: int
    Age_Category: int
    Height_cm: float
    Weight_kg: float
    BMI: float
    Smoking_History: int
    Alcohol_Consumption: float
    Fruit_Consumption: float
    Green_Vegetables_Consumption: float
    FriedPotato_Consumption: float

@app.post("/predict/arthritis")
def predict_arthritis(data: PatientData2):
    
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    input_scaled = scaler2.transform(input_df)    # Scale features

    # proba = model2.predict_proba(input_scaled)[:, 1][0] # Predict probability
    proba = model2.predict(input_scaled).flatten()[0]
    
    pred = model2.predict(input_scaled)[0]    # Predict class

    return {
        "ncd_probability": float(proba),
        "ncd_prediction": int(pred)
    }



#---------------------------------Model3--------------------------------------------------

class PatientData3(BaseModel):# Defining the input data model
    #Unnamed: 0: int
    #index: int
    Age: int
    Gender: int
    Air_Pollution: int
    Alcohol_use: int
    Dust_Allergy: int
    OccuPational_Hazards: int
    Genetic_Risk: int
    chronic_Lung_Disease: int
    Balanced_Diet: int
    Obesity: int
    Smoking: int
    Passive_Smoker: int
    Chest_Pain: int
    Coughing_of_Blood: int
    Fatigue: int
    Weight_Loss: int
    Shortness_of_Breath: int
    Wheezing: int
    Swallowing_Difficulty: int
    Clubbing_of_Finger_Nails: int
    Frequent_Cold: int
    Dry_Cough: int
    Snoring: int
    #Level: int

@app.post("/predict/lung_cancer")
def predict_lung_cancer(data: PatientData3):
    
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    input_scaled = scaler3.transform(input_df)    # Scale features

    # proba = model3.predict_proba(input_scaled)[:, 1][0] # Predict probability
    proba = model3.predict(input_scaled).flatten()[0]
    
    pred = model3.predict(input_scaled)[0]    # Predict class

    return {
        "ncd_probability": float(proba),
        "ncd_prediction": int(pred)
    }




    
#---------------------------------Model4--------------------------------------------------

class PatientData4(BaseModel):# Defining the input data model
    #Unnamed: 0: int
    Sex: int
    PhysicalHealthDays: float
    MentalHealthDays: float
    LastCheckupTime: int
    PhysicalActivities: int
    SleepHours: float
    RemovedTeeth: int
    HadHeartAttack: int
    HadAngina: int
    HadStroke: int
    #HadAsthma: int
    HadSkinCancer: int
    HadCOPD: int
    HadDepressiveDisorder: int
    HadKidneyDisease: int
    HadArthritis: int
    HadDiabetes: int
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
   

@app.post("/predict/asthma")
def predict_asthma(data: PatientData4):
    
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    input_scaled = scaler4.transform(input_df)    # Scale features

    # proba = model4.predict_proba(input_scaled)[:, 1][0] # Predict probability
    probs = model4.predict_proba(input_scaled)
    prob_pos = probs[0][1]

    return {
        "ncd_probability": float(prob_pos),
        "ncd_prediction": int(prob_pos)
    }







#---------------------------------Model5--------------------------------------------------

class PatientData5(BaseModel):# Defining the input data model
    gender: float
    age: float
    #hypertension: int
    #heart_disease: int
    smoking_history: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
    #diabetes: int

@app.post("/predict/diabetes")
def predict_diabetes(data: PatientData5):
    
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    input_scaled = scaler5.transform(input_df)    # Scale features

    # proba = model5.predict_proba(input_scaled)[:, 1][0] # Predict probability
    proba = model5.predict(input_scaled).flatten()[0]
    
    pred = model5.predict(input_scaled)[0]    # Predict class

    return {
        "ncd_probability": float(proba),
        "ncd_prediction": int(pred)
    }

