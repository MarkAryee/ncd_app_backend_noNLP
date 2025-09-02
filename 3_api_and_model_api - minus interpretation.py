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


#from flask_cors import CORS


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



# -------------------- Map Defifnitions (for custom answers fields) --------------------
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

# Hypertension - categorical
sexMap = {"Female": 1, "Male": 0}
smokingYesNoMap = {"Yes": 1, "No": 0}
bpMedsMap = {"Yes": 1, "No": 0}
strokeHistoryMap = {"Yes": 1, "No": 0}
hypertensionHistoryMap = {"Yes": 1, "No": 0}
diabetesMap = {"Yes": 1, "No": 0}

# Arthritis
arthritisSexMap = {"Male": 1, "Female": 0}
arthritisAgeMap = {
    '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4,
    '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8,
    '65-69': 9, '70-74': 10, '75-79': 11, '80 or older': 12
}
checkupMap = {"Yes": 1, "No": 0}
exerciseMap = {"Yes": 1, "No": 0}
smokingHistoryMap = {"Never": 0, "Former": 1, "Current": 2}




lastCheckupTimeMap = {
    "Less than 1 year": 0,
    "1-2 years": 1,
    "2-5 years": 2,
    "5+ years": 3,
}
removedTeethMap = {
    "None": 0,
    "1 to 5": 1,
    "6 or more but not all": 2,
    "All": 3,
}
smokerStatusMap = {
    "Never smoked": 0,
    "Former smoker": 1,
    "Current smoker": 2,
}
eCigaretteUsageMap = {
    "Never used": 0,
    "Former user": 1,
    "Current user": 2,
}
raceEthnicityMap = {
    "White": 0,
    "Black": 1,
    "Hispanic": 2,
    "Asian": 3,
    "Other": 4,
}
ageCategoryMap = {
    "18-29": 0,
    "30-39": 1,
    "40-49": 2,
    "50-59": 3,
    "60-69": 4,
    "70-79": 5,
    "80+": 6,
}
tetanusMap = {
    "Yes, within 10 years": 1,
    "Yes, over 10 years": 2,
    "No": 0,
}
covidPosMap = {
    "Yes": 1,
    "No": 0,
}

# Disease-specific categorical maps
diabetesMedicationMap = {
    "Yes": 1,
    "No": 0,
}
hypertensionTreatmentMap = {
    "Yes": 1,
    "No": 0,
}
asthmaSeverityMap = {
    "Mild": 0,
    "Moderate": 1,
    "Severe": 2,
}
cancerTypeMap = {
    "Lung": 0,
    "Breast": 1,
    "Prostate": 2,
    "Colon": 3,
    "Other": 4,
}

# Alcohol for arthritis (number per week → numeric_fields)

#----------------- Map definitions----------------------------------------







#------------------------------FieldsGroupping based on Answer Types

    # -------------------- Fields by type --------------------
numeric_fields = {
    # Shared numeric
    "PhysicalHealthDays", "MentalHealthDays", "SleepHours", "PhysicalActivities",
    "HeightInMeters", "WeightInKilograms", "BMI",
    
    # Hypertension"AgeCategory", "Age_Category",
    "age", "Age", "cigsPerDay", "totChol", "sysBP", "diaBP", "heartRate", "glucose",
    "SystolicBP", "DiastolicBP",

    # Diabetes
    "FastingBloodSugar", "HbA1c","HbA1c_level", "RandomBloodSugar", "blood_glucose_level",

    # Heart Disease
    "Cholesterol", "LDL", "HDL",

    # CKD
    "SerumCreatinine", "eGFR", "UrineAlbumin",

    # Arthritis lifestyle nutrition
    "Alcohol_Consumption", "Height_cm", "Weight_kg",
    "Fruit_Consumption", "Green_Vegetables_Consumption", "FriedPotato_Consumption",
    
    
    "PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI", "bmi","HeightCM",
    "WeightKG", "AgeYears", "HeartRate", "SystolicBP", "DiastolicBP",
    "BloodSugarLevel", "CholesterolLevel", "LDLCholesterol", "HDLCholesterol", "Triglycerides",
    "DialysisSessionsPerWeek", "GlomerularFiltrationRate", "CreatinineLevel", "HbA1c", "WhiteBloodCellCount",
    "RedBloodCellCount", "PlateletCount", "OxygenSaturation", "FastingBloodGlucose", "PostprandialGlucose",
    "ExerciseMinutesPerWeek", "WorkPhysicalDays", "WalkMinutesPerDay", "CigarettesPerDay", "AlcoholDrinksPerWeek",
    "AsthmaAttacksPastYear", "HospitalVisitsPastYear", "ERVisitsPastYear", "MedicationsCount", "MissedWorkDaysPastMonth",
    "DoctorVisitsPastYear", "DentalVisitsPastYear"
}





yes_no_fields = {
    # Common disease fields
    "HadAngina", "HadStroke", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
    "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
    "DifficultyErrands", "ChestScan", "AlcoholDrinkers", "HIVTesting",
    "FluVaxLast12", "PneumoVaxEver", "HighRiskLastYear",

    # Hypertension
    "is_smoking", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",

    # Arthritis
     "Exercise",
    
     "Asthma", "KidneyDisease", "Diabetes", "Hypertension",
    "Arthritis", "Depression", "COPD", "Cancer",
    "HadStroke", "HeartDisease", "PhysicalActivity",
    "AlcoholDrinking", "DifficultyWalking", "SleepDisorder",
    "HearingIssue", "BlindOrVisionIssue", "HIVTesting",
    "FluVaxLast12", "PneumoVaxEver", "BloodPressureMeds",
    
    
    
    
     "FluVaxLast12", "PneumoVaxEver", "HIVTesting", "CovidPos", "HighRiskLastYear",
    "HadAsthma", "HasHypertension", "HasDiabetes", "HasHeartDisease", "HadStroke",
    "KidneyDisease", "LiverDisease", "CancerEver", "SkinCancerEver", "ChronicObstructivePulmonaryDisease",
    "Depression", "Arthritis", "VisionDifficulty", "HearingDifficulty", "MobilityDifficulty",
    "SelfCareDifficulty", "CognitiveDifficulty", "ADLDifficulty", "HealthInsurance", "HasPrimaryDoctor",
    "CurrentlyPregnant", "TakingMedication", "OnDialysis", "HadSurgeryPastYear", "HospitalizedPastYear"

}

# fields with special classification answers
# ------------------ CENTRALIZED CATEGORICAL MAPS (ALL 5 DISEASES) ------------------
categorical_maps = {
    # ---------- Hypertension ----------
    "sex": {"Female": 1, "Male": 0},
    "is_smoking": {"Yes": 1, "No": 0},
    "BPMeds": {"Yes": 1, "No": 0},
    "prevalentStroke": {"Yes": 1, "No": 0},
    "prevalentHyp": {"Yes": 1, "No": 0},
    "diabetes": {"Yes": 1, "No": 0},

    # ---------- Arthritis ----------
    "Sex": {"Male": 1, "Female": 0},
    '''
    "Age_Category": {
        "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4,
        "45-49": 5, "50-54": 6, "55-59": 7, "60-64": 8, "65-69": 9,
        "70-74": 10, "75-79": 11, "80 or older": 12
    }, 
    
    "AgeCategory": {
        "Age 18 to 24": 0, "Age 25 to 29": 1, "Age 30 to 34": 2, "Age 35 to 39": 3,
        "Age 40 to 44": 4, "Age 45 to 49": 5, "Age 50 to 54": 6, "Age 55 to 59": 7,
        "Age 60 to 64": 8, "Age 65 to 69": 9, "Age 70 to 74": 10,
        "Age 75 to 79": 11, "Age 80 or older": 12
    },    
    

    '''
    "Checkup": { 'last year': 0, 'last 2 years': 1, 'last year': 2, 'last 2 years': 3, 'Never': 4 },
    "Exercise": {"Yes": 1, "No": 0},
    "Diabetes":  { 'Yes': 3, 'No': 0, 'pre-diabetic': 1, 'only during pregnancy': 2 },
    "Skin_Cancer": {"Yes": 1, "No": 0},
    "Other_Cancer": {"Yes": 1, "No": 0},
    "Smoking_History": {"Yes": 1, "No":0},

    # ---------- Lung Cancer ----------
    "Gender": {"Male": 1, "Female": 2},
    "Air_Pollution": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8},
    "Alcohol_use": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8},
    "Dust_Allergy": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8},
    "OccuPational_Hazards": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8},
    
    "Genetic_Risk": {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Usually':5, 'Almost always':6, 'Always':7} ,
    "Balanced_Diet": {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Usually':5, 'Almost always':6, 'Always':7},
    "Obesity": {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Usually':5, 'Almost always':6, 'Always':7},
    
    "Smoking": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8} ,
    "Passive_Smoker": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8} ,
    
    "chronic_Lung_Disease": {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Usually':5, 'Almost always':6, 'Always':7},
    
    "Chest_Pain": {'Never':1, 'Almost never':2,'Rarely':3, 'Occasionally':4, 'Sometimes':5, 'Often':6, 'Usually':7, 'Almost always':8, 'Always':9} ,
    "Coughing_of_Blood": {'Never':1, 'Almost never':2,'Rarely':3, 'Occasionally':4, 'Sometimes':5, 'Often':6, 'Usually':7, 'Almost always':8, 'Always':9} ,
    "Fatigue":{'Never':1, 'Almost never':2,'Rarely':3, 'Occasionally':4, 'Sometimes':5, 'Often':6, 'Usually':7, 'Almost always':8, 'Always':9} ,
    
    "Weight_Loss": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8} ,
    
    "Shortness_of_Breath": {'Never':1, 'Almost never':2,'Rarely':3, 'Occasionally':4, 'Sometimes':5, 'Often':6, 'Usually':7, 'Almost always':8, 'Always':9} ,
    
    "Wheezing": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8} ,
    "Swallowing_Difficulty": {'Never':1, 'Rarely':2, 'Occasionally':3, 'Sometimes':4, 'Often':5, 'Usually':6, 'Almost always':7, 'Always':8} ,
    
    "Clubbing_of_Finger_Nails": {'Never':1, 'Almost never':2,'Rarely':3, 'Occasionally':4, 'Sometimes':5, 'Often':6, 'Usually':7, 'Almost always':8, 'Always':9},
    
    "Frequent_Cold": {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Usually':5, 'Almost always':6, 'Always':7},
    "Dry_Cough": {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Usually':5, 'Almost always':6, 'Always':7},
    "Snoring": {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Usually':5, 'Almost always':6, 'Always':7},

    # ---------- Asthma ----------
    "RaceEthnicityCategory": {
        "Multiracial, Non-Hispanic": 0,
        "Other race only, Non-Hispanic": 1,
        "Black only, Non-Hispanic": 2,
        "Hispanic": 3,
        "White only, Non-Hispanic": 4
    },
   
    "Sex": {"Male": 1, "Female": 0},

    "LastCheckupTime": {
        "5 or more years ago": 0,
        "Within past 5 years (2 years but less than 5 years ago)": 1,
        "Within past 2 years (1 year but less than 2 years ago)": 2,
        "Within past year (anytime less than 12 months ago)": 3
    },
    # NOTE: Using the encoding from your Asthma form (reversed vs your old dict)
    "RemovedTeeth": {
        "All": 3,
        "6 or more, but not all": 2,
        "1 to 5": 1,
        "None of them": 0
    },

    "DeafOrHardOfHearing": {"Yes": 1, "No": 0},
    "BlindOrVisionDifficulty": {"Yes": 1, "No": 0},
    "DifficultyConcentrating": {"Yes": 1, "No": 0},
    "DifficultyWalking": {"Yes": 1, "No": 0},
    "DifficultyDressingBathing": {"Yes": 1, "No": 0},
    "DifficultyErrands": {"Yes": 1, "No": 0},

    "SmokerStatus": {
        # Full BRFSS-style options
        "Never smoked": 0,
        "Former smoker": 1,
        "Current smoker - now smokes some days": 2,
        "Current smoker - now smokes every day": 3,
        # Short labels used in your Diabetes form
        "Never": 0,
        "Former": 1,
        "Some days": 2,
        "Every day": 3
    },
    "ECigaretteUsage": {
        "Never used e-cigarettes in my entire life": 0,
        "Not at all (right now)": 1,
        "Use them some days": 2,
        "Use them every day": 3
    },
    "AlcoholDrinkers": {"Yes": 1, "No": 0},

    "HadAngina": {"Yes": 1, "No": 0},
    "HadStroke": {"Yes": 1, "No": 0},
    "HadHeartAttack": {"Yes": 1, "No": 0},
    "HadCOPD": {"Yes": 1, "No": 0},
    "HadDepressiveDisorder": {"Yes": 1, "No": 0},
    "HadArthritis": {"Yes": 1, "No": 0},
    "HadKidneyDisease": {"Yes": 1, "No": 0},
    "HadSkinCancer": {"Yes": 1, "No": 0},
    "HadDiabetes": {"Yes": 1, "No": 0},

    "ChestScan": {"Yes": 1, "No": 0},
    "HIVTesting": {"Yes": 1, "No": 0},
    "FluVaxLast12": {"Yes": 1, "No": 0},
    "PneumoVaxEver": {"Yes": 1, "No": 0},
    "TetanusLast10Tdap": {
        "No, did not receive any tetanus shot in the past 10 years": 0,
        "Yes, received tetanus shot, but not Tdap": 1,
        "Yes, received tetanus shot but not sure what type": 2,
        "Yes, received Tdap": 3
    },
    "HighRiskLastYear": {"Yes": 1, "No": 0},

    "CovidPos": {
        "No": 0,
        "Tested positive using home test without a health professional": 1,
        "Yes": 2
    },

    # ---------- Diabetes ----------
    "gender": {"Male": 1, "Female": 0},
    "hypertension": {"Yes": 1, "No": 0},
    "heart_disease": {"Yes": 1, "No": 0},
    "ChestScan_diabetes": {"Yes": 1, "No": 0},  # if you want a disease-specific alias
    # (But since your field key in Diabetes is exactly "ChestScan", you'll just use "ChestScan" above.)

    "smoking_history": {"Never": 0, "Former": 1, "Current": 2},
    
    # "SmokerStatus" handled above (shared) "Smoking_History": {"Never": 0, "Former": 1, "Current": 2},
}
















# -------------------- API Route --------------------
@app.post("/parse_speech")
async def parse_speech(request: Request):
    data = await request.json()
    spoken_text = data.get("text", "")
    question_key = data.get("questionKey", "")

    value = None
   
        
        # Mapping for if using the voice form---one-by-one
        # *Mapping for if using the fill-in form-is done in the Angular
           
     #----------------------------Mapping the spoken text to processable values----------------------------------

        
     #-------Field by type (General Yes/No or Number Fields) Grouped Earlier--Get those out of the way------------------------------------------------------
       # In the fields categorised as numeric(measurements) earlier, take only the numbers from them. #CENTRALIZING
    if question_key in numeric_fields:
        value = extract_number(spoken_text)

    elif question_key in yes_no_fields:
        value = extract_boolean(spoken_text)
    
    
        
        #-----------Fields with custom options---------------------
  
        # categorical maps
    elif question_key in categorical_maps:
        label = classify(spoken_text, list(categorical_maps[question_key].keys()))
        value = categorical_maps[question_key].get(label)
      
    #Age Catergory is a special case, there's an infinte customisation - in - 80+  
    elif question_key in ["AgeCategory", "Age_Category"]:
        num = extract_number(spoken_text)
        value = age_to_category(num)
        
    else:
        value = None  # fallback

    #return jsonify({"extractedValue": value})
    return {"extractedValue": value}
    # ---------------------------- Mapping ----------------------------------

#Age Catergory is a special case, there's an infinte customisation - in - 80+. After being numberic, its categorised here
def age_to_category(age):
    age = int(age)
    if 18 <= age <= 24: return 0
    elif 25 <= age <= 29: return 1
    elif 30 <= age <= 34: return 2
    elif 35 <= age <= 39: return 3
    elif 40 <= age <= 44: return 4
    elif 45 <= age <= 49: return 5
    elif 50 <= age <= 54: return 6
    elif 55 <= age <= 59: return 7
    elif 60 <= age <= 64: return 8
    elif 65 <= age <= 69: return 9
    elif 70 <= age <= 74: return 10
    elif 75 <= age <= 79: return 11
    elif age >= 80: return 12
    else: return None  # for ages below 18 or invalid input

 # -------------------- API Route --------------------





#---------------------------------Model--------------------------------------------------

class PatientData(BaseModel):# Define the input data model
    #education: float
    
    #prevalentStroke: int
    #prevalentHyp: int
    #diabetes: int
    age: int
    sex: int
    is_smoking: int
    cigsPerDay: float
    BPMeds: float
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

