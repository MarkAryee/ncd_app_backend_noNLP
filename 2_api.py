from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import re

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

def classify(text, labels):
    if not text or not labels:
        return None
    result = classifier(text, candidate_labels=labels)
    return labels[result["scores"].index(max(result["scores"]))]

# -------------------- Mappings --------------------
genderMap = { 'Female': 0, 'Male': 1 }
yesNoMap = { 'No': 0, 'Yes': 1 }
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

# -------------------- API Route --------------------
@app.post("/parse_speech")
async def parse_speech(request: Request):
    data = await request.json()
    spoken_text = data.get("text", "")
    question_key = data.get("questionKey", "")

    value = None

    if question_key in numeric_fields:
        value = extract_number(spoken_text)

    elif question_key in yes_no_fields:
        value = extract_boolean(spoken_text)

    elif question_key == "Sex":
        label = classify(spoken_text, list(genderMap.keys()))
        value = genderMap.get(label)

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
