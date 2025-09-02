import re
from typing import Optional, Union, Dict
from transformers import pipeline

# -------------------- Zero-shot classifier --------------------
classifier = pipeline(
    "zero-shot-classification",
    model=(
        "./models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/"
        "snapshots/6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
    ),
    tokenizer=(
        "./models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/"
        "snapshots/6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
    ),
    framework="pt",
)

# -------------------- Utilities --------------------
def extract_number(text: str) -> Optional[float]:
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None

def extract_boolean(text: str) -> Optional[int]:
    l = text.lower()
    if "yes" in l:
        return 1
    if "no" in l:
        return 0
    return None

def classify_category(text: str, labels: list[str]) -> Optional[str]:
    if not text.strip():
        return None
    result = classifier(text, labels)
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

# -------------------- Field-specific extraction --------------------
def extract_field(key: str, text: str) -> Union[int, float, None]:
    key = key.strip()

    yes_no_fields = {
        "HadAngina", "HadStroke", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
        "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
        "DifficultyErrands", "ChestScan", "AlcoholDrinkers", "HIVTesting",
        "FluVaxLast12", "PneumoVaxEver", "HighRiskLastYear"
    }

    numeric_fields = {
        "PhysicalHealthDays", "MentalHealthDays", "SleepHours",
        "PhysicalActivities", "HeightInMeters", "WeightInKilograms", "BMI"
    }

    if key in numeric_fields:
        return extract_number(text)

    if key in yes_no_fields:
        return extract_boolean(text)

    if key == "Sex":
        label = classify_category(text, list(genderMap.keys()))
        return genderMap.get(label)

    if key == "LastCheckupTime":
        label = classify_category(text, list(lastCheckupTimeMap.keys()))
        return lastCheckupTimeMap.get(label)

    if key == "RemovedTeeth":
        label = classify_category(text, list(removedTeethMap.keys()))
        return removedTeethMap.get(label)

    if key == "SmokerStatus":
        label = classify_category(text, list(smokerStatusMap.keys()))
        return smokerStatusMap.get(label)

    if key == "ECigaretteUsage":
        label = classify_category(text, list(eCigaretteUsageMap.keys()))
        return eCigaretteUsageMap.get(label)

    if key == "RaceEthnicityCategory":
        label = classify_category(text, list(raceEthnicityMap.keys()))
        return raceEthnicityMap.get(label)

    if key == "AgeCategory":
        label = classify_category(text, list(ageCategoryMap.keys()))
        return ageCategoryMap.get(label)

    if key == "TetanusLast10Tdap":
        label = classify_category(text, list(tetanusMap.keys()))
        return tetanusMap.get(label)

    if key == "CovidPos":
        label = classify_category(text, list(covidPosMap.keys()))
        return covidPosMap.get(label)

    return None

# -------------------- Quick test --------------------
if __name__ == "__main__":
    # demo
    print("Sex:", extract_field("Sex", "I'm a boy"))
    print("SleepHours:", extract_field("SleepHours", "About 8 hours each night"))
    print("RaceEthnicityCategory:", extract_field("RaceEthnicityCategory", "I'm black"))
    print("HadStroke:", extract_field("HadStroke", "No I haven't"))
    print("TetanusLast10Tdap:", extract_field("TetanusLast10Tdap", "Yes, I had a Tdap"))
