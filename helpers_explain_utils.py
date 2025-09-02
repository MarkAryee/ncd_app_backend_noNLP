


# ----------------------------
# Centralized SHAP + Interpretation
# ----------------------------
import shap
import numpy as np
import pandas as pd

     
def explain_prediction(model, scaler, input_df, X, disease_name, proba_lowT, proba_modT):
   
    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict probability and class
    proba = float(model.predict(input_scaled).flatten()[0])
    pred = int(model.predict(input_scaled)[0])

    # -------------------------------------------------------------------------
    # SHAP Explainability
    # ----------------------------
    background = scaler.transform(input_df.sample(1, random_state=42).values)  # small ref sample
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(input_scaled)

    # shap_values is [class0, class1] for binary
    shap_for_user = shap_values[0][0]
    
   
    feature_names = list(X.columns)
    
    # Contributions
    feature_contribs = sorted(
        zip(feature_names, shap_for_user),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    
    
    
    
    # ----------------------------
    # 6. Build narrative instead of printing
    # ----------------------------
    narrative_parts = []

    #narrative_parts.append("\n--- Feature contributions to this prediction ---")
    feature_contribs = sorted(
        zip(feature_names, shap_for_user),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for feat, val in feature_contribs:
        sign = "↑" if val > 0 else "↓"
        #narrative_parts.append(f"{feat}: {val:.4f} ({sign} risk)")

    # ----------------------------
    # 6b. Interpretation of result
    # ----------------------------
    #narrative_parts.append("\n--- Final Interpretation ---")
    percent = proba * 100
    proba_low = proba_lowT
    proba_mod = proba_modT
    #proba_high = proba * 100

    if proba < proba_low:
        category = "Low"
        verdict = ("<b> Keep maintaining a healthy lifestyle </b> <br><br>"
                   "<i> Your predicted risk is low.This means your features look similar to people in the dataset who rarely developed hypertension</i>"
                   )
    elif proba < proba_mod:
        category = "Moderate"
        verdict = ("<b> Consider making lifestyle improvements </b> <br><br> "
                   "<i> Your predicted risk is moderate.Even though the number looks small in absolute terms, it is higher than average for people of similar profile in the dataset. That’s why it’s considered moderate. </i>"
                   )
    else:
        category = "High"
        verdict = ("<b> You may need to consult a healthcare professional </b> <br><br>"
                   "<i> Your predicted risk is high. Compared to similar people in the dataset, your submission places you in a group that developed hypertension more often. </i>"
                   )

    #narrative_parts.append(f"📊 Predicted risk: {percent:.1f}% → Category: {category}")
    #narrative_parts.append(f"📝 Verdict: {verdict}")
    narrative_parts.append(f"{verdict}")

    # ----------------------------
    # 6c. Why this result? (using SHAP)
    # ----------------------------
    #narrative_parts.append("\n--- Why this result? ---")
    #narrative_parts.append("These features most strongly influenced the prediction:")

    for feat, val in feature_contribs[:5]:  # top 5 reasons
        if val > 0:
            reason = "<i> pushed the risk upward <i>"
        else:
            reason = "<i> helped lower the risk <i>"
        narrative_parts.append(f"Your {feat} → {reason}")
        #narrative_parts.append(f"Your {feat} {val:.0f} → {reason}")

    # ----------------------------
    # FINAL NARRATIVE OUTPUT
    # ----------------------------
    final_narrative = "\n".join(narrative_parts)
    print(final_narrative)
    return {
        "ncd_probability": proba,
        "ncd_prediction": pred,
        "final_narrative": final_narrative,
    }
        
  
  
  
  
  
  
  
  
  
  
  

#from transformers import pipeline
import re



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import joblib
import pandas as pd
import shap
from tensorflow.keras.models import load_model


  
  
def explain_prediction_tabnet(model, scaler, input_df, X, disease_name, proba_lowT, proba_modT):
    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict probability and class
    proba = float(model.predict_proba(input_scaled)[0][1])  # class 1 prob
    pred = int(proba >= 0.5)

    # -------------------------------------------------------------------------
    # SHAP Explainability with KernelExplainer (works for TabNet)
    # -------------------------------------------------------------------------
    background = scaler.transform(X.sample(50, random_state=42))  # small reference sample

    explainer = shap.KernelExplainer(model.predict_proba, background)
    explain_matrix, masks = model.explain(input_scaled)

    shap_values = explainer.shap_values(input_scaled)

    # shap_values is [array_for_class0, array_for_class1]
    # Get shap values for single sample
    shap_for_user = explain_matrix[0]  # shape: (n_features,)

    # Ensure it's a 1D array
    shap_for_user = np.ravel(shap_for_user)
    feature_names = list(X.columns)
    # Now zip with feature names safely
    feature_contribs = sorted(
        zip(feature_names, shap_for_user),
        key=lambda x: abs(x[1]),
        reverse=True
    )


    # Contributions
    feature_contribs = sorted(
        zip(feature_names, shap_for_user),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # ----------------------------
    # Build Narrative
    # ----------------------------
    narrative_parts = []

    percent = proba * 100
    if proba < proba_lowT:
        category = "Low"
        verdict = ("<b> Keep maintaining a healthy lifestyle </b> <br><br>"
                   "<i> Your predicted risk is low.This means your features look similar to people in the dataset who rarely developed hypertension</i>"
                   )
    elif proba < proba_modT:
        category = "Moderate"
        verdict = ("<b> Consider making lifestyle improvements </b> <br><br> "
                   "<i> Your predicted risk is moderate.Even though the number looks small in absolute terms, it is higher than average for people of similar profile in the dataset. That’s why it’s considered moderate. </i>"
                   )
    else:
        category = "High"
        verdict = ("<b> You may need to consult a healthcare professional </b> <br><br>"
                   "<i> Your predicted risk is high. Compared to similar people in the dataset, your submission places you in a group that developed hypertension more often. </i>"
                   )

    #narrative_parts.append(f"📊 Predicted risk: {percent:.1f}% → Category: {category}")
    narrative_parts.append(f"{verdict}")

    #narrative_parts.append("\n--- Why this result? ---")
    for feat, val in feature_contribs[:5]:
        if val > 0:
            reason = "pushed the risk upward"
        else:
            reason = "helped lower the risk"
        narrative_parts.append(f"Your {feat} → {reason}")
        #narrative_parts.append(f"- {feat}: {val:.4f} → {reason}")

    final_narrative = "\n".join(narrative_parts)
    print(final_narrative)

    return {
        "ncd_probability": proba,
        "ncd_prediction": pred,
        "final_narrative": final_narrative,
    }
  
  
  
  
  
  
  
  
  
  