import streamlit as st
import joblib
import pandas as pd
from io import StringIO

# -----------------------------
# Load trained artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    models = joblib.load("demo_models.pkl")
    risks  = joblib.load("demo_risks.pkl")
    return models, risks

demo_models, demo_risks = load_artifacts()

# -----------------------------
# UI – Title and Introduction
# -----------------------------
st.title("Clinical Protocol Risk Classifier (Demo)")

st.markdown(
    """
This application provides a **demonstration-only risk classification** of clinical protocol texts using
a pre-trained machine learning model. Users may upload a `.txt` file or paste protocol content directly
to obtain probabilistic estimates for different risk categories.

The system performs **inference only**: no training occurs within the app, and no uploaded text is stored.
All predictions are **illustrative**, based on a small and imbalanced experimental dataset, and **must not**
be used for real ethical, regulatory, or clinical decision-making.
"""
)

# -----------------------------
# Inputs
# -----------------------------
text_input = st.text_area("Paste protocol text here", height=180)
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

protocol_text = None
if uploaded_file is not None:
    protocol_text = uploaded_file.read().decode("utf-8", errors="ignore")
elif text_input.strip():
    protocol_text = text_input.strip()

# -----------------------------
# Fixed risk name translation (Spanish -> English)
# -----------------------------
RISK_TRANSLATION = {
    "riesgo_resolucion_8430": "Regulatory Risk (Resolution 8430)",
    "riesgo_poblacion_vulnerable": "Vulnerable Population Risk",
    "riesgo_naturaleza_alcance": "Nature and Scope Risk",
    "riesgo_biologico": "Biological Risk",
    "riesgo_tratamiento_datos": "Data Processing Risk",
    "riesgo_sistemas_seguridad_informacion": "Information Security Systems Risk",
    "Financial Risk": "Financial Risk",
}

# -----------------------------
# Prediction and Table
# -----------------------------
if protocol_text:
    rows = []
    for risk in demo_risks:
        model = demo_models[risk]
        p = model.predict_proba([protocol_text])[0][1]  # probability of class "risk present"

        rows.append({
            "risk_raw": risk,
            "probability": p,
            "present_flag": int(p >= 0.5),
        })

    df = pd.DataFrame(rows)

    # Deterministic translation using the fixed mapping
    df["Risk"] = df["risk_raw"].map(RISK_TRANSLATION).fillna(df["risk_raw"])

    # Convert probability to percentage
    df["Probability"] = (df["probability"] * 100).round(1).astype(str) + "%"

    # Convert binary flag to Yes / No
    df["Present"] = df["present_flag"].map({1: "Yes", 0: "No"})

    # Final clean table
    df = df[["Risk", "Probability", "Present"]]

    st.subheader("Predictions")
    st.dataframe(df, use_container_width=True)

    st.info(
        "Demo model: trained on a small, imbalanced dataset. "
        "Predictions are illustrative only and not for real regulatory use."
    )

else:
    st.warning("Provide text or upload a .txt file.")
