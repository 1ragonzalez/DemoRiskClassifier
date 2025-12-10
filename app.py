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
# Risk name translation (Spanish → English)
# -----------------------------
RISK_TRANSLATION = {
    "riesgo_etico": "Ethical Risk",
    "riesgo_financiero": "Financial Risk",
    "riesgo_legal": "Legal Risk",
    "riesgo_privacidad": "Privacy Risk",
    "riesgo_seguridad": "Safety Risk"
}

# -----------------------------
# Prediction and Table
# -----------------------------
if protocol_text:
    rows = []
    for risk in demo_risks:
        m = demo_models[risk]
        p = m.predict_proba([protocol_text])[0][1]  # probability of class "risk present"

        rows.append({
            "risk": risk,
            "probability": p,
            "present_flag": int(p >= 0.5)
        })

    df = pd.DataFrame(rows)

    # Translate risk names to English (safe fallback if not found)
    df["risk"] = df["risk"].map(lambda x: RISK_TRANSLATION.get(x, x))

    # Convert probability to percentage
    df["Probability"] = (df["probability"] * 100).round(1).astype(str) + "%"

    # Convert binary flag to Yes / No
    df["Present"] = df["present_flag"].map({1: "Yes", 0: "No"})

    # Final clean table (English only)
    df = df[["risk", "Probability", "Present"]].rename(
        columns={"risk": "Risk"}
    )

    st.subheader("Predictions")
    st.dataframe(df, use_container_width=True)

    st.info(
        "Demo model: trained on a small, imbalanced dataset. "
        "Predictions are illustrative only and not for real regulatory use."
    )

else:
    st.warning("Provide text or upload a .txt file.")
