import streamlit as st
import joblib
import pandas as pd
from io import StringIO

@st.cache_resource
def load_artifacts():
    models = joblib.load("demo_models.pkl")
    risks  = joblib.load("demo_risks.pkl")
    return models, risks

demo_models, demo_risks = load_artifacts()

st.title("Clinical Protocol Risk Classifier (Demo)")
st.markdown("Upload a .txt file or paste protocol text to classify risks.")

text_input = st.text_area("Paste protocol text here", height=180)
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

protocol_text = None
if uploaded_file is not None:
    protocol_text = uploaded_file.read().decode("utf-8", errors="ignore")
elif text_input.strip():
    protocol_text = text_input.strip()

if protocol_text:
    rows = []
    for risk in demo_risks:
        m = demo_models[risk]
        p = m.predict_proba([protocol_text])[0][1]
        rows.append({
            "risk": risk,
            "probability": p,
            "binary_flag": int(p >= 0.5)
        })

    df = pd.DataFrame(rows).sort_values("probability", ascending=False)

    st.subheader("Predictions")
    st.dataframe(df, use_container_width=True)

    st.info("Demo model: trained on a small dataset. Predictions are illustrative only.")

else:
    st.warning("Provide text or upload a .txt file.")
