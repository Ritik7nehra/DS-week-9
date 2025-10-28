 import streamlit as st
import pickle
import pandas as pd
import numpy as np
from apputil import GroupEstimate


# ======= PAGE TITLE =======
st.write("""
# Week 9: Sentiment Analysis & GroupEstimate Demo
This app demonstrates:
- A simple **Sentiment Analysis model**
- A custom **GroupEstimate** class that computes mean/median by group
""")


# ======= LOAD SENTIMENT MODEL =======
MODEL_PATH = "sentiment_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    """Load saved sentiment model."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# Try loading the model
try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ sentiment_model.pkl not found. Please train and save the model first.")
    model_loaded = False


# ======= SENTIMENT ANALYSIS SECTION =======
st.subheader("Sentiment Prediction")
st.write("Enter text below to predict if the sentiment is **Positive** or **Negative**:")

user_text = st.text_area("Text Input", placeholder="Type your sentence here...")

if st.button("Predict Sentiment"):
    if not model_loaded:
        st.warning("Model not loaded.")
    elif not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        prediction = model.predict([user_text])[0]
        prob = model.predict_proba([user_text])[0][prediction]
        label = "😊 Positive" if prediction == 1 else "☹️ Negative"
        st.success(f"Prediction: **{label}** (Confidence: {prob:.2f})")


# ======= GROUPESTIMATE DEMO SECTION =======
st.subheader("GroupEstimate Example")
st.write("Demo of predicting group-level mean or median values.")

if st.button("Run GroupEstimate Demo"):
    # Example dataset
    X = pd.DataFrame({
        "country": ["Guatemala", "Mexico", "Mexico", "Guatemala"],
        "roast": ["Light", "Medium", "Medium", "Light"]
    })
    y = np.array([88, 90, 92, 89])

    gm = GroupEstimate(estimate="mean")
    gm.fit(X, y)

    new_data = [["Guatemala", "Light"], ["Mexico", "Medium"], ["Canada", "Dark"]]
    preds = gm.predict(new_data)

    st.write("Predictions for new data:")
    result_df = pd.DataFrame(new_data, columns=["country", "roast"])
    result_df["Predicted Rating"] = preds
    st.dataframe(result_df)
