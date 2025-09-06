import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="wide")

# -----------------------
# Load trained model
# -----------------------
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------
# Sidebar: patient features
# -----------------------
st.sidebar.header("Patient Features")

age = st.sidebar.number_input("age", 1, 120, value=50)
chest_pain = st.sidebar.number_input("chest_pain", 0, 3, value=1)
resting_BP = st.sidebar.number_input("resting_BP ", 50, 250, value=120)  # note the space
serum_cholestoral = st.sidebar.number_input("serum_cholestoral", 100, 400, value=200)
max_heart_rate = st.sidebar.number_input("max_heart_rate", 60, 220, value=150)
oldpeak = st.sidebar.number_input("oldpeak", 0.0, 10.0, value=1.0)
num_major_vessels = st.sidebar.number_input("num_major_vessels", 0, 4, value=0)
thal = st.sidebar.number_input("thal", 0, 3, value=2)
fitness_index = st.sidebar.number_input("fitness_index", 0, 100, value=50)
oldpeak_slope = st.sidebar.number_input("oldpeak_slope", 0, 3, value=2)

# -----------------------
# Prepare input dataframe
# -----------------------
input_df = pd.DataFrame([{
    "age": age,
    "chest_pain": chest_pain,
    "resting_BP ": resting_BP,
    "serum_cholestoral": serum_cholestoral,
    "max_heart_rate": max_heart_rate,
    "oldpeak": oldpeak,
    "num_major_vessels": num_major_vessels,
    "thal": thal,
    "fitness_index": fitness_index,
    "oldpeak_slope": oldpeak_slope
}])

# -----------------------
# Prediction
# -----------------------
if st.sidebar.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[:, 1][0]

    st.subheader("Prediction Result")
    if pred == 1:
        st.success(f"The patient is predicted to have Heart Disease (Probability: {prob:.2f})")
    else:
        st.info(f"The patient is predicted NOT to have Heart Disease (Probability: {prob:.2f})")

# -----------------------
# Show input data for verification
# -----------------------
st.subheader("Input Patient Data")
st.dataframe(input_df)