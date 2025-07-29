import streamlit as st
import pandas as pd
import joblib

# Loading trained model and encoders
model = joblib.load("best_model.pkl")
education_encoder = joblib.load("education_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")

# Page config
st.set_page_config(page_title="Salary Class Predictor", page_icon="💼", layout="wide")

#background
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f9f9f9, #e6ecf0);
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #125e9e;
    }
    .stMarkdown h1 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>💼 Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Predict whether an employee earns >50K or ≤50K based on input features</h4>", unsafe_allow_html=True)
st.markdown("---")

# Dropdown values
education_options = ["HS-grad", "Some-college", "Assoc", "Bachelors", "Masters", "PhD"]
occupation_options = [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
]

# Form layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Personal Info")
    age = st.slider("Age", 18, 65, 30)
    education = st.selectbox("Education Level", education_options)
    occupation = st.selectbox("Job Role", occupation_options)

with col2:
    st.subheader("🕒 Work Info")
    hours_per_week = st.slider("Hours per Week", 1, 80, 40)
    experience = st.slider("Years of Experience", 0, 40, 5)

# Input data values
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# Encode
input_df['education'] = education_encoder.transform(input_df['education'])
input_df['occupation'] = occupation_encoder.transform(input_df['occupation'])


st.markdown("### 🔍 Input Summary")
st.dataframe(input_df.style.set_properties(**{'text-align': 'center'}), use_container_width=True)


st.markdown("### 🔮 Prediction")
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    if prediction[0] == '>50K':
        st.success("🎉 This employee is predicted to earn >50K.")
    else:
        st.info("💼 This employee is predicted to earn ≤50K.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.9em;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
