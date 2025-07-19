import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model & preprocessor
model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

st.set_page_config(page_title=" Employee Salary Predictor", layout="wide")

st.title(" Employee Salary Prediction App")
st.markdown("Use this app to predict salary based on employee information.")

# Sidebar form input
st.sidebar.header(" Enter Employee Details")

def user_input_features():
    age = st.sidebar.slider('Age', 18, 65, 30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
    experience = st.sidebar.slider('Years of Experience', 0, 40, 5)
    education = st.sidebar.selectbox('Education Level', ['High School', 'Bachelor', 'Master', 'PhD'])
    job = st.sidebar.selectbox('Job Title', ['Data Analyst', 'Data Scientist', 'ML Engineer', 'Software Engineer'])
    city = st.sidebar.selectbox('City', ['Hyderabad', 'Bangalore', 'Mumbai', 'Delhi', 'Chennai'])
    seniority = st.sidebar.selectbox('Seniority Level', ['Junior', 'Mid', 'Senior', 'Lead'])
    industry = st.sidebar.selectbox('Industry', ['Tech', 'Finance', 'Healthcare', 'Retail', 'Education'])
    remote = st.sidebar.selectbox('Remote Status', ['Remote', 'In-Office', 'Hybrid'])

    data = {
        'Age': age,
        'Gender': gender,
        'Years of Experience': experience,
        'Education Level': education,
        'Job Title': job,
        'City': city,
        'Seniority Level': seniority,
        'Industry': industry,
        'Remote Status': remote
    }
    return pd.DataFrame([data])


input_df = user_input_features()

# Show input
st.subheader(' Input Data')
st.write(input_df)

# Preprocess input
X = preprocessor.transform(input_df)

# Predict
prediction = model.predict(X)[0]

# Output
st.subheader(" Predicted Salary:")
st.success(f"₹ {prediction:,.2f}")

# Add footer
st.markdown("---")
st.markdown(" Built by Navitha | IBM SkillBuild Internship Project")

