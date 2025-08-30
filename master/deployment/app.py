import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="your-username/tourism_model",  # 🔹 replace with your HF repo ID
    filename="tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction App")
st.write("""
This app predicts whether a customer is likely to purchase a tourism package 
based on their demographic, travel, and interaction details.
Please fill in the information below to get a prediction.
""")

# --- User Inputs ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=10.0)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
passport = st.selectbox("Has Passport", [0, 1])
satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Owns a Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=1000000.0, value=25000.0)

# --- Assemble input into DataFrame ---
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# --- Prediction ---
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "✅ Will Purchase" if prediction == 1 else "❌ Will Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
