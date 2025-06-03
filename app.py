import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('rf_model.pkl')

st.title("✈️ Airline Passenger Satisfaction Predictor")

# Sidebar input
st.sidebar.header("Passenger Info")

# Input features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"])
age = st.sidebar.slider("Age", 7, 85, 30)
flight_distance = st.sidebar.slider("Flight Distance", 31, 5000, 500)

# Service ratings
wifi = st.sidebar.slider("Inflight wifi service (0–5)", 0, 5, 3)
booking = st.sidebar.slider("Ease of Online booking (0–5)", 0, 5, 3)
online_boarding = st.sidebar.slider("Online boarding (0–5)", 0, 5, 3)
seat_comfort = st.sidebar.slider("Seat comfort (0–5)", 0, 5, 3)
entertainment = st.sidebar.slider("Inflight entertainment (0–5)", 0, 5, 3)
baggage = st.sidebar.slider("Baggage handling (0–5)", 0, 5, 3)
cleanliness = st.sidebar.slider("Cleanliness (0–5)", 0, 5, 3)

# Manual encoding to match training format
input_data = pd.DataFrame({
    'Age': [age],
    'Flight Distance': [flight_distance],
    'Inflight wifi service': [wifi],
    'Ease of Online booking': [booking],
    'Online boarding': [online_boarding],
    'Seat comfort': [seat_comfort],
    'Inflight entertainment': [entertainment],
    'Baggage handling': [baggage],
    'Cleanliness': [cleanliness],
    'Gender_Male': [1 if gender == "Male" else 0],
    'Customer Type_Loyal Customer': [1 if customer_type == "Loyal Customer" else 0],
    'Type of Travel_Personal Travel': [1 if travel_type == "Personal Travel" else 0],
    'Class_Eco': [1 if travel_class == "Eco" else 0],
    'Class_Eco Plus': [1 if travel_class == "Eco Plus" else 0]
})

# Predict
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ This passenger is likely **Satisfied**.")
    else:
        st.error("⚠️ This passenger is likely **Dissatisfied**.")
