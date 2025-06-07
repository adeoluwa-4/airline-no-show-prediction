import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Fallback model training if needed
def train_and_save_model():
    df = pd.read_csv('data/train.csv')
    df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore', inplace=True)
    df['Arrival Delay in Minutes'].fillna(0, inplace=True)
    df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'rf_model.pkl')
    joblib.dump(list(X.columns), 'feature_names.pkl')

# Train if models not found
if not os.path.exists("rf_model.pkl") or not os.path.exists("feature_names.pkl"):
    train_and_save_model()

# Load trained model
model = joblib.load("rf_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# UI Layout
st.set_page_config(page_title="Airline Passenger Satisfaction Predictor", layout="centered")
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.write("Fill out the passenger information below to predict their satisfaction level.")

# Input fields
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
with col2:
    age = st.slider("Age", 7, 85, 30)
    flight_distance = st.slider("Flight Distance", 31, 5000, 500)

st.subheader("Rate the services (0–5):")
wifi = st.slider("Inflight wifi service", 0, 5, 3)
booking = st.slider("Ease of Online booking", 0, 5, 3)
online_boarding = st.slider("Online boarding", 0, 5, 3)
seat_comfort = st.slider("Seat comfort", 0, 5, 3)
entertainment = st.slider("Inflight entertainment", 0, 5, 3)
baggage = st.slider("Baggage handling", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)

# Build model input
input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
input_data["Age"] = age
input_data["Flight Distance"] = flight_distance
input_data["Inflight wifi service"] = wifi
input_data["Ease of Online booking"] = booking
input_data["Online boarding"] = online_boarding
input_data["Seat comfort"] = seat_comfort
input_data["Inflight entertainment"] = entertainment
input_data["Baggage handling"] = baggage
input_data["Cleanliness"] = cleanliness
input_data["Gender_Male"] = 1 if gender == "Male" else 0
input_data["Customer Type_disloyal Customer"] = 1 if customer_type == "disloyal Customer" else 0
input_data["Type of Travel_Personal Travel"] = 1 if travel_type == "Personal Travel" else 0
input_data["Class_Eco"] = 1 if travel_class == "Eco" else 0
input_data["Class_Eco Plus"] = 1 if travel_class == "Eco Plus" else 0

# Prediction
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ This passenger is likely **Satisfied**.")
    else:
        st.error("⚠️ This passenger is likely **Dissatisfied**.")
        st.markdown("### 🤔 Possible reasons for dissatisfaction:")
        st.markdown("""
        - Lower service ratings (wifi, entertainment, comfort)
        - Economy class or no loyalty status
        - Longer flight distance with poor amenities
        """)
