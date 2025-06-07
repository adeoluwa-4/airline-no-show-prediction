import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Fallback: define model training logic if no .pkl files exist
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

# Trigger fallback if needed
if not os.path.exists("rf_model.pkl") or not os.path.exists("feature_names.pkl"):
    train_and_save_model()

# Load model and features
model = joblib.load("rf_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Streamlit UI
st.set_page_config(page_title="Airline Satisfaction Predictor", layout="centered")
st.title("✈️ Airline Passenger Satisfaction Predictor")

# Sidebar inputs
st.sidebar.header("Passenger Info")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="gender_selectbox")
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"], key="customer_type_selectbox")
travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"], key="travel_type_selectbox")
travel_class = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"], key="travel_class_selectbox")
age = st.sidebar.slider("Age", 7, 85, 30)
flight_distance = st.sidebar.slider("Flight Distance", 31, 5000, 500)

# Service ratings
wifi = st.sidebar.slider("Inflight wifi service", 0, 5, 3)
booking = st.sidebar.slider("Ease of Online booking", 0, 5, 3)
online_boarding = st.sidebar.slider("Online boarding", 0, 5, 3)
seat_comfort = st.sidebar.slider("Seat comfort", 0, 5, 3)
entertainment = st.sidebar.slider("Inflight entertainment", 0, 5, 3)
baggage = st.sidebar.slider("Baggage handling", 0, 5, 3)
cleanliness = st.sidebar.slider("Cleanliness", 0, 5, 3)

# Build input
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

# Predict and display result
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ This passenger is likely **Satisfied**.")
    else:
        st.error("⚠️ This passenger is likely **Dissatisfied**.")
        st.markdown("### 🤔 Why might this passenger be dissatisfied?")
        st.markdown("""
        - Low ratings in service categories (wifi, comfort, entertainment)
        - Economy or Economy Plus class
        - Long flight with basic amenities
        - Not a loyal or frequent flyer
        """)
