import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('data/train.csv')
df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore', inplace=True)
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)
df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Train-test split
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and feature names
joblib.dump(model, 'rf_model.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')  # 👈 THIS is what's missing
print("✅ Model and feature names saved.")
