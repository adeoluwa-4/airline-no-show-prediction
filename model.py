import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save_model():
    df = pd.read_csv('data/train.csv')

    # Clean data
    df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore', inplace=True)
    df['Arrival Delay in Minutes'].fillna(0, inplace=True)
    df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

    # Encode
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model + features
    joblib.dump(model, 'rf_model.pkl')
    joblib.dump(list(X.columns), 'feature_names.pkl')

# Uncomment to test manually
# train_and_save_model()
