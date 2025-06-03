import pandas as pd

# Load the data
df = pd.read_csv('data/train.csv')

# Drop unnecessary columns
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

# Fill missing Arrival Delay with median
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)

# Convert satisfaction to binary
df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# Preview cleaned data
print("Cleaned Data Preview:")
print(df.head())

# Check updated summary
print("\nUpdated Column Info:")
print(df.info())

# Confirm no missing values
print("\nRemaining Missing Values:")
print(df.isnull().sum())
