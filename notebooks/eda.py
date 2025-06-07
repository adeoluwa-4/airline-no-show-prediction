import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/train.csv')

# Clean the data
df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore', inplace=True)
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)
df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# Set Seaborn style
sns.set(style="whitegrid")

# Plot 1: Satisfaction Rate by Class
plt.figure(figsize=(6, 4))
sns.barplot(x='Class', y='satisfaction', data=df)
plt.title("Satisfaction Rate by Class")
plt.ylabel("Proportion Satisfied")
plt.savefig('notebooks/satisfaction_by_class.png')
plt.close()

# Plot 2: Satisfaction Rate by Type of Travel
plt.figure(figsize=(6, 4))
sns.barplot(x='Type of Travel', y='satisfaction', data=df)
plt.title("Satisfaction Rate by Travel Type")
plt.ylabel("Proportion Satisfied")
plt.savefig('notebooks/satisfaction_by_travel_type.png')
plt.close()

# Plot 3: Satisfaction Rate by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Gender', y='satisfaction', data=df)
plt.title("Satisfaction Rate by Gender")
plt.ylabel("Proportion Satisfied")
plt.savefig('notebooks/satisfaction_by_gender.png')
plt.close()

# Plot 4: Correlation Heatmap (numerical features only)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.savefig('notebooks/correlation_heatmap.png')
plt.close()

print("✅ EDA plots saved in the notebooks/ folder.")
