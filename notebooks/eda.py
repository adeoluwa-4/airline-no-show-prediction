import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (adjust file name if needed)
df = pd.read_csv('data/train.csv')

# Clean minimally just in case
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)
df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# Set style
sns.set(style="whitegrid")

# Plot 1: Satisfaction by Class
plt.figure(figsize=(6,4))
sns.barplot(x='Class', y='satisfaction', data=df)
plt.title("Satisfaction Rate by Flight Class")
plt.savefig('notebooks/satisfaction_by_class.png')
plt.close()

# Plot 2: Satisfaction by Type of Travel
plt.figure(figsize=(6,4))
sns.barplot(x='Type of Travel', y='satisfaction', data=df)
plt.title("Satisfaction Rate by Travel Type")
plt.savefig('notebooks/satisfaction_by_travel_type.png')
plt.close()

# Plot 3: Satisfaction by Gender
plt.figure(figsize=(6,4))
sns.barplot(x='Gender', y='satisfaction', data=df)
plt.title("Satisfaction Rate by Gender")
plt.savefig('notebooks/satisfaction_by_gender.png')
plt.close()

# Plot 4: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig('notebooks/correlation_heatmap.png')
plt.close()

print("✅ All EDA plots saved in the notebooks/ folder.")
