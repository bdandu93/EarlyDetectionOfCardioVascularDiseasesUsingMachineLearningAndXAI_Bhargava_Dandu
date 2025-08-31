import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Define column names
# =========================
column_names = [
    "age",
    "gender",
    "chest_pain",
    "resting_BP ",
    "serum_cholestoral",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "slope",
    "num_major_vessels",
    "thal",
    "target"
]

# =========================
# Load datasets from CSV
# =========================
# Ensure these files are in the same folder as this script
cleave_data = pd.read_csv("processed.cleveland.data", delimiter=',', header=None, names=column_names)
hung_data   = pd.read_csv("processed.hungarian.data", delimiter=',', header=None, names=column_names)
swiss_data  = pd.read_csv("processed.switzerland.data", delimiter=',', header=None, names=column_names)

# =========================
# Replace '?' with NaN
# =========================
for df in [cleave_data, hung_data, swiss_data]:
    df.replace('?', np.nan, inplace=True)

# =========================
# Combine datasets
# =========================
heart_data = pd.concat([cleave_data, hung_data, swiss_data], ignore_index=True)

# Convert to numeric
for col in heart_data.columns:
    heart_data[col] = pd.to_numeric(heart_data[col], errors='coerce')

# =========================
# EDA
# =========================
print("\nFirst 10 rows:\n", heart_data.head(10))
print("\nLast 10 rows:\n", heart_data.tail(10))
print("\nDataset shape:", heart_data.shape)
print("\nDataset info:")
print(heart_data.info())
print("\nDataset statistics:\n", heart_data.describe())
print("\nMissing values per column:\n", heart_data.isnull().sum())

# =========================
# UNIVARIATE ANALYSIS
# =========================

# Gender distribution (doughnut plot)
gender_counts = heart_data['gender'].value_counts()
labels = ['Male', 'Female']
colors = ['#66b3ff', '#ff9999']

plt.figure(figsize=(5, 5))
plt.pie(gender_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=140, wedgeprops={'width': 0.4})
plt.title('Gender Distribution')
plt.show()

# Age distribution
plt.figure(figsize=(8, 4))
sns.histplot(heart_data['age'], kde=True, color='skyblue', bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Chest pain types
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='chest_pain', data=heart_data, palette='Set2')
plt.title('Chest Pain Types')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + 0.3, p.get_height() + 1))
plt.show()

# Serum cholesterol distribution
plt.figure(figsize=(8, 4))
sns.boxenplot(x=heart_data['serum_cholestoral'], color='salmon')
plt.title('Serum Cholesterol Distribution')
plt.xlabel('Serum Cholesterol (mg/dl)')
plt.show()

# Fasting blood sugar
fbs_counts = heart_data['fasting_blood_sugar'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.stem(fbs_counts.index, fbs_counts.values, basefmt=" ")
plt.xticks([0, 1], ['≤120 mg/dl', '>120 mg/dl'])
plt.title('Fasting Blood Sugar Levels')
plt.xlabel('Fasting Blood Sugar')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Resting ECG
plt.figure(figsize=(6, 4))
sns.stripplot(x='resting_ecg', data=heart_data, jitter=True, palette='cool', size=6)
plt.title('Resting ECG Results')
plt.xlabel('ECG Result Category')
plt.show()

# Max heart rate
plt.figure(figsize=(8, 4))
sns.kdeplot(heart_data['max_heart_rate'], shade=True, color='green')
plt.title('Maximum Heart Rate Achieved')
plt.xlabel('Max Heart Rate')
plt.show()

# Exercise-induced angina
labels = ['No', 'Yes']
values = heart_data['exercise_angina'].value_counts().sort_index()
colors = ['#a1dab4', '#41b6c4']
plt.figure(figsize=(5, 5))
plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Exercise-Induced Angina')
plt.show()

# Oldpeak distribution
plt.figure(figsize=(8, 4))
sns.violinplot(x=heart_data['oldpeak'], color='orchid')
plt.title('Oldpeak (ST Depression) Distribution')
plt.xlabel('Oldpeak Value')
plt.show()

# Slope distribution
slope_counts = heart_data['slope'].value_counts().sort_index()
labels = ['Upsloping', 'Flat', 'Downsloping']
plt.figure(figsize=(7, 4))
sns.barplot(x=slope_counts.values, y=labels, palette='Blues_r')
plt.title('Slope of Peak Exercise ST Segment')
plt.xlabel('Count')
plt.ylabel('Slope Type')
plt.show()

# Major vessels
vessel_counts = heart_data['num_major_vessels'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.step(vessel_counts.index, vessel_counts.values, where='mid', linewidth=2, color='purple')
plt.scatter(vessel_counts.index, vessel_counts.values, color='purple')
plt.title('Number of Major Vessels Colored by Fluoroscopy')
plt.xlabel('Number of Vessels')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Thalassemia
plt.figure(figsize=(6, 4))
sns.countplot(x='thal', data=heart_data, palette='pastel')
plt.title('Thalassemia Types')
plt.xlabel('Thalassemia (3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)')
plt.ylabel('Count')
plt.show()

# =========================
# BI-VARIATE ANALYSIS
# =========================
plt.figure(figsize=(12, 8))
sns.heatmap(heart_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title("Correlation Heatmap of Heart Disease Features")
plt.tight_layout()
plt.show()

sns.pairplot(heart_data, hue='target', palette='coolwarm')
plt.suptitle("Pairplot of Heart Disease Dataset", y=1.02)
plt.show()

# Scatter plot: Age vs Max Heart Rate vs Target
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='max_heart_rate', hue='target', data=heart_data, s=100)
plt.title('Age vs. Max Heart Rate with Target Labels')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.legend(title='Target')
plt.show()

# Age distribution by gender
plt.figure(figsize=(10, 6))
sns.histplot(x='age', hue='gender', data=heart_data, palette='muted', multiple='stack', bins=15)
plt.title('Age Distribution by Gender')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.show()

# =========================
# Save processed dataset
# =========================
final_dataset = 'heart_disease_combined.csv'
heart_data.to_csv(final_dataset, index=False)
print(f"\n✅ Combined dataset saved as {final_dataset}")