# heart_preprocessing.py
# Script to preprocess heart disease dataset using MLP regression and KNN imputation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ML imports
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# KNN imputation imports
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


def main():
    # Load dataset
    heart_data = pd.read_csv("heart_disease_combined.csv")
    print("===== Head of Dataset =====")
    print(heart_data.head(20))

    print("\n===== Dataset Info =====")
    print(heart_data.info())

    print("\n===== Dataset Statistics =====")
    print(heart_data.describe())

    print("\n===== Null Values Before Processing =====")
    print(heart_data.isnull().sum())

    # Fill missing values in numerical columns with mean
    columns_to_fill = ['resting_BP ', 'serum_cholestoral',
                       'resting_ecg', 'max_heart_rate',
                       'exercise_angina', 'oldpeak']
    for column in columns_to_fill:
        heart_data[column].fillna(heart_data[column].mean(), inplace=True)

    print("\n===== Null Values After Mean Imputation =====")
    print(heart_data.isnull().sum())

    # Serum Cholestoral Fix using ML regression
    print("\n===== Serum Cholestoral Imputation =====")
    print("Number of records with serum_cholestoral = 0:",
          heart_data['serum_cholestoral'].value_counts()[0])

    train_data = heart_data[heart_data['serum_cholestoral'] != 0]
    predict_data = heart_data[heart_data['serum_cholestoral'] == 0]

    X = train_data[['age', 'resting_BP ', 'max_heart_rate']]
    y = train_data['serum_cholestoral']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )

    mlp_model.fit(X_train_scaled, y_train)
    y_pred_val = mlp_model.predict(X_val_scaled)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print("\n===== Evaluation Results =====")
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation MAE : {mae:.2f}")
    print(f"Validation R² Score: {r2:.2f}")

    # Predict missing serum cholestoral values
    X_predict_scaled = scaler.transform(predict_data[['age', 'resting_BP ', 'max_heart_rate']])
    predicted_cholestrol = mlp_model.predict(X_predict_scaled)
    predicted_cholestrol_rounded = np.round(predicted_cholestrol).astype(int)

    heart_data.loc[heart_data['serum_cholestoral'] == 0, 'serum_cholestoral'] = predicted_cholestrol_rounded

    print("\n===== Null Values After Cholestoral Imputation =====")
    print(heart_data.isnull().sum())

    # KNN Imputation for categorical values
    print("\n===== KNN Imputation =====")
    categorical_cols = ['slope', 'num_major_vessels', 'thal', 'fasting_blood_sugar']

    label_encoders = {}
    for col in categorical_cols:
        if heart_data[col].dtype == 'object' or str(heart_data[col].dtype) == 'category':
            le = LabelEncoder()
            heart_data[col] = heart_data[col].astype(str)
            heart_data[col] = le.fit_transform(heart_data[col])
            label_encoders[col] = le

    knn_imputer = KNNImputer(n_neighbors=5)
    heart_data[categorical_cols] = knn_imputer.fit_transform(heart_data[categorical_cols])

    for col, le in label_encoders.items():
        heart_data[col] = heart_data[col].round().astype(int)
        heart_data[col] = le.inverse_transform(heart_data[col])

    print("Missing values after KNN imputation:\n", heart_data[categorical_cols].isnull().sum())

    # Final check
    print("\n===== Final Null Values =====")
    print(heart_data.isnull().sum())

    # Correlation heatmap
    correlation_matrix = heart_data.corr()
    top_features = correlation_matrix['target'].sort_values(ascending=False)[:10].index
    plt.figure(figsize=(8, 5))
    sns.heatmap(heart_data[top_features].corr(), annot=True, cmap='coolwarm',
                linewidths=0.1, annot_kws={"fontsize": 6})
    plt.title('Heatmap of Top Features Correlated with Target Class\n', fontsize=14)
    plt.show()

    # Save preprocessed dataset
    heart_data.to_csv("preprocessed_dataset.csv", index=False)
    print("\n===== Preprocessed dataset saved to preprocessed_dataset.csv =====")


if __name__ == "__main__":
    main()