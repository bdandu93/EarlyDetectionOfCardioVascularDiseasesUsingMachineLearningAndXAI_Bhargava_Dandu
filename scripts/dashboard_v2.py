# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance

# ----------------------------
# Load dataset
# ----------------------------
st.title("❤️ Heart Disease Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("/Users/bdandu/Desktop/DKIT/M.Sc_(DKIT)/Dissertation/finalProject/dataset/pre processed dataset/preprocessed_dataset.csv")
    return df

heart_data = load_data()
st.write("### Dataset Preview", heart_data.head())

# ----------------------------
# Preprocess
# ----------------------------
X_disease = heart_data.drop(columns='target')
y = heart_data['target']

scaler = MinMaxScaler(feature_range=(0,1)).fit_transform(X_disease)
X = pd.DataFrame(scaler, columns=X_disease.columns)

# Feature engineering
X['age_group'] = pd.cut(X['age'], bins=[0, 40, 55, 70, 100], labels=[0,1,2,3])
X['high_chol'] = (X['serum_cholestoral'] > 240).astype(int)
X['high_bp'] = (X['resting_BP '] > 140).astype(int)
X['fitness_index'] = X['thal'] / X['age']
X['oldpeak_slope'] = X['oldpeak'] * X['slope']
X['age_exang'] = X['age'] * X['exercise_angina']
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['category','object']).columns:
    X[col] = X[col].fillna(X[col].mode()[0])
X = X.astype(np.float32)

# Feature selection with RFE
rf = RandomForestClassifier()
rfe = RFE(estimator=rf, n_features_to_select=10)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
X_reduced = X[selected_features]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Train Models
# ----------------------------
rf_final = RandomForestClassifier(random_state=42, n_estimators=200)
rf_final.fit(X_train, y_train)
ada_model = AdaBoostClassifier(
    estimator=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
    n_estimators=100, learning_rate=0.5, random_state=42
)
ada_model.fit(X_train, y_train)
gb_model = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
)
gb_model.fit(X_train, y_train)

models = {
    "Random Forest": rf_final,
    "AdaBoost": ada_model,
    "Gradient Boosting": gb_model
}

# ----------------------------
# Dashboard UI
# ----------------------------
option = st.selectbox("Choose a model", list(models.keys()))
model = models[option]
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

# Evaluation metrics
st.subheader(f"{option} Performance")
st.write("Accuracy:", round(accuracy_score(y_test, y_pred),3))
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = [f"Class {i}" for i in np.unique(y)]
st.write("Confusion Matrix")
st.dataframe(pd.DataFrame(cm, index=[f"Actual {l}" for l in labels],
                               columns=[f"Pred {l}" for l in labels]))

# Precision-Recall Curve
from sklearn.preprocessing import label_binarize

classes = np.unique(y)
if len(classes) > 2:
    y_test_bin = label_binarize(y_test, classes=classes)
    y_scores_bin = model.predict_proba(X_test)

    fig, ax = plt.subplots()
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_scores_bin[:, i])
        avg_prec = average_precision_score(y_test_bin[:, i], y_scores_bin[:, i])
        ax.plot(recall, precision, label=f"Class {cls} (AP={avg_prec:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multiclass Precision-Recall Curves")
    ax.legend()
    st.pyplot(fig)

# Feature Importances
if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    importances.plot(kind='bar', ax=ax)
    st.pyplot(fig)

# Permutation Importances
st.subheader("Permutation Importances")
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots()
perm_importances.plot(kind='barh', ax=ax)
st.pyplot(fig)