import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

# Optional for deployment
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
from pyngrok import ngrok

# ----------------------------
# Load dataset
# ----------------------------
heart_data = pd.read_csv("preprocessed_dataset.csv")

# Split into X and y
X_disease = heart_data.drop(columns='target')
y = heart_data['target']

# ----------------------------
# Feature scaling
# ----------------------------
scaler = MinMaxScaler(feature_range=(0,1)).fit_transform(X_disease)
X = pd.DataFrame(scaler, columns=X_disease.columns)

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# ----------------------------
# Random Forest baseline
# ----------------------------
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("RandomForestClassifier Accuracy:", round(rf.score(X_test, y_test),3))

# ----------------------------
# Hyperparameter tuning (Bayesian optimization)
# ----------------------------
params = {
    'n_estimators': Integer(50, 500),
    'criterion': Categorical(['gini', 'entropy', 'log_loss']),
    'max_features': Categorical(['sqrt', 'log2'])
}
rf_bayes = RandomForestClassifier(random_state=42)
bayes_cv = BayesSearchCV(
    estimator=rf_bayes,
    search_spaces=params,
    n_iter=32,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    random_state=42
)
bayes_cv.fit(X_train, y_train)
print("Best Parameters:", bayes_cv.best_params_)
print("Best CV Score:", bayes_cv.best_score_)

# ----------------------------
# Feature Engineering
# ----------------------------
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

# ----------------------------
# Feature selection with RFE
# ----------------------------
rfe = RFE(estimator=rf, n_features_to_select=10)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
X_reduced = X[selected_features]

# ----------------------------
# Train final models
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest
rf_final = RandomForestClassifier(random_state=42, n_estimators=200)
rf_final.fit(X_train, y_train)
y_pred = rf_final.predict(X_test)

# AdaBoost
ada_model = AdaBoostClassifier(
    estimator=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(name, y_true, y_pred, y_scores=None):
    print(f"\n{name} Performance")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    if y_scores is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_prec = average_precision_score(y_true, y_scores)
        print(f"Average Precision (PR AUC): {avg_prec:.3f}")
        plt.plot(recall, precision, label=f"{name} (AP={avg_prec:.2f})")

plt.figure(figsize=(8,6))
evaluate_model("Random Forest", y_test, y_pred, rf_final.predict_proba(X_test)[:,1])
evaluate_model("AdaBoost", y_test, y_pred_ada, ada_model.predict_proba(X_test)[:,1])
evaluate_model("Gradient Boosting", y_test, y_pred_gb, gb_model.predict_proba(X_test)[:,1])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Save models
# ----------------------------
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_final, f)
with open("adaboost_model.pkl", "wb") as f:
    pickle.dump(ada_model, f)
with open("gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(gb_model, f)
print("Models saved successfully!")

# ----------------------------
# Feature importance plots
# ----------------------------
def plot_feature_importances(model, model_name, X_train):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(8,5))
        importances.plot(kind='bar')
        plt.title(f"{model_name} - Feature Importances")
        plt.ylabel("Importance")
        plt.show()

plot_feature_importances(rf_final, "Random Forest", X_train)
plot_feature_importances(ada_model, "AdaBoost", X_train)
plot_feature_importances(gb_model, "Gradient Boosting", X_train)

# ----------------------------
# Permutation importance
# ----------------------------
def plot_permutation_importance(model, model_name, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    plt.figure(figsize=(8,5))
    importances.plot(kind='barh')
    plt.title(f"{model_name} - Permutation Importance")
    plt.xlabel("Mean Importance")
    plt.show()

plot_permutation_importance(rf_final, "Random Forest", X_test, y_test)
plot_permutation_importance(ada_model, "AdaBoost", X_test, y_test)
plot_permutation_importance(gb_model, "Gradient Boosting", X_test, y_test)