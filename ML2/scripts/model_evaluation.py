# model_evaluation.py
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data and models
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

dt_model = joblib.load('models/dt_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')

# Predictions
dt_preds = dt_model.predict(X_test)
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
svm_preds = svm_model.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Evaluate models
dt_eval = evaluate_model(y_test, dt_preds)
lr_eval = evaluate_model(y_test, lr_preds)
rf_eval = evaluate_model(y_test, rf_preds)
svm_eval = evaluate_model(y_test, svm_preds)

evaluation_results = {
    'Decision Tree': dt_eval,
    'Logistic Regression': lr_eval,
    'Random Forest': rf_eval,
    'SVM': svm_eval
}

# Save evaluation results
import json
with open('evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f)

print("Model evaluation complete and results saved.")
