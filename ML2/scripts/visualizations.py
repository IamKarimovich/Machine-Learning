# visualization.py
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json
import os  # Import os module to check if files exist
import plotly.express as px
import pandas as pd

# Load preprocessed data and evaluation results
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
with open('evaluation_results.json', 'r') as f:
    evaluation_results = json.load(f)

# Load models
dt_model = joblib.load('models/dt_best_model.pkl')
lr_model = joblib.load('models/lr_best_model.pkl')
rf_model = joblib.load('models/rf_best_model.pkl')
svm_model = joblib.load('models/svm_best_model.pkl')

# Check if the directory for saving images exists, if not, create it
if not os.path.exists('images'):
    os.makedirs('images')

# Get probability scores
dt_probs = dt_model.predict_proba(X_test)[:, 1]
lr_probs = lr_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]
svm_probs = svm_model.predict_proba(X_test)[:, 1]

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    image_path = f'images/{title.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(image_path)  # Save the image
    plt.close()  # Close the plot to prevent it from displaying in the console
    return image_path

# ROC Curve
def plot_roc_curve(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    image_path = f'images/{title.lower().replace(" ", "_")}_roc_curve.png'
    plt.savefig(image_path)  # Save the image
    plt.close()  # Close the plot to prevent it from displaying in the console
    return image_path

# Feature Importance for Random Forest
feature_importances = rf_model.feature_importances_
features = [f'feature_{i}' for i in range(1, 31)] + ['mean_radius_area']
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Interactive barplot for feature importances
fig = px.bar(importance_df, x='Importance', y='Feature', title='Feature Importances for Random Forest', orientation='h')
feature_importance_path = 'images/feature_importance.png'
fig.write_image(feature_importance_path)  # Save the image

# Print image paths for debugging
print("Confusion Matrix Images:")
print(plot_confusion_matrix(y_test, dt_model.predict(X_test), 'Decision Tree Confusion Matrix'))
print(plot_confusion_matrix(y_test, lr_model.predict(X_test), 'Logistic Regression Confusion Matrix'))
print(plot_confusion_matrix(y_test, rf_model.predict(X_test), 'Random Forest Confusion Matrix'))
print(plot_confusion_matrix(y_test, svm_model.predict(X_test), 'SVM Confusion Matrix'))
print("\nROC Curve Images:")
print(plot_roc_curve(y_test, dt_probs, 'Decision Tree ROC Curve'))
print(plot_roc_curve(y_test, lr_probs, 'Logistic Regression ROC Curve'))
print(plot_roc_curve(y_test, rf_probs, 'Random Forest ROC Curve'))
print(plot_roc_curve(y_test, svm_probs, 'SVM ROC Curve'))
print("\nFeature Importance Image:")
print(feature_importance_path)

print("Visualizations complete.")
