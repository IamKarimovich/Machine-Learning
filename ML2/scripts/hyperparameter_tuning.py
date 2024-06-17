# hyperparameter_tuning.py
import numpy as np
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Load preprocessed data
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

# Initialize models
dt_model = DecisionTreeClassifier(random_state=42)
lr_model = LogisticRegression(max_iter=10000, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)

# Hyperparameter tuning
dt_params = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
lr_params = {'C': [0.1, 1, 10, 100]}
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 10]}
svm_params = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}

dt_grid = GridSearchCV(dt_model, dt_params, cv=5, scoring='accuracy')
lr_grid = GridSearchCV(lr_model, lr_params, cv=5, scoring='accuracy')
