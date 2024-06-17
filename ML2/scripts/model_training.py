# model_training.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

# Load preprocessed data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

# Initialize models
dt_model = DecisionTreeClassifier(random_state=42)
lr_model = LogisticRegression(max_iter=10000, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)

# Train original models
dt_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Save original models
joblib.dump(dt_model, 'models/dt_model.pkl')
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')

# Define hyperparameter grids
dt_params = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
lr_params = {'C': [0.1, 1, 10, 100]}
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 10]}
svm_params = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}

# Perform hyperparameter tuning
dt_grid = GridSearchCV(dt_model, dt_params, cv=5, scoring='accuracy')
lr_grid = GridSearchCV(lr_model, lr_params, cv=5, scoring='accuracy')
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='accuracy')
svm_grid = GridSearchCV(svm_model, svm_params, cv=5, scoring='accuracy')

dt_grid.fit(X_train, y_train)
lr_grid.fit(X_train, y_train)
rf_grid.fit(X_train, y_train)
svm_grid.fit(X_train, y_train)

# Save best models
joblib.dump(dt_grid.best_estimator_, 'models/dt_best_model.pkl')
joblib.dump(lr_grid.best_estimator_, 'models/lr_best_model.pkl')
joblib.dump(rf_grid.best_estimator_, 'models/rf_best_model.pkl')
joblib.dump(svm_grid.best_estimator_, 'models/svm_best_model.pkl')

print("Model training complete and models saved.")
