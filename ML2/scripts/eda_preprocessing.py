



# eda_preprocessing.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('data/breast_cancer_data.csv')

# Check for missing values
missing_values = data.isnull().sum().sum()
print(f"Total missing values: {missing_values}")

# Basic statistics
print(data.describe())

# Pairplot to visualize the relationships
sns.pairplot(data, hue='Diagnosis', vars=data.columns[1:6])
plt.show()

# Correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Data distribution
data.hist(bins=20, figsize=(20, 15))
plt.show()

# Feature Engineering: Add new features
data['mean_radius_area'] = data['feature_1'] * data['feature_10']  # Example feature

# Split the data into features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler object
joblib.dump(scaler, 'models/scaler.pkl')

# Save the preprocessed data
np.save('data/X_train.npy', X_train_scaled)
np.save('data/X_test.npy', X_test_scaled)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print("EDA and preprocessing complete, data saved.")
