# data_retrieval.py
import pandas as pd

print("Welcome to the Breast Cancer Diagnosis Project!")

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)

# Drop the ID column
data = data.drop('ID', axis=1)

# Map diagnosis to binary values
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Save the data for later use
data.to_csv('data/breast_cancer_data.csv', index=False)
print("Data retrieval complete and saved to 'breast_cancer_data.csv'")
