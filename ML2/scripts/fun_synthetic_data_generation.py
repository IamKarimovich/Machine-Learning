# fun_synthetic_data_generation.py
import numpy as np
import pandas as pd

# Function to generate synthetic data
def generate_synthetic_data(n_samples=1000):
    # Generate synthetic features
    synthetic_features = np.random.rand(n_samples, 30)  # Generating random values for features

    # Generate synthetic target labels
    synthetic_labels = np.random.randint(0, 2, size=n_samples)  # Generating random binary labels

    # Combine features and labels into a DataFrame
    synthetic_data = pd.DataFrame(synthetic_features, columns=[f'feature_{i}' for i in range(1, 31)])
    synthetic_data['Diagnosis'] = synthetic_labels

    return synthetic_data

# Save synthetic data
synthetic_data = generate_synthetic_data()
synthetic_data.to_csv('data/synthetic_data.csv', index=False)

print("Synthetic data generation complete and saved.")
