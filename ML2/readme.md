Breast Cancer Diagnosis Project
Overview
This project aims to predict the diagnosis of breast cancer using various machine learning algorithms and compare their performance. The dataset used for this project is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset from the UCI Machine Learning Repository.

Project Structure
The project is structured as follows:

1.data_retrieval.py: Script to retrieve and preprocess the dataset.
2.eda_preprocessing.py: Script for exploratory data analysis (EDA) and preprocessing.
3.fun_synthetic_data_generation.py: Script to generate synthetic data.
4.hyperparameter_tuning.py: Script for hyperparameter tuning of various models.
5.model_training.py: Script to train and save the models.
6.model_evaluation.py: Script to evaluate the models.
7.visualization.py: Script for visualizing results.
8.report_generation.py: Script to generate a PDF report.
9.web_app.py: Streamlit web application for breast cancer diagnosis prediction.


Data Retrieval
The dataset is retrieved from the UCI Machine Learning Repository and saved locally as data/breast_cancer_data.csv.

Exploratory Data Analysis and Preprocessing
EDA is conducted to understand the data distribution and relationships between features. Preprocessing steps include handling missing values, feature scaling, and feature engineering.

Synthetic Data Generation
Synthetic data is generated for additional experimentation and testing. This data is saved as data/synthetic_data.csv.

Model Training
Four machine learning algorithms are trained:

Decision Tree
Logistic Regression
Random Forest
Support Vector Machine (SVM)
Models are trained on the preprocessed data and saved for future use.

Hyperparameter Tuning
Hyperparameter tuning is performed to optimize the performance of the models using GridSearchCV.

Model Evaluation
The models are evaluated using metrics such as accuracy, precision, recall, and F1 score. Evaluation results are saved in evaluation_results.json.

Visualization
Various visualizations are created to interpret the model performance, including confusion matrices and ROC curves.

Report Generation
A PDF report summarizing the project is generated using the fpdf library. The report includes sections on data retrieval, EDA, model training, evaluation, and conclusions.

Streamlit Web Application
A Streamlit web application is provided for user-friendly interaction with the trained models. Users can input feature values to get a breast cancer diagnosis prediction.

----------------------------------------------------------------
To run the web application: in terminal run this command.
----- streamlit run web_app.py
----------------------------------------------------------------

Results
The project demonstrated that Logistic Regression and SVM showed the best performance for this dataset.

Future Work
Future improvements could include exploring more advanced algorithms, incorporating additional data sources, and deploying the model for real-time diagnosis.

Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any questions or suggestions, please open an issue or contact ruslankarimovinfo@gmail.com


Directory Structure:

ML2/
├── data/
│   ├── breast_cancer_data.csv
│   ├── synthetic_data.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
├── models/
│   ├── scaler.pkl
│   ├── dt_model.pkl
│   ├── lr_model.pkl
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── dt_best_model.pkl
│   ├── lr_best_model.pkl
│   ├── rf_best_model.pkl
│   ├── svm_best_model.pkl
├── reports/
│   └── report.pdf
├── scripts/├──
|   ├── data_retrieval.py
|   ├── eda_preprocessing.py
|   ├── fun_synthetic_data_generation.py
|   ├── hyperparameter_tuning.py
|   ├── model_training.py
|   ├── model_evaluation.py
|   ├── visualization.py
|   ├── report_generation.py
|   ├── web_app.py
|   images/
|   ├── decision_tree_confusion_matrix_confusion_matrix.png
|   ├── decision_tree_roc_curve_roc_curve.png
|   └── .......
|
└── README.md

