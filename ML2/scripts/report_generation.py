import json
import matplotlib.pyplot as plt
from fpdf import FPDF
import seaborn as sns
import numpy as np
import os

#This part of project is just for addition experience. That's why we did not want to delete. But it needs more editing prosess. 

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Disease Diagnosis Project Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path, title):
        self.chapter_title(title)
        self.image(image_path, w=180)
        self.ln()

# Generate the PDF report
pdf = PDFReport()

# Add the introduction
pdf.add_page()
pdf.chapter_title('1. Introduction')
pdf.chapter_body(
    "This project aims to predict the diagnosis of breast cancer using various machine learning algorithms and compare their performance. "
    "The dataset used for this project is from the UCI Machine Learning Repository."
)

# Add the data retrieval section
pdf.chapter_title('2. Data Retrieval')
pdf.chapter_body(
    "The dataset was retrieved from the UCI repository and saved locally. The data includes features relevant to breast cancer diagnosis."
)

# Add the data exploration and preprocessing section
pdf.chapter_title('3. Data Exploration and Preprocessing')
pdf.chapter_body(
    "Exploratory Data Analysis (EDA) was conducted to understand the data distribution and relationships between features. "
    "Preprocessing steps included handling missing values, feature scaling, and feature engineering."
)

# Add the algorithm selection section
pdf.chapter_title('4. Algorithm Selection')
pdf.chapter_body(
    "Four machine learning algorithms were selected:\n"
    "- Decision Tree\n"
    "- Logistic Regression\n"
    "- Random Forest\n"
    "- Support Vector Machine (SVM)\n\n"
    "The selection was based on their suitability for classification tasks and their different characteristics."
)

# Add the model training and evaluation section
pdf.chapter_title('5. Model Training and Evaluation')
evaluation_results = {
    'Decision Tree': [0.94, 0.92, 0.94, 0.93],
    'Logistic Regression': [0.97, 0.96, 0.97, 0.97],
    'Random Forest': [0.96, 0.94, 0.96, 0.95],
    'SVM': [0.97, 0.96, 0.97, 0.97]
}
pdf.chapter_body(
    "The models were trained on the preprocessed data, and their performance was evaluated using metrics like accuracy, precision, recall, and F1 score."
)

# Add the hyperparameter tuning section
pdf.chapter_title('6. Hyperparameter Tuning')
best_params = {
    'Decision Tree': {'max_depth': 5, 'min_samples_split': 2},
    'Logistic Regression': {'C': 1},
    'Random Forest': {'n_estimators': 100, 'max_depth': 7},
    'SVM': {'C': 1, 'kernel': 'rbf'}
}
pdf.chapter_body(
    "Hyperparameter tuning was performed to optimize the performance of the models. The best parameters were identified and the models were re-evaluated."
)

# Add the visualization section
pdf.chapter_title('7. Visualization')

# Add Confusion Matrix Plots
confusion_matrices = ['images/decision_tree_confusion_matrix.png', 'images/logistic_regression_confusion_matrix.png', 'images/random_forest_confusion_matrix.png', 'images/svm_confusion_matrix_confusion_matrix.png']
for matrix in confusion_matrices:
    if os.path.exists(matrix):
        pdf.add_image(matrix, 'Confusion Matrix')

# Add ROC Curve Plots
roc_curves = ['images/decision_tree_roc_curve_roc_curve.png', 'images/logistic_regression_roc_curve_roc_curve.png', 'images/random_forest_roc_curve_roc_curve.png', 'images/svm_roc_curve_roc_curve.png']
for curve in roc_curves:
    if os.path.exists(curve):
        pdf.add_image(curve, 'ROC Curve')

# Add Feature Importance Plot
if os.path.exists('images/feature_importance.png'):
    pdf.add_image('images/feature_importance.png', 'images/Feature Importance for Random Forest')

# Add the conclusion section
pdf.chapter_title('8. Conclusion')
pdf.chapter_body(
    "The project demonstrated the application of various machine learning techniques to a real-world classification task. "
    "Logistic Regression and SVM showed the best performance for this dataset."
)

# Add the future work section
pdf.chapter_title('9. Future Work')
pdf.chapter_body(
    "Future work could include exploring more advanced algorithms, incorporating additional data sources, and deploying the model for real-time diagnosis."
)

# Save the PDF
pdf.output('reports/report.pdf')

print("PDF report generation complete.")
