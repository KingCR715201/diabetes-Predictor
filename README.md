# Diabetes Prediction Using Machine Learning (Random Forest)

## Project Title
Diabetes Prediction Using Machine Learning

---

## Problem Statement
Diabetes is a chronic disease that requires early detection to prevent severe health complications. Manual diagnosis based solely on individual medical parameters can be error-prone and time-consuming. There is a need for an automated system that can analyze multiple health indicators simultaneously and assist in predicting whether a person is diabetic or non-diabetic.

---

## Objective
The primary objective of this project is to build an efficient machine learning model that predicts the presence of diabetes in a patient based on medical attributes. The project aims to improve prediction reliability by handling missing values, outliers, and non-linear feature relationships.

---

## Dataset Description
The dataset used in this project is the PIMA Indians Diabetes Dataset. It consists of medical diagnostic measurements for female patients of Pima Indian heritage.

### Attributes:
- Pregnancies: Number of pregnancies
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skinfold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg / height in m²)
- DiabetesPedigreeFunction: Diabetes genetic function
- Age: Age of the patient
- Outcome: Class variable (0 = Non-Diabetic, 1 = Diabetic)

---

## Methodology / Approach
1. Load the diabetes dataset.
2. Identify and handle invalid zero values in medical attributes.
3. Replace missing values using median imputation to reduce the impact of outliers.
4. Split the dataset into training and testing sets using stratified sampling.
5. Train a Random Forest Classifier to capture non-linear relationships.
6. Evaluate the model using accuracy and classification metrics.
7. Save the trained model for future predictions.
8. Test the model with sample patient input.

---

## Tools & Technologies Used
- Programming Language: Python
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
- Machine Learning Algorithm: Random Forest Classifier
- Development Environment: VS Code / PyCharm / Jupyter Notebook

---

## Steps to Run the Project
1. Install the required dependencies:
   ```bash
   pip install numpy pandas scikit-learn
## Results / Output
The model achieves reliable prediction accuracy on the test dataset.

Outputs include:
1. Model accuracy score
2. Classification report (precision, recall, F1-score)
3. Feature importance values
4. Final prediction (Diabetic / Non-Diabetic)
5. Probability of diabetes for a given input
