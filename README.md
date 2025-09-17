# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

This repository contains a data science project that uses various machine learning models to predict the likelihood of a patient having heart disease based on their medical attributes.

## 📋 Table of Contents
1. [Problem Definition](#-problem-definition)
2. [Data Source](#-data-source)
3. [Evaluation Metric](#-evaluation-metric)
4. [Features & Data Dictionary](#-features--data-dictionary)
5. [Project Workflow](#-project-workflow)
6. [Final Results](#-final-results)
7. [Technologies Used](#-technologies-used)
8. [How to Run](#-how-to-run)
9. [License](#-license)

## 🎯 Problem Definition
The central question this project aims to answer is:
> Given a patient's clinical parameters, can we build a machine learning model to accurately predict whether or not they have heart disease?

## 📊 Data Source
The original data was sourced from the Cleveland database at the UCI Machine Learning Repository. A version is also available on Kaggle.
* **UCI Machine Learning Repository:** [Heart Disease Data Set](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
* **Kaggle:** [Heart Disease Classification Dataset](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)

## 📈 Evaluation Metric
The primary goal for this proof-of-concept is to achieve **95% accuracy** in predicting the presence of heart disease.

## 📝 Features & Data Dictionary
The dataset consists of 13 input features and 1 target variable.

| Feature    | Description                                           | Notes                                                      |
| :--------- | :---------------------------------------------------- | :--------------------------------------------------------- |
| **age** | Age in years                                          |                                                            |
| **sex** | (1 = male; 0 = female)                                |                                                            |
| **cp** | Chest Pain Type                                       | 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic   |
| **trestbps** | Resting blood pressure (mm Hg)                        | > 130-140 is cause for concern                             |
| **chol** | Serum cholesterol (mg/dl)                             | > 200 is cause for concern                                 |
| **fbs** | Fasting blood sugar > 120 mg/dl                       | (1 = true; 0 = false)                                      |
| **restecg**| Resting electrocardiographic results                | 0: Normal, 1: ST-T wave abnormality, 2: Hypertrophy        |
| **thalach**| Maximum heart rate achieved                           |                                                            |
| **exang** | Exercise-induced angina                               | (1 = yes; 0 = no)                                          |
| **oldpeak**| ST depression induced by exercise relative to rest    |                                                            |
| **slope** | Slope of the peak exercise ST segment                 | 0: Upsloping, 1: Flatsloping, 2: Downsloping               |
| **ca** | Number of major vessels (0-3) colored by fluoroscopy  |                                                            |
| **thal** | Thallium stress result                                | 1/3: Normal, 6: Fixed defect, 7: Reversible defect         |
| **target** | **Has heart disease** | **(1 = yes; 0 = no)** - **The Target Variable** |

## 🛠️ Project Workflow

The project followed a structured machine learning pipeline:

### 1. Exploratory Data Analysis (EDA)
The dataset was thoroughly examined to understand its structure. This included checking for missing values (none were found), analyzing the class balance of the target variable, and creating visualizations to explore relationships between features.

### 2. Model Selection and Baseline Results
To identify the most promising algorithm, three different models were initially trained with their default parameters to establish a fair baseline.

| Model | Baseline Accuracy |
| :--- | :--- |
| **Logistic Regression** | **83.61%** |
| Random Forest Classifier | 81.97% |
| K-Nearest Neighbors (KNN) | 63.93% |

Based on these results, the K-Nearest Neighbors model was eliminated due to significantly lower performance. **Logistic Regression** was selected as the primary candidate for in-depth hyperparameter tuning.

### 3. Hyperparameter Tuning
The selected models (Logistic Regression and Random Forest) were systematically tuned using `RandomizedSearchCV` and `GridSearchCV` to find the optimal combination of hyperparameters that maximized performance.

### 4. Final Model Evaluation
The best-performing model (tuned Logistic Regression) was evaluated comprehensively using multiple metrics, including a ROC curve, confusion matrix, classification report, and robust cross-validated scores.

### 5. Feature Importance
The `coef_` attribute of the final Logistic Regression model was used to determine which medical features were most predictive of heart disease.

## 🏆 Final Results

After extensive tuning and evaluation, the **Logistic Regression** model was finalized. The model's performance, validated using 5-fold cross-validation, is robust and provides a strong predictive signal.

The final cross-validated metrics for the tuned model were:

| Metric    | Score         |
| :-------- | :------------ |
| Accuracy  | 84.48%        |
| Precision | 82.08%        |
| Recall    | 92.12%        |
| F1-Score  | 86.73%        |

The model's performance on the test set is visualized in the confusion matrix below:

![Confusion Matrix](https://i.imgur.com/l7xjn3I.png)

## 🚀 Technologies Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-313131?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3274A1?style=for-the-badge)

## 💻 How to Run
To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder-name>
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(If no `requirements.txt` is provided, install: `pip install numpy pandas scikit-learn matplotlib seaborn jupyter`)*

3.  **Download the dataset:**
    The `heart-disease.csv` file should be located in a `data/` subdirectory. If not, download it from the Kaggle link provided above.

4.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open the main notebook file (`.ipynb`) and run the cells to see the complete analysis.
