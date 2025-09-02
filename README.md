# Heart Disease Prediction Using Machine Learning

A machine learning project that predicts the likelihood of heart disease in patients based on various clinical and demographic features. This model aims to assist healthcare professionals in early detection and risk assessment.

## 🎯 Project Overview

Heart disease remains one of the leading causes of death worldwide. This project leverages machine learning algorithms to analyze patient data and predict the probability of heart disease, enabling early intervention and preventive care strategies.

## 📊 Dataset

The model is trained on a comprehensive dataset containing the following features:

- **Age**: Patient's age in years
- **Sex**: Gender (1 = male, 0 = female)
- **Chest Pain Type**: Type of chest pain experienced
- **Resting Blood Pressure**: Blood pressure at rest (mmHg)
- **Cholesterol**: Serum cholesterol level (mg/dl)
- **Fasting Blood Sugar**: Blood sugar level after fasting
- **Resting ECG**: Electrocardiogram results at rest
- **Max Heart Rate**: Maximum heart rate achieved during exercise
- **Exercise Induced Angina**: Presence of exercise-induced chest pain
- **ST Depression**: Exercise-induced ST depression
- **Slope**: Slope of peak exercise ST segment
- **Number of Major Vessels**: Vessels colored by fluoroscopy
- **Thalassemia**: Blood disorder indicator

**Target Variable**: Presence of heart disease (1 = disease, 0 = no disease)

## 🚀 Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Exploratory Data Analysis**: Detailed visualization and statistical analysis
- **Multiple ML Algorithms**: Implementation of various classification algorithms
- **Model Evaluation**: Performance comparison using multiple metrics
- **Feature Importance**: Analysis of most predictive features
- **Cross-Validation**: Robust model validation techniques
- **Hyperparameter Tuning**: Optimized model parameters for best performance

## 🔧 Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## 📈 Model Performance

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|---------|----------|
| Random Forest | 85.2% | 0.87 | 0.83 | 0.85 |
| Logistic Regression | 82.1% | 0.84 | 0.80 | 0.82 |
| SVM | 83.7% | 0.85 | 0.82 | 0.83 |
| Gradient Boosting | 86.1% | 0.88 | 0.84 | 0.86 |

*Best performing model: **Gradient Boosting** with 86.1% accuracy*

## 🗂️ Repository Structure

```
heart-disease-prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and preprocessed data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── models/
│   ├── best_model.pkl
│   └── model_comparison.json
│
├── visualizations/
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   └── roc_curves.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebooks:
```bash
jupyter notebook
```

### Usage

1. **Data Exploration**: Start with `01_data_exploration.ipynb` to understand the dataset
2. **Preprocessing**: Run `02_data_preprocessing.ipynb` to clean and prepare the data
3. **Training**: Execute `03_model_training.ipynb` to train multiple ML models
4. **Evaluation**: Use `04_model_evaluation.ipynb` to compare model performance

### Making Predictions

```python
import pickle
import pandas as pd

# Load the trained model
with open('models/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Prepare your data (example)
patient_data = pd.DataFrame({
    'age': [45],
    'sex': [1],
    'chest_pain_type': [2],
    'resting_bp': [120],
    'cholesterol': [240],
    # ... add other features
})

# Make prediction
prediction = model.predict(patient_data)
probability = model.predict_proba(patient_data)

print(f"Heart Disease Risk: {prediction[0]}")
print(f"Probability: {probability[0][1]:.2%}")
```

## 📊 Key Insights

- **Most Important Features**: Chest pain type, maximum heart rate, and ST depression are the strongest predictors
- **Age Factor**: Risk increases significantly after age 50
- **Gender Differences**: Males show higher risk patterns in the dataset
- **Exercise Correlation**: Patients with exercise-induced angina have higher disease probability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

- **Author**: [Saumya Khushlani]
- **Email**: [saumyakhushlani9@gmail.com]

## 🙏 Acknowledgments

- Dataset source and contributors
- Open-source community for tools and libraries
- Healthcare professionals who provided domain insights

---

⭐ If you found this project helpful, please give it a star!
