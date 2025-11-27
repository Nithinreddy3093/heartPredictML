# Heart Disease Risk Prediction System

## Overview
A machine learning-powered healthcare analytics application that predicts heart disease risk based on patient medical parameters. Built with Python, Scikit-learn, and Streamlit.

## Project Structure
```
├── app.py                  # Main Streamlit application
├── model.py                # ML model training and prediction module
├── data/
│   └── heart.csv           # UCI Heart Disease dataset
├── patient_history.json    # Patient prediction history storage
├── .streamlit/
│   └── config.toml         # Streamlit configuration
└── pyproject.toml          # Python dependencies
```

## Features

### Core Features
- **Data Exploration**: Visualize dataset statistics, distributions, and correlations
- **Model Training**: Train Logistic Regression and Random Forest classifiers
- **Risk Prediction**: Interactive patient data input for heart disease risk assessment
- **Model Comparison**: Side-by-side comparison of model performance metrics

### Advanced Features
- **Cross-Validation**: 5-fold stratified cross-validation for robust model evaluation
- **ROC Curves**: Receiver Operating Characteristic curves with AUC scores
- **Patient History**: Track and save prediction records with file-based persistence
- **Report Export**: Download prediction reports as text files

## Tech Stack
- Python 3.11
- Streamlit (Web UI)
- Pandas & NumPy (Data processing)
- Scikit-learn (Machine Learning)
- Matplotlib & Seaborn (Visualization)

## Dataset
Uses the UCI Heart Disease dataset with 13 features:
- age, sex, chest pain type, resting blood pressure
- cholesterol, fasting blood sugar, resting ECG
- max heart rate, exercise induced angina
- ST depression, slope, major vessels, thalassemia

## Running the Application
The application runs on port 5000 using Streamlit:
```bash
streamlit run app.py --server.port 5000
```

## Pages
1. **Home**: Overview and key metrics
2. **Data Exploration**: Dataset visualization and statistics
3. **Model Training**: Model performance metrics and feature importance
4. **Cross-Validation**: 5-fold CV results with fold-by-fold breakdown
5. **ROC Curves**: ROC curve visualization and AUC comparison
6. **Risk Prediction**: Patient input form with risk assessment
7. **Patient History**: Saved prediction records
8. **Model Comparison**: Side-by-side model comparison

## Recent Changes
- Initial implementation (Nov 2025)
- Data preprocessing with StandardScaler
- Logistic Regression and Random Forest models
- Interactive prediction interface with probability scores
- Comprehensive visualization dashboard
- Cross-validation with proper pipeline isolation
- ROC curves and AUC score visualization
- Patient history tracking with JSON persistence
- Downloadable prediction reports
