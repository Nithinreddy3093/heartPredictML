Heart Disease Risk Prediction System

Machine Learning for Healthcare Analytics
Python · Pandas · Scikit-Learn · Matplotlib · (Optional: Streamlit / Flask)

🚀 Project Overview

A simple, resume-friendly ML project that predicts the risk of heart disease using clinical features.
This repository contains a full pipeline: data ingestion → preprocessing → model training → evaluation → a lightweight prediction interface. Ideal to demonstrate healthcare interest and ML engineering skills for internships like Evernorth.

Resume-ready one-liner:
Heart Disease Risk Prediction System — Built an ML pipeline (data cleaning, feature engineering, model training, and a simple UI) using the UCI Heart Disease dataset to predict patient risk, achieving ~85% accuracy and producing explainable feature insights.

📁 Repository Structure (suggested)
heart-disease-prediction/
├── data/
│   └── heart.csv                  # (or script to download dataset)
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app/
│   └── app.py                     # Streamlit or Flask app
├── requirements.txt
├── README.md
└── LICENSE

📥 Dataset

Source: UCI Heart Disease dataset (commonly the Cleveland dataset).
Columns typically include: age, sex, cp (chest pain), trestbps (resting blood pressure), chol, fbs, restecg, thalach (max heart rate), exang, oldpeak, slope, ca, thal, target.

NOTE: If using a preprocessed CSV (data/heart.csv), include a small data/README.md describing provenance and preprocessing steps.

🧰 Tech Stack

Python 3.8+

pandas, numpy

scikit-learn

matplotlib / seaborn (visualization)

joblib (model persistence)

Optional: streamlit or Flask for UI

requirements.txt (example)

pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit    # optional
flask        # optional, choose one

⚙️ Key Components & Workflow

Data Loading — load CSV from data/ or script to fetch UCI dataset.

Exploratory Data Analysis — summary stats, missing values, feature correlations, visualizations.

Preprocessing — encoding categorical features, imputation (if needed), scaling (StandardScaler).

Feature Selection — correlation check, (optional) recursive feature elimination.

Modeling — Logistic Regression and Random Forest (baseline + stronger).

Evaluation — Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC.

Persistence — save trained model with joblib.dump.

Interface — simple Streamlit/Flask app to input patient data and show risk prediction + key feature importance.

📌 Quick Start — Setup & Run

Clone repo

git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction


Install dependencies

python -m venv venv
source venv/bin/activate    # mac/linux
# venv\Scripts\activate     # windows
pip install -r requirements.txt


Run Jupyter Notebook (EDA & training)

jupyter notebook notebooks/EDA_and_Modeling.ipynb


Train model from script

python src/train.py --data data/heart.csv --output models/heart_model.joblib


Run Streamlit demo (optional)

streamlit run app/app.py


Use Flask API (optional)

export FLASK_APP=app/app.py
flask run
# Then call POST /predict with JSON

🧪 Example: Train & Evaluate (high-level)

Split data: train_test_split(test_size=0.2, random_state=42)

Models tried: LogisticRegression, RandomForestClassifier

Typical result (example):

Accuracy: ~85%

Precision / Recall / F1: reported in notebooks/ after training
(Actual numbers vary by preprocessing and train/test split — include your final numbers in the notebook & README.)

🔍 Sample Prediction (curl)

If you expose a Flask API POST /predict that accepts JSON:

curl -X POST http://127.0.0.1:5000/predict \
 -H "Content-Type: application/json" \
 -d '{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
  "slope": 0, "ca": 0, "thal": 1
 }'


Response example:

{ "risk": "High", "probability": 0.87 }

✅ What to Highlight on Your Resume

Project title with tech stack: Heart Disease Risk Prediction System — Python, Scikit-Learn, Streamlit

One-line impact: Built an explainable ML model for heart disease risk scoring using clinical features (UCI dataset); achieved ~85% accuracy and deployed a minimal predictive interface.

Bullet points: data preprocessing, model(s) used, evaluation metrics, and domain relevance (healthcare analytics, risk scoring).

📈 Visuals to Include in Repo / Presentation

Correlation heatmap (features vs target)

ROC curve & AUC for the final model

Confusion matrix

Feature importance (for RandomForest)

Example prediction screenshots (if UI included)

♻️ Extensions & Future Work (resume talking points)

Use larger, real-world healthcare datasets (MIMIC, if available with proper approvals)

Build time-series patient risk models (if longitudinal data available)

Add model explainability: SHAP or LIME for per-prediction explanations

Deploy model to cloud (Heroku / AWS / Azure) with CI/CD pipeline

Integrate basic privacy-conscious measures (masking PHI, encrypting data in transit)

📝 Notes on Ethics & Data Privacy

This is a demo/educational project. Do not use it for real medical decision-making.

If working with real patient data, follow legal and ethical rules (HIPAA/GDPR equivalents, institutional review, anonymization).

📬 Contact / Attribution

Built by: Your Name — include GitHub link and email

Dataset: UCI Heart Disease dataset (Cleveland) — cite appropriately in data/README.md
