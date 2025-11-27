import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from datetime import datetime

FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
TARGET_NAME = 'target'

FEATURE_DESCRIPTIONS = {
    'age': 'Age (years)',
    'sex': 'Sex (0=Female, 1=Male)',
    'cp': 'Chest Pain Type (0-3)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)',
    'restecg': 'Resting ECG Results (0-2)',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (0=No, 1=Yes)',
    'oldpeak': 'ST Depression Induced by Exercise',
    'slope': 'Slope of Peak Exercise ST Segment (0-2)',
    'ca': 'Number of Major Vessels Colored by Fluoroscopy (0-4)',
    'thal': 'Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Unknown)'
}

class HeartDiseasePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict] = {}
        self.cv_scores: Dict[str, Dict] = {}
        self.roc_data: Dict[str, Dict] = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_train_scaled: Optional[np.ndarray] = None
        self.X_test_scaled: Optional[np.ndarray] = None
        self.data: Optional[pd.DataFrame] = None
        
    def load_data(self, filepath: str = 'data/heart.csv') -> pd.DataFrame:
        self.data = pd.read_csv(filepath)
        return self.data
    
    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        X = self.data[FEATURE_NAMES]
        y = self.data[TARGET_NAME]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self) -> Tuple:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = model
        
        y_pred = model.predict(self.X_test_scaled)
        y_prob = model.predict_proba(self.X_test_scaled)[:, 1]
        
        self.metrics['Logistic Regression'] = self._calculate_metrics(y_pred, y_prob)
        self._calculate_cv_scores('Logistic Regression', model)
        self._calculate_roc_data('Logistic Regression', y_prob)
        
        return model, self.metrics['Logistic Regression']
    
    def train_random_forest(self, n_estimators: int = 100) -> Tuple:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['Random Forest'] = model
        
        y_pred = model.predict(self.X_test_scaled)
        y_prob = model.predict_proba(self.X_test_scaled)[:, 1]
        
        self.metrics['Random Forest'] = self._calculate_metrics(y_pred, y_prob)
        self._calculate_cv_scores('Random Forest', model)
        self._calculate_roc_data('Random Forest', y_prob)
        
        return model, self.metrics['Random Forest']
    
    def _calculate_metrics(self, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0.0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0.0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0.0),
            'auc_roc': roc_auc_score(self.y_test, y_prob),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def _calculate_cv_scores(self, model_name: str, model) -> None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        if self.data is None:
            return
            
        X = self.data[FEATURE_NAMES]
        y = self.data[TARGET_NAME]
        
        if model_name == 'Logistic Regression':
            cv_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        else:
            cv_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
        
        cv_accuracy = cross_val_score(cv_pipeline, X, y, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(cv_pipeline, X, y, cv=cv, scoring='precision')
        cv_recall = cross_val_score(cv_pipeline, X, y, cv=cv, scoring='recall')
        cv_f1 = cross_val_score(cv_pipeline, X, y, cv=cv, scoring='f1')
        cv_roc_auc = cross_val_score(cv_pipeline, X, y, cv=cv, scoring='roc_auc')
        
        self.cv_scores[model_name] = {
            'accuracy': {'mean': cv_accuracy.mean(), 'std': cv_accuracy.std(), 'scores': cv_accuracy.tolist()},
            'precision': {'mean': cv_precision.mean(), 'std': cv_precision.std(), 'scores': cv_precision.tolist()},
            'recall': {'mean': cv_recall.mean(), 'std': cv_recall.std(), 'scores': cv_recall.tolist()},
            'f1_score': {'mean': cv_f1.mean(), 'std': cv_f1.std(), 'scores': cv_f1.tolist()},
            'roc_auc': {'mean': cv_roc_auc.mean(), 'std': cv_roc_auc.std(), 'scores': cv_roc_auc.tolist()}
        }
    
    def _calculate_roc_data(self, model_name: str, y_prob: np.ndarray) -> None:
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        self.roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    def predict(self, model_name: str, input_data: List) -> Tuple[int, np.ndarray]:
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        input_scaled = self.scaler.transform(input_df)
        
        model = self.models[model_name]
        prediction = int(model.predict(input_scaled)[0])
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_scaled)[0]
        else:
            probability = np.array([1 - prediction, prediction])
        
        return prediction, probability
    
    def get_feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if model_name == 'Random Forest':
            importance = model.feature_importances_
        elif model_name == 'Logistic Regression':
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        importance_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def get_data_summary(self) -> Optional[Dict]:
        if self.data is None:
            return None
        
        summary = {
            'total_samples': len(self.data),
            'features': len(FEATURE_NAMES),
            'positive_cases': int(self.data[TARGET_NAME].sum()),
            'negative_cases': int(len(self.data) - self.data[TARGET_NAME].sum()),
            'positive_rate': float(self.data[TARGET_NAME].mean() * 100)
        }
        return summary
    
    def get_cv_scores(self, model_name: str) -> Optional[Dict]:
        return self.cv_scores.get(model_name)
    
    def get_roc_data(self, model_name: str) -> Optional[Dict]:
        return self.roc_data.get(model_name)


class PatientHistoryManager:
    def __init__(self, history_file: str = 'patient_history.json'):
        self.history_file = history_file
        self.history: List[Dict] = []
        self._load_history()
    
    def _load_history(self) -> None:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.history = []
        else:
            self.history = []
    
    def _save_history(self) -> None:
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_prediction(self, patient_data: Dict, prediction: int, probability: List[float], model_name: str) -> Dict:
        record = {
            'id': len(self.history) + 1,
            'timestamp': datetime.now().isoformat(),
            'patient_data': patient_data,
            'prediction': prediction,
            'risk_probability': probability[1],
            'model_used': model_name,
            'risk_level': 'HIGH' if prediction == 1 else 'LOW'
        }
        self.history.append(record)
        self._save_history()
        return record
    
    def get_all_history(self) -> List[Dict]:
        return self.history
    
    def get_history_df(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        
        records = []
        for h in self.history:
            record = {
                'ID': h['id'],
                'Timestamp': h['timestamp'],
                'Model': h['model_used'],
                'Risk Level': h['risk_level'],
                'Risk Probability': f"{h['risk_probability']*100:.1f}%",
                'Age': h['patient_data'].get('age', 'N/A'),
                'Sex': 'Male' if h['patient_data'].get('sex', 0) == 1 else 'Female',
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def clear_history(self) -> None:
        self.history = []
        self._save_history()
    
    def get_record_by_id(self, record_id: int) -> Optional[Dict]:
        for h in self.history:
            if h['id'] == record_id:
                return h
        return None


def get_predictor() -> HeartDiseasePredictor:
    predictor = HeartDiseasePredictor()
    predictor.load_data()
    predictor.preprocess_data()
    predictor.train_logistic_regression()
    predictor.train_random_forest()
    return predictor


def get_history_manager() -> PatientHistoryManager:
    return PatientHistoryManager()
