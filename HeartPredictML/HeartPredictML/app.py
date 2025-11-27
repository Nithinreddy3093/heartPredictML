import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime
from model import (
    HeartDiseasePredictor, FEATURE_NAMES, FEATURE_DESCRIPTIONS, 
    get_predictor, get_history_manager, PatientHistoryManager
)

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_pdf_report(patient_data, prediction, probability, model_name, risk_level):
    report = f"""
================================================================================
                    HEART DISEASE RISK PREDICTION REPORT
================================================================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Used: {model_name}

--------------------------------------------------------------------------------
                              PREDICTION RESULT
--------------------------------------------------------------------------------

Risk Level: {risk_level}
Risk Probability: {probability[1]*100:.1f}%
No Disease Probability: {probability[0]*100:.1f}%

--------------------------------------------------------------------------------
                              PATIENT INFORMATION
--------------------------------------------------------------------------------

"""
    for feature, value in patient_data.items():
        desc = FEATURE_DESCRIPTIONS.get(feature, feature)
        report += f"{desc}: {value}\n"
    
    report += f"""
--------------------------------------------------------------------------------
                                 DISCLAIMER
--------------------------------------------------------------------------------

This prediction is for educational purposes only and should not be used 
as a substitute for professional medical advice. Always consult with a 
healthcare professional for medical decisions.

================================================================================
                        END OF REPORT
================================================================================
"""
    return report

@st.cache_resource
def load_predictor():
    return get_predictor()

@st.cache_resource
def load_history_manager():
    return get_history_manager()

predictor = load_predictor()
history_manager = load_history_manager()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Data Exploration", "Model Training", "Cross-Validation", "ROC Curves", "Risk Prediction", "Patient History", "Model Comparison"]
)

if page == "Home":
    st.title("Heart Disease Risk Prediction System")
    st.markdown("### Machine Learning for Healthcare Analytics")
    
    st.markdown("""
    This system uses machine learning algorithms to predict the risk of heart disease 
    based on various medical parameters. The prediction models are trained on the 
    UCI Heart Disease dataset.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    summary = predictor.get_data_summary()
    
    with col1:
        st.metric("Total Patients", summary['total_samples'])
    with col2:
        st.metric("Features Analyzed", summary['features'])
    with col3:
        st.metric("Heart Disease Rate", f"{summary['positive_rate']:.1f}%")
    
    st.markdown("---")
    
    st.markdown("### Key Features Analyzed")
    
    col1, col2 = st.columns(2)
    
    features_list = list(FEATURE_DESCRIPTIONS.items())
    mid = len(features_list) // 2
    
    with col1:
        for feature, desc in features_list[:mid]:
            st.markdown(f"- **{feature}**: {desc}")
    
    with col2:
        for feature, desc in features_list[mid:]:
            st.markdown(f"- **{feature}**: {desc}")
    
    st.markdown("---")
    
    st.markdown("### How It Works")
    st.markdown("""
    1. **Data Collection**: Patient medical data is collected including age, blood pressure, cholesterol levels, etc.
    2. **Data Preprocessing**: Features are scaled and normalized for optimal model performance.
    3. **Model Training**: Multiple ML algorithms (Logistic Regression, Random Forest) are trained on the dataset.
    4. **Risk Prediction**: New patient data can be input to predict heart disease risk with probability scores.
    """)

elif page == "Data Exploration":
    st.title("Data Exploration")
    st.markdown("### Understanding the Heart Disease Dataset")
    
    data = predictor.data
    
    st.subheader("Dataset Overview")
    st.dataframe(data.head(20))
    
    st.subheader("Statistical Summary")
    st.dataframe(data.describe())
    
    st.markdown("---")
    
    st.subheader("Target Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        target_counts = data['target'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(target_counts, labels=['No Heart Disease', 'Heart Disease'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Heart Disease Distribution')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        target_counts = data['target'].value_counts().sort_index()
        ax.bar(['No Heart Disease', 'Heart Disease'], target_counts.values, color=colors)
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel('Count')
        ax.set_title('Heart Disease Count')
        for i, v in enumerate(target_counts.values):
            ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    st.subheader("Feature Distributions")
    
    selected_feature = st.selectbox("Select a feature to visualize", FEATURE_NAMES)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for target_val, color, label in [(0, '#2ecc71', 'No Heart Disease'), (1, '#e74c3c', 'Heart Disease')]:
            subset = data[data['target'] == target_val][selected_feature]
            ax.hist(subset, alpha=0.6, color=color, label=label, bins=20)
        ax.set_title(f'Distribution of {FEATURE_DESCRIPTIONS[selected_feature]}')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        data_no_disease = data[data['target'] == 0][selected_feature]
        data_disease = data[data['target'] == 1][selected_feature]
        
        bp = ax.boxplot([data_no_disease, data_disease], 
                       labels=['No Heart Disease', 'Heart Disease'],
                       patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_title(f'{FEATURE_DESCRIPTIONS[selected_feature]} by Target')
        ax.set_ylabel(selected_feature)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    st.subheader("Feature Correlation Heatmap")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.2f', ax=ax, square=True)
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    plt.close()

elif page == "Model Training":
    st.title("Model Training & Evaluation")
    st.markdown("### Training Machine Learning Models")
    
    summary = predictor.get_data_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", int(summary['total_samples'] * 0.8))
    with col2:
        st.metric("Testing Samples", int(summary['total_samples'] * 0.2))
    with col3:
        st.metric("Features", summary['features'])
    with col4:
        st.metric("Target Classes", 2)
    
    st.markdown("---")
    
    model_choice = st.selectbox(
        "Select Model to View",
        ["Logistic Regression", "Random Forest"]
    )
    
    metrics = predictor.metrics[model_choice]
    
    st.subheader(f"{model_choice} Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
    with col5:
        st.metric("AUC-ROC", f"{metrics['auc_roc']:.2%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_choice} - Confusion Matrix')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Feature Importance")
        importance_df = predictor.get_feature_importance(model_choice)
        if importance_df is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors_gradient = plt.cm.get_cmap('RdYlGn')(np.linspace(0.2, 0.8, len(importance_df)))[::-1]
            ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors_gradient)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_choice} - Feature Importance')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    st.subheader("Model Interpretation")
    
    if model_choice == "Logistic Regression":
        st.markdown("""
        **Logistic Regression** is a statistical model that uses a logistic function to model 
        a binary dependent variable. In this case, it predicts the probability of heart disease 
        based on the input features.
        
        - **Pros**: Interpretable, fast training, works well with linearly separable data
        - **Cons**: May underperform with complex non-linear relationships
        """)
    else:
        st.markdown("""
        **Random Forest** is an ensemble learning method that constructs multiple decision trees 
        during training and outputs the mode of the classes for classification.
        
        - **Pros**: Handles non-linear relationships, robust to overfitting, provides feature importance
        - **Cons**: Less interpretable than logistic regression, can be slower with many trees
        """)

elif page == "Cross-Validation":
    st.title("Cross-Validation Results")
    st.markdown("### 5-Fold Stratified Cross-Validation")
    
    st.markdown("""
    Cross-validation provides a more robust estimate of model performance by training and 
    evaluating the model on different subsets of the data. This helps identify if the model 
    is overfitting or underfitting.
    """)
    
    st.markdown("---")
    
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest"]
    )
    
    cv_scores = predictor.get_cv_scores(model_choice)
    
    if cv_scores:
        st.subheader(f"{model_choice} - Cross-Validation Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_list = [
            ('accuracy', 'Accuracy', col1),
            ('precision', 'Precision', col2),
            ('recall', 'Recall', col3),
            ('f1_score', 'F1 Score', col4),
            ('roc_auc', 'AUC-ROC', col5)
        ]
        
        for metric_key, metric_name, col in metrics_list:
            with col:
                mean_val = cv_scores[metric_key]['mean']
                std_val = cv_scores[metric_key]['std']
                st.metric(
                    metric_name,
                    f"{mean_val:.2%}",
                    delta=f"+/- {std_val:.2%}"
                )
        
        st.markdown("---")
        
        st.subheader("Fold-by-Fold Performance")
        
        fold_data = []
        for i in range(5):
            fold_row = {'Fold': f'Fold {i+1}'}
            for metric_key, metric_name, _ in metrics_list:
                fold_row[metric_name] = f"{cv_scores[metric_key]['scores'][i]:.2%}"
            fold_data.append(fold_row)
        
        mean_row = {'Fold': 'Mean'}
        std_row = {'Fold': 'Std Dev'}
        for metric_key, metric_name, _ in metrics_list:
            mean_row[metric_name] = f"{cv_scores[metric_key]['mean']:.2%}"
            std_row[metric_name] = f"{cv_scores[metric_key]['std']:.2%}"
        fold_data.append(mean_row)
        fold_data.append(std_row)
        
        fold_df = pd.DataFrame(fold_data)
        st.dataframe(fold_df, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("Cross-Validation Score Distribution")
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        for idx, (metric_key, metric_name, _) in enumerate(metrics_list):
            scores = cv_scores[metric_key]['scores']
            mean_score = cv_scores[metric_key]['mean']
            
            axes[idx].bar(range(1, 6), scores, color='#3498db', alpha=0.7)
            axes[idx].axhline(y=mean_score, color='#e74c3c', linestyle='--', linewidth=2, label='Mean')
            axes[idx].set_xlabel('Fold')
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(metric_name)
            axes[idx].set_ylim(0, 1)
            axes[idx].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        st.subheader("Model Comparison - Cross-Validation")
        
        comparison_data = []
        for model_name in ["Logistic Regression", "Random Forest"]:
            model_cv = predictor.get_cv_scores(model_name)
            if model_cv:
                row = {'Model': model_name}
                for metric_key, metric_name, _ in metrics_list:
                    row[metric_name] = f"{model_cv[metric_key]['mean']:.2%} (+/- {model_cv[metric_key]['std']:.2%})"
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True)

elif page == "ROC Curves":
    st.title("ROC Curves & AUC Analysis")
    st.markdown("### Receiver Operating Characteristic Curves")
    
    st.markdown("""
    The ROC curve illustrates the diagnostic ability of a binary classifier system. 
    The AUC (Area Under the Curve) provides an aggregate measure of performance across 
    all possible classification thresholds.
    """)
    
    st.markdown("---")
    
    st.subheader("ROC Curve Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'Logistic Regression': '#3498db', 'Random Forest': '#2ecc71'}
    
    for model_name in ["Logistic Regression", "Random Forest"]:
        roc_data = predictor.get_roc_data(model_name)
        if roc_data:
            ax.plot(roc_data['fpr'], roc_data['tpr'], 
                   color=colors[model_name], 
                   linewidth=2,
                   label=f"{model_name} (AUC = {roc_data['auc']:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    st.subheader("AUC Scores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lr_roc = predictor.get_roc_data("Logistic Regression")
        if lr_roc:
            st.metric("Logistic Regression AUC", f"{lr_roc['auc']:.4f}")
    
    with col2:
        rf_roc = predictor.get_roc_data("Random Forest")
        if rf_roc:
            st.metric("Random Forest AUC", f"{rf_roc['auc']:.4f}")
    
    st.markdown("---")
    
    st.subheader("Individual ROC Curves")
    
    col1, col2 = st.columns(2)
    
    for col, model_name, color in [(col1, "Logistic Regression", '#3498db'), 
                                    (col2, "Random Forest", '#2ecc71')]:
        with col:
            st.markdown(f"#### {model_name}")
            roc_data = predictor.get_roc_data(model_name)
            if roc_data:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.fill_between(roc_data['fpr'], roc_data['tpr'], alpha=0.3, color=color)
                ax.plot(roc_data['fpr'], roc_data['tpr'], color=color, linewidth=2)
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'AUC = {roc_data["auc"]:.4f}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
    
    st.markdown("---")
    
    st.subheader("Understanding ROC Curves")
    
    st.markdown("""
    **Key Metrics:**
    - **AUC = 1.0**: Perfect classifier
    - **AUC = 0.5**: Random classifier (no discrimination)
    - **AUC > 0.9**: Excellent discrimination
    - **AUC 0.8-0.9**: Good discrimination
    - **AUC 0.7-0.8**: Fair discrimination
    - **AUC < 0.7**: Poor discrimination
    
    **Interpretation:**
    - The closer the ROC curve follows the left border and then the top border, the better
    - A larger AUC indicates better model performance
    - The diagonal line represents a random classifier with no predictive power
    """)

elif page == "Risk Prediction":
    st.title("Heart Disease Risk Prediction")
    st.markdown("### Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (years)", 20, 100, 50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                                "Non-anginal Pain", "Asymptomatic"][x])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                          format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                              format_func=lambda x: ["Normal", "ST-T Abnormality", 
                                                     "Left Ventricular Hypertrophy"][x])
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col3:
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                           format_func=lambda x: ["Normal", "Fixed Defect", 
                                                  "Reversible Defect", "Unknown"][x])
    
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    patient_data = dict(zip(FEATURE_NAMES, input_data))
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_choice = st.radio("Select Prediction Model", 
                               ["Logistic Regression", "Random Forest"],
                               horizontal=True)
    
    with col2:
        save_to_history = st.checkbox("Save to History", value=True)
    
    if st.button("Predict Risk", type="primary", use_container_width=True):
        prediction, probability = predictor.predict(model_choice, input_data)
        
        if save_to_history:
            history_manager.add_prediction(patient_data, prediction, probability.tolist(), model_choice)
            st.cache_resource.clear()
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("### High Risk of Heart Disease")
                risk_level = "HIGH"
                risk_color = "#e74c3c"
            else:
                st.success("### Low Risk of Heart Disease")
                risk_level = "LOW"
                risk_color = "#2ecc71"
        
        with col2:
            risk_prob = probability[1] * 100
            st.metric("Risk Probability", f"{risk_prob:.1f}%")
        
        st.markdown("---")
        
        st.subheader("Risk Probability Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh(['Risk Level'], [probability[1]], color=risk_color, height=0.5)
            ax.barh(['Risk Level'], [probability[0]], left=[probability[1]], color='#95a5a6', height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
            ax.legend(['Heart Disease Risk', 'No Heart Disease'], loc='upper right')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Probability Breakdown")
            st.markdown(f"- **Heart Disease**: {probability[1]*100:.1f}%")
            st.markdown(f"- **No Heart Disease**: {probability[0]*100:.1f}%")
        
        st.markdown("---")
        
        st.subheader("Patient Input Summary")
        
        input_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Value': input_data,
            'Description': [FEATURE_DESCRIPTIONS[f] for f in FEATURE_NAMES]
        })
        st.dataframe(input_df, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("Download Report")
        
        report_content = generate_pdf_report(patient_data, prediction, probability, model_choice, risk_level)
        
        st.download_button(
            label="Download Prediction Report (TXT)",
            data=report_content,
            file_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        st.info("""
        **Disclaimer**: This prediction is for educational purposes only and should not be used 
        as a substitute for professional medical advice. Always consult with a healthcare 
        professional for medical decisions.
        """)

elif page == "Patient History":
    st.title("Patient History")
    st.markdown("### Prediction History Tracking")
    
    history_df = history_manager.get_history_df()
    
    if history_df.empty:
        st.info("No predictions have been saved yet. Make a prediction and enable 'Save to History' to start tracking.")
    else:
        st.subheader("Saved Predictions")
        st.dataframe(history_df, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("Risk Distribution in History")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = history_df['Risk Level'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71' if level == 'LOW' else '#e74c3c' for level in risk_counts.index]
            ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Risk Level Distribution')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Summary Statistics")
            total = len(history_df)
            high_risk = len(history_df[history_df['Risk Level'] == 'HIGH'])
            low_risk = len(history_df[history_df['Risk Level'] == 'LOW'])
            
            st.metric("Total Predictions", total)
            st.metric("High Risk Cases", high_risk)
            st.metric("Low Risk Cases", low_risk)
        
        st.markdown("---")
        
        st.subheader("View Detailed Record")
        
        record_ids = [h['id'] for h in history_manager.get_all_history()]
        if record_ids:
            selected_id = st.selectbox("Select Record ID", record_ids)
            
            if st.button("View Details"):
                record = history_manager.get_record_by_id(selected_id)
                if record:
                    st.markdown(f"**Timestamp:** {record['timestamp']}")
                    st.markdown(f"**Model Used:** {record['model_used']}")
                    st.markdown(f"**Risk Level:** {record['risk_level']}")
                    st.markdown(f"**Risk Probability:** {record['risk_probability']*100:.1f}%")
                    
                    st.markdown("**Patient Data:**")
                    patient_df = pd.DataFrame({
                        'Feature': list(record['patient_data'].keys()),
                        'Value': list(record['patient_data'].values()),
                        'Description': [FEATURE_DESCRIPTIONS.get(f, f) for f in record['patient_data'].keys()]
                    })
                    st.dataframe(patient_df, hide_index=True)
        
        st.markdown("---")
        
        if st.button("Clear All History", type="secondary"):
            history_manager.clear_history()
            st.cache_resource.clear()
            st.rerun()

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("### Comparing Machine Learning Models")
    
    models = list(predictor.metrics.keys())
    
    comparison_data = []
    for model in models:
        metrics = predictor.metrics[model]
        comparison_data.append({
            'Model': model,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'AUC-ROC': metrics['auc_roc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.subheader("Performance Metrics Comparison")
    
    styled_df = comparison_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']:
        styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(styled_df, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.15
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
        
        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            values = comparison_df[metric].values
            ax.bar(x + i*width, values, width, label=metric, color=color)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        for idx, model in enumerate(models):
            cm = predictor.metrics[model]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'])
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_title(f'{model}')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    st.subheader("Feature Importance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Logistic Regression")
        importance_lr = predictor.get_feature_importance("Logistic Regression")
        if importance_lr is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors_blue = plt.cm.get_cmap('Blues')(np.linspace(0.3, 0.9, len(importance_lr)))[::-1]
            ax.barh(importance_lr['Feature'], importance_lr['Importance'], color=colors_blue)
            ax.set_xlabel('Importance (Absolute Coefficient)')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.markdown("#### Random Forest")
        importance_rf = predictor.get_feature_importance("Random Forest")
        if importance_rf is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors_green = plt.cm.get_cmap('Greens')(np.linspace(0.3, 0.9, len(importance_rf)))[::-1]
            ax.barh(importance_rf['Feature'], importance_rf['Importance'], color=colors_green)
            ax.set_xlabel('Importance (Gini Importance)')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    st.subheader("Model Recommendation")
    
    best_model = comparison_df.loc[comparison_df['AUC-ROC'].idxmax(), 'Model']
    best_auc = comparison_df['AUC-ROC'].max()
    
    st.success(f"""
    **Recommended Model: {best_model}**
    
    Based on the evaluation metrics, **{best_model}** achieves the highest AUC-ROC of **{best_auc:.2%}** 
    on the test dataset. However, the choice of model may depend on specific requirements:
    
    - **For interpretability**: Choose Logistic Regression
    - **For handling complex patterns**: Choose Random Forest
    - **For balanced precision/recall**: Compare F1 scores
    - **For overall discrimination**: Compare AUC-ROC scores
    """)
