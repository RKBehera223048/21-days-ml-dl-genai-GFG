import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Day 4: Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Medical/Healthcare theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #ff6b6b 0%, #c44569 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        color: white;
    }
    .stMetric label {
        color: #ffffff !important;
        font-weight: 600;
    }
    h1 {
        color: #ffffff;
        font-weight: 700;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    h2 {
        color: #ffffff;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    h3 {
        color: #f0f0f0;
        font-weight: 500;
    }
    .highlight-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #ff6b6b;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .health-good {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        color: white;
    }
    .health-risk {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b6b 0%, #c44569 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load heart disease dataset
@st.cache_data
def load_heart_disease_data():
    # Create a sample heart disease dataset based on UCI Heart Disease dataset
    np.random.seed(42)
    n_samples = 920
    
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
        'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),
        'resting_bp': np.random.randint(90, 200, n_samples),
        'cholesterol': np.random.randint(126, 564, n_samples),
        'fasting_blood_sugar': np.random.choice([0, 1], n_samples),  # 0: <=120, 1: >120
        'rest_ecg': np.random.choice([0, 1, 2], n_samples),
        'max_heart_rate': np.random.randint(71, 202, n_samples),
        'exercise_angina': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples),
        'st_slope': np.random.choice([0, 1, 2], n_samples),
        'num_major_vessels': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'thalassemia': np.random.choice([0, 1, 2, 3], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on features (simplified logic)
    risk_score = (
        (df['age'] > 60) * 0.3 +
        (df['sex'] == 1) * 0.2 +
        (df['chest_pain_type'] > 0) * 0.2 +
        (df['cholesterol'] > 240) * 0.3 +
        (df['max_heart_rate'] < 120) * 0.2 +
        (df['exercise_angina'] == 1) * 0.3 +
        (df['oldpeak'] > 2) * 0.2 +
        np.random.uniform(0, 0.3, n_samples)
    )
    
    df['target'] = (risk_score > 0.6).astype(int)
    
    return df

# Train classification models
@st.cache_resource
def train_classification_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        predictions[name] = y_pred
        probabilities[name] = y_prob
    
    return results, predictions, probabilities

# Header
st.markdown("<h1 style='text-align: center;'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #f0f0f0;'>Day 4: AI in Healthcare - Building a Life-Saving Predictor</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 Project Overview")
    st.image("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🏥 Classification vs Regression</li>
            <li>📊 Binary classification</li>
            <li>🔍 Feature importance in healthcare</li>
            <li>🎯 Multiple classifiers comparison</li>
            <li>📈 ROC-AUC analysis</li>
            <li>🔢 Confusion matrix interpretation</li>
            <li>⚖️ Precision-Recall tradeoffs</li>
            <li>🧪 Model evaluation metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚕️ Medical Features")
    st.info("""
    **Patient Attributes:**
    - Age, Sex
    - Chest Pain Type
    - Blood Pressure
    - Cholesterol Levels
    - Heart Rate
    - Exercise Angina
    - And more...
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 4 of 21</strong></p>
        <p>GeeksforGeeks Course</p>
    </div>
    """, unsafe_allow_html=True)

# Load and prepare data
df = load_heart_disease_data()

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
with st.spinner('🔄 Training healthcare AI models...'):
    results, predictions, probabilities = train_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Overview",
    "🔍 EDA",
    "🤖 Model Training",
    "📈 Model Comparison",
    "🏥 Diagnosis Tool",
    "💡 Insights"
])

# Tab 1: Data Overview
with tab1:
    st.markdown("## 📊 Heart Disease Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        st.metric("Features", len(X.columns))
    with col3:
        disease_count = df['target'].sum()
        st.metric("Heart Disease", f"{disease_count}")
    with col4:
        healthy_count = len(df) - disease_count
        st.metric("Healthy", f"{healthy_count}")
    with col5:
        disease_rate = (disease_count / len(df) * 100)
        st.metric("Disease Rate", f"{disease_rate:.1f}%")
    
    st.markdown("### 📋 Sample Patient Data")
    display_df = df.head(10).copy()
    display_df['target'] = display_df['target'].map({0: 'Healthy', 1: 'Heart Disease'})
    st.dataframe(display_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Feature Descriptions")
        feature_desc = pd.DataFrame({
            'Feature': [
                'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
                'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
                'exercise_angina', 'oldpeak', 'st_slope', 
                'num_major_vessels', 'thalassemia', 'target'
            ],
            'Description': [
                'Age of patient (years)',
                'Sex (0: Female, 1: Male)',
                'Chest pain type (0-3)',
                'Resting blood pressure (mm Hg)',
                'Serum cholesterol (mg/dl)',
                'Fasting blood sugar > 120 mg/dl',
                'Resting ECG results (0-2)',
                'Maximum heart rate achieved',
                'Exercise induced angina (0: No, 1: Yes)',
                'ST depression induced by exercise',
                'Slope of peak exercise ST segment',
                'Number of major vessels (0-4)',
                'Thalassemia (0-3)',
                'Diagnosis (0: Healthy, 1: Disease)'
            ]
        })
        st.dataframe(feature_desc, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📊 Summary Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

# Tab 2: EDA
with tab2:
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Target Distribution")
        target_counts = df['target'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Healthy', 'Heart Disease'],
            values=target_counts.values,
            hole=0.4,
            marker=dict(colors=['#11998e', '#ff6b6b']),
            textinfo='label+percent+value',
            textfont=dict(size=14, color='white')
        )])
        fig.update_layout(
            title="Heart Disease Distribution",
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 👥 Age Distribution by Diagnosis")
        fig = px.histogram(
            df, x='age', color='target',
            nbins=30,
            title='Age Distribution',
            labels={'age': 'Age', 'target': 'Diagnosis'},
            color_discrete_map={0: '#11998e', 1: '#ff6b6b'},
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 👫 Gender Analysis")
        gender_disease = df.groupby(['sex', 'target']).size().reset_index(name='count')
        gender_disease['sex'] = gender_disease['sex'].map({0: 'Female', 1: 'Male'})
        gender_disease['target'] = gender_disease['target'].map({0: 'Healthy', 1: 'Disease'})
        
        fig = px.bar(
            gender_disease, x='sex', y='count', color='target',
            title='Heart Disease by Gender',
            labels={'sex': 'Gender', 'count': 'Count'},
            color_discrete_map={'Healthy': '#11998e', 'Disease': '#ff6b6b'},
            barmode='group'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 💓 Max Heart Rate Distribution")
        fig = px.box(
            df, x='target', y='max_heart_rate',
            title='Heart Rate by Diagnosis',
            labels={'target': 'Diagnosis', 'max_heart_rate': 'Max Heart Rate'},
            color='target',
            color_discrete_map={0: '#11998e', 1: '#ff6b6b'}
        )
        fig.update_xaxes(ticktext=['Healthy', 'Disease'], tickvals=[0, 1])
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### 🔥 Feature Correlation Heatmap")
    correlation_matrix = df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix of Medical Features'
    )
    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance correlation with target
    st.markdown("### 🎯 Feature Correlation with Heart Disease")
    target_corr = correlation_matrix['target'].drop('target').sort_values(ascending=False)
    
    fig = go.Figure(go.Bar(
        x=target_corr.values,
        y=target_corr.index,
        orientation='h',
        marker=dict(
            color=target_corr.values,
            colorscale='RdYlGn',
            showscale=True
        )
    ))
    fig.update_layout(
        title='Feature Correlation with Heart Disease Risk',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Features',
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Model Training
with tab3:
    st.markdown("## 🤖 Classification Model Training")
    
    st.markdown("### 📚 Models Trained")
    st.info("""
    - **Logistic Regression**: Linear probabilistic classifier
    - **Decision Tree**: Rule-based decision making
    - **Random Forest**: Ensemble of decision trees
    - **K-Nearest Neighbors**: Instance-based learning
    - **Support Vector Machine**: Maximum margin classifier
    - **Gradient Boosting**: Advanced ensemble method
    """)
    
    # Model performance table
    st.markdown("### 📊 Model Performance Metrics")
    
    performance_data = []
    for name, result in results.items():
        performance_data.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values('Accuracy', ascending=False)
    
    st.dataframe(
        performance_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen'),
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Accuracy Comparison")
        acc_data = {name: result['accuracy'] for name, result in results.items()}
        
        fig = px.bar(
            x=list(acc_data.keys()),
            y=list(acc_data.values()),
            title='Accuracy by Model',
            labels={'x': 'Model', 'y': 'Accuracy'},
            color=list(acc_data.values()),
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400,
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 F1-Score Comparison")
        f1_data = {name: result['f1'] for name, result in results.items()}
        
        fig = px.bar(
            x=list(f1_data.keys()),
            y=list(f1_data.values()),
            title='F1-Score by Model',
            labels={'x': 'Model', 'y': 'F1-Score'},
            color=list(f1_data.values()),
            color_continuous_scale='Greens'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400,
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model_result = results[best_model_name]
    
    st.markdown(f"### 🏆 Best Performing Model: {best_model_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{best_model_result['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{best_model_result['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{best_model_result['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{best_model_result['f1']:.4f}")

# Tab 4: Model Comparison
with tab4:
    st.markdown("## 📈 Detailed Model Comparison")
    
    model_select = st.selectbox("Select Model for Detailed Analysis:", list(results.keys()))
    
    selected_result = results[model_select]
    cm = selected_result['confusion_matrix']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔢 Confusion Matrix")
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Healthy', 'Disease'],
            y=['Healthy', 'Disease'],
            text_auto=True,
            color_continuous_scale='Reds',
            title=f'{model_select} - Confusion Matrix'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics explanation
        st.markdown("### 📊 Metrics Breakdown")
        tn, fp, fn, tp = cm.ravel()
        
        metrics_df = pd.DataFrame({
            'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
            'Count': [tn, fp, fn, tp],
            'Description': [
                'Correctly predicted healthy',
                'Incorrectly predicted disease',
                'Missed disease cases',
                'Correctly predicted disease'
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        
        metrics_values = [
            selected_result['accuracy'],
            selected_result['precision'],
            selected_result['recall'],
            selected_result['f1']
        ]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                marker=dict(
                    color=metrics_values,
                    colorscale='Reds',
                    showscale=False
                ),
                text=[f'{v:.3f}' for v in metrics_values],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title=f'{model_select} - Performance Metrics',
            yaxis_title='Score',
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400,
            yaxis_range=[0, 1.1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### ℹ️ Metric Explanations")
        st.markdown("""
        - **Accuracy**: Overall correct predictions
        - **Precision**: Accuracy of disease predictions
        - **Recall**: % of actual disease cases found
        - **F1-Score**: Balance between precision and recall
        """)
    
    # ROC Curve
    if probabilities[model_select] is not None:
        st.markdown("### 📊 ROC Curve Analysis")
        
        fpr, tpr, _ = roc_curve(y_test, probabilities[model_select])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='#ff6b6b', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))
        fig.update_layout(
            title=f'{model_select} - ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 5: Diagnosis Tool
with tab5:
    st.markdown("## 🏥 Heart Disease Risk Assessment Tool")
    
    st.markdown("### 👤 Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 20, 85, 50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        chest_pain = st.selectbox("Chest Pain Type", 
                                  ["Type 0", "Type 1", "Type 2", "Type 3"])
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        rest_ecg = st.selectbox("Resting ECG", ["Normal", "Abnormal 1", "Abnormal 2"])
        max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    
    with col3:
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
        st_slope = st.selectbox("ST Slope", ["Type 0", "Type 1", "Type 2"])
        major_vessels = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thalassemia = st.selectbox("Thalassemia", ["Type 0", "Type 1", "Type 2", "Type 3"])
    
    # Convert inputs
    sex_val = 1 if sex == "Male" else 0
    chest_pain_val = int(chest_pain.split()[-1])
    fasting_bs_val = 1 if fasting_bs == "Yes" else 0
    rest_ecg_val = 0 if rest_ecg == "Normal" else (1 if rest_ecg == "Abnormal 1" else 2)
    exercise_angina_val = 1 if exercise_angina == "Yes" else 0
    st_slope_val = int(st_slope.split()[-1])
    thalassemia_val = int(thalassemia.split()[-1])
    
    # Make prediction
    input_data = np.array([[age, sex_val, chest_pain_val, resting_bp, cholesterol,
                           fasting_bs_val, rest_ecg_val, max_hr, exercise_angina_val,
                           oldpeak, st_slope_val, major_vessels, thalassemia_val]])
    input_scaled = scaler.transform(input_data)
    
    st.markdown("### 🔮 Diagnosis Results from All Models")
    
    # Get predictions from all models
    model_predictions = []
    for name, result in results.items():
        pred = result['model'].predict(input_scaled)[0]
        prob = result['model'].predict_proba(input_scaled)[0] if hasattr(result['model'], 'predict_proba') else None
        model_predictions.append({
            'model': name,
            'prediction': pred,
            'probability': prob[1] if prob is not None else None
        })
    
    # Calculate ensemble prediction
    ensemble_pred = np.mean([p['prediction'] for p in model_predictions])
    ensemble_risk = ensemble_pred if ensemble_pred > 0.5 else 1 - ensemble_pred
    
    # Display predictions
    pred_cols = st.columns(3)
    
    for idx, pred_info in enumerate(model_predictions):
        col_idx = idx % 3
        with pred_cols[col_idx]:
            diagnosis = "Disease Risk" if pred_info['prediction'] == 1 else "Healthy"
            prob_text = f"{pred_info['probability']*100:.1f}%" if pred_info['probability'] else "N/A"
            color_class = "health-risk" if pred_info['prediction'] == 1 else "health-good"
            
            st.markdown(f"""
            <div class="{color_class}">
                <h4 style='margin:0; color: white;'>{pred_info['model']}</h4>
                <h2 style='margin:10px 0; color: white;'>{diagnosis}</h2>
                <p style='margin:0; color: white;'>Confidence: {prob_text}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Ensemble prediction
    st.markdown("---")
    ensemble_diagnosis = "Heart Disease Risk Detected" if ensemble_pred > 0.5 else "Low Risk - Healthy"
    ensemble_color = "health-risk" if ensemble_pred > 0.5 else "health-good"
    
    st.markdown(f"""
    <div class="{ensemble_color}" style="font-size: 1.2em;">
        <h3 style='color: white; margin: 0;'>🏥 Ensemble Consensus</h3>
        <h1 style='color: white; margin: 10px 0; font-size: 2.5em;'>{ensemble_diagnosis}</h1>
        <p style='color: white; margin: 0; font-size: 1.3em;'>Risk Score: {ensemble_risk*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚠️ Medical Disclaimer")
    st.warning("""
    **Important**: This is a demonstration tool for educational purposes only. 
    It should NOT be used for actual medical diagnosis. Always consult with 
    qualified healthcare professionals for medical advice and diagnosis.
    """)

# Tab 6: Insights
with tab6:
    st.markdown("## 💡 Key Insights & Learnings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 Model Performance Insights</h3>
            <ul>
                <li><strong>Best Model:</strong> {} with {:.2f}% accuracy</li>
                <li><strong>Most Reliable:</strong> High precision reduces false alarms</li>
                <li><strong>Sensitivity:</strong> High recall catches more disease cases</li>
                <li><strong>Balance:</strong> F1-score indicates overall effectiveness</li>
                <li><strong>Ensemble Power:</strong> Combining models improves reliability</li>
            </ul>
        </div>
        """.format(
            best_model_name,
            best_model_result['accuracy'] * 100
        ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🏥 Medical Risk Factors</h3>
            <ul>
                <li><strong>Age:</strong> Risk increases with age</li>
                <li><strong>Gender:</strong> Males generally at higher risk</li>
                <li><strong>Cholesterol:</strong> High levels indicate risk</li>
                <li><strong>Blood Pressure:</strong> Hypertension is a major factor</li>
                <li><strong>Heart Rate:</strong> Abnormal rates are concerning</li>
                <li><strong>Exercise Angina:</strong> Pain during exercise is a red flag</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>📚 Technical Skills Learned</h3>
            <ul>
                <li><strong>Classification:</strong> Binary prediction (disease/no disease)</li>
                <li><strong>Model Comparison:</strong> Evaluating multiple algorithms</li>
                <li><strong>Metrics:</strong> Accuracy, Precision, Recall, F1-score</li>
                <li><strong>Confusion Matrix:</strong> Understanding prediction errors</li>
                <li><strong>ROC-AUC:</strong> Model performance visualization</li>
                <li><strong>Feature Importance:</strong> Identifying key predictors</li>
                <li><strong>Data Scaling:</strong> Standardization for ML models</li>
                <li><strong>Medical AI:</strong> Healthcare applications of ML</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>⚖️ Classification Metrics Explained</h3>
            <ul>
                <li><strong>Accuracy:</strong> Overall correct predictions</li>
                <li><strong>Precision:</strong> Of predicted diseases, how many correct?</li>
                <li><strong>Recall:</strong> Of actual diseases, how many detected?</li>
                <li><strong>F1-Score:</strong> Harmonic mean of precision & recall</li>
                <li><strong>True Positive:</strong> Correctly identified disease</li>
                <li><strong>False Positive:</strong> Healthy predicted as disease</li>
                <li><strong>False Negative:</strong> Disease missed (most critical!)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary table
    st.markdown("### 📊 Complete Model Performance Summary")
    
    summary_data = []
    for name, result in results.items():
        tn, fp, fn, tp = result['confusion_matrix'].ravel()
        summary_data.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'True Positives': tp,
            'False Negatives': fn
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🚀 Real-World Healthcare Applications</h3>
        <ul>
            <li><strong>Early Detection:</strong> Identifying at-risk patients early</li>
            <li><strong>Screening:</strong> Mass population health screenings</li>
            <li><strong>Risk Assessment:</strong> Personalized risk profiles</li>
            <li><strong>Treatment Planning:</strong> Guiding medical interventions</li>
            <li><strong>Resource Allocation:</strong> Prioritizing high-risk patients</li>
            <li><strong>Preventive Care:</strong> Lifestyle modification recommendations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>❤️ Heart Disease Prediction System Complete!</h3>
    <p><strong>Day 4 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>AI in Healthcare - Building a Life-Saving Predictor</p>
</div>
""", unsafe_allow_html=True)
