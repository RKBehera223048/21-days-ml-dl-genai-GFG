import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Day 7: Customer Churn", page_icon="📱", layout="wide")

st.markdown("""<style>
.main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
.stMetric {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; color: white;}
h1 {color: #ffffff; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);}
h2, h3 {color: #f0f0f0;}
.highlight-box {background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; border-left: 6px solid #667eea;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Feature engineering
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years'])
    df['TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] != 'No').sum(axis=1)
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    return df

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 0,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob
        }
    return results

st.markdown("<h1 style='text-align: center;'>📱 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Day 7: Preventing Customer Churn with Feature Engineering</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1556740758-90de374c12ad?w=400", use_container_width=True)
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
        <h4 style='color: #ffffff;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🔧 Feature engineering</li>
            <li>📊 Churn analysis</li>
            <li>🎯 Classification models</li>
            <li>📈 ROC-AUC evaluation</li>
            <li>💼 Business metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #f0f0f0;'><strong>Day 7 of 21</strong></p>", unsafe_allow_html=True)

df = load_data()

if df is not None:
    # Prepare features
    label_encoders = {}
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    df_encoded['Churn'] = LabelEncoder().fit_transform(df['Churn'])
    
    feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'AvgMonthlySpend'] + categorical_cols
    X = df_encoded[feature_cols]
    y = df_encoded['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with st.spinner('🔄 Training models...'):
        results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analysis", "🤖 Models", "💡 Insights"])
    
    with tab1:
        st.markdown("## 📊 Customer Churn Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            churned = df['Churn'].value_counts().get('Yes', 0)
            st.metric("Churned", f"{churned:,}")
        with col3:
            churn_rate = (churned / len(df) * 100)
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col4:
            st.metric("Avg Tenure", f"{df['tenure'].mean():.0f} mo")
        with col5:
            st.metric("Avg Monthly", f"${df['MonthlyCharges'].mean():.0f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            churn_counts = df['Churn'].value_counts()
            fig = px.pie(values=churn_counts.values, names=churn_counts.index, 
                        title='Churn Distribution', color_discrete_sequence=['#51cf66', '#ff6b6b'])
            fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').sum())
            fig = px.bar(x=contract_churn.index, y=contract_churn.values, title='Churn by Contract Type',
                        labels={'x': 'Contract', 'y': 'Churned Customers'}, color=contract_churn.values,
                        color_continuous_scale='Reds')
            fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## 🔍 Churn Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='Churn', y='tenure', title='Tenure by Churn Status',
                        color='Churn', color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'})
            fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Churn', y='MonthlyCharges', title='Monthly Charges by Churn',
                        color='Churn', color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'})
            fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)')
            st.plotly_chart(fig, use_container_width=True)
        
        fig = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn', title='Tenure vs Monthly Charges',
                        color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'}, opacity=0.6)
        fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## 🤖 Model Performance")
        
        perf_data = []
        for name, result in results.items():
            perf_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1']:.4f}",
                'ROC-AUC': f"{result['roc_auc']:.4f}"
            })
        
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
        
        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        best_model = results[best_model_name]
        
        st.markdown(f"### 🏆 Best Model: {best_model_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cm = best_model['confusion_matrix']
            fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=['No Churn', 'Churn'], 
                          y=['No Churn', 'Churn'], text_auto=True, color_continuous_scale='Reds')
            fig.update_layout(title='Confusion Matrix', paper_bgcolor='rgba(255,255,255,0.9)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if best_model['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, best_model['probabilities'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={best_model["roc_auc"]:.3f})', line=dict(color='#667eea', width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
                fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                                paper_bgcolor='rgba(255,255,255,0.9)')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## 💡 Key Insights")
        
        st.markdown(f"""
        <div class="highlight-box">
            <h3>📊 Churn Insights</h3>
            <ul>
                <li><strong>Churn Rate:</strong> {(churned/len(df)*100):.1f}% of customers churned</li>
                <li><strong>High Risk:</strong> Month-to-month contracts have highest churn</li>
                <li><strong>Tenure Effect:</strong> Shorter tenure = higher churn risk</li>
                <li><strong>Best Model:</strong> {best_model_name} with {best_model['roc_auc']:.1%} ROC-AUC</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>💼 Business Recommendations</h3>
            <ul>
                <li>🎯 Target high-risk customers with retention offers</li>
                <li>📞 Proactive outreach in first 12 months</li>
                <li>💰 Incentivize longer-term contracts</li>
                <li>🔧 Improve service quality for at-risk segments</li>
                <li>📊 Monitor churn indicators monthly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: white;'><h3>📱 Customer Churn Prediction Complete!</h3><p>Day 7 of 21 | GeeksforGeeks Course</p></div>", unsafe_allow_html=True)
