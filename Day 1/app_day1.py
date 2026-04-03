import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Day 1: Titanic EDA",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }
    h2 {
        color: #2563eb;
        font-weight: 600;
    }
    h3 {
        color: #3b82f6;
        font-weight: 500;
    }
    .highlight-box {
        background-color: #dbeafe;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Titanic-Dataset.csv')
        return df
    except:
        st.error("⚠️ Please ensure 'Titanic-Dataset.csv' is in the same directory as this script!")
        return None

# Header
st.markdown("<h1 style='text-align: center;'>🚢 Day 1: Titanic Survival Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #64748b;'>End-to-End Exploratory Data Analysis (EDA)</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/300px-RMS_Titanic_3.jpg", use_container_width=True)
    st.markdown("### 📚 What I Learned")
    st.markdown("""
    - 🔍 **Data Cleaning** - Handling missing values
    - 📊 **Statistical Analysis** - Understanding distributions
    - 🎨 **Data Visualization** - Creating meaningful plots
    - ⚙️ **Feature Engineering** - Creating new insights
    - 🔗 **Correlation Analysis** - Finding relationships
    - 📈 **Survival Patterns** - Identifying key factors
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Project Objective")
    st.info("Analyze the Titanic dataset to discover patterns and factors that influenced passenger survival rates.")

# Load data
df = load_data()

if df is not None:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dataset Overview", 
        "🔍 Data Quality", 
        "📈 Statistical Analysis",
        "🎨 Visualizations",
        "🔗 Correlations",
        "💡 Key Insights"
    ])
    
    # Tab 1: Dataset Overview
    with tab1:
        st.markdown("## 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Passengers", len(df))
        with col2:
            st.metric("Survived", df['Survived'].sum())
        with col3:
            st.metric("Died", len(df) - df['Survived'].sum())
        with col4:
            st.metric("Survival Rate", f"{(df['Survived'].mean()*100):.1f}%")
        
        st.markdown("### 📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📏 Dataset Dimensions")
            st.info(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
            
        with col2:
            st.markdown("### 🏷️ Column Names")
            st.write(", ".join(df.columns.tolist()))
        
        st.markdown("### 📝 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Tab 2: Data Quality
    with tab2:
        st.markdown("## 🔍 Data Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ❌ Missing Values")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                # Visualize missing data
                fig = px.bar(missing_df, x='Column', y='Missing %', 
                           title='Missing Data Percentage by Column',
                           color='Missing %',
                           color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values in the dataset!")
        
        with col2:
            st.markdown("### 📊 Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.markdown("### 🔢 Unique Values per Column")
            unique_df = pd.DataFrame({
                'Column': df.columns,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(unique_df, use_container_width=True)
    
    # Tab 3: Statistical Analysis
    with tab3:
        st.markdown("## 📈 Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👥 Passenger Class Distribution")
            pclass_counts = df['Pclass'].value_counts().sort_index()
            fig = px.pie(values=pclass_counts.values, 
                        names=['1st Class', '2nd Class', '3rd Class'],
                        title='Passenger Distribution by Class',
                        color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 👫 Gender Distribution")
            gender_counts = df['Sex'].value_counts()
            fig = px.pie(values=gender_counts.values, 
                        names=gender_counts.index.str.capitalize(),
                        title='Passenger Distribution by Gender',
                        color_discrete_sequence=['#3b82f6', '#ec4899'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎂 Age Distribution")
            fig = px.histogram(df, x='Age', nbins=30,
                             title='Age Distribution of Passengers',
                             labels={'Age': 'Age (years)', 'count': 'Number of Passengers'},
                             color_discrete_sequence=['#2563eb'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 💰 Fare Distribution")
            fig = px.histogram(df, x='Fare', nbins=50,
                             title='Fare Distribution',
                             labels={'Fare': 'Fare ($)', 'count': 'Number of Passengers'},
                             color_discrete_sequence=['#10b981'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Visualizations
    with tab4:
        st.markdown("## 🎨 Advanced Visualizations")
        
        # Survival Analysis
        st.markdown("### ⚖️ Survival Analysis by Different Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival by Class
            survival_class = df.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
            fig = px.bar(survival_class, x='Pclass', y='Count', color='Survived',
                        title='Survival by Passenger Class',
                        labels={'Pclass': 'Passenger Class', 'Count': 'Number of Passengers'},
                        color_discrete_map={0: '#ef4444', 1: '#10b981'},
                        barmode='group')
            fig.update_layout(legend=dict(title='Survived', orientation='h', y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
            
            # Survival by Gender
            survival_gender = df.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
            fig = px.bar(survival_gender, x='Sex', y='Count', color='Survived',
                        title='Survival by Gender',
                        labels={'Sex': 'Gender', 'Count': 'Number of Passengers'},
                        color_discrete_map={0: '#ef4444', 1: '#10b981'},
                        barmode='group')
            fig.update_layout(legend=dict(title='Survived', orientation='h', y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Survival Rate by Class
            survival_rate_class = df.groupby('Pclass')['Survived'].mean() * 100
            fig = px.bar(x=survival_rate_class.index, y=survival_rate_class.values,
                        title='Survival Rate by Passenger Class',
                        labels={'x': 'Passenger Class', 'y': 'Survival Rate (%)'},
                        color=survival_rate_class.values,
                        color_continuous_scale='Greens')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Survival Rate by Gender
            survival_rate_gender = df.groupby('Sex')['Survived'].mean() * 100
            fig = px.bar(x=survival_rate_gender.index, y=survival_rate_gender.values,
                        title='Survival Rate by Gender',
                        labels={'x': 'Gender', 'y': 'Survival Rate (%)'},
                        color=survival_rate_gender.values,
                        color_continuous_scale='Blues')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Age vs Fare scatter plot
        st.markdown("### 🔍 Age vs Fare by Survival Status")
        fig = px.scatter(df.dropna(subset=['Age']), x='Age', y='Fare', 
                        color='Survived', size='Fare',
                        title='Age vs Fare Colored by Survival',
                        labels={'Survived': 'Survived'},
                        color_discrete_map={0: '#ef4444', 1: '#10b981'},
                        hover_data=['Pclass', 'Sex'])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Family size analysis
        st.markdown("### 👨‍👩‍👧‍👦 Family Size Analysis")
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        col1, col2 = st.columns(2)
        with col1:
            family_survival = df.groupby('FamilySize')['Survived'].mean() * 100
            fig = px.line(x=family_survival.index, y=family_survival.values,
                         title='Survival Rate by Family Size',
                         labels={'x': 'Family Size', 'y': 'Survival Rate (%)'},
                         markers=True)
            fig.update_traces(line_color='#2563eb', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            family_counts = df['FamilySize'].value_counts().sort_index()
            fig = px.bar(x=family_counts.index, y=family_counts.values,
                        title='Distribution of Family Sizes',
                        labels={'x': 'Family Size', 'y': 'Count'},
                        color=family_counts.values,
                        color_continuous_scale='Purples')
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Correlations
    with tab5:
        st.markdown("## 🔗 Correlation Analysis")
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add FamilySize if not already present
        if 'FamilySize' not in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔥 Correlation Heatmap")
            fig = px.imshow(corr_matrix, 
                           text_auto='.2f',
                           aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Correlation Matrix of Numeric Features')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Correlation with Survival")
            survival_corr = corr_matrix['Survived'].drop('Survived').sort_values(ascending=False)
            
            fig = go.Figure(go.Bar(
                x=survival_corr.values,
                y=survival_corr.index,
                orientation='h',
                marker=dict(
                    color=survival_corr.values,
                    colorscale='RdYlGn',
                    showscale=True
                )
            ))
            fig.update_layout(
                title='Feature Correlation with Survival',
                xaxis_title='Correlation Coefficient',
                yaxis_title='Features',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Key Insights
    with tab6:
        st.markdown("## 💡 Key Insights & Learnings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
                <h3>🎯 Major Findings</h3>
                <ul>
                    <li><strong>Gender Impact:</strong> Women had significantly higher survival rates (~74%) compared to men (~19%)</li>
                    <li><strong>Class Matters:</strong> 1st class passengers had better survival rates (~63%) than 3rd class (~24%)</li>
                    <li><strong>Age Factor:</strong> Children had higher survival rates than adults</li>
                    <li><strong>Family Size:</strong> Small families (2-4 members) had better survival rates than solo travelers or very large families</li>
                    <li><strong>Fare Correlation:</strong> Higher fare (proxy for wealth) correlated with better survival</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>📚 Technical Skills Learned</h3>
                <ul>
                    <li><strong>Data Cleaning:</strong> Handling missing values in Age, Cabin, and Embarked columns</li>
                    <li><strong>Feature Engineering:</strong> Created FamilySize feature from SibSp and Parch</li>
                    <li><strong>Statistical Analysis:</strong> Computing descriptive statistics and distributions</li>
                    <li><strong>Visualization:</strong> Creating interactive plots with Plotly</li>
                    <li><strong>Correlation Analysis:</strong> Understanding relationships between variables</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Summary Statistics Box
            st.markdown("### 📊 Quick Stats")
            
            metrics_data = {
                'Metric': [
                    'Overall Survival Rate',
                    'Female Survival Rate',
                    'Male Survival Rate',
                    '1st Class Survival Rate',
                    '2nd Class Survival Rate',
                    '3rd Class Survival Rate',
                    'Average Age',
                    'Average Fare'
                ],
                'Value': [
                    f"{df['Survived'].mean()*100:.1f}%",
                    f"{df[df['Sex']=='female']['Survived'].mean()*100:.1f}%",
                    f"{df[df['Sex']=='male']['Survived'].mean()*100:.1f}%",
                    f"{df[df['Pclass']==1]['Survived'].mean()*100:.1f}%",
                    f"{df[df['Pclass']==2]['Survived'].mean()*100:.1f}%",
                    f"{df[df['Pclass']==3]['Survived'].mean()*100:.1f}%",
                    f"{df['Age'].mean():.1f} years",
                    f"${df['Fare'].mean():.2f}"
                ]
            }
            
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>🔍 Data Quality Notes</h3>
                <ul>
                    <li><strong>Missing Age:</strong> ~20% of passengers have missing age data</li>
                    <li><strong>Missing Cabin:</strong> ~77% of cabin information is missing</li>
                    <li><strong>Missing Embarked:</strong> Only 2 passengers have missing embarkation port</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>🚀 Next Steps</h3>
                <ul>
                    <li>Build predictive models for survival prediction</li>
                    <li>Implement advanced imputation techniques for missing data</li>
                    <li>Create additional engineered features</li>
                    <li>Perform deeper statistical hypothesis testing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p><strong>Day 1 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>🚢 Titanic Survival Analysis - Exploratory Data Analysis</p>
</div>
""", unsafe_allow_html=True)
