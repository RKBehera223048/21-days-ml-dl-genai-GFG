import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Day 3: House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
        border-left: 6px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .prediction-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load California Housing dataset
@st.cache_data
def load_california_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df, housing.feature_names, housing.target

# Train models
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        predictions[name] = y_pred
    
    return results, predictions

# Header
st.markdown("<h1 style='text-align: center;'>🏠 House Price Prediction with Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #f0f0f0;'>Day 3: Predicting Housing Market Trends with AI</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Project Overview")
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🔢 Regression fundamentals</li>
            <li>📊 Feature engineering</li>
            <li>🧹 Data preprocessing</li>
            <li>📈 Model training & evaluation</li>
            <li>🎯 Multiple ML algorithms</li>
            <li>📉 Performance metrics (RMSE, MAE, R²)</li>
            <li>🔍 Feature importance analysis</li>
            <li>🎨 Advanced visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset selection
    st.markdown("### 📊 Dataset")
    dataset_choice = st.selectbox(
        "Choose Dataset:",
        ["California Housing (Demo)"]
    )
    
    st.info("📌 Using California Housing dataset with 8 features predicting median house values in California districts.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 3 of 21</strong></p>
        <p>GeeksforGeeks Course</p>
    </div>
    """, unsafe_allow_html=True)

# Load data
df, feature_names, target = load_california_data()

# Prepare data
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
with st.spinner('🔄 Training models...'):
    results, predictions = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Overview",
    "🔍 EDA",
    "🤖 Model Training",
    "📈 Model Comparison",
    "🎯 Predictions",
    "💡 Insights"
])

# Tab 1: Data Overview
with tab1:
    st.markdown("## 📊 California Housing Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Features", len(feature_names))
    with col3:
        st.metric("Avg Price", f"${y.mean():.2f}K")
    with col4:
        st.metric("Min Price", f"${y.min():.2f}K")
    with col5:
        st.metric("Max Price", f"${y.max():.2f}K")
    
    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Feature Descriptions")
        feature_desc = pd.DataFrame({
            'Feature': feature_names + ['MedHouseVal'],
            'Description': [
                'Median income in block group',
                'Median house age in block group',
                'Average number of rooms per household',
                'Average number of bedrooms per household',
                'Block group population',
                'Average household size',
                'Block group latitude',
                'Block group longitude',
                'Median house value (Target - in $100,000s)'
            ]
        })
        st.dataframe(feature_desc, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📊 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("### ℹ️ Data Info")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': [df[col].notna().sum() for col in df.columns],
            'Data Type': [str(df[col].dtype) for col in df.columns]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

# Tab 2: EDA
with tab2:
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 House Price Distribution")
        fig = px.histogram(
            df, x='MedHouseVal',
            nbins=50,
            title='Distribution of Median House Values',
            labels={'MedHouseVal': 'Median House Value ($100K)', 'count': 'Frequency'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏚️ House Age Distribution")
        fig = px.histogram(
            df, x='HouseAge',
            nbins=40,
            title='Distribution of House Age',
            labels={'HouseAge': 'Median House Age', 'count': 'Frequency'},
            color_discrete_sequence=['#f5576c']
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Income vs House Price")
        fig = px.scatter(
            df.sample(5000), x='MedInc', y='MedHouseVal',
            title='Median Income vs House Value',
            labels={'MedInc': 'Median Income', 'MedHouseVal': 'House Value ($100K)'},
            color='MedHouseVal',
            color_continuous_scale='Viridis',
            opacity=0.6
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🛏️ Rooms vs House Price")
        fig = px.scatter(
            df.sample(5000), x='AveRooms', y='MedHouseVal',
            title='Average Rooms vs House Value',
            labels={'AveRooms': 'Average Rooms', 'MedHouseVal': 'House Value ($100K)'},
            color='MedInc',
            color_continuous_scale='Plasma',
            opacity=0.6
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)',
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
        title='Correlation Matrix of All Features',
        labels=dict(color="Correlation")
    )
    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with target
    st.markdown("### 🎯 Feature Correlation with House Price")
    target_corr = correlation_matrix['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=False)
    
    fig = go.Figure(go.Bar(
        x=target_corr.values,
        y=target_corr.index,
        orientation='h',
        marker=dict(
            color=target_corr.values,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Correlation")
        )
    ))
    fig.update_layout(
        title='Feature Importance (Correlation with Target)',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Features',
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Model Training
with tab3:
    st.markdown("## 🤖 Model Training & Evaluation")
    
    st.markdown("### 📚 Models Trained")
    st.info("""
    - **Linear Regression**: Simple baseline model
    - **Ridge Regression**: Linear model with L2 regularization
    - **Lasso Regression**: Linear model with L1 regularization
    - **Random Forest**: Ensemble of decision trees
    - **Gradient Boosting**: Advanced ensemble method
    """)
    
    # Model performance table
    st.markdown("### 📊 Model Performance Metrics")
    
    performance_data = []
    for name, result in results.items():
        performance_data.append({
            'Model': name,
            'RMSE': f"{result['rmse']:.4f}",
            'MAE': f"{result['mae']:.4f}",
            'R² Score': f"{result['r2']:.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values('R² Score', ascending=False)
    
    # Highlight best model
    st.dataframe(
        performance_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                          .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 RMSE Comparison")
        rmse_data = {name: result['rmse'] for name, result in results.items()}
        
        fig = px.bar(
            x=list(rmse_data.keys()),
            y=list(rmse_data.values()),
            title='Root Mean Squared Error by Model',
            labels={'x': 'Model', 'y': 'RMSE'},
            color=list(rmse_data.values()),
            color_continuous_scale='Reds_r'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 R² Score Comparison")
        r2_data = {name: result['r2'] for name, result in results.items()}
        
        fig = px.bar(
            x=list(r2_data.keys()),
            y=list(r2_data.values()),
            title='R² Score by Model',
            labels={'x': 'Model', 'y': 'R² Score'},
            color=list(r2_data.values()),
            color_continuous_scale='Greens'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model details
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model_result = results[best_model_name]
    
    st.markdown(f"### 🏆 Best Performing Model: {best_model_name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"{best_model_result['rmse']:.4f}")
    with col2:
        st.metric("MAE", f"{best_model_result['mae']:.4f}")
    with col3:
        st.metric("R² Score", f"{best_model_result['r2']:.4f}")

# Tab 4: Model Comparison
with tab4:
    st.markdown("## 📈 Model Comparison & Analysis")
    
    # Predictions vs Actual
    st.markdown("### 🎯 Predictions vs Actual Values")
    
    model_select = st.selectbox("Select Model to Visualize:", list(results.keys()))
    
    selected_predictions = predictions[model_select]
    
    # Create comparison plot
    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': selected_predictions
    })
    
    fig = px.scatter(
        comparison_df.sample(min(1000, len(comparison_df))),
        x='Actual',
        y='Predicted',
        title=f'{model_select}: Predicted vs Actual House Prices',
        labels={'Actual': 'Actual Price ($100K)', 'Predicted': 'Predicted Price ($100K)'},
        opacity=0.6
    )
    
    # Add diagonal line
    max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
    min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Residual plot
    st.markdown("### 📊 Residual Analysis")
    
    residuals = y_test.values - selected_predictions
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            x=selected_predictions,
            y=residuals,
            title='Residual Plot',
            labels={'x': 'Predicted Values', 'y': 'Residuals'},
            opacity=0.6
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            x=residuals,
            nbins=50,
            title='Distribution of Residuals',
            labels={'x': 'Residual', 'y': 'Frequency'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (for Random Forest and Gradient Boosting)
    if model_select in ['Random Forest', 'Gradient Boosting']:
        st.markdown("### 🎯 Feature Importance")
        
        model = results[model_select]['model']
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {model_select}',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 5: Predictions
with tab5:
    st.markdown("## 🎯 Make Your Own Predictions")
    
    st.markdown("### 🏠 Input House Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        med_inc = st.slider("Median Income (in $10,000s)", 
                           float(X['MedInc'].min()), 
                           float(X['MedInc'].max()), 
                           float(X['MedInc'].mean()))
        
        house_age = st.slider("House Age (years)", 
                             float(X['HouseAge'].min()), 
                             float(X['HouseAge'].max()), 
                             float(X['HouseAge'].mean()))
        
        ave_rooms = st.slider("Average Rooms", 
                             float(X['AveRooms'].min()), 
                             min(float(X['AveRooms'].max()), 15.0), 
                             float(X['AveRooms'].mean()))
        
        ave_bedrms = st.slider("Average Bedrooms", 
                              float(X['AveBedrms'].min()), 
                              min(float(X['AveBedrms'].max()), 5.0), 
                              float(X['AveBedrms'].mean()))
    
    with col2:
        population = st.slider("Population", 
                              float(X['Population'].min()), 
                              min(float(X['Population'].max()), 5000.0), 
                              float(X['Population'].mean()))
        
        ave_occup = st.slider("Average Occupancy", 
                             float(X['AveOccup'].min()), 
                             min(float(X['AveOccup'].max()), 10.0), 
                             float(X['AveOccup'].mean()))
        
        latitude = st.slider("Latitude", 
                            float(X['Latitude'].min()), 
                            float(X['Latitude'].max()), 
                            float(X['Latitude'].mean()))
        
        longitude = st.slider("Longitude", 
                             float(X['Longitude'].min()), 
                             float(X['Longitude'].max()), 
                             float(X['Longitude'].mean()))
    
    # Make prediction
    input_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, 
                           population, ave_occup, latitude, longitude]])
    input_scaled = scaler.transform(input_data)
    
    st.markdown("### 🔮 Predictions from All Models")
    
    pred_cols = st.columns(len(results))
    
    for idx, (name, result) in enumerate(results.items()):
        with pred_cols[idx]:
            prediction = result['model'].predict(input_scaled)[0]
            st.markdown(f"""
            <div class="prediction-card">
                <h4 style='margin:0; color: white;'>{name}</h4>
                <h2 style='margin:10px 0; color: white;'>${prediction:.2f}K</h2>
                <p style='margin:0; color: white; font-size: 0.9em;'>Predicted Price</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Average prediction
    avg_prediction = np.mean([result['model'].predict(input_scaled)[0] for result in results.values()])
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 20px; text-align: center; 
                box-shadow: 0 8px 16px rgba(0,0,0,0.3); margin: 20px 0;'>
        <h3 style='color: white; margin: 0;'>📊 Ensemble Average Prediction</h3>
        <h1 style='color: white; margin: 10px 0; font-size: 3em;'>${avg_prediction:.2f}K</h1>
        <p style='color: white; margin: 0; font-size: 1.2em;'>Average across all models</p>
    </div>
    """, unsafe_allow_html=True)

# Tab 6: Insights
with tab6:
    st.markdown("## 💡 Key Insights & Learnings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 Model Performance Insights</h3>
            <ul>
                <li><strong>Best Model:</strong> {} achieved R² of {:.4f}</li>
                <li><strong>Accuracy:</strong> Models can predict prices within ${:.2f}K on average</li>
                <li><strong>Linear Models:</strong> Good baseline but limited by linear assumptions</li>
                <li><strong>Ensemble Methods:</strong> Random Forest & Gradient Boosting perform best</li>
                <li><strong>Regularization:</strong> Ridge & Lasso help prevent overfitting</li>
            </ul>
        </div>
        """.format(
            best_model_name,
            best_model_result['r2'],
            best_model_result['mae']
        ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>📊 Feature Insights</h3>
            <ul>
                <li><strong>Top Predictor:</strong> Median Income (correlation: {:.3f})</li>
                <li><strong>Location Matters:</strong> Latitude & Longitude are important</li>
                <li><strong>House Age:</strong> Negative correlation with price</li>
                <li><strong>Room Count:</strong> Positive impact on house value</li>
                <li><strong>Population:</strong> Minimal direct effect on prices</li>
            </ul>
        </div>
        """.format(
            df.corr()['MedHouseVal']['MedInc']
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>📚 Technical Skills Learned</h3>
            <ul>
                <li><strong>Data Preprocessing:</strong> Scaling, normalization, train-test split</li>
                <li><strong>Feature Engineering:</strong> Understanding feature relationships</li>
                <li><strong>Model Selection:</strong> Comparing multiple algorithms</li>
                <li><strong>Regression Metrics:</strong> RMSE, MAE, R² interpretation</li>
                <li><strong>Visualization:</strong> Residual plots, scatter plots, heatmaps</li>
                <li><strong>Cross-validation:</strong> Model evaluation techniques</li>
                <li><strong>Ensemble Methods:</strong> Random Forest, Gradient Boosting</li>
                <li><strong>Regularization:</strong> Ridge & Lasso for model stability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🔍 Evaluation Metrics Explained</h3>
            <ul>
                <li><strong>RMSE:</strong> Average prediction error in $100K units (lower is better)</li>
                <li><strong>MAE:</strong> Mean absolute error, easier to interpret</li>
                <li><strong>R² Score:</strong> % of variance explained (closer to 1 is better)</li>
                <li><strong>Residuals:</strong> Difference between actual and predicted values</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary table
    st.markdown("### 📈 Model Performance Summary")
    
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'RMSE': f"${result['rmse']:.4f}K",
            'MAE': f"${result['mae']:.4f}K",
            'R² Score': f"{result['r2']:.4f}",
            'Accuracy': f"{result['r2']*100:.2f}%",
            'Avg Error': f"${result['mae']:.2f}K"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🚀 Real-World Applications</h3>
        <ul>
            <li><strong>Real Estate:</strong> Property valuation and pricing</li>
            <li><strong>Investment:</strong> Identifying undervalued properties</li>
            <li><strong>Lending:</strong> Mortgage risk assessment</li>
            <li><strong>Urban Planning:</strong> Understanding housing market dynamics</li>
            <li><strong>Market Analysis:</strong> Trend prediction and forecasting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🏠 House Price Prediction Project Complete!</h3>
    <p><strong>Day 3 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>Predicting Housing Market Trends with Machine Learning</p>
</div>
""", unsafe_allow_html=True)
