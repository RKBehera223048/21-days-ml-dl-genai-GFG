import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Day 6: Time Series Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Aviation/Time theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #2C5364 0%, #0F2027 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        color: white;
    }
    .stMetric label {
        color: #ffffff !important;
        font-weight: 600;
    }
    h1 {
        color: #4fc3f7;
        font-weight: 700;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
    }
    h2 {
        color: #81d4fa;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h3 {
        color: #b3e5fc;
        font-weight: 500;
    }
    .highlight-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #4fc3f7;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .forecast-card {
        background: linear-gradient(135deg, #4fc3f7 0%, #2C5364 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        color: white;
        margin: 10px 0;
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
        background: linear-gradient(135deg, #4fc3f7 0%, #2C5364 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../Datasets/airline_passenger_timeseries.csv')
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        return df
    except:
        try:
            df = pd.read_csv('airline_passenger_timeseries.csv')
            df['Month'] = pd.to_datetime(df['Month'])
            df.set_index('Month', inplace=True)
            return df
        except:
            st.error("⚠️ Please ensure 'airline_passenger_timeseries.csv' is available!")
            return None

# Perform ADF test
def adf_test(series):
    result = adfuller(series)
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Stationary': result[1] < 0.05
    }

# Fit ARIMA model
@st.cache_data
def fit_arima_model(train_data, order=(1,1,1)):
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    return fitted_model

# Fit Exponential Smoothing
@st.cache_data
def fit_exp_smoothing(train_data, seasonal_periods=12):
    model = ExponentialSmoothing(
        train_data,
        seasonal_periods=seasonal_periods,
        trend='add',
        seasonal='add'
    )
    fitted_model = model.fit()
    return fitted_model

# Header
st.markdown("<h1 style='text-align: center;'>✈️ Airline Passenger Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #b3e5fc;'>Day 6: Time Series Analysis & Future Predictions</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Project Overview")
    st.image("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #b3e5fc;'>
            <li>📈 Time series fundamentals</li>
            <li>📊 Stationarity testing (ADF)</li>
            <li>🔍 Trend & seasonality decomposition</li>
            <li>📉 ACF & PACF analysis</li>
            <li>🎯 ARIMA modeling</li>
            <li>⏰ Exponential Smoothing</li>
            <li>🔮 Future forecasting</li>
            <li>📏 Forecast evaluation metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Forecasting parameters
    st.markdown("### ⚙️ Forecasting Settings")
    
    test_size = st.slider("Test Set Size (%)", 10, 30, 20)
    forecast_periods = st.slider("Forecast Periods (months)", 6, 36, 12)
    
    st.markdown("#### ARIMA Parameters")
    p = st.slider("p (AR order)", 0, 5, 1)
    d = st.slider("d (Differencing)", 0, 2, 1)
    q = st.slider("q (MA order)", 0, 5, 1)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #b3e5fc;'>
        <p><strong>Day 6 of 21</strong></p>
        <p>GeeksforGeeks Course</p>
    </div>
    """, unsafe_allow_html=True)

# Load data
df = load_data()

if df is not None:
    # Split data
    test_size_count = int(len(df) * test_size / 100)
    train = df.iloc[:-test_size_count]
    test = df.iloc[-test_size_count:]
    
    # Fit models
    with st.spinner('🔄 Training time series models...'):
        arima_model = fit_arima_model(train['Passengers'], order=(p, d, q))
        exp_model = fit_exp_smoothing(train['Passengers'])
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🔍 Time Series Analysis",
        "📈 Stationarity Tests",
        "🤖 Model Training",
        "🔮 Forecasting",
        "💡 Insights"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown("## 📊 Airline Passenger Data Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Months", len(df))
        with col2:
            st.metric("Time Range", f"{df.index[0].year}-{df.index[-1].year}")
        with col3:
            st.metric("Avg Passengers", f"{df['Passengers'].mean():.0f}")
        with col4:
            st.metric("Min", f"{df['Passengers'].min()}")
        with col5:
            st.metric("Max", f"{df['Passengers'].max()}")
        
        # Time series plot
        st.markdown("### 📈 Passenger Traffic Over Time")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Passengers'],
            mode='lines',
            name='Passengers',
            line=dict(color='#4fc3f7', width=2)
        ))
        fig.update_layout(
            title='Monthly Airline Passengers (1949-1960)',
            xaxis_title='Date',
            yaxis_title='Number of Passengers (thousands)',
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Sample Data")
            st.dataframe(df.head(20), use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.markdown("### 📅 Data Information")
            info_df = pd.DataFrame({
                'Attribute': ['Start Date', 'End Date', 'Frequency', 'Total Points', 'Missing Values'],
                'Value': [
                    df.index[0].strftime('%Y-%m'),
                    df.index[-1].strftime('%Y-%m'),
                    'Monthly',
                    len(df),
                    df['Passengers'].isna().sum()
                ]
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    # Tab 2: Time Series Analysis
    with tab2:
        st.markdown("## 🔍 Time Series Components Analysis")
        
        # Seasonal decomposition
        st.markdown("### 📊 Seasonal Decomposition")
        
        decomposition = seasonal_decompose(df['Passengers'], model='multiplicative', period=12)
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Passengers'], name='Original', line=dict(color='#4fc3f7')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=decomposition.trend, name='Trend', line=dict(color='#ff6b6b')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=decomposition.seasonal, name='Seasonal', line=dict(color='#51cf66')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=decomposition.resid, name='Residual', line=dict(color='#ffd43b')), row=4, col=1)
        
        fig.update_layout(
            height=900,
            showlegend=False,
            paper_bgcolor='rgba(255,255,255,0.9)',
            title_text="Time Series Decomposition"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Decomposition Components:**
        - **Trend**: Long-term progression (upward growth visible)
        - **Seasonal**: Regular periodic fluctuations (yearly pattern)
        - **Residual**: Random noise after removing trend and seasonality
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📅 Monthly Pattern Analysis")
            
            df_monthly = df.copy()
            df_monthly['Month'] = df_monthly.index.month
            monthly_avg = df_monthly.groupby('Month')['Passengers'].mean()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig = px.bar(
                x=month_names,
                y=monthly_avg.values,
                title='Average Passengers by Month',
                labels={'x': 'Month', 'y': 'Avg Passengers'},
                color=monthly_avg.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Yearly Growth Trend")
            
            df_yearly = df.copy()
            df_yearly['Year'] = df_yearly.index.year
            yearly_avg = df_yearly.groupby('Year')['Passengers'].mean()
            
            fig = px.line(
                x=yearly_avg.index,
                y=yearly_avg.values,
                title='Average Passengers by Year',
                labels={'x': 'Year', 'y': 'Avg Passengers'},
                markers=True
            )
            fig.update_traces(line_color='#4fc3f7', line_width=3)
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot by month
        st.markdown("### 📊 Seasonal Variation (Box Plot)")
        
        df_box = df.copy()
        df_box['Month'] = df_box.index.month_name()
        
        fig = px.box(
            df_box,
            x='Month',
            y='Passengers',
            title='Passenger Distribution by Month',
            color='Month',
            category_orders={'Month': ['January', 'February', 'March', 'April', 'May', 'June',
                                      'July', 'August', 'September', 'October', 'November', 'December']}
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Stationarity Tests
    with tab3:
        st.markdown("## 📈 Stationarity Analysis")
        
        st.info("""
        **What is Stationarity?**
        A time series is stationary if its statistical properties (mean, variance) don't change over time.
        Most forecasting models require stationary data.
        """)
        
        # ADF test on original data
        st.markdown("### 🔍 Augmented Dickey-Fuller Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Series")
            adf_original = adf_test(df['Passengers'])
            
            result_color = 'green' if adf_original['Stationary'] else 'red'
            result_text = 'Stationary ✓' if adf_original['Stationary'] else 'Non-Stationary ✗'
            
            st.markdown(f"**Result:** <span style='color:{result_color}; font-size:1.2em;'>{result_text}</span>", unsafe_allow_html=True)
            
            adf_df = pd.DataFrame({
                'Metric': ['ADF Statistic', 'p-value', '1% Critical Value', '5% Critical Value', '10% Critical Value'],
                'Value': [
                    f"{adf_original['ADF Statistic']:.4f}",
                    f"{adf_original['p-value']:.4f}",
                    f"{adf_original['Critical Values']['1%']:.4f}",
                    f"{adf_original['Critical Values']['5%']:.4f}",
                    f"{adf_original['Critical Values']['10%']:.4f}"
                ]
            })
            st.dataframe(adf_df, use_container_width=True, hide_index=True)
            
            if adf_original['p-value'] > 0.05:
                st.warning("⚠️ p-value > 0.05: Series is non-stationary. Differencing required!")
            else:
                st.success("✓ p-value < 0.05: Series is stationary!")
        
        with col2:
            st.markdown("#### First Differenced Series")
            df_diff = df['Passengers'].diff().dropna()
            adf_diff = adf_test(df_diff)
            
            result_color = 'green' if adf_diff['Stationary'] else 'red'
            result_text = 'Stationary ✓' if adf_diff['Stationary'] else 'Non-Stationary ✗'
            
            st.markdown(f"**Result:** <span style='color:{result_color}; font-size:1.2em;'>{result_text}</span>", unsafe_allow_html=True)
            
            adf_diff_df = pd.DataFrame({
                'Metric': ['ADF Statistic', 'p-value', '1% Critical Value', '5% Critical Value', '10% Critical Value'],
                'Value': [
                    f"{adf_diff['ADF Statistic']:.4f}",
                    f"{adf_diff['p-value']:.4f}",
                    f"{adf_diff['Critical Values']['1%']:.4f}",
                    f"{adf_diff['Critical Values']['5%']:.4f}",
                    f"{adf_diff['Critical Values']['10%']:.4f}"
                ]
            })
            st.dataframe(adf_diff_df, use_container_width=True, hide_index=True)
            
            if adf_diff['p-value'] > 0.05:
                st.warning("⚠️ May need second differencing")
            else:
                st.success("✓ First differencing makes series stationary!")
        
        # Plot comparison
        st.markdown("### 📊 Visual Comparison")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original Series', 'First Differenced Series'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Passengers'], name='Original', line=dict(color='#4fc3f7')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[1:], y=df_diff, name='Differenced', line=dict(color='#ff6b6b')), row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ACF and PACF
        st.markdown("### 📉 ACF & PACF Plots")
        
        st.info("""
        **Autocorrelation Functions:**
        - **ACF**: Shows correlation with lagged values
        - **PACF**: Shows partial correlation (removes indirect correlations)
        - Used to determine ARIMA parameters (p, q)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ACF (Autocorrelation)")
            acf_values = acf(df_diff, nlags=40)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color='#4fc3f7'))
            fig.add_hline(y=1.96/np.sqrt(len(df_diff)), line_dash="dash", line_color="red")
            fig.add_hline(y=-1.96/np.sqrt(len(df_diff)), line_dash="dash", line_color="red")
            fig.update_layout(
                title='ACF Plot',
                xaxis_title='Lag',
                yaxis_title='Correlation',
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### PACF (Partial Autocorrelation)")
            pacf_values = pacf(df_diff, nlags=40)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='#ff6b6b'))
            fig.add_hline(y=1.96/np.sqrt(len(df_diff)), line_dash="dash", line_color="red")
            fig.add_hline(y=-1.96/np.sqrt(len(df_diff)), line_dash="dash", line_color="red")
            fig.update_layout(
                title='PACF Plot',
                xaxis_title='Lag',
                yaxis_title='Partial Correlation',
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Model Training
    with tab4:
        st.markdown("## 🤖 Time Series Model Training")
        
        st.markdown(f"### 📊 Train-Test Split")
        st.info(f"""
        - **Training Data:** {len(train)} months ({100-test_size}%)
        - **Test Data:** {len(test)} months ({test_size}%)
        - **Train Period:** {train.index[0].strftime('%Y-%m')} to {train.index[-1].strftime('%Y-%m')}
        - **Test Period:** {test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')}
        """)
        
        # ARIMA Model
        st.markdown("### 📈 ARIMA Model")
        st.markdown(f"**Model Configuration:** ARIMA({p}, {d}, {q})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Summary")
            st.text(arima_model.summary().as_text()[:1000] + "...")
        
        with col2:
            st.markdown("#### Parameter Explanation")
            st.markdown(f"""
            - **p = {p}**: Number of autoregressive terms
            - **d = {d}**: Degree of differencing
            - **q = {q}**: Number of moving average terms
            
            **AIC:** {arima_model.aic:.2f} (lower is better)
            **BIC:** {arima_model.bic:.2f} (lower is better)
            """)
            
            st.info("""
            **Model Selection Criteria:**
            - Lower AIC/BIC indicates better model
            - Balance between goodness of fit and complexity
            - Try different (p,d,q) combinations
            """)
        
        # Predictions on test set
        arima_pred = arima_model.forecast(steps=len(test))
        exp_pred = exp_model.forecast(steps=len(test))
        
        # Calculate metrics
        arima_rmse = np.sqrt(mean_squared_error(test['Passengers'], arima_pred))
        arima_mae = mean_absolute_error(test['Passengers'], arima_pred)
        arima_mape = mean_absolute_percentage_error(test['Passengers'], arima_pred) * 100
        
        exp_rmse = np.sqrt(mean_squared_error(test['Passengers'], exp_pred))
        exp_mae = mean_absolute_error(test['Passengers'], exp_pred)
        exp_mape = mean_absolute_percentage_error(test['Passengers'], exp_pred) * 100
        
        # Model comparison
        st.markdown("### 📊 Model Performance Comparison")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ARIMA RMSE", f"{arima_rmse:.2f}")
            st.metric("Exp Smooth RMSE", f"{exp_rmse:.2f}")
        with col2:
            st.metric("ARIMA MAE", f"{arima_mae:.2f}")
            st.metric("Exp Smooth MAE", f"{exp_mae:.2f}")
        with col3:
            st.metric("ARIMA MAPE", f"{arima_mape:.2f}%")
            st.metric("Exp Smooth MAPE", f"{exp_mape:.2f}%")
        
        # Performance table
        performance_df = pd.DataFrame({
            'Model': ['ARIMA', 'Exponential Smoothing'],
            'RMSE': [f"{arima_rmse:.2f}", f"{exp_rmse:.2f}"],
            'MAE': [f"{arima_mae:.2f}", f"{exp_mae:.2f}"],
            'MAPE': [f"{arima_mape:.2f}%", f"{exp_mape:.2f}%"]
        })
        
        st.dataframe(performance_df, use_container_width=True, hide_index=True)
        
        # Plot predictions vs actual
        st.markdown("### 📈 Test Set Predictions vs Actual")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train['Passengers'], name='Training Data', line=dict(color='#4fc3f7')))
        fig.add_trace(go.Scatter(x=test.index, y=test['Passengers'], name='Actual Test', line=dict(color='#51cf66', width=3)))
        fig.add_trace(go.Scatter(x=test.index, y=arima_pred, name=f'ARIMA Prediction', line=dict(color='#ff6b6b', dash='dash')))
        fig.add_trace(go.Scatter(x=test.index, y=exp_pred, name='Exp Smooth Prediction', line=dict(color='#ffd43b', dash='dash')))
        
        fig.update_layout(
            title='Model Predictions on Test Set',
            xaxis_title='Date',
            yaxis_title='Passengers',
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Forecasting
    with tab5:
        st.markdown("## 🔮 Future Forecasting")
        
        st.markdown(f"### 📅 {forecast_periods}-Month Forecast")
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
        
        # Make forecasts
        arima_future = arima_model.forecast(steps=forecast_periods)
        exp_future = exp_model.forecast(steps=forecast_periods)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'ARIMA Forecast': arima_future.values,
            'Exp Smooth Forecast': exp_future.values,
            'Average Forecast': (arima_future.values + exp_future.values) / 2
        })
        
        # Display forecast table
        st.markdown("### 📊 Forecast Table")
        display_forecast = forecast_df.copy()
        display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m')
        display_forecast['ARIMA Forecast'] = display_forecast['ARIMA Forecast'].round(0).astype(int)
        display_forecast['Exp Smooth Forecast'] = display_forecast['Exp Smooth Forecast'].round(0).astype(int)
        display_forecast['Average Forecast'] = display_forecast['Average Forecast'].round(0).astype(int)
        
        st.dataframe(display_forecast, use_container_width=True, hide_index=True)
        
        # Plot forecast
        st.markdown("### 📈 Forecast Visualization")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Passengers'],
            name='Historical Data',
            line=dict(color='#4fc3f7', width=2)
        ))
        
        # ARIMA forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=arima_future,
            name='ARIMA Forecast',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
        
        # Exp Smoothing forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=exp_future,
            name='Exp Smooth Forecast',
            line=dict(color='#ffd43b', width=2, dash='dash')
        ))
        
        # Average forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=(arima_future + exp_future) / 2,
            name='Ensemble Average',
            line=dict(color='#51cf66', width=3)
        ))
        
        fig.update_layout(
            title=f'Airline Passenger Forecast ({forecast_periods} months ahead)',
            xaxis_title='Date',
            yaxis_title='Passengers (thousands)',
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=600,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast insights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_forecast = forecast_df['Average Forecast'].mean()
            st.markdown(f"""
            <div class="forecast-card">
                <h4 style='margin:0; color: white;'>Avg Forecast</h4>
                <h2 style='margin:10px 0; color: white;'>{avg_forecast:.0f}</h2>
                <p style='margin:0; color: white;'>Passengers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            min_forecast = forecast_df['Average Forecast'].min()
            st.markdown(f"""
            <div class="forecast-card">
                <h4 style='margin:0; color: white;'>Min Forecast</h4>
                <h2 style='margin:10px 0; color: white;'>{min_forecast:.0f}</h2>
                <p style='margin:0; color: white;'>Passengers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            max_forecast = forecast_df['Average Forecast'].max()
            st.markdown(f"""
            <div class="forecast-card">
                <h4 style='margin:0; color: white;'>Max Forecast</h4>
                <h2 style='margin:10px 0; color: white;'>{max_forecast:.0f}</h2>
                <p style='margin:0; color: white;'>Passengers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            growth_rate = ((max_forecast - min_forecast) / min_forecast * 100)
            st.markdown(f"""
            <div class="forecast-card">
                <h4 style='margin:0; color: white;'>Growth Rate</h4>
                <h2 style='margin:10px 0; color: white;'>{growth_rate:.1f}%</h2>
                <p style='margin:0; color: white;'>Peak vs Trough</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Download forecast
        st.markdown("### 📥 Download Forecast")
        csv = display_forecast.to_csv(index=False)
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name=f"passenger_forecast_{forecast_periods}months.csv",
            mime="text/csv"
        )
    
    # Tab 6: Insights
    with tab6:
        st.markdown("## 💡 Key Insights & Learnings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
                <h3>📈 Time Series Insights</h3>
                <ul>
                    <li><strong>Trend:</strong> Strong upward growth in passengers over time</li>
                    <li><strong>Seasonality:</strong> Clear yearly pattern with summer peaks</li>
                    <li><strong>Peak Months:</strong> July-August (vacation season)</li>
                    <li><strong>Low Months:</strong> November-February (winter)</li>
                    <li><strong>Growth Rate:</strong> Consistent year-over-year increase</li>
                    <li><strong>Volatility:</strong> Seasonal fluctuations around 20-30%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="highlight-box">
                <h3>🤖 Model Performance</h3>
                <ul>
                    <li><strong>Best Model:</strong> {'ARIMA' if arima_rmse < exp_rmse else 'Exponential Smoothing'}</li>
                    <li><strong>ARIMA RMSE:</strong> {arima_rmse:.2f}</li>
                    <li><strong>Exp Smooth RMSE:</strong> {exp_rmse:.2f}</li>
                    <li><strong>Accuracy:</strong> {(100 - min(arima_mape, exp_mape)):.1f}%</li>
                    <li><strong>Forecast Horizon:</strong> {forecast_periods} months</li>
                    <li><strong>Ensemble Benefit:</strong> Averaging improves reliability</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight-box">
                <h3>📚 Technical Skills Learned</h3>
                <ul>
                    <li><strong>Time Series Basics:</strong> Trend, seasonality, noise</li>
                    <li><strong>Stationarity:</strong> ADF test and differencing</li>
                    <li><strong>Decomposition:</strong> Breaking down components</li>
                    <li><strong>ARIMA Modeling:</strong> (p,d,q) parameter selection</li>
                    <li><strong>Exponential Smoothing:</strong> Holt-Winters method</li>
                    <li><strong>ACF/PACF:</strong> Autocorrelation analysis</li>
                    <li><strong>Forecasting:</strong> Multi-step ahead predictions</li>
                    <li><strong>Model Evaluation:</strong> RMSE, MAE, MAPE metrics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>🎯 Business Applications</h3>
                <ul>
                    <li><strong>Capacity Planning:</strong> Aircraft and crew scheduling</li>
                    <li><strong>Revenue Management:</strong> Dynamic pricing strategies</li>
                    <li><strong>Resource Allocation:</strong> Staff and gate assignments</li>
                    <li><strong>Inventory:</strong> Fuel and supplies forecasting</li>
                    <li><strong>Marketing:</strong> Campaign timing optimization</li>
                    <li><strong>Strategic Planning:</strong> Fleet expansion decisions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("### 📊 Comprehensive Summary")
        
        summary_df = pd.DataFrame({
            'Metric': [
                'Historical Average',
                'Historical Min',
                'Historical Max',
                'Forecast Average',
                'Forecast Min',
                'Forecast Max',
                'ARIMA RMSE',
                'ARIMA MAPE',
                'Best Month (Historical)',
                'Worst Month (Historical)'
            ],
            'Value': [
                f"{df['Passengers'].mean():.0f}",
                f"{df['Passengers'].min():.0f}",
                f"{df['Passengers'].max():.0f}",
                f"{forecast_df['Average Forecast'].mean():.0f}",
                f"{forecast_df['Average Forecast'].min():.0f}",
                f"{forecast_df['Average Forecast'].max():.0f}",
                f"{arima_rmse:.2f}",
                f"{arima_mape:.2f}%",
                df['Passengers'].idxmax().strftime('%Y-%m'),
                df['Passengers'].idxmin().strftime('%Y-%m')
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🚀 Key Takeaways</h3>
            <ul>
                <li><strong>Time Series is Everywhere:</strong> Sales, stock prices, weather, demand</li>
                <li><strong>Components Matter:</strong> Understanding trend + seasonality is crucial</li>
                <li><strong>Stationarity is Key:</strong> Most models need stationary data</li>
                <li><strong>Multiple Models:</strong> Try different approaches and ensemble</li>
                <li><strong>Validation is Critical:</strong> Always test on held-out data</li>
                <li><strong>Business Context:</strong> Interpret forecasts with domain knowledge</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4fc3f7; padding: 20px;'>
    <h3>✈️ Time Series Forecasting Complete!</h3>
    <p style='color: #b3e5fc;'><strong>Day 6 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p style='color: #81d4fa;'>Predicting Future Store Sales with AI</p>
</div>
""", unsafe_allow_html=True)
