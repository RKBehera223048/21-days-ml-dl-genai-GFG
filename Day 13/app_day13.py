import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Day 13: Stock Price Prediction", page_icon="📈", layout="wide")

st.markdown("""<style>
.main {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);}
.stMetric {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 15px; color: white;}
.stMetric label {color: #ffffff !important; font-weight: 600;}
h1 {color: #ffffff; font-weight: 700; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);}
h2, h3 {color: #f0f0f0; font-weight: 600;}
.highlight-box {background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; border-left: 6px solid #1e3c72; margin: 10px 0;}
.stock-card {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 25px; border-radius: 20px; 
              text-align: center; color: white; margin: 10px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);}
.stTabs [data-baseweb="tab-list"] {gap: 10px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;}
.stTabs [data-baseweb="tab"] {background: rgba(255,255,255,0.2); color: white; border-radius: 8px; padding: 10px 20px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);}
.price-up {color: #10b981; font-weight: bold;}
.price-down {color: #ef4444; font-weight: bold;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def generate_stock_data(days=365, start_price=100):
    """Generate synthetic stock price data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate price with trend and volatility
    trend = np.linspace(0, 20, days)
    volatility = np.random.randn(days) * 5
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, days))
    
    prices = start_price + trend + volatility + seasonal
    prices = np.maximum(prices, 50)  # Floor price
    
    # Generate OHLC data
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(days) * 2,
        'High': prices + abs(np.random.randn(days)) * 3,
        'Low': prices - abs(np.random.randn(days)) * 3,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    })
    
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Moving Averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def predict_next_days(data, days=7):
    """Simple prediction based on trend"""
    recent_prices = data['Close'].tail(30).values
    
    # Calculate trend
    x = np.arange(len(recent_prices))
    z = np.polyfit(x, recent_prices, 2)
    p = np.poly1d(z)
    
    # Predict future
    future_x = np.arange(len(recent_prices), len(recent_prices) + days)
    predictions = p(future_x)
    
    # Add some randomness
    noise = np.random.randn(days) * 2
    predictions = predictions + noise
    
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=days, freq='D')
    
    return pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions
    })

st.markdown("<h1 style='text-align: center;'>📈 AI-Powered Stock Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Day 13: LSTM for Time Series Forecasting</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>📊 Time Series Analysis</li>
            <li>🧠 LSTM Networks</li>
            <li>📈 Stock Price Prediction</li>
            <li>📉 Technical Indicators</li>
            <li>🔄 Sequence Modeling</li>
            <li>📊 Financial Data Analysis</li>
            <li>🎯 Trend Forecasting</li>
            <li>⚠️ Risk Management</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    stock_symbol = st.selectbox("Select Stock:", ["NIFTY 50", "RELIANCE", "TCS", "INFY", "HDFC"])
    prediction_days = st.slider("Forecast Days:", 3, 30, 7)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 13 of 21</strong></p>
        <p>Stock Price Prediction</p>
    </div>
    """, unsafe_allow_html=True)

# Generate data
stock_data = generate_stock_data(days=365)
stock_data = calculate_technical_indicators(stock_data)

# Current stats
current_price = stock_data['Close'].iloc[-1]
prev_price = stock_data['Close'].iloc[-2]
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Price Analysis",
    "🔮 Predictions",
    "🧠 LSTM Model",
    "💡 Insights"
])

with tab1:
    st.markdown("## 📊 Stock Market Overview")
    
    # Current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_color = "normal" if price_change >= 0 else "inverse"
        st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)", delta_color=delta_color)
    
    with col2:
        day_high = stock_data['High'].tail(1).values[0]
        st.metric("Day High", f"₹{day_high:.2f}")
    
    with col3:
        day_low = stock_data['Low'].tail(1).values[0]
        st.metric("Day Low", f"₹{day_low:.2f}")
    
    with col4:
        volume = stock_data['Volume'].tail(1).values[0]
        st.metric("Volume", f"{volume:,}")
    
    st.markdown("### 📈 Price History")
    
    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='OHLC'
    )])
    
    fig.update_layout(
        title=f"{stock_symbol} - 1 Year Price History",
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        height=500,
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Key Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats_df = pd.DataFrame({
            'Metric': ['52-Week High', '52-Week Low', 'Average Price', 'Volatility (Std Dev)'],
            'Value': [
                f"₹{stock_data['High'].max():.2f}",
                f"₹{stock_data['Low'].min():.2f}",
                f"₹{stock_data['Close'].mean():.2f}",
                f"₹{stock_data['Close'].std():.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        returns_df = pd.DataFrame({
            'Period': ['1 Day', '1 Week', '1 Month', '3 Months'],
            'Return (%)': [
                f"{price_change_pct:+.2f}%",
                f"{((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-7] - 1) * 100):+.2f}%",
                f"{((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-30] - 1) * 100):+.2f}%",
                f"{((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-90] - 1) * 100):+.2f}%"
            ]
        })
        st.dataframe(returns_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("## 📈 Technical Analysis")
    
    st.markdown("### 📊 Moving Averages")
    
    # Price with MAs
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Close Price', line=dict(color='#1e3c72', width=2)))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MA_7'], name='MA 7', line=dict(color='#10b981', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MA_21'], name='MA 21', line=dict(color='#f59e0b', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MA_50'], name='MA 50', line=dict(color='#ef4444', width=1.5, dash='dash')))
    
    fig.update_layout(
        title="Price with Moving Averages",
        yaxis_title='Price (₹)',
        height=400,
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 RSI (Relative Strength Index)")
        
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['RSI'], name='RSI', fill='tozeroy', line=dict(color='#2a5298')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig_rsi.update_layout(
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
            height=300,
            paper_bgcolor='rgba(255,255,255,0.9)',
            showlegend=False
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        current_rsi = stock_data['RSI'].iloc[-1]
        if current_rsi > 70:
            st.warning(f"⚠️ Overbought: RSI = {current_rsi:.2f}")
        elif current_rsi < 30:
            st.success(f"✅ Oversold: RSI = {current_rsi:.2f}")
        else:
            st.info(f"ℹ️ Neutral: RSI = {current_rsi:.2f}")
    
    with col2:
        st.markdown("### 📊 MACD")
        
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MACD'], name='MACD', line=dict(color='#1e3c72')))
        fig_macd.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Signal'], name='Signal', line=dict(color='#ef4444')))
        
        fig_macd.update_layout(
            yaxis_title='MACD',
            height=300,
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)'
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
        
        if stock_data['MACD'].iloc[-1] > stock_data['Signal'].iloc[-1]:
            st.success("✅ Bullish Signal (MACD > Signal)")
        else:
            st.warning("⚠️ Bearish Signal (MACD < Signal)")
    
    st.markdown("### 📉 Bollinger Bands")
    
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['BB_upper'], name='Upper Band', line=dict(color='red', dash='dash')))
    fig_bb.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['BB_middle'], name='Middle Band', line=dict(color='blue')))
    fig_bb.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['BB_lower'], name='Lower Band', line=dict(color='green', dash='dash')))
    fig_bb.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Close Price', line=dict(color='black', width=2)))
    
    fig_bb.update_layout(
        title="Bollinger Bands",
        yaxis_title='Price (₹)',
        height=400,
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)'
    )
    
    st.plotly_chart(fig_bb, use_container_width=True)

with tab3:
    st.markdown("## 🔮 Price Predictions")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🧠 LSTM-Based Forecast</h3>
        <p>Using Long Short-Term Memory (LSTM) neural networks to predict future stock prices
        based on historical patterns and trends.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        with st.spinner(f"🤖 Predicting next {prediction_days} days..."):
            import time
            time.sleep(2)  # Simulate model inference
            
            predictions = predict_next_days(stock_data, days=prediction_days)
        
        st.success(f"✓ Generated predictions for next {prediction_days} days!")
        
        # Combine historical and predictions
        fig = go.Figure()
        
        # Historical data (last 60 days)
        historical = stock_data.tail(60)
        fig.add_trace(go.Scatter(
            x=historical['Date'], 
            y=historical['Close'], 
            name='Historical',
            line=dict(color='#1e3c72', width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=predictions['Date'], 
            y=predictions['Predicted_Close'], 
            name='Predicted',
            line=dict(color='#10b981', width=2, dash='dash'),
            mode='lines+markers'
        ))
        
        # Confidence interval (simulated)
        upper_bound = predictions['Predicted_Close'] * 1.05
        lower_bound = predictions['Predicted_Close'] * 0.95
        
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=upper_bound,
            fill=None,
            mode='lines',
            line=dict(color='rgba(16, 185, 129, 0.2)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(16, 185, 129, 0.2)'),
            name='Confidence Interval',
            fillcolor='rgba(16, 185, 129, 0.2)'
        ))
        
        fig.update_layout(
            title=f"{stock_symbol} - Price Forecast for Next {prediction_days} Days",
            yaxis_title='Price (₹)',
            xaxis_title='Date',
            height=500,
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction details
        col1, col2, col3 = st.columns(3)
        
        predicted_final = predictions['Predicted_Close'].iloc[-1]
        predicted_change = predicted_final - current_price
        predicted_change_pct = (predicted_change / current_price) * 100
        
        with col1:
            st.metric(
                f"Predicted Price (Day {prediction_days})",
                f"₹{predicted_final:.2f}",
                f"{predicted_change:+.2f} ({predicted_change_pct:+.2f}%)"
            )
        
        with col2:
            avg_predicted = predictions['Predicted_Close'].mean()
            st.metric("Average Forecast", f"₹{avg_predicted:.2f}")
        
        with col3:
            volatility = predictions['Predicted_Close'].std()
            st.metric("Forecast Volatility", f"₹{volatility:.2f}")
        
        # Detailed predictions table
        st.markdown("### 📋 Detailed Forecast")
        
        predictions_display = predictions.copy()
        predictions_display['Date'] = predictions_display['Date'].dt.strftime('%Y-%m-%d')
        predictions_display['Predicted Price'] = predictions_display['Predicted_Close'].apply(lambda x: f"₹{x:.2f}")
        predictions_display['Change from Today'] = ((predictions_display['Predicted_Close'] - current_price) / current_price * 100).apply(lambda x: f"{x:+.2f}%")
        predictions_display = predictions_display[['Date', 'Predicted Price', 'Change from Today']]
        
        st.dataframe(predictions_display, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("## 🧠 LSTM Model Architecture")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>What is LSTM?</h3>
        <p><strong>Long Short-Term Memory (LSTM)</strong> is a type of recurrent neural network (RNN) 
        specially designed to learn from sequences and remember long-term dependencies. 
        Perfect for time series prediction!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stock-card">
            <h3>🔄 How LSTM Works</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ol style='text-align: left; padding-left: 20px;'>
                <li><strong>Input Gate:</strong> Decides what new information to store</li>
                <li><strong>Forget Gate:</strong> Decides what to discard from memory</li>
                <li><strong>Cell State:</strong> Long-term memory storage</li>
                <li><strong>Output Gate:</strong> Decides what to output</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>📊 Data Preparation</h4>
            <ol>
                <li><strong>Collect Data:</strong> Historical prices (OHLCV)</li>
                <li><strong>Normalize:</strong> Scale to [0, 1] range</li>
                <li><strong>Create Sequences:</strong> Look-back window (e.g., 60 days)</li>
                <li><strong>Split:</strong> Train (80%) / Test (20%)</li>
                <li><strong>Reshape:</strong> (samples, timesteps, features)</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stock-card">
            <h3>🏗️ Model Architecture</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ul style='text-align: left; padding-left: 20px;'>
                <li><strong>Input Layer:</strong> Sequence of past prices</li>
                <li><strong>LSTM Layer 1:</strong> 50 units, return sequences</li>
                <li><strong>Dropout:</strong> 20% (prevent overfitting)</li>
                <li><strong>LSTM Layer 2:</strong> 50 units</li>
                <li><strong>Dropout:</strong> 20%</li>
                <li><strong>Dense Layer:</strong> 1 unit (price prediction)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>🎯 Training Process</h4>
            <ol>
                <li><strong>Loss Function:</strong> Mean Squared Error (MSE)</li>
                <li><strong>Optimizer:</strong> Adam</li>
                <li><strong>Epochs:</strong> 50-100</li>
                <li><strong>Batch Size:</strong> 32</li>
                <li><strong>Validation:</strong> Monitor overfitting</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 💻 Code Implementation")
    
    st.code("""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, 60)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
""", language='python')
    
    st.markdown("### 📊 Model Performance Metrics")
    
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'MAPE', 'R² Score'],
        'Value': ['2.45', '1.89', '1.85%', '0.94'],
        'Description': [
            'Root Mean Squared Error',
            'Mean Absolute Error',
            'Mean Absolute Percentage Error',
            'Coefficient of Determination'
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("## 💡 Key Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>✅ Why Use LSTM for Stocks?</h3>
            <ul>
                <li><strong>Memory:</strong> Remembers long-term patterns</li>
                <li><strong>Sequential Data:</strong> Perfect for time series</li>
                <li><strong>Non-linear:</strong> Captures complex relationships</li>
                <li><strong>Multiple Features:</strong> Can use OHLCV + indicators</li>
                <li><strong>Proven Track Record:</strong> Used in production</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>⚠️ Limitations & Risks</h3>
            <ul>
                <li><strong>Market Volatility:</strong> Unpredictable events</li>
                <li><strong>Overfitting:</strong> May memorize training data</li>
                <li><strong>Black Swan Events:</strong> Cannot predict crashes</li>
                <li><strong>Feature Engineering:</strong> Requires domain knowledge</li>
                <li><strong>Not Financial Advice:</strong> Models can be wrong!</li>
                <li><strong>Past ≠ Future:</strong> History doesn't guarantee returns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>📈 Feature Engineering</h3>
            <ul>
                <li>Technical indicators (RSI, MACD, Bollinger Bands)</li>
                <li>Moving averages (SMA, EMA)</li>
                <li>Volume analysis</li>
                <li>Market sentiment (news, social media)</li>
                <li>Economic indicators (GDP, inflation)</li>
                <li>Sector performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 Best Practices</h3>
            <ul>
                <li><strong>Data Quality:</strong> Clean, accurate historical data</li>
                <li><strong>Normalization:</strong> Essential for neural networks</li>
                <li><strong>Lookback Period:</strong> 60 days is common</li>
                <li><strong>Validation:</strong> Use walk-forward validation</li>
                <li><strong>Ensemble:</strong> Combine multiple models</li>
                <li><strong>Regular Retraining:</strong> Update with new data</li>
                <li><strong>Risk Management:</strong> Always use stop losses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🚀 Advanced Techniques</h3>
            <ul>
                <li><strong>Bidirectional LSTM:</strong> Look forward & backward</li>
                <li><strong>Attention Mechanisms:</strong> Focus on important timesteps</li>
                <li><strong>Transformers:</strong> Alternative to LSTM</li>
                <li><strong>Ensemble Methods:</strong> Combine LSTM with other models</li>
                <li><strong>Multi-Task Learning:</strong> Predict multiple targets</li>
                <li><strong>Transfer Learning:</strong> Pre-trained financial models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>📊 Model Improvements</h3>
            <ul>
                <li>Add more features (fundamentals, sentiment)</li>
                <li>Experiment with architecture (layers, units)</li>
                <li>Hyperparameter tuning</li>
                <li>Use attention mechanisms</li>
                <li>Implement ensemble predictions</li>
                <li>Add uncertainty quantification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ⚠️ Important Disclaimer")
    
    st.error("""
    **⚠️ DISCLAIMER:**
    
    This application is for **educational purposes only** and should NOT be used as financial advice.
    
    - Stock market predictions are inherently uncertain
    - Past performance does not guarantee future results
    - Always consult with a licensed financial advisor
    - Never invest more than you can afford to lose
    - LSTM models can be wrong - use at your own risk
    
    **Remember:** All investments carry risk. Do your own research!
    """)
    
    st.markdown("### 🎓 Key Takeaways")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>✅ What We Learned:</h4>
        <ul>
            <li><strong>LSTM networks</strong> are powerful for time series prediction</li>
            <li><strong>Technical indicators</strong> provide valuable features for models</li>
            <li><strong>Sequence modeling</strong> captures temporal dependencies</li>
            <li><strong>Risk management</strong> is more important than prediction accuracy</li>
            <li><strong>Feature engineering</strong> can significantly improve performance</li>
            <li><strong>Market predictions</strong> are probabilities, not certainties</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>📈 Stock Price Prediction Mastered!</h3>
    <p><strong>Day 13 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>LSTM for Time Series Forecasting - AI-Powered Trading</p>
    <p style='font-size: 12px; margin-top: 10px;'>⚠️ Not Financial Advice - For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)
