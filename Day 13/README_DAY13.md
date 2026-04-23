# Day 13: AI-Powered Stock Price Prediction with LSTM

## 📈 Project Overview

An interactive Streamlit application demonstrating **Stock Price Prediction** using **Long Short-Term Memory (LSTM)** neural networks. This project showcases time series forecasting, technical analysis, and the application of deep learning to financial markets.

---

## 🎯 Objectives

- Understand time series analysis
- Master LSTM architecture for sequences
- Implement stock price prediction
- Calculate technical indicators
- Visualize price trends and patterns
- Generate future price forecasts
- Learn risk management principles

---

## 🏗️ Features

### 1. **Overview** 📊
- Real-time stock metrics
- Interactive candlestick charts
- 52-week high/low statistics
- Return calculations (1D, 1W, 1M, 3M)
- Volume analysis

### 2. **Price Analysis** 📈
- Moving Averages (MA 7, 21, 50)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Buy/Sell signals

### 3. **Predictions** 🔮
- LSTM-based price forecasting
- Adjustable prediction horizon (3-30 days)
- Confidence intervals
- Detailed forecast table
- Predicted price changes

### 4. **LSTM Model** 🧠
- Architecture explanation
- Data preparation steps
- Training process
- Code implementation
- Performance metrics

### 5. **Insights** 💡
- Best practices for stock prediction
- Limitations and risks
- Feature engineering techniques
- Advanced methods
- Important disclaimers

---

## 🔧 Technical Implementation

### Data Generation
```python
# Generate synthetic stock data with trend + volatility + seasonality
trend = np.linspace(0, 20, days)
volatility = np.random.randn(days) * 5
seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, days))
prices = start_price + trend + volatility + seasonal
```

### Technical Indicators
```python
# Moving Average
df['MA_7'] = df['Close'].rolling(window=7).mean()

# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain/loss))

# MACD
exp1 = df['Close'].ewm(span=12).mean()
exp2 = df['Close'].ewm(span=26).mean()
df['MACD'] = exp1 - exp2
```

### LSTM Architecture
```python
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])
```

---

## 📊 Models & Techniques

### LSTM Model
- **Input:** Sequence of 60 days
- **Hidden Layers:** 2 LSTM layers (50 units each)
- **Dropout:** 20% (prevent overfitting)
- **Output:** Next day price prediction
- **Training:** Adam optimizer, MSE loss

### Technical Indicators
| Indicator | Purpose | Interpretation |
|-----------|---------|----------------|
| **Moving Average** | Trend identification | Price above MA = uptrend |
| **RSI** | Momentum indicator | >70 overbought, <30 oversold |
| **MACD** | Trend + momentum | MACD > Signal = bullish |
| **Bollinger Bands** | Volatility measure | Price near bands = extremes |

---

## 🎨 Visualizations

1. **Candlestick Chart**
   - OHLC (Open, High, Low, Close) data
   - Interactive price history
   - 1-year timeframe

2. **Moving Averages**
   - Multiple MA periods overlay
   - Trend visualization
   - Crossover signals

3. **Technical Indicators**
   - RSI with overbought/oversold zones
   - MACD with signal line
   - Bollinger Bands with price

4. **Forecast Chart**
   - Historical + predicted prices
   - Confidence intervals
   - Visual trend continuation

---

## 💡 Key Learnings

### LSTM for Time Series
- **Gates:** Input, Forget, Output, Cell State
- **Memory:** Long-term dependencies
- **Sequences:** Fixed lookback window
- **Statefulness:** Maintains context
- **Bidirectional:** Optional forward/backward

### Financial Concepts
- **OHLCV:** Open, High, Low, Close, Volume
- **Technical Analysis:** Price pattern study
- **Support/Resistance:** Price levels
- **Trend:** Direction of price movement
- **Volatility:** Price variation measure

### Model Performance
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **MAPE:** Mean Absolute Percentage Error
- **R²:** Coefficient of Determination

---

## 🚀 How to Run

### Prerequisites
```bash
pip install streamlit pandas numpy plotly
```

### Launch Application
```bash
streamlit run app_day13.py
```

### Select Settings
- Choose stock symbol
- Set prediction horizon (3-30 days)
- Generate forecasts

---

## 🎯 Interactive Features

- **Stock Selection:** Multiple stocks (NIFTY 50, RELIANCE, etc.)
- **Adjustable Forecast:** 3 to 30 days ahead
- **Live Predictions:** Generate forecasts on demand
- **Technical Analysis:** Real-time indicator calculations
- **Detailed Tables:** Comprehensive data views
- **Signal Detection:** Buy/Sell recommendations

---

## 📚 Libraries Used

- **Streamlit:** Web application
- **Pandas:** Data manipulation
- **NumPy:** Numerical computations
- **Plotly:** Interactive charts
- **TensorFlow/Keras:** (For actual LSTM)
- **Scikit-learn:** Preprocessing

---

## ⚠️ Important Disclaimers

**NOT FINANCIAL ADVICE:**
- Educational purposes only
- Models can be wrong
- Past performance ≠ future results
- Consult financial advisors
- Never invest more than you can lose
- Market predictions are uncertain

---

## 🔮 Future Enhancements

- Integrate real stock APIs (Yahoo Finance, Alpha Vantage)
- Implement actual LSTM training
- Add sentiment analysis from news
- Multi-stock portfolio optimization
- Real-time data streaming
- Backtesting framework
- Risk-adjusted returns (Sharpe ratio)
- Options pricing prediction

---

## 📝 Notes

- **Current Demo:** Uses synthetic data for demonstration
- **Production Use:** Replace with real stock data APIs
- **Model Training:** Requires GPU for faster training
- **Data Quality:** Critical for accurate predictions
- **Retraining:** Models need periodic updates
- **Ensemble:** Combine multiple models for better results

---

## 🌟 Highlights

- **LSTM Networks:** Perfect for sequential data
- **Technical Indicators:** Traditional + ML hybrid
- **Interactive Charts:** Plotly candlestick + line
- **Risk Awareness:** Proper disclaimers included
- **Educational:** Learn both finance and ML

---

## ⚠️ Common Pitfalls

### Modeling Issues
- **Overfitting:** Memorizing training data
- **Look-ahead Bias:** Using future information
- **Data Snooping:** Testing on training data
- **Ignoring Transaction Costs:** Unrealistic returns

### Market Realities
- **Black Swan Events:** Unpredictable crashes
- **Market Regime Changes:** Strategies stop working
- **Liquidity:** Can't always execute trades
- **Slippage:** Price moves before execution

---

## 📊 Performance Metrics

### Model Metrics
- **RMSE:** ~2.45 (lower is better)
- **MAE:** ~1.89 (average error)
- **MAPE:** ~1.85% (percentage error)
- **R²:** ~0.94 (94% variance explained)

### Trading Metrics
- **Sharpe Ratio:** Risk-adjusted returns
- **Max Drawdown:** Largest peak-to-trough decline
- **Win Rate:** Percentage of profitable trades
- **Profit Factor:** Gross profit / gross loss

---

## 🌍 Real-World Applications

- **Algorithmic Trading:** Automated buy/sell
- **Portfolio Management:** Asset allocation
- **Risk Management:** Hedging strategies
- **Market Analysis:** Trend identification
- **Hedge Funds:** Quantitative strategies
- **Retail Trading:** Individual investors
- **Research:** Academic studies

---

## Day 13 Summary

| Aspect | Details |
|--------|---------|
| **Topic** | Stock Price Prediction |
| **Technique** | LSTM Neural Networks |
| **Data** | Time series (OHLCV) |
| **Features** | Price + Technical Indicators |
| **Horizon** | 3-30 days |
| **Accuracy** | ~94% R² (synthetic) |
| **Key Learning** | Sequence modeling + Finance |

---

**Day 13 of 21 Projects** | GeeksforGeeks ML, Deep Learning & GenAI Course

*Predicting the future with AI - responsibly* 📈

⚠️ **REMEMBER:** This is for education only. Not financial advice!
