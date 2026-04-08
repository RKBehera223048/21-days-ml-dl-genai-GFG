# Day 3: House Price Prediction with AI 🏠

## 🎯 Project Overview
An interactive machine learning dashboard for predicting house prices using multiple regression algorithms. Features comprehensive EDA, model comparison, and real-time predictions with the California Housing dataset.

## ✨ Features

### 📊 Data Overview Tab
- **Key Metrics:** Total samples, feature count, price statistics
- **Sample Data Preview:** View raw dataset
- **Feature Descriptions:** Detailed explanation of each feature
- **Summary Statistics:** Statistical overview of all features
- **Data Info:** Column types and non-null counts

### 🔍 Exploratory Data Analysis Tab
- **Price Distribution:** Histogram of house values
- **Age Distribution:** House age patterns
- **Income vs Price:** Scatter plot showing relationship
- **Rooms vs Price:** Average rooms impact analysis
- **Correlation Heatmap:** Full feature correlation matrix
- **Feature Importance:** Correlation with target variable

### 🤖 Model Training Tab
- **5 ML Models Trained:**
  - Linear Regression (baseline)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
  - Random Forest (ensemble)
  - Gradient Boosting (advanced ensemble)
- **Performance Metrics:** RMSE, MAE, R² for each model
- **Visual Comparisons:** Bar charts for RMSE and R² scores
- **Best Model Highlighting:** Automatic identification

### 📈 Model Comparison Tab
- **Predictions vs Actual:** Scatter plot with perfect prediction line
- **Residual Analysis:** Residual plot and distribution
- **Feature Importance:** For tree-based models (RF & GB)
- **Model Selection:** Interactive model comparison
- **Error Analysis:** Comprehensive residual diagnostics

### 🎯 Predictions Tab
- **Interactive Input:** 8 sliders for all features
  - Median Income
  - House Age
  - Average Rooms
  - Average Bedrooms
  - Population
  - Average Occupancy
  - Latitude & Longitude
- **Multi-Model Predictions:** Get predictions from all 5 models
- **Ensemble Average:** Combined prediction across models
- **Beautiful Cards:** Gradient-styled prediction displays

### 💡 Key Insights Tab
- Model performance insights
- Feature importance analysis
- Technical skills learned summary
- Evaluation metrics explained
- Real-world applications
- Performance summary table

## 🎨 Design Features
- **Purple Gradient Theme:** Modern purple-pink gradient background
- **Glassmorphism:** Translucent cards with backdrop blur effects
- **Animated Metrics:** Gradient-filled metric cards
- **Interactive Charts:** All visualizations built with Plotly
- **Responsive Layout:** Adapts to all screen sizes
- **Prediction Cards:** Beautiful gradient cards for predictions

## 🛠️ Installation

1. **Install required packages:**
```bash
pip install streamlit plotly pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 How to Run

1. **Navigate to Day 3 folder:**
```bash
cd "c:\Users\Rasak\Desktop\coding\GFG course Project\Day 3"
```

2. **Run the Streamlit app:**
```bash
streamlit run app_day3.py
```

3. **Open your browser** at `http://localhost:8501`

## 📋 Requirements
- Python 3.7+
- streamlit
- plotly
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 📁 Dataset
**California Housing Dataset** (Built into scikit-learn)
- **Samples:** 20,640 housing districts
- **Features:** 8 numeric features
  - MedInc: Median income
  - HouseAge: Median house age
  - AveRooms: Average rooms per household
  - AveBedrms: Average bedrooms per household
  - Population: Block group population
  - AveOccup: Average household size
  - Latitude: Block latitude
  - Longitude: Block longitude
- **Target:** MedHouseVal (Median house value in $100,000s)

## 🎯 What I Learned

### Machine Learning Concepts
- ✅ **Regression fundamentals** - Predicting continuous values
- ✅ **Feature engineering** - Understanding feature relationships
- ✅ **Model selection** - Comparing multiple algorithms
- ✅ **Ensemble methods** - Random Forest & Gradient Boosting
- ✅ **Regularization** - Ridge & Lasso techniques
- ✅ **Cross-validation** - Train-test split strategies

### Technical Skills
- ✅ **Data preprocessing** - Scaling, normalization
- ✅ **Scikit-learn** - Model training and evaluation
- ✅ **Performance metrics** - RMSE, MAE, R² interpretation
- ✅ **Visualization** - Residual plots, scatter plots, heatmaps
- ✅ **Feature importance** - Identifying key predictors
- ✅ **Interactive predictions** - Real-time model inference

### Evaluation Metrics
- **RMSE (Root Mean Squared Error):** Average prediction error
- **MAE (Mean Absolute Error):** Mean absolute deviation
- **R² Score:** Percentage of variance explained (0-1)
- **Residual Analysis:** Understanding prediction errors

## 🔑 Key Findings

### Model Performance
- **Best Model:** Gradient Boosting or Random Forest typically perform best
- **R² Scores:** Usually 0.75-0.85 range (75-85% variance explained)
- **Average Error:** Models predict within $40K-60K of actual prices
- **Linear Models:** Good baselines but limited by linear assumptions
- **Ensemble Methods:** Significantly outperform simple models

### Feature Insights
- **Top Predictor:** Median Income (strongest correlation ~0.68)
- **Location:** Latitude & Longitude are crucial factors
- **House Age:** Generally negative correlation with price
- **Rooms:** Positive impact but with diminishing returns
- **Population:** Minimal direct effect on individual prices

### Business Insights
- Income is the strongest single predictor of house value
- Coastal locations command premium prices
- Newer houses generally valued higher
- Room count matters but bedroom ratio is important
- Dense populations don't always mean higher prices

## 📊 Model Comparison

| Model | Typical R² | Strengths | Use Case |
|-------|-----------|-----------|----------|
| Linear Regression | ~0.60 | Fast, interpretable | Baseline, simple problems |
| Ridge Regression | ~0.62 | Handles multicollinearity | Correlated features |
| Lasso Regression | ~0.62 | Feature selection | Many features, some irrelevant |
| Random Forest | ~0.80 | Robust, non-linear | Complex relationships |
| Gradient Boosting | ~0.82 | Highest accuracy | Production models |

## 🎬 Visualizations Included
1. **Histograms:** Price and age distributions
2. **Scatter Plots:** Income vs price, rooms vs price
3. **Heatmap:** Full correlation matrix
4. **Bar Charts:** Model performance comparisons
5. **Prediction Plot:** Actual vs predicted values
6. **Residual Plot:** Error analysis
7. **Feature Importance:** Tree-based model insights

## 🚀 Real-World Applications
- **Real Estate Valuation:** Automated property pricing
- **Investment Analysis:** Identifying undervalued properties
- **Mortgage Lending:** Risk assessment and loan amounts
- **Urban Planning:** Understanding market dynamics
- **Market Forecasting:** Predicting future trends

## 📈 Next Steps
- Implement advanced feature engineering
- Try deep learning models (Neural Networks)
- Add cross-validation for robust evaluation
- Hyperparameter tuning for optimal performance
- Deploy model as API for production use

---

**Part of:** 21 Projects, 21 Days: ML, Deep Learning & GenAI - GeeksforGeeks  
**Day:** 3 of 21  
**Topic:** House Price Prediction with Regression Models
