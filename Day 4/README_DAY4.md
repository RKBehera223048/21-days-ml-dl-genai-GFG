# Day 4: Heart Disease Prediction 🏥❤️

## 🎯 Project Overview
An interactive healthcare AI dashboard for predicting heart disease risk using multiple classification algorithms. Features comprehensive medical data analysis, model comparison, and a real-time diagnosis tool.

## ✨ Features

### 📊 Data Overview Tab
- **Patient Statistics:** Total patients, disease/healthy counts, disease rate
- **Sample Patient Data:** View anonymized patient records
- **Feature Descriptions:** Detailed medical attribute explanations
- **Summary Statistics:** Statistical overview of all medical features

### 🔍 Exploratory Data Analysis Tab
- **Disease Distribution:** Pie chart showing healthy vs disease ratio
- **Age Analysis:** Age distribution by diagnosis
- **Gender Analysis:** Disease prevalence by gender
- **Heart Rate Analysis:** Box plots comparing healthy vs diseased patients
- **Correlation Heatmap:** Full medical feature correlation matrix
- **Risk Factor Analysis:** Features most correlated with heart disease

### 🤖 Model Training Tab
- **6 Classification Models:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Support Vector Machine
  - Gradient Boosting
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score
- **Visual Comparisons:** Side-by-side model performance
- **Best Model Identification:** Automatic highlighting

### 📈 Model Comparison Tab
- **Confusion Matrix:** Detailed error analysis
- **Performance Breakdown:** TP, FP, TN, FN counts
- **ROC Curve:** Area Under Curve analysis
- **Metric Explanations:** Understanding each metric's importance
- **Model Selection:** Interactive comparison tool

### 🏥 Diagnosis Tool Tab
- **Interactive Patient Input:** 13 medical parameters
  - Age, Sex
  - Chest Pain Type
  - Blood Pressure
  - Cholesterol
  - Fasting Blood Sugar
  - Resting ECG
  - Max Heart Rate
  - Exercise Angina
  - ST Depression
  - ST Slope
  - Major Vessels
  - Thalassemia
- **Multi-Model Diagnosis:** Predictions from all 6 models
- **Ensemble Prediction:** Consensus across all models
- **Risk Score:** Percentage-based risk assessment
- **Medical Disclaimer:** Important safety notice

### 💡 Key Insights Tab
- Model performance insights
- Medical risk factor analysis
- Technical skills learned
- Classification metrics explained
- Healthcare applications
- Complete performance summary

## 🎨 Design Features
- **Medical Theme:** Red gradient reflecting heart/healthcare
- **Color-Coded Results:** Green for healthy, red for disease risk
- **Professional Cards:** Healthcare-appropriate styling
- **Interactive Charts:** All visualizations with Plotly
- **Responsive Layout:** Works on all devices
- **Clear Metrics:** Easy-to-understand health indicators

## 🛠️ Installation

1. **Install required packages:**
```bash
pip install streamlit plotly pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 How to Run

1. **Navigate to Day 4 folder:**
```bash
cd "c:\Users\Rasak\Desktop\coding\GFG course Project\Day 4"
```

2. **Run the Streamlit app:**
```bash
streamlit run app_day4.py
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
**Heart Disease Dataset** (UCI-based simulation)
- **Samples:** 920 patient records
- **Features:** 13 medical attributes
- **Target:** Binary (0: Healthy, 1: Heart Disease)

### Medical Features:
1. **Age:** Patient age in years
2. **Sex:** 0 = Female, 1 = Male
3. **Chest Pain Type:** 0-3 (different types)
4. **Resting BP:** Blood pressure at rest (mm Hg)
5. **Cholesterol:** Serum cholesterol (mg/dl)
6. **Fasting Blood Sugar:** > 120 mg/dl (0 = No, 1 = Yes)
7. **Rest ECG:** Resting electrocardiographic results (0-2)
8. **Max Heart Rate:** Maximum heart rate achieved
9. **Exercise Angina:** Exercise induced angina (0 = No, 1 = Yes)
10. **Oldpeak:** ST depression induced by exercise
11. **ST Slope:** Slope of peak exercise ST segment (0-2)
12. **Major Vessels:** Number of major vessels colored (0-4)
13. **Thalassemia:** Blood disorder type (0-3)

## 🎯 What I Learned

### Classification Concepts
- ✅ **Binary Classification** - Two-class prediction problems
- ✅ **Classification Metrics** - Accuracy, Precision, Recall, F1
- ✅ **Confusion Matrix** - Understanding prediction errors
- ✅ **ROC-AUC Analysis** - Model discrimination ability
- ✅ **Ensemble Methods** - Combining multiple models
- ✅ **Medical AI Ethics** - Responsible healthcare ML

### Technical Skills
- ✅ **Multi-Model Comparison** - Training & evaluating 6 algorithms
- ✅ **Probability Predictions** - Getting confidence scores
- ✅ **Feature Importance** - Identifying key risk factors
- ✅ **Medical Data Analysis** - Healthcare-specific preprocessing
- ✅ **Interactive Diagnosis** - Real-time prediction interface
- ✅ **Data Scaling** - Standardization for ML models

### Evaluation Metrics Deep Dive
- **Accuracy:** Overall correctness (good for balanced data)
- **Precision:** Of predicted positives, how many correct? (reduce false alarms)
- **Recall:** Of actual positives, how many found? (catch all diseases - most critical!)
- **F1-Score:** Harmonic mean balancing precision & recall
- **True Positive (TP):** Correctly identified disease ✓
- **False Positive (FP):** Healthy predicted as disease ✗
- **True Negative (TN):** Correctly identified healthy ✓
- **False Negative (FN):** Disease missed ✗✗ (MOST DANGEROUS!)

## 🔑 Key Findings

### Model Performance
- **Best Models:** Random Forest and Gradient Boosting typically excel
- **Typical Accuracy:** 75-85% on test data
- **Critical Metric:** Recall (we want to catch all disease cases!)
- **Precision vs Recall:** Balance depends on use case
- **Ensemble Benefit:** Combining models increases reliability

### Medical Risk Factors
- **Age:** Strongest predictor - risk increases with age
- **Gender:** Males at higher risk than females
- **Cholesterol:** High levels strongly correlated with disease
- **Blood Pressure:** Hypertension is major risk factor
- **Exercise Angina:** Pain during exercise is red flag
- **Heart Rate:** Both too high and too low can indicate problems

### Healthcare Insights
- Early detection is critical for treatment success
- Multiple risk factors compound the danger
- Lifestyle modifications can reduce risk significantly
- Regular screening essential for high-risk individuals
- AI can assist but never replace medical professionals

## 📊 Model Comparison Table

| Model | Strengths | Best For | Speed |
|-------|-----------|----------|-------|
| Logistic Regression | Fast, interpretable | Baseline, understanding | ⚡⚡⚡ |
| Decision Tree | Visual, easy to explain | Clinical decision support | ⚡⚡⚡ |
| Random Forest | Robust, accurate | Production deployment | ⚡⚡ |
| KNN | Simple, no training | Small datasets | ⚡⚡ |
| SVM | Powerful, flexible | High-dimensional data | ⚡ |
| Gradient Boosting | Highest accuracy | Competition, research | ⚡ |

## 🎬 Visualizations Included
1. **Pie Charts:** Disease distribution
2. **Histograms:** Age, heart rate distributions
3. **Box Plots:** Comparing diseased vs healthy
4. **Heatmap:** Feature correlation matrix
5. **Bar Charts:** Model performance comparisons
6. **Confusion Matrix:** Error analysis
7. **ROC Curves:** Model discrimination ability

## ⚠️ Medical Disclaimer
**IMPORTANT**: This application is for **educational purposes only**. It is NOT a medical device and should NOT be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice, diagnosis, and treatment.

## 🚀 Real-World Healthcare Applications
- **Screening Programs:** Population-level risk assessment
- **Emergency Triage:** Quick risk evaluation in ER
- **Preventive Care:** Identifying high-risk patients early
- **Treatment Planning:** Guiding intervention strategies
- **Research:** Understanding disease patterns
- **Resource Allocation:** Prioritizing limited healthcare resources

## 🔬 Why This Matters
- **Heart disease** is the leading cause of death worldwide
- **Early detection** dramatically improves survival rates
- **AI assistance** can help overworked healthcare systems
- **Risk prediction** enables preventive interventions
- **Cost reduction** through early intervention vs late-stage treatment
- **Access to care** improved through automated screening

## 📈 Future Enhancements
- Integration with electronic health records (EHR)
- Deep learning models (Neural Networks)
- Temporal analysis (disease progression over time)
- Explainable AI (SHAP values, LIME)
- Multi-class classification (disease severity levels)
- Real-time monitoring integration

---

**Part of:** 21 Projects, 21 Days: ML, Deep Learning & GenAI - GeeksforGeeks  
**Day:** 4 of 21  
**Topic:** Heart Disease Prediction with Classification Models

**⚕️ Remember:** AI is a tool to assist healthcare professionals, not replace them!
