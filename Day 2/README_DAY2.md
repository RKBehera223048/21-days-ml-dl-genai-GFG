# Day 2: Netflix Content Strategy Analysis 🎬

## 🎯 Project Overview
An interactive Netflix-themed Streamlit dashboard analyzing Netflix's content strategy, including trends in content production, popular genres, ratings, and global distribution patterns.

## ✨ Features

### 📊 Overview Tab
- Total content metrics (titles, movies, TV shows, countries)
- Content type distribution (pie chart)
- Rating distribution analysis
- Sample content preview
- Data quality overview with missing value analysis

### 🎬 Content Analysis Tab
- **Movies Analysis:**
  - Duration distribution histogram
  - Statistical summary (avg, median, min, max)
  - Total movie count
- **TV Shows Analysis:**
  - Season distribution
  - Statistical summary
  - Total TV show count
- **Release Year Trends:**
  - Interactive line chart showing production trends since 2000

### 🌍 Global Insights Tab
- Top 15 content producing countries
- Content type distribution by country (stacked bars)
- Top 15 most prolific directors
- Global reach analysis

### 📈 Trends Over Time Tab
- Content addition by year (area chart)
- Content addition by month
- Yearly trends split by content type
- Growth pattern analysis

### 🎭 Genre Analysis Tab
- Top 20 genres on Netflix
- Genre distribution by content type
- Top 5 genre trends over time
- Interactive genre comparisons

### 💡 Key Insights Tab
- Content strategy insights
- Technical skills learned
- Statistical highlights
- Data quality observations
- Business insights
- Quick reference summary table

## 🎨 Design Features
- **Netflix Theme:** Dark background (#141414) with signature red (#E50914)
- **Custom CSS:** Netflix-inspired styling throughout
- **Interactive Charts:** All visualizations built with Plotly
- **Responsive Layout:** Adapts to different screen sizes
- **Professional Metrics:** Gradient cards with shadow effects

## 🛠️ Installation

1. **Install required packages:**
```bash
pip install streamlit plotly pandas numpy matplotlib seaborn
```

## 🚀 How to Run

1. **Navigate to Day 2 folder:**
```bash
cd "c:\Users\Rasak\Desktop\coding\GFG course Project\Day 2"
```

2. **Run the Streamlit app:**
```bash
streamlit run app_day2.py
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

## 📁 Required Files
- `app_day2.py` - Main Streamlit application
- `netflix_titles.csv` - Dataset file (should be in ../Datasets/ folder)

## 🎯 What I Learned

### Content Analysis Skills
- ✅ Analyzing content distribution patterns
- ✅ Time series trend analysis
- ✅ Genre popularity analysis
- ✅ Global content distribution mapping
- ✅ Comparative analysis (Movies vs TV Shows)
- ✅ String processing for categorical data

### Technical Skills
- ✅ Advanced Plotly visualizations
- ✅ Custom theming with Netflix branding
- ✅ String manipulation (split, explode operations)
- ✅ Date/time data handling
- ✅ Aggregation and grouping techniques
- ✅ Interactive dashboard design

## 🔑 Key Findings
- **Content Distribution:** Movies dominate (~70%) vs TV Shows (~30%)
- **Global Reach:** Content from 100+ countries
- **Top Producer:** United States leads content production
- **Popular Genres:** International content and Dramas are most common
- **Growth Pattern:** Significant content addition increase in recent years
- **Duration:** Average movie length is ~90-100 minutes
- **TV Shows:** Most have 1-2 seasons

## 📊 Dataset Information
- **Total Titles:** 8,000+ entries
- **Columns:** 12 features including type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description
- **Missing Data:** Some director, cast, and country information missing
- **Time Range:** Content from various decades up to 2021

## 🎬 Visualizations Included
1. **Pie Charts:** Content type distribution
2. **Bar Charts:** Ratings, countries, directors, genres
3. **Histograms:** Movie duration, TV show seasons
4. **Line Charts:** Trends over time, genre evolution
5. **Area Charts:** Content growth over years
6. **Stacked Bar Charts:** Content type by country/genre
7. **Interactive Tables:** Summary statistics and metrics

---

**Part of:** 21 Projects, 21 Days: ML, Deep Learning & GenAI - GeeksforGeeks  
**Day:** 2 of 21  
**Topic:** Netflix Content Strategy Analysis - EDA
