import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Day 2: Netflix Content Strategy",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix theme
st.markdown("""
    <style>
    .main {
        background-color: #141414;
    }
    .stMetric {
        background: linear-gradient(135deg, #E50914 0%, #831010 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(229, 9, 20, 0.3);
        color: white;
    }
    .stMetric label {
        color: #ffffff !important;
    }
    .stMetric .css-1xarl3l {
        color: #ffffff !important;
    }
    h1 {
        color: #E50914;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    h2 {
        color: #E50914;
        font-weight: 600;
    }
    h3 {
        color: #ffffff;
        font-weight: 500;
    }
    .highlight-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #E50914;
        margin: 10px 0;
        color: white;
    }
    .netflix-card {
        background: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E50914;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../Datasets/netflix_titles.csv')
        return df
    except:
        try:
            df = pd.read_csv('netflix_titles.csv')
            return df
        except:
            st.error("⚠️ Please ensure 'netflix_titles.csv' is available!")
            return None

# Data preprocessing
@st.cache_data
def preprocess_data(df):
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['month_name'] = df['date_added'].dt.month_name()
    return df

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🎬 Netflix Content Strategy Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #999;'>Day 2: Cracking the Code of Netflix's Content</h3>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Project Overview")
    st.image("https://images.ctfassets.net/y2ske730sjqp/1aONibCke6nileyep0xkqs/2c401b05a07288746ddf3bd3943fbc76/BrandAssets_Logos_01-Wordmark.jpg?w=940", 
             use_container_width=True)
    
    st.markdown("""
    <div style='background: #1a1a1a; padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #E50914; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: white;'>
            <li>📊 Content trend analysis</li>
            <li>🌍 Global content distribution</li>
            <li>🎭 Genre popularity analysis</li>
            <li>📈 Time series analysis</li>
            <li>🎬 Movies vs TV Shows patterns</li>
            <li>☁️ WordCloud generation</li>
            <li>📉 Rating distribution analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999;'>
        <p><strong>Day 2 of 21</strong></p>
        <p>GeeksforGeeks Course</p>
    </div>
    """, unsafe_allow_html=True)

# Load data
df = load_data()

if df is not None:
    df = preprocess_data(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🎬 Content Analysis", 
        "🌍 Global Insights",
        "📈 Trends Over Time",
        "🎭 Genre Analysis",
        "💡 Key Findings"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.markdown("## 📊 Netflix Content Overview")
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Titles", f"{len(df):,}")
        with col2:
            movies_count = len(df[df['type'] == 'Movie'])
            st.metric("Movies", f"{movies_count:,}")
        with col3:
            tv_count = len(df[df['type'] == 'TV Show'])
            st.metric("TV Shows", f"{tv_count:,}")
        with col4:
            countries = df['country'].dropna().str.split(',').explode().nunique()
            st.metric("Countries", f"{countries:,}")
        with col5:
            years_range = f"{int(df['release_year'].min())}-{int(df['release_year'].max())}"
            st.metric("Years Range", years_range)
        
        st.markdown("### 📋 Sample Netflix Content")
        display_df = df[['type', 'title', 'director', 'country', 'release_year', 'rating', 'duration']].head(10)
        st.dataframe(display_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📺 Content Type Distribution")
            type_counts = df['type'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.4,
                marker=dict(colors=['#E50914', '#831010']),
                textinfo='label+percent',
                textfont=dict(size=14, color='white')
            )])
            fig.update_layout(
                title="Movies vs TV Shows",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⭐ Rating Distribution")
            rating_counts = df['rating'].value_counts().head(10)
            
            fig = px.bar(
                x=rating_counts.values,
                y=rating_counts.index,
                orientation='h',
                title='Top 10 Content Ratings',
                labels={'x': 'Count', 'y': 'Rating'},
                color=rating_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Quality
        st.markdown("### 🔍 Data Quality Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            if len(missing_df) > 0:
                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing %',
                    title='Missing Data Percentage',
                    color='Missing %',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Content Analysis
    with tab2:
        st.markdown("## 🎬 Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎥 Movies Analysis")
            movies_df = df[df['type'] == 'Movie'].copy()
            
            # Duration analysis for movies
            movies_df['duration_min'] = movies_df['duration'].str.extract('(\d+)').astype(float)
            
            fig = px.histogram(
                movies_df.dropna(subset=['duration_min']),
                x='duration_min',
                nbins=50,
                title='Movie Duration Distribution',
                labels={'duration_min': 'Duration (minutes)', 'count': 'Number of Movies'},
                color_discrete_sequence=['#E50914']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Movie statistics
            st.markdown("#### 📊 Movie Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Average Duration', 'Median Duration', 'Min Duration', 'Max Duration', 'Total Movies'],
                'Value': [
                    f"{movies_df['duration_min'].mean():.0f} min",
                    f"{movies_df['duration_min'].median():.0f} min",
                    f"{movies_df['duration_min'].min():.0f} min",
                    f"{movies_df['duration_min'].max():.0f} min",
                    f"{len(movies_df):,}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📺 TV Shows Analysis")
            tv_df = df[df['type'] == 'TV Show'].copy()
            
            # Season analysis for TV shows
            tv_df['num_seasons'] = tv_df['duration'].str.extract('(\d+)').astype(float)
            
            season_counts = tv_df['num_seasons'].value_counts().sort_index().head(10)
            
            fig = px.bar(
                x=season_counts.index,
                y=season_counts.values,
                title='TV Shows by Number of Seasons',
                labels={'x': 'Number of Seasons', 'y': 'Count'},
                color=season_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # TV Show statistics
            st.markdown("#### 📊 TV Show Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Average Seasons', 'Median Seasons', 'Min Seasons', 'Max Seasons', 'Total TV Shows'],
                'Value': [
                    f"{tv_df['num_seasons'].mean():.1f}",
                    f"{tv_df['num_seasons'].median():.0f}",
                    f"{tv_df['num_seasons'].min():.0f}",
                    f"{tv_df['num_seasons'].max():.0f}",
                    f"{len(tv_df):,}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Release Year Analysis
        st.markdown("### 📅 Content by Release Year")
        
        release_year_type = df.groupby(['release_year', 'type']).size().reset_index(name='count')
        release_year_type = release_year_type[release_year_type['release_year'] >= 2000]
        
        fig = px.line(
            release_year_type,
            x='release_year',
            y='count',
            color='type',
            title='Content Production Trends (2000 onwards)',
            labels={'release_year': 'Release Year', 'count': 'Number of Titles', 'type': 'Content Type'},
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#831010'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Global Insights
    with tab3:
        st.markdown("## 🌍 Global Content Distribution")
        
        # Top producing countries
        countries_list = df['country'].dropna().str.split(',').explode().str.strip()
        top_countries = countries_list.value_counts().head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🗺️ Top 15 Content Producing Countries")
            fig = px.bar(
                x=top_countries.values,
                y=top_countries.index,
                orientation='h',
                title='Content by Country',
                labels={'x': 'Number of Titles', 'y': 'Country'},
                color=top_countries.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎬 Content Type by Top Countries")
            
            # Get content type distribution for top 10 countries
            top_10_countries = top_countries.head(10).index.tolist()
            
            country_type_data = []
            for country in top_10_countries:
                movies = len(df[(df['country'].str.contains(country, na=False)) & (df['type'] == 'Movie')])
                tv = len(df[(df['country'].str.contains(country, na=False)) & (df['type'] == 'TV Show')])
                country_type_data.append({'Country': country, 'Movies': movies, 'TV Shows': tv})
            
            country_type_df = pd.DataFrame(country_type_data)
            
            fig = go.Figure(data=[
                go.Bar(name='Movies', x=country_type_df['Country'], y=country_type_df['Movies'], 
                       marker_color='#E50914'),
                go.Bar(name='TV Shows', x=country_type_df['Country'], y=country_type_df['TV Shows'], 
                       marker_color='#831010')
            ])
            fig.update_layout(
                barmode='stack',
                title='Movies vs TV Shows by Country',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500,
                xaxis_title='Country',
                yaxis_title='Number of Titles'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Director Analysis
        st.markdown("### 🎬 Top Directors on Netflix")
        directors_list = df['director'].dropna().str.split(',').explode().str.strip()
        top_directors = directors_list.value_counts().head(15)
        
        fig = px.bar(
            x=top_directors.values,
            y=top_directors.index,
            orientation='h',
            title='Top 15 Most Prolific Directors',
            labels={'x': 'Number of Titles', 'y': 'Director'},
            color=top_directors.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Trends Over Time
    with tab4:
        st.markdown("## 📈 Netflix Content Addition Trends")
        
        # Filter data with valid dates
        df_with_dates = df.dropna(subset=['year_added'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📅 Content Added by Year")
            yearly_additions = df_with_dates['year_added'].value_counts().sort_index()
            
            fig = px.area(
                x=yearly_additions.index,
                y=yearly_additions.values,
                title='Netflix Content Growth Over Years',
                labels={'x': 'Year', 'y': 'Number of Titles Added'},
                color_discrete_sequence=['#E50914']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📆 Content Added by Month")
            monthly_additions = df_with_dates['month_name'].value_counts()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_additions = monthly_additions.reindex([m for m in month_order if m in monthly_additions.index])
            
            fig = px.bar(
                x=monthly_additions.index,
                y=monthly_additions.values,
                title='Content Addition by Month',
                labels={'x': 'Month', 'y': 'Number of Titles'},
                color=monthly_additions.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Yearly trend by content type
        st.markdown("### 📊 Yearly Content Addition by Type")
        yearly_type = df_with_dates.groupby(['year_added', 'type']).size().reset_index(name='count')
        
        fig = px.bar(
            yearly_type,
            x='year_added',
            y='count',
            color='type',
            title='Movies vs TV Shows Added Each Year',
            labels={'year_added': 'Year', 'count': 'Number of Titles', 'type': 'Content Type'},
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#831010'},
            barmode='group'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Genre Analysis
    with tab5:
        st.markdown("## 🎭 Genre Analysis")
        
        # Extract and count genres
        genres_list = df['listed_in'].dropna().str.split(',').explode().str.strip()
        top_genres = genres_list.value_counts().head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Top 20 Genres on Netflix")
            fig = px.bar(
                x=top_genres.values,
                y=top_genres.index,
                orientation='h',
                title='Most Popular Genres',
                labels={'x': 'Number of Titles', 'y': 'Genre'},
                color=top_genres.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎬 Genre Distribution by Content Type")
            
            # Get top 10 genres for cleaner visualization
            top_10_genres = top_genres.head(10).index.tolist()
            
            genre_type_data = []
            for genre in top_10_genres:
                movies = len(df[(df['listed_in'].str.contains(genre, na=False)) & (df['type'] == 'Movie')])
                tv = len(df[(df['listed_in'].str.contains(genre, na=False)) & (df['type'] == 'TV Show')])
                genre_type_data.append({'Genre': genre, 'Movies': movies, 'TV Shows': tv})
            
            genre_type_df = pd.DataFrame(genre_type_data)
            
            fig = go.Figure(data=[
                go.Bar(name='Movies', y=genre_type_df['Genre'], x=genre_type_df['Movies'], 
                       orientation='h', marker_color='#E50914'),
                go.Bar(name='TV Shows', y=genre_type_df['Genre'], x=genre_type_df['TV Shows'], 
                       orientation='h', marker_color='#831010')
            ])
            fig.update_layout(
                barmode='stack',
                title='Top 10 Genres: Movies vs TV Shows',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600,
                xaxis_title='Number of Titles',
                yaxis_title='Genre'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Genre trends over time
        st.markdown("### 📈 Top Genre Trends Over Time")
        
        # Select top 5 genres
        top_5_genres = top_genres.head(5).index.tolist()
        
        # Create year-genre data
        df_with_year = df.dropna(subset=['year_added'])
        genre_year_data = []
        
        for year in sorted(df_with_year['year_added'].unique()):
            year_df = df_with_year[df_with_year['year_added'] == year]
            for genre in top_5_genres:
                count = year_df['listed_in'].str.contains(genre, na=False).sum()
                genre_year_data.append({'Year': year, 'Genre': genre, 'Count': count})
        
        genre_year_df = pd.DataFrame(genre_year_data)
        
        fig = px.line(
            genre_year_df,
            x='Year',
            y='Count',
            color='Genre',
            title='Top 5 Genre Trends Over Time',
            labels={'Count': 'Number of Titles'},
            markers=True
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Key Findings
    with tab6:
        st.markdown("## 💡 Key Insights & Learnings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
                <h3>🎯 Content Strategy Insights</h3>
                <ul>
                    <li><strong>Content Split:</strong> Movies dominate Netflix catalog (~{:.1f}% vs {:.1f}% TV Shows)</li>
                    <li><strong>Global Reach:</strong> Content from {} countries available</li>
                    <li><strong>Top Producer:</strong> {} leads with {:,} titles</li>
                    <li><strong>Content Rating:</strong> Most content is rated {} and {}</li>
                    <li><strong>Average Movie:</strong> {:.0f} minutes duration</li>
                    <li><strong>TV Shows:</strong> Most have {:.0f} season(s)</li>
                    <li><strong>Popular Genre:</strong> {} dominates the platform</li>
                    <li><strong>Peak Addition:</strong> Most content added in recent years</li>
                </ul>
            </div>
            """.format(
                (len(df[df['type'] == 'Movie']) / len(df) * 100),
                (len(df[df['type'] == 'TV Show']) / len(df) * 100),
                countries,
                top_countries.index[0],
                top_countries.values[0],
                df['rating'].value_counts().index[0],
                df['rating'].value_counts().index[1],
                movies_df['duration_min'].mean(),
                tv_df['num_seasons'].median(),
                top_genres.index[0]
            ), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>📊 Statistical Highlights</h3>
                <ul>
                    <li><strong>Total Titles:</strong> {:,} pieces of content</li>
                    <li><strong>Release Years:</strong> {} to {}</li>
                    <li><strong>Directors:</strong> {:,} unique directors</li>
                    <li><strong>Genres:</strong> {} different genre categories</li>
                    <li><strong>Ratings:</strong> {} different content ratings</li>
                </ul>
            </div>
            """.format(
                len(df),
                int(df['release_year'].min()),
                int(df['release_year'].max()),
                df['director'].dropna().str.split(',').explode().nunique(),
                df['listed_in'].dropna().str.split(',').explode().nunique(),
                df['rating'].nunique()
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight-box">
                <h3>📚 Technical Skills Learned</h3>
                <ul>
                    <li><strong>Data Cleaning:</strong> Handling missing values and text data</li>
                    <li><strong>String Processing:</strong> Splitting and exploding comma-separated values</li>
                    <li><strong>Time Series Analysis:</strong> Analyzing trends over years and months</li>
                    <li><strong>Categorical Analysis:</strong> Genre, country, and rating distributions</li>
                    <li><strong>Text Extraction:</strong> Extracting numeric values from duration strings</li>
                    <li><strong>Advanced Visualizations:</strong> Interactive Plotly charts</li>
                    <li><strong>Comparative Analysis:</strong> Movies vs TV Shows patterns</li>
                    <li><strong>Aggregation:</strong> Grouping and counting techniques</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>🔍 Data Quality Observations</h3>
                <ul>
                    <li><strong>Director Data:</strong> ~{:.1f}% missing director information</li>
                    <li><strong>Cast Data:</strong> ~{:.1f}% missing cast information</li>
                    <li><strong>Country Data:</strong> ~{:.1f}% missing country information</li>
                    <li><strong>Date Added:</strong> ~{:.1f}% missing date information</li>
                </ul>
            </div>
            """.format(
                (df['director'].isna().sum() / len(df) * 100),
                (df['cast'].isna().sum() / len(df) * 100),
                (df['country'].isna().sum() / len(df) * 100),
                (df['date_added'].isna().sum() / len(df) * 100)
            ), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>🚀 Business Insights</h3>
                <ul>
                    <li><strong>Content Strategy:</strong> Focus on international content diversification</li>
                    <li><strong>Production Trend:</strong> Significant increase in original content</li>
                    <li><strong>Genre Balance:</strong> Wide variety to cater to different audiences</li>
                    <li><strong>Regional Focus:</strong> Strong presence in US, India, and UK markets</li>
                    <li><strong>Content Age:</strong> Mix of classic and recent releases</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary metrics
        st.markdown("### 📈 Quick Reference Table")
        summary_df = pd.DataFrame({
            'Metric': [
                'Total Content',
                'Movies',
                'TV Shows',
                'Countries',
                'Directors',
                'Total Genres',
                'Average Movie Duration',
                'Median TV Show Seasons',
                'Most Common Rating',
                'Top Genre'
            ],
            'Value': [
                f"{len(df):,}",
                f"{len(df[df['type'] == 'Movie']):,}",
                f"{len(df[df['type'] == 'TV Show']):,}",
                f"{countries:,}",
                f"{df['director'].dropna().str.split(',').explode().nunique():,}",
                f"{df['listed_in'].dropna().str.split(',').explode().nunique():,}",
                f"{movies_df['duration_min'].mean():.0f} min",
                f"{tv_df['num_seasons'].median():.0f}",
                f"{df['rating'].value_counts().index[0]}",
                f"{top_genres.index[0]}"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #E50914; padding: 20px;'>
    <h3>🎬 Netflix Content Strategy Analysis Complete!</h3>
    <p style='color: #999;'><strong>Day 2 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p style='color: #666;'>Exploring trends, genres, and global content distribution</p>
</div>
""", unsafe_allow_html=True)
