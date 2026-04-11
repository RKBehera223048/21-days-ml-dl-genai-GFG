import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Day 5: Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Shopping/Retail theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    .segment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../Datasets/Mall_Customers.csv')
        return df
    except:
        try:
            df = pd.read_csv('Mall_Customers.csv')
            return df
        except:
            st.error("⚠️ Please ensure 'Mall_Customers.csv' is available!")
            return None

# Perform clustering
@st.cache_data
def perform_clustering(X_scaled, n_clusters=5):
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hierarchical.fit_predict(X_scaled)
    
    # Calculate metrics
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    hier_silhouette = silhouette_score(X_scaled, hier_labels)
    
    kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)
    hier_db = davies_bouldin_score(X_scaled, hier_labels)
    
    kmeans_ch = calinski_harabasz_score(X_scaled, kmeans_labels)
    hier_ch = calinski_harabasz_score(X_scaled, hier_labels)
    
    return {
        'kmeans': {
            'labels': kmeans_labels,
            'centroids': kmeans.cluster_centers_,
            'silhouette': kmeans_silhouette,
            'davies_bouldin': kmeans_db,
            'calinski_harabasz': kmeans_ch
        },
        'hierarchical': {
            'labels': hier_labels,
            'silhouette': hier_silhouette,
            'davies_bouldin': hier_db,
            'calinski_harabasz': hier_ch
        }
    }

# Find optimal K using elbow method
@st.cache_data
def find_optimal_k(X_scaled, max_k=10):
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    return list(K_range), inertias, silhouette_scores

# Header
st.markdown("<h1 style='text-align: center;'>🛍️ Smart Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #f0f0f0;'>Day 5: Unlocking Customer Personas with AI Clustering</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Project Overview")
    st.image("https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🔄 Unsupervised learning</li>
            <li>📊 K-Means clustering</li>
            <li>🌳 Hierarchical clustering</li>
            <li>📈 Elbow method</li>
            <li>🎯 Silhouette analysis</li>
            <li>👥 Customer personas</li>
            <li>📉 Feature scaling importance</li>
            <li>🔍 Cluster interpretation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clustering parameters
    st.markdown("### ⚙️ Clustering Settings")
    n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    
    clustering_features = st.multiselect(
        "Select Features for Clustering:",
        ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
        default=['Annual Income (k$)', 'Spending Score (1-100)']
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 5 of 21</strong></p>
        <p>GeeksforGeeks Course</p>
    </div>
    """, unsafe_allow_html=True)

# Load data
df = load_data()

if df is not None and len(clustering_features) > 0:
    # Prepare data for clustering
    X = df[clustering_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    with st.spinner('🔄 Performing customer segmentation...'):
        k_range, inertias, silhouette_scores = find_optimal_k(X_scaled)
        clustering_results = perform_clustering(X_scaled, n_clusters)
    
    # Add cluster labels to dataframe
    df['KMeans_Cluster'] = clustering_results['kmeans']['labels']
    df['Hierarchical_Cluster'] = clustering_results['hierarchical']['labels']
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🔍 EDA",
        "🎯 Optimal Clusters",
        "📈 Clustering Results",
        "👥 Customer Personas",
        "💡 Insights"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown("## 📊 Mall Customer Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            st.metric("Male", f"{len(df[df['Gender']=='Male'])}")
        with col3:
            st.metric("Female", f"{len(df[df['Gender']=='Female'])}")
        with col4:
            st.metric("Avg Age", f"{df['Age'].mean():.0f}")
        with col5:
            st.metric("Avg Income", f"${df['Annual Income (k$)'].mean():.0f}K")
        
        st.markdown("### 📋 Sample Customer Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Dataset Information")
            info_df = pd.DataFrame({
                'Feature': df.columns,
                'Data Type': [str(df[col].dtype) for col in df.columns],
                'Non-Null': [df[col].notna().sum() for col in df.columns],
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📊 Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
    
    # Tab 2: EDA
    with tab2:
        st.markdown("## 🔍 Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👥 Gender Distribution")
            gender_counts = df['Gender'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=gender_counts.index,
                values=gender_counts.values,
                hole=0.4,
                marker=dict(colors=['#667eea', '#f093fb']),
                textinfo='label+percent+value'
            )])
            fig.update_layout(
                title="Customer Gender Distribution",
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🎂 Age Distribution")
            fig = px.histogram(
                df, x='Age', nbins=30,
                title='Customer Age Distribution',
                labels={'Age': 'Age', 'count': 'Number of Customers'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Income Distribution")
            fig = px.histogram(
                df, x='Annual Income (k$)', nbins=30,
                title='Annual Income Distribution',
                labels={'Annual Income (k$)': 'Annual Income ($1000)', 'count': 'Count'},
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🛒 Spending Score Distribution")
            fig = px.histogram(
                df, x='Spending Score (1-100)', nbins=30,
                title='Spending Score Distribution',
                labels={'Spending Score (1-100)': 'Spending Score', 'count': 'Count'},
                color_discrete_sequence=['#f093fb']
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        st.markdown("### 📊 Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df, x='Annual Income (k$)', y='Spending Score (1-100)',
                color='Gender',
                title='Income vs Spending Score',
                labels={'Annual Income (k$)': 'Annual Income ($1000)'},
                color_discrete_map={'Male': '#667eea', 'Female': '#f093fb'}
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df, x='Age', y='Spending Score (1-100)',
                color='Gender',
                title='Age vs Spending Score',
                color_discrete_map={'Male': '#667eea', 'Female': '#f093fb'}
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D scatter plot
        st.markdown("### 🌐 3D Customer Distribution")
        fig = px.scatter_3d(
            df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
            color='Gender',
            title='3D Customer Distribution',
            color_discrete_map={'Male': '#667eea', 'Female': '#f093fb'},
            opacity=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Optimal Clusters
    with tab3:
        st.markdown("## 🎯 Finding Optimal Number of Clusters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 Elbow Method")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=k_range, y=inertias,
                mode='lines+markers',
                marker=dict(size=10, color='#667eea'),
                line=dict(width=3, color='#667eea'),
                name='Inertia'
            ))
            fig.update_layout(
                title='Elbow Method - Finding Optimal K',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Within-Cluster Sum of Squares (Inertia)',
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **How to read the Elbow curve:**
            - Look for the "elbow" point where the curve bends
            - That's where adding more clusters gives diminishing returns
            - Typically between 3-6 clusters for this dataset
            """)
        
        with col2:
            st.markdown("### 📈 Silhouette Score Analysis")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=k_range, y=silhouette_scores,
                mode='lines+markers',
                marker=dict(size=10, color='#f093fb'),
                line=dict(width=3, color='#f093fb'),
                name='Silhouette Score'
            ))
            fig.update_layout(
                title='Silhouette Score by K',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Silhouette Score',
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Silhouette Score interpretation:**
            - Ranges from -1 to 1
            - Higher score = better defined clusters
            - > 0.5 is good, > 0.7 is excellent
            - Peak indicates optimal cluster count
            """)
        
        # Metrics comparison
        st.markdown("### 📊 Clustering Quality Metrics")
        
        optimal_k_idx = silhouette_scores.index(max(silhouette_scores))
        optimal_k = k_range[optimal_k_idx]
        
        st.success(f"📌 Suggested Optimal K: **{optimal_k}** (based on highest Silhouette Score: {max(silhouette_scores):.3f})")
        
        metrics_df = pd.DataFrame({
            'K': k_range,
            'Inertia': [f"{i:.2f}" for i in inertias],
            'Silhouette Score': [f"{s:.3f}" for s in silhouette_scores]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Tab 4: Clustering Results
    with tab4:
        st.markdown("## 📈 Clustering Results Visualization")
        
        # Algorithm selection
        algo_select = st.selectbox("Select Clustering Algorithm:", ['K-Means', 'Hierarchical'])
        
        cluster_col = 'KMeans_Cluster' if algo_select == 'K-Means' else 'Hierarchical_Cluster'
        results = clustering_results['kmeans'] if algo_select == 'K-Means' else clustering_results['hierarchical']
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", f"{results['silhouette']:.3f}")
        with col2:
            st.metric("Davies-Bouldin Index", f"{results['davies_bouldin']:.3f}")
        with col3:
            st.metric("Calinski-Harabasz Score", f"{results['calinski_harabasz']:.2f}")
        
        st.info("""
        **Metrics explanation:**
        - **Silhouette Score**: Higher is better (0.5-1.0 = good separation)
        - **Davies-Bouldin**: Lower is better (measures cluster separation)
        - **Calinski-Harabasz**: Higher is better (ratio of between/within cluster dispersion)
        """)
        
        # 2D Clustering Visualization
        if len(clustering_features) >= 2:
            st.markdown(f"### 🎨 Customer Segments - {algo_select}")
            
            fig = px.scatter(
                df, 
                x=clustering_features[0], 
                y=clustering_features[1],
                color=cluster_col,
                title=f'Customer Segments using {algo_select}',
                color_continuous_scale='Viridis',
                hover_data=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            )
            
            # Add centroids for K-Means
            if algo_select == 'K-Means':
                centroids = scaler.inverse_transform(results['centroids'])
                fig.add_trace(go.Scatter(
                    x=centroids[:, 0], 
                    y=centroids[:, 1] if len(centroids[0]) > 1 else [0]*len(centroids),
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='x', line=dict(width=2, color='white')),
                    name='Centroids',
                    showlegend=True
                ))
            
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Visualization if 3 features selected
        if len(clustering_features) == 3:
            st.markdown("### 🌐 3D Cluster Visualization")
            
            fig = px.scatter_3d(
                df,
                x=clustering_features[0],
                y=clustering_features[1],
                z=clustering_features[2],
                color=cluster_col,
                title=f'3D Customer Segments - {algo_select}',
                color_continuous_scale='Viridis',
                opacity=0.7
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster distribution
        st.markdown("### 📊 Cluster Size Distribution")
        
        cluster_counts = df[cluster_col].value_counts().sort_index()
        
        fig = px.bar(
            x=[f'Cluster {i}' for i in cluster_counts.index],
            y=cluster_counts.values,
            title='Number of Customers per Cluster',
            labels={'x': 'Cluster', 'y': 'Number of Customers'},
            color=cluster_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Customer Personas
    with tab5:
        st.markdown("## 👥 Customer Persona Analysis")
        
        algo_choice = st.radio("Select Algorithm for Persona Analysis:", ['K-Means', 'Hierarchical'], horizontal=True)
        cluster_col = 'KMeans_Cluster' if algo_choice == 'K-Means' else 'Hierarchical_Cluster'
        
        # Calculate cluster statistics
        cluster_stats = df.groupby(cluster_col).agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
            'Gender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mixed'
        }).round(1)
        
        cluster_counts = df[cluster_col].value_counts().sort_index()
        cluster_stats['Count'] = cluster_counts
        
        # Define personas based on income and spending
        def define_persona(row):
            income = row['Annual Income (k$)']
            spending = row['Spending Score (1-100)']
            
            if income >= 60 and spending >= 60:
                return "💎 High Value Customers", "High income, high spending - VIP segment"
            elif income >= 60 and spending < 40:
                return "🎯 Potential Targets", "High income, low spending - untapped potential"
            elif income < 40 and spending >= 60:
                return "⚡ Impulsive Buyers", "Low income, high spending - price sensitive"
            elif income < 40 and spending < 40:
                return "💰 Budget Shoppers", "Low income, low spending - value seekers"
            else:
                return "⚖️ Standard Customers", "Moderate income and spending"
        
        # Display personas
        st.markdown("### 🎭 Customer Segment Personas")
        
        for cluster_id in range(n_clusters):
            if cluster_id in cluster_stats.index:
                row = cluster_stats.loc[cluster_id]
                persona_name, persona_desc = define_persona(row)
                
                col1, col2, col3 = st.columns([1, 2, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="segment-card">
                        <h2 style='color: white; margin: 0;'>Cluster {cluster_id}</h2>
                        <h3 style='color: white; margin: 10px 0;'>{persona_name}</h3>
                        <p style='color: white; margin: 0;'>{int(row['Count'])} customers</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 📊 Segment Profile")
                    profile_df = pd.DataFrame({
                        'Metric': ['Average Age', 'Avg Income', 'Avg Spending', 'Dominant Gender', 'Size'],
                        'Value': [
                            f"{row['Age']:.0f} years",
                            f"${row['Annual Income (k$)']:.0f}K",
                            f"{row['Spending Score (1-100)']:.0f}/100",
                            row['Gender'],
                            f"{int(row['Count'])} customers"
                        ]
                    })
                    st.dataframe(profile_df, use_container_width=True, hide_index=True)
                
                with col3:
                    st.markdown("#### 💡 Marketing Strategy")
                    income = row['Annual Income (k$)']
                    spending = row['Spending Score (1-100)']
                    
                    if income >= 60 and spending >= 60:
                        st.success("""
                        - Premium product offerings
                        - Exclusive membership programs
                        - Personalized luxury experiences
                        - Loyalty rewards
                        """)
                    elif income >= 60 and spending < 40:
                        st.info("""
                        - Targeted promotion campaigns
                        - Product value demonstration
                        - Quality over quantity messaging
                        - Engagement incentives
                        """)
                    elif income < 40 and spending >= 60:
                        st.warning("""
                        - Payment plan options
                        - Budget-friendly alternatives
                        - Impulse buy promotions
                        - Flash sales and discounts
                        """)
                    elif income < 40 and spending < 40:
                        st.error("""
                        - Value-for-money products
                        - Bulk purchase discounts
                        - Basic product line focus
                        - Seasonal clearance sales
                        """)
                    else:
                        st.info("""
                        - Balanced product mix
                        - Standard promotions
                        - Regular customer engagement
                        - Cross-selling opportunities
                        """)
                
                st.markdown("---")
        
        # Heatmap of cluster characteristics
        st.markdown("### 🔥 Cluster Characteristics Heatmap")
        
        heatmap_data = cluster_stats[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].T
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Cluster", y="Feature", color="Value"),
            x=[f'Cluster {i}' for i in heatmap_data.columns],
            y=heatmap_data.index,
            text_auto='.1f',
            color_continuous_scale='Viridis',
            title='Feature Values Across Clusters'
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Insights
    with tab6:
        st.markdown("## 💡 Key Insights & Business Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
                <h3>🎯 Segmentation Insights</h3>
                <ul>
                    <li><strong>Cluster Count:</strong> {} distinct customer segments identified</li>
                    <li><strong>Best Algorithm:</strong> {} (Silhouette: {:.3f})</li>
                    <li><strong>Largest Segment:</strong> {} customers in Cluster {}</li>
                    <li><strong>Most Profitable:</strong> High income + high spending segment</li>
                    <li><strong>Growth Opportunity:</strong> High income + low spending group</li>
                </ul>
            </div>
            """.format(
                n_clusters,
                'K-Means' if clustering_results['kmeans']['silhouette'] > clustering_results['hierarchical']['silhouette'] else 'Hierarchical',
                max(clustering_results['kmeans']['silhouette'], clustering_results['hierarchical']['silhouette']),
                max(cluster_counts.values),
                cluster_counts.idxmax()
            ), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>📊 Customer Behavior Patterns</h3>
                <ul>
                    <li><strong>Income-Spending Correlation:</strong> Not always linear</li>
                    <li><strong>Gender Patterns:</strong> Different shopping behaviors observed</li>
                    <li><strong>Age Factor:</strong> Younger customers may spend differently</li>
                    <li><strong>Sweet Spots:</strong> Multiple profitable segments exist</li>
                    <li><strong>Diversity:</strong> Wide range of customer types in mall</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight-box">
                <h3>📚 Technical Skills Learned</h3>
                <ul>
                    <li><strong>Unsupervised Learning:</strong> Clustering without labels</li>
                    <li><strong>K-Means Algorithm:</strong> Centroid-based clustering</li>
                    <li><strong>Hierarchical Clustering:</strong> Dendrogram-based approach</li>
                    <li><strong>Elbow Method:</strong> Finding optimal K</li>
                    <li><strong>Silhouette Analysis:</strong> Cluster quality evaluation</li>
                    <li><strong>Feature Scaling:</strong> StandardScaler importance</li>
                    <li><strong>PCA:</strong> Dimensionality reduction concepts</li>
                    <li><strong>Business Intelligence:</strong> Data-driven decisions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight-box">
                <h3>🚀 Business Recommendations</h3>
                <ul>
                    <li><strong>Personalization:</strong> Tailor marketing to each segment</li>
                    <li><strong>Resource Allocation:</strong> Focus on high-value customers</li>
                    <li><strong>Product Strategy:</strong> Different products for different segments</li>
                    <li><strong>Pricing:</strong> Segment-specific pricing strategies</li>
                    <li><strong>Retention:</strong> Loyalty programs for key segments</li>
                    <li><strong>Acquisition:</strong> Target lookalike audiences</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("### 📈 Segmentation Summary")
        
        summary_df = pd.DataFrame({
            'Segment': [f'Cluster {i}' for i in range(n_clusters)],
            'Size': [int(cluster_counts[i]) if i in cluster_counts.index else 0 for i in range(n_clusters)],
            'Avg Income': [f"${cluster_stats.loc[i, 'Annual Income (k$)']:.0f}K" if i in cluster_stats.index else "N/A" for i in range(n_clusters)],
            'Avg Spending': [f"{cluster_stats.loc[i, 'Spending Score (1-100)']:.0f}" if i in cluster_stats.index else "N/A" for i in range(n_clusters)],
            'Avg Age': [f"{cluster_stats.loc[i, 'Age']:.0f}" if i in cluster_stats.index else "N/A" for i in range(n_clusters)]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🎓 Key Takeaways</h3>
            <ul>
                <li><strong>Customer Diversity:</strong> Not all customers are the same - segmentation reveals this</li>
                <li><strong>Strategic Marketing:</strong> One-size-fits-all doesn't work - personalize!</li>
                <li><strong>Data-Driven Decisions:</strong> Use clustering to inform business strategy</li>
                <li><strong>Continuous Improvement:</strong> Re-segment periodically as customers evolve</li>
                <li><strong>Actionable Insights:</strong> Convert clusters into marketing personas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    if df is None:
        st.error("❌ Unable to load dataset. Please check file location.")
    else:
        st.warning("⚠️ Please select at least one feature for clustering from the sidebar.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🛍️ Customer Segmentation Project Complete!</h3>
    <p><strong>Day 5 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>Smart Segmentation - Unlocking Customer Personas with AI</p>
</div>
""", unsafe_allow_html=True)
