import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Day 17: Intelligent Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with search engine blue-purple gradient theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
    }
    
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .highlight-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .search-result {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2575fc;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .search-title {
        color: #1a0dab;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .search-url {
        color: #006621;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .search-snippet {
        color: #545454;
        font-size: 14px;
        line-height: 1.5;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .crawler-log {
        background: #2d3748;
        color: #68d391;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1523961131990-5ea7c61b2107?w=800&h=400&fit=crop", 
             use_container_width=True)
    
    st.markdown("### 🔍 Day 17: Intelligent Search Engine")
    st.markdown("**Build Your Own Web Crawler & Search**")
    
    st.markdown("---")
    st.markdown("### 🎯 What I Learned")
    st.markdown("""
    - 🕷️ Web crawling fundamentals
    - 🔍 Search algorithm design
    - 📊 PageRank & ranking algorithms
    - 🗃️ Inverted index creation
    - 🧠 TF-IDF & relevance scoring
    - 🌐 Distributed crawling
    - ⚡ Search optimization
    - 🤖 AI-powered search (semantic search)
    """)
    
    st.markdown("---")
    st.info("**Day 17 of 21** - GeeksforGeeks ML & GenAI Course")

# Sample crawled data
@st.cache_data
def get_sample_crawled_data():
    return [
        {
            "id": 1,
            "url": "https://example.com/machine-learning-basics",
            "title": "Introduction to Machine Learning | Complete Guide",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. Learn the basics of supervised learning, unsupervised learning, and reinforcement learning. Includes Python code examples with scikit-learn.",
            "keywords": ["machine learning", "AI", "python", "scikit-learn", "tutorial"],
            "depth": 0,
            "pagerank": 0.95,
            "backlinks": 45,
            "crawl_time": "2024-01-15 10:30:00"
        },
        {
            "id": 2,
            "url": "https://example.com/deep-learning-guide",
            "title": "Deep Learning Tutorial - Neural Networks Explained",
            "content": "Deep learning uses neural networks with multiple layers to learn hierarchical representations. Covers CNNs, RNNs, LSTMs, and Transformers. Learn to build models with TensorFlow and PyTorch for image classification and NLP tasks.",
            "keywords": ["deep learning", "neural networks", "tensorflow", "pytorch", "CNN", "RNN"],
            "depth": 1,
            "pagerank": 0.88,
            "backlinks": 32,
            "crawl_time": "2024-01-15 10:31:15"
        },
        {
            "id": 3,
            "url": "https://example.com/nlp-transformers",
            "title": "Natural Language Processing with Transformers",
            "content": "Transformers revolutionized NLP with attention mechanisms. Learn about BERT, GPT, T5, and how to fine-tune models for text classification, question answering, and text generation. Includes HuggingFace tutorials.",
            "keywords": ["NLP", "transformers", "BERT", "GPT", "attention", "huggingface"],
            "depth": 1,
            "pagerank": 0.91,
            "backlinks": 38,
            "crawl_time": "2024-01-15 10:32:30"
        },
        {
            "id": 4,
            "url": "https://example.com/computer-vision-opencv",
            "title": "Computer Vision with OpenCV and Deep Learning",
            "content": "Computer vision enables machines to understand images and videos. Learn object detection with YOLO, image segmentation, face recognition, and optical character recognition using OpenCV and deep learning frameworks.",
            "keywords": ["computer vision", "opencv", "YOLO", "object detection", "image processing"],
            "depth": 2,
            "pagerank": 0.82,
            "backlinks": 28,
            "crawl_time": "2024-01-15 10:33:45"
        },
        {
            "id": 5,
            "url": "https://example.com/reinforcement-learning",
            "title": "Reinforcement Learning: From Basics to Advanced",
            "content": "Reinforcement learning teaches agents to make decisions through trial and error. Covers Q-learning, Deep Q-Networks (DQN), Policy Gradients, and Actor-Critic methods. Build game-playing AI with OpenAI Gym.",
            "keywords": ["reinforcement learning", "Q-learning", "DQN", "policy gradient", "OpenAI gym"],
            "depth": 1,
            "pagerank": 0.79,
            "backlinks": 22,
            "crawl_time": "2024-01-15 10:35:00"
        },
        {
            "id": 6,
            "url": "https://example.com/data-preprocessing",
            "title": "Data Preprocessing for Machine Learning",
            "content": "Data preprocessing is crucial for ML success. Learn data cleaning, handling missing values, feature scaling, encoding categorical variables, and feature engineering techniques with pandas and scikit-learn.",
            "keywords": ["data preprocessing", "feature engineering", "pandas", "data cleaning"],
            "depth": 2,
            "pagerank": 0.75,
            "backlinks": 19,
            "crawl_time": "2024-01-15 10:36:15"
        },
        {
            "id": 7,
            "url": "https://example.com/mlops-deployment",
            "title": "MLOps: Deploying Machine Learning Models to Production",
            "content": "MLOps combines ML with DevOps for production systems. Learn model deployment with Docker, Kubernetes, monitoring with Prometheus, CI/CD pipelines, and A/B testing for ML models.",
            "keywords": ["MLOps", "deployment", "docker", "kubernetes", "monitoring"],
            "depth": 1,
            "pagerank": 0.86,
            "backlinks": 30,
            "crawl_time": "2024-01-15 10:37:30"
        },
        {
            "id": 8,
            "url": "https://example.com/python-for-ml",
            "title": "Python Programming for Machine Learning",
            "content": "Python is the most popular language for ML. Master NumPy for numerical computing, Pandas for data analysis, Matplotlib for visualization, and scikit-learn for building ML models. Includes 50+ code examples.",
            "keywords": ["python", "numpy", "pandas", "matplotlib", "scikit-learn"],
            "depth": 0,
            "pagerank": 0.93,
            "backlinks": 42,
            "crawl_time": "2024-01-15 10:38:45"
        },
        {
            "id": 9,
            "url": "https://example.com/generative-ai",
            "title": "Generative AI: GANs, VAEs, and Diffusion Models",
            "content": "Generative AI creates new content from learned patterns. Explore Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Diffusion Models like Stable Diffusion for image and text generation.",
            "keywords": ["generative AI", "GANs", "VAE", "diffusion models", "stable diffusion"],
            "depth": 2,
            "pagerank": 0.89,
            "backlinks": 35,
            "crawl_time": "2024-01-15 10:40:00"
        },
        {
            "id": 10,
            "url": "https://example.com/time-series-forecasting",
            "title": "Time Series Forecasting with ARIMA and LSTM",
            "content": "Time series forecasting predicts future values from historical data. Learn ARIMA, SARIMA, Prophet, and LSTM networks for stock price prediction, demand forecasting, and anomaly detection in time series.",
            "keywords": ["time series", "forecasting", "ARIMA", "LSTM", "prophet"],
            "depth": 2,
            "pagerank": 0.77,
            "backlinks": 24,
            "crawl_time": "2024-01-15 10:41:15"
        }
    ]

# TF-IDF scoring
def calculate_tfidf_score(query, document):
    """Calculate TF-IDF based relevance score"""
    query_terms = query.lower().split()
    doc_text = (document['title'] + ' ' + document['content'] + ' ' + ' '.join(document['keywords'])).lower()
    
    score = 0
    for term in query_terms:
        if term in doc_text:
            # Simple TF calculation
            tf = doc_text.count(term)
            score += tf
            
            # Bonus for title matches
            if term in document['title'].lower():
                score += 3
            
            # Bonus for keyword matches
            if term in ' '.join(document['keywords']).lower():
                score += 2
    
    # PageRank boost
    score = score * document['pagerank']
    
    return score

# Search function
def search_documents(query, documents, max_results=10):
    """Search documents and return ranked results"""
    if not query:
        return []
    
    # Calculate scores
    results = []
    for doc in documents:
        score = calculate_tfidf_score(query, doc)
        if score > 0:
            results.append({
                'document': doc,
                'score': score
            })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:max_results]

# Main app
st.title("🔍 Intelligent Internet Search Engine")
st.markdown("### Build Your Own Web Crawler & Search System")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Search Demo", 
    "🕷️ Web Crawler", 
    "🧠 Search Algorithms",
    "🏗️ Architecture",
    "💡 Insights"
])

with tab1:
    st.header("🔍 Search Engine Demo")
    
    # Search box
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., machine learning tutorial",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Popular searches
    st.markdown("**Popular Searches:**")
    popular_cols = st.columns(5)
    popular_queries = [
        "machine learning",
        "deep learning",
        "NLP transformers",
        "computer vision",
        "python ML"
    ]
    
    selected_popular = None
    for i, query in enumerate(popular_queries):
        with popular_cols[i]:
            if st.button(query, key=f"pop_{i}", use_container_width=True):
                selected_popular = query
    
    if selected_popular:
        search_query = selected_popular
        search_button = True
    
    # Perform search
    if search_query or search_button:
        documents = get_sample_crawled_data()
        results = search_documents(search_query, documents)
        
        if results:
            st.markdown(f"### Found {len(results)} results for '{search_query}'")
            st.markdown(f"*Search completed in 0.{np.random.randint(10, 99)} seconds*")
            
            # Display results
            for i, result in enumerate(results, 1):
                doc = result['document']
                score = result['score']
                
                st.markdown('<div class="search-result">', unsafe_allow_html=True)
                
                # Result number and title
                st.markdown(f"<div class='search-title'>{i}. {doc['title']}</div>", unsafe_allow_html=True)
                
                # URL
                st.markdown(f"<div class='search-url'>{doc['url']}</div>", unsafe_allow_html=True)
                
                # Snippet with highlighted terms
                snippet = doc['content'][:200] + "..."
                for term in search_query.lower().split():
                    snippet = re.sub(f"({term})", r"**\1**", snippet, flags=re.IGNORECASE)
                st.markdown(f"<div class='search-snippet'>{snippet}</div>", unsafe_allow_html=True)
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⭐ PageRank: {doc['pagerank']:.2f}")
                with col2:
                    st.caption(f"🔗 {doc['backlinks']} backlinks")
                with col3:
                    st.caption(f"📊 Relevance: {score:.1f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"No results found for '{search_query}'")
            st.info("💡 Try different keywords or broader search terms")
    
    # Search stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    documents = get_sample_crawled_data()
    with col1:
        st.metric("Indexed Pages", len(documents))
    with col2:
        avg_rank = np.mean([d['pagerank'] for d in documents])
        st.metric("Avg PageRank", f"{avg_rank:.2f}")
    with col3:
        total_keywords = sum([len(d['keywords']) for d in documents])
        st.metric("Total Keywords", total_keywords)
    with col4:
        avg_backlinks = np.mean([d['backlinks'] for d in documents])
        st.metric("Avg Backlinks", f"{avg_backlinks:.0f}")

with tab2:
    st.header("🕷️ Web Crawler Simulator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Crawler Configuration")
        
        start_url = st.text_input("Start URL:", value="https://wikipedia.org")
        max_depth = st.slider("Maximum Depth:", 0, 5, 2)
        max_pages = st.slider("Maximum Pages:", 10, 1000, 100)
        threads = st.slider("Concurrent Threads:", 1, 10, 4)
        
        respect_robots = st.checkbox("Respect robots.txt", value=True)
        follow_redirects = st.checkbox("Follow Redirects", value=True)
        
        if st.button("🚀 Start Crawling", type="primary", use_container_width=True):
            st.session_state['crawling'] = True
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.get('crawling', False):
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.markdown("#### 📊 Crawling Progress")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
                status_text.text(f"Crawled {i+1} pages...")
            
            st.success("✅ Crawling Complete!")
            
            # Crawler stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages Crawled", max_pages)
                st.metric("Avg Depth", f"{max_depth/2:.1f}")
            with col2:
                st.metric("Time Taken", f"{np.random.randint(5, 30)} sec")
                st.metric("Pages/sec", f"{max_pages/np.random.randint(5, 30):.1f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.session_state['crawling'] = False
        else:
            st.info("👆 Configure crawler settings and click 'Start Crawling'")
    
    # Crawler visualizations
    st.markdown("---")
    st.markdown("#### 📊 Crawler Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Depth distribution
        documents = get_sample_crawled_data()
        depth_df = pd.DataFrame([{'Depth': d['depth'], 'Count': 1} for d in documents])
        depth_count = depth_df.groupby('Depth').count().reset_index()
        
        fig1 = px.bar(depth_count, x='Depth', y='Count',
                      title='Pages by Crawl Depth',
                      color='Count',
                      color_continuous_scale='Purples')
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # PageRank distribution
        pagerank_df = pd.DataFrame([
            {'URL': d['url'].split('/')[-1], 'PageRank': d['pagerank']} 
            for d in documents
        ])
        
        fig2 = px.bar(pagerank_df, x='URL', y='PageRank',
                      title='PageRank Distribution',
                      color='PageRank',
                      color_continuous_scale='Blues')
        fig2.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Crawler log
    st.markdown("#### 📝 Crawler Log")
    crawler_log = """
[2024-01-15 10:30:00] Starting crawler...
[2024-01-15 10:30:01] Fetching https://wikipedia.org
[2024-01-15 10:30:02] Found 150 links on page
[2024-01-15 10:30:03] Filtering by depth and robots.txt
[2024-01-15 10:30:04] Queueing 45 valid URLs
[2024-01-15 10:30:05] Crawling https://wikipedia.org/machine_learning
[2024-01-15 10:30:06] Extracted title: Introduction to Machine Learning
[2024-01-15 10:30:07] Indexed 1 page (depth=0, pagerank=0.95)
[2024-01-15 10:30:08] Crawling https://wikipedia.org/deep_learning
[2024-01-15 10:30:09] Extracted title: Deep Learning Guide
[2024-01-15 10:30:10] Indexed 2 pages (depth=1, pagerank=0.88)
[2024-01-15 10:30:11] Processing complete. Total: 100 pages
"""
    st.markdown(f'<div class="crawler-log">{crawler_log}</div>', unsafe_allow_html=True)

with tab3:
    st.header("🧠 Search Algorithms Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📚 Core Algorithms")
        
        st.markdown("**1. TF-IDF (Term Frequency - Inverse Document Frequency)**")
        st.markdown("""
        Measures word importance in documents:
        - **TF**: How often a term appears in a document
        - **IDF**: How rare a term is across all documents
        - **TF-IDF**: TF × IDF (higher = more important)
        """)
        
        st.code("""
# TF-IDF Calculation
def calculate_tfidf(term, document, all_docs):
    # Term Frequency
    tf = document.count(term) / len(document)
    
    # Inverse Document Frequency
    docs_with_term = sum(1 for doc in all_docs 
                         if term in doc)
    idf = log(len(all_docs) / (1 + docs_with_term))
    
    # TF-IDF Score
    tfidf = tf * idf
    return tfidf
        """, language='python')
        
        st.markdown("**2. PageRank**")
        st.markdown("""
        Google's algorithm for ranking web pages:
        - More backlinks = higher rank
        - Quality of backlinks matters
        - Iterative calculation
        """)
        
        st.code("""
# Simplified PageRank
def calculate_pagerank(page, all_pages, damping=0.85):
    incoming_links = [p for p in all_pages 
                      if page in p.outgoing_links]
    
    rank = (1 - damping) / len(all_pages)
    
    for link in incoming_links:
        rank += damping * (link.rank / len(link.outgoing_links))
    
    return rank
        """, language='python')
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Modern Search Techniques")
        
        st.markdown("**3. BM25 (Best Matching 25)**")
        st.markdown("""
        Advanced ranking function:
        - Improved version of TF-IDF
        - Considers document length
        - Saturation of term frequency
        """)
        
        st.code("""
# BM25 Scoring
def bm25_score(query_terms, doc, all_docs, k1=1.5, b=0.75):
    score = 0
    doc_len = len(doc)
    avg_doc_len = mean([len(d) for d in all_docs])
    
    for term in query_terms:
        tf = doc.count(term)
        idf = log((len(all_docs) - df(term) + 0.5) / 
                  (df(term) + 0.5))
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        
        score += idf * (numerator / denominator)
    
    return score
        """, language='python')
        
        st.markdown("**4. Semantic Search**")
        st.markdown("""
        AI-powered understanding of meaning:
        - Uses embeddings (BERT, Sentence-BERT)
        - Understands context and synonyms
        - Vector similarity search
        """)
        
        st.code("""
# Semantic Search with Embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents and query
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(
    query_embedding, 
    doc_embeddings
)[0]

# Rank by similarity
ranked_docs = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True
)
        """, language='python')
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Algorithm comparison
    st.markdown("---")
    st.markdown("#### ⚖️ Search Algorithm Comparison")
    
    algo_data = {
        'Algorithm': ['TF-IDF', 'BM25', 'PageRank', 'Semantic Search', 'Hybrid'],
        'Accuracy': [75, 82, 70, 90, 95],
        'Speed': ['Fast', 'Fast', 'Medium', 'Slow', 'Medium'],
        'Understands Meaning': ['No', 'No', 'No', 'Yes', 'Yes'],
        'Requires Links': ['No', 'No', 'Yes', 'No', 'Optional'],
        'Best For': ['Keyword search', 'General search', 'Web ranking', 'Contextual search', 'All-purpose']
    }
    
    algo_df = pd.DataFrame(algo_data)
    st.dataframe(algo_df, use_container_width=True, hide_index=True)
    
    # Accuracy visualization
    fig = px.bar(algo_df, x='Algorithm', y='Accuracy',
                 title='Search Algorithm Accuracy Comparison',
                 color='Accuracy',
                 color_continuous_scale='Viridis')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("🏗️ Search Engine Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🏛️ Complete System Architecture")
        st.markdown("""
        ```
        ┌─────────────────────────────────┐
        │       Web Crawler               │
        │  - URL frontier management      │
        │  - Robots.txt compliance        │
        │  - Distributed crawling         │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    HTML Parser & Extractor      │
        │  - Text extraction              │
        │  - Link extraction              │
        │  - Metadata extraction          │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      Indexing Engine            │
        │  - Inverted index creation      │
        │  - TF-IDF calculation           │
        │  - Document storage             │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      Ranking System             │
        │  - PageRank computation         │
        │  - BM25 scoring                 │
        │  - Machine learning ranking     │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      Query Processor            │
        │  - Query parsing                │
        │  - Spell correction             │
        │  - Query expansion              │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      Search API                 │
        │  - Result retrieval             │
        │  - Result ranking               │
        │  - Snippet generation           │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      Frontend UI                │
        │  - Search box                   │
        │  - Results display              │
        │  - Pagination                   │
        └─────────────────────────────────┘
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔧 Implementation Example")
        st.code("""
# Complete Search Engine
from crawl4ai import WebCrawler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SearchEngine:
    def __init__(self):
        self.crawler = WebCrawler()
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
    
    def crawl(self, start_url, max_pages=100):
        '''Crawl websites'''
        urls_to_crawl = [start_url]
        crawled = set()
        
        while urls_to_crawl and len(crawled) < max_pages:
            url = urls_to_crawl.pop(0)
            
            if url in crawled:
                continue
            
            # Crawl page
            result = self.crawler.run(url)
            
            # Extract content
            doc = {
                'url': url,
                'title': result.title,
                'content': result.text,
                'links': result.links
            }
            
            self.documents.append(doc)
            crawled.add(url)
            
            # Add new URLs
            urls_to_crawl.extend(result.links[:10])
    
    def build_index(self):
        '''Build inverted index'''
        texts = [d['title'] + ' ' + d['content'] 
                 for d in self.documents]
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=10):
        '''Search and rank documents'''
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarity
        scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        # Rank documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'document': self.documents[idx],
                    'score': scores[idx]
                })
        
        return results

# Usage
engine = SearchEngine()
engine.crawl('https://example.com', max_pages=100)
engine.build_index()

results = engine.search('machine learning tutorial')
for r in results:
    print(f"{r['document']['title']} - Score: {r['score']:.2f}")
        """, language='python')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Technology stack
    st.markdown("---")
    st.markdown("#### 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🕷️ Crawling**")
        st.markdown("• Scrapy")
        st.markdown("• Crawl4AI")
        st.markdown("• Beautiful Soup")
        st.markdown("• Selenium")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🗃️ Storage**")
        st.markdown("• Elasticsearch")
        st.markdown("• MongoDB")
        st.markdown("• PostgreSQL")
        st.markdown("• Redis (cache)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔍 Search**")
        st.markdown("• scikit-learn")
        st.markdown("• Sentence-BERT")
        st.markdown("• Whoosh")
        st.markdown("• Apache Solr")
        st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.header("💡 Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ✅ Search Engine Best Practices")
        st.markdown("""
        **1. Crawling** 🕷️
        - Respect robots.txt
        - Implement rate limiting
        - Use distributed crawling
        - Handle errors gracefully
        - Store crawl timestamps
        
        **2. Indexing** 📚
        - Build inverted index
        - Use compression
        - Shard across servers
        - Update incrementally
        - Store metadata
        
        **3. Ranking** 📊
        - Combine multiple signals
        - Use machine learning
        - Personalize results
        - Consider freshness
        - A/B test algorithms
        
        **4. Performance** ⚡
        - Cache popular queries
        - Use CDN for static content
        - Optimize database queries
        - Load balance servers
        - Monitor latency
        
        **5. User Experience** 👥
        - Fast response time (<100ms)
        - Relevant results
        - Good snippets
        - Did-you-mean suggestions
        - Related searches
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Real-World Applications")
        st.markdown("""
        **1. E-commerce Search** 🛒
        - Product discovery
        - Faceted search
        - Personalized recommendations
        - **Example**: Amazon, eBay
        
        **2. Enterprise Search** 🏢
        - Internal document search
        - Knowledge base
        - Employee directory
        - **Example**: Google Workspace
        
        **3. Academic Search** 📚
        - Research papers
        - Citations
        - Scholar profiles
        - **Example**: Google Scholar
        
        **4. News Search** 📰
        - Recent articles
        - Topic clustering
        - Trending stories
        - **Example**: Google News
        
        **5. Code Search** 💻
        - Repository search
        - Function lookup
        - Example code
        - **Example**: GitHub Search
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Challenges")
        st.markdown("""
        - **Scale**: Billions of pages to index
        - **Spam**: Detecting and filtering spam
        - **Speed**: Sub-second response times
        - **Relevance**: Understanding user intent
        - **Freshness**: Keeping index updated
        - **Personalization**: User-specific results
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("---")
    st.markdown("#### 📊 Search Engine Performance")
    
    perf_data = {
        'Metric': ['Pages Indexed', 'Query Latency', 'Crawl Rate', 'Index Size', 'Daily Queries'],
        'Small Engine': ['1M', '200ms', '100/sec', '10GB', '10K'],
        'Medium Engine': ['100M', '100ms', '1K/sec', '1TB', '1M'],
        'Large Engine (Google)': ['100B+', '<100ms', '100K+/sec', '100PB+', '8.5B']
    }
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True, height=250)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🎓 Day 17 Complete: Intelligent Internet Search Engine</h3>
    <p>Build the next generation of search with AI and web crawling!</p>
</div>
""", unsafe_allow_html=True)
