import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Day 18: RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with RAG chatbot teal gradient theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 50%, #00d2ff 100%);
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
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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
        border-left: 5px solid #11998e;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 5px solid #667eea;
    }
    
    .bot-message {
        background: white;
        color: #333;
        border-left: 5px solid #11998e;
    }
    
    .context-box {
        background: #f0f9ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0ea5e9;
        margin: 10px 0;
        font-size: 14px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .code-block {
        background: #2d3748;
        color: #68d391;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1531746790731-6c087fecd65a?w=800&h=400&fit=crop", 
             use_container_width=True)
    
    st.markdown("### 💬 Day 18: RAG Chatbot")
    st.markdown("**Chat with Your Knowledge Base**")
    
    st.markdown("---")
    st.markdown("### 🎯 What I Learned")
    st.markdown("""
    - 🔍 Retrieval-Augmented Generation (RAG)
    - 📚 Vector embeddings & similarity search
    - 🗃️ Vector databases (ChromaDB, FAISS)
    - 🧠 LLM integration for context-aware responses
    - 📊 Document chunking strategies
    - 🔗 Semantic search techniques
    - ⚡ Real-time knowledge retrieval
    - 🤖 Building conversational AI systems
    """)
    
    st.markdown("---")
    st.info("**Day 18 of 21** - GeeksforGeeks ML & GenAI Course")

# Load hospital reviews data
@st.cache_data
def load_reviews_data():
    try:
        df = pd.read_csv(r"reviews.csv")
    except:
        try:
            df = pd.read_csv(r"../Day 18/reviews.csv")
        except:
            # Create sample data
            df = pd.DataFrame({
                'review_id': range(100),
                'review': [
                    f"Sample hospital review {i}. The medical staff was professional. The facilities were clean."
                    for i in range(100)
                ],
                'physician_name': [f"Dr. Smith {i%10}" for i in range(100)],
                'hospital_name': ["Wallace-Hamilton"] * 100,
                'patient_name': [f"Patient {i}" for i in range(100)]
            })
    return df

# Simple embedding simulation (in production, use OpenAI/HuggingFace)
def create_simple_embedding(text):
    """Create a simple embedding based on word frequency"""
    # Lowercase and tokenize
    words = re.findall(r'\w+', text.lower())
    
    # Create a simple vector based on key medical terms
    key_terms = ['medical', 'staff', 'doctor', 'nurse', 'facility', 'clean', 'care', 
                 'treatment', 'professional', 'attentive', 'wait', 'time', 'experience',
                 'positive', 'negative', 'excellent', 'poor', 'good', 'bad']
    
    embedding = []
    for term in key_terms:
        embedding.append(words.count(term))
    
    return np.array(embedding)

# Calculate similarity
def calculate_similarity(query_embedding, doc_embedding):
    """Calculate cosine similarity between embeddings"""
    dot_product = np.dot(query_embedding, doc_embedding)
    norm_query = np.linalg.norm(query_embedding)
    norm_doc = np.linalg.norm(doc_embedding)
    
    if norm_query == 0 or norm_doc == 0:
        return 0
    
    return dot_product / (norm_query * norm_doc)

# RAG retrieval
def retrieve_relevant_documents(query, documents, top_k=3):
    """Retrieve most relevant documents using simple similarity"""
    query_embedding = create_simple_embedding(query)
    
    similarities = []
    for i, doc in enumerate(documents):
        doc_embedding = create_simple_embedding(doc)
        similarity = calculate_similarity(query_embedding, doc_embedding)
        similarities.append((i, similarity, doc))
    
    # Sort by similarity and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Generate response
def generate_rag_response(query, relevant_docs):
    """Generate response using retrieved context"""
    # Combine context
    context = "\n\n".join([doc[2] for doc in relevant_docs])
    
    # Simple template-based response (in production, use LLM)
    if "positive" in query.lower() or "good" in query.lower():
        response = "Based on the reviews, many patients had positive experiences. "
    elif "negative" in query.lower() or "bad" in query.lower() or "problem" in query.lower():
        response = "Some patients mentioned concerns in their reviews. "
    elif "doctor" in query.lower() or "physician" in query.lower() or "medical staff" in query.lower():
        response = "Regarding the medical staff, here's what patients said: "
    elif "facility" in query.lower() or "hospital" in query.lower():
        response = "About the hospital facilities, patients noted: "
    elif "wait" in query.lower() or "time" in query.lower():
        response = "Concerning wait times and scheduling: "
    else:
        response = "Based on the patient reviews in our database: "
    
    # Add specific insights from context
    if relevant_docs:
        response += f"\n\nKey feedback includes:\n"
        for i, (idx, score, doc) in enumerate(relevant_docs, 1):
            snippet = doc[:150] + "..." if len(doc) > 150 else doc
            response += f"{i}. {snippet}\n"
    
    return response, context

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main app
st.title("💬 RAG Chatbot - Chat with Your Knowledge Base")
st.markdown("### Retrieval-Augmented Generation for Intelligent Q&A")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat Demo", 
    "🔍 How RAG Works", 
    "🏗️ Architecture",
    "📊 Knowledge Base",
    "💡 Insights"
])

with tab1:
    st.header("💬 Chat with Hospital Reviews Knowledge Base")
    
    # Load reviews
    df = load_reviews_data()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### Ask Questions About Hospital Reviews")
        
        # Example questions
        st.markdown("**Try these questions:**")
        example_questions = [
            "What do patients say about the medical staff?",
            "Are there any negative reviews?",
            "What are common complaints about wait times?",
            "Tell me about the hospital facilities",
            "What are the most positive reviews?"
        ]
        
        example_cols = st.columns(3)
        selected_example = None
        for i, question in enumerate(example_questions):
            with example_cols[i % 3]:
                if st.button(question, key=f"ex_q_{i}", use_container_width=True):
                    selected_example = question
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input
        user_query = st.text_area(
            "Your question:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="e.g., What do patients say about their experience?"
        )
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            send_button = st.button("🚀 Send Query", type="primary", use_container_width=True)
        with col_b:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process query
        if send_button and user_query:
            with st.spinner("🔍 Searching knowledge base..."):
                # Retrieve relevant documents
                documents = df['review'].tolist()
                relevant_docs = retrieve_relevant_documents(user_query, documents, top_k=3)
                
                # Generate response
                response, context = generate_rag_response(user_query, relevant_docs)
                
                # Add to history
                st.session_state.chat_history.append({
                    'query': user_query,
                    'response': response,
                    'relevant_docs': relevant_docs,
                    'context': context
                })
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                # User message
                st.markdown(f'<div class="chat-message user-message">', unsafe_allow_html=True)
                st.markdown(f"**👤 You:** {chat['query']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Bot response
                st.markdown(f'<div class="chat-message bot-message">', unsafe_allow_html=True)
                st.markdown(f"**🤖 Assistant:** {chat['response']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show retrieved context
                with st.expander(f"📄 View Retrieved Context (Similarity Scores)"):
                    for j, (idx, score, doc) in enumerate(chat['relevant_docs'], 1):
                        st.markdown(f"**Document {j}** (Similarity: {score:.3f})")
                        st.markdown(f'<div class="context-box">{doc}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📊 Knowledge Base Stats")
        
        st.metric("Total Documents", f"{len(df):,}")
        
        # Average review length
        avg_length = df['review'].str.len().mean()
        st.metric("Avg Review Length", f"{avg_length:.0f} chars")
        
        # Unique entities
        st.metric("Physicians", df['physician_name'].nunique())
        st.metric("Hospitals", df['hospital_name'].nunique())
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # RAG Process visualization
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 RAG Process")
        st.markdown("""
        **1. Query** 📝  
        User asks a question
        
        **2. Embed** 🔢  
        Convert to vector
        
        **3. Search** 🔍  
        Find similar documents
        
        **4. Retrieve** 📚  
        Get top-k results
        
        **5. Augment** ➕  
        Add context to prompt
        
        **6. Generate** 🤖  
        LLM creates response
        """)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("🔍 How RAG (Retrieval-Augmented Generation) Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📚 What is RAG?")
        st.markdown("""
        **Retrieval-Augmented Generation (RAG)** combines:
        - **Information Retrieval**: Search for relevant documents
        - **Language Generation**: LLM creates contextual responses
        
        **Why RAG?**
        - ✅ Reduces hallucinations (LLM making things up)
        - ✅ Provides up-to-date information
        - ✅ Cites sources for transparency
        - ✅ Works with proprietary/private data
        - ✅ No need to fine-tune LLM
        
        **RAG vs Fine-tuning:**
        
        | Aspect | RAG | Fine-tuning |
        |--------|-----|-------------|
        | Setup | Fast (hours) | Slow (days/weeks) |
        | Cost | Low ($) | High ($$$) |
        | Updates | Real-time | Re-train needed |
        | Accuracy | High | Very High |
        | Use Case | Knowledge bases | Domain expertise |
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔢 Vector Embeddings")
        st.markdown("""
        **Embeddings convert text to numbers:**
        
        ```
        "The doctor was excellent" 
        → [0.12, -0.34, 0.56, ..., 0.89] (384 dimensions)
        ```
        
        **Why embeddings?**
        - Capture semantic meaning
        - Similar texts → Similar vectors
        - Enable fast similarity search
        
        **Popular embedding models:**
        - OpenAI: text-embedding-ada-002
        - Sentence-BERT: all-MiniLM-L6-v2
        - Cohere: embed-multilingual
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 RAG Pipeline (6 Steps)")
        
        st.code("""
# Step 1: Document Ingestion
documents = [
    "The medical staff was excellent...",
    "Wait times were too long...",
    # ... more documents
]

# Step 2: Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# Step 3: Create Embeddings
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents(chunks)

# Step 4: Store in Vector Database
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Step 5: Retrieval
query = "What did patients say about doctors?"
relevant_docs = vectorstore.similarity_search(
    query, 
    k=3  # Get top 3 most relevant
)

# Step 6: Generation with Context
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

response = qa_chain.run(query)
print(response)
        """, language='python')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Similarity search visualization
    st.markdown("---")
    st.markdown("#### 📊 Similarity Search Visualization")
    
    # Create sample embeddings for visualization
    np.random.seed(42)
    query_vec = np.array([0.8, 0.6])
    doc_vecs = np.random.rand(10, 2)
    
    # Calculate similarities
    similarities = [calculate_similarity(query_vec, doc) for doc in doc_vecs]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Plot documents
    fig.add_trace(go.Scatter(
        x=doc_vecs[:, 0],
        y=doc_vecs[:, 1],
        mode='markers+text',
        marker=dict(size=12, color=similarities, colorscale='Viridis', showscale=True,
                   colorbar=dict(title="Similarity")),
        text=[f'Doc {i+1}' for i in range(10)],
        textposition="top center",
        name='Documents'
    ))
    
    # Plot query
    fig.add_trace(go.Scatter(
        x=[query_vec[0]],
        y=[query_vec[1]],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='star'),
        text=['Query'],
        textposition="top center",
        name='Query'
    ))
    
    fig.update_layout(
        title='Vector Space: Query vs Documents (2D Projection)',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Interpretation**: Documents closer to the query (red star) have higher similarity scores and are retrieved first.")

with tab3:
    st.header("🏗️ RAG System Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🏛️ Complete RAG Architecture")
        st.markdown("""
        ```
        ┌─────────────────────────────────┐
        │      Data Sources               │
        │  - PDFs, Docs, CSVs             │
        │  - Databases, APIs              │
        │  - Websites, Knowledge Bases    │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Document Processing          │
        │  - Text extraction              │
        │  - Cleaning & normalization     │
        │  - Metadata extraction          │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      Chunking Strategy          │
        │  - Recursive splitting          │
        │  - Semantic chunking            │
        │  - Overlap for context          │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Embedding Generation         │
        │  - OpenAI / HuggingFace         │
        │  - Batch processing             │
        │  - Dimension: 384-1536          │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Vector Database              │
        │  - Chroma / FAISS / Pinecone    │
        │  - Indexing (HNSW, IVF)         │
        │  - Metadata filtering           │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      User Query                 │
        │  - Natural language question    │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Query Embedding              │
        │  - Same model as documents      │
        │  - Consistent vector space      │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Similarity Search            │
        │  - Cosine similarity            │
        │  - Top-k retrieval (k=3-10)     │
        │  - Metadata filtering           │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │   Context Augmentation          │
        │  - Combine retrieved docs       │
        │  - Format prompt template       │
        │  - Add system instructions      │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      LLM Generation             │
        │  - GPT-4 / Claude / Llama       │
        │  - Temperature: 0-0.3           │
        │  - Grounded response            │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Response with Citations      │
        │  - Answer + source documents    │
        └─────────────────────────────────┘
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔧 Production Implementation")
        st.code("""
# Complete RAG System with LangChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGSystem:
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key
        )
        self.llm = OpenAI(
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.vectorstore = None
    
    def ingest_documents(self, file_paths):
        '''Load and process documents'''
        documents = []
        
        # Load documents
        for path in file_paths:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        
        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return len(chunks)
    
    def create_qa_chain(self):
        '''Create RAG QA chain'''
        # Custom prompt template
        template = '''
        Use the following context to answer the question.
        If you don't know, say "I don't know based on 
        the provided context."
        
        Context: {context}
        
        Question: {question}
        
        Answer: '''
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def query(self, question):
        '''Query the RAG system'''
        qa_chain = self.create_qa_chain()
        result = qa_chain({"query": question})
        
        return {
            'answer': result['result'],
            'sources': result['source_documents']
        }

# Usage
rag = RAGSystem(openai_api_key="your-key")

# Ingest documents
num_chunks = rag.ingest_documents([
    "document1.pdf",
    "document2.pdf"
])
print(f"Ingested {num_chunks} chunks")

# Query
result = rag.query("What are the main findings?")
print("Answer:", result['answer'])
print("\\nSources:")
for doc in result['sources']:
    print(f"- {doc.metadata['source']}")
        """, language='python')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Technology stack
    st.markdown("---")
    st.markdown("#### 🛠️ RAG Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🗃️ Vector DBs**")
        st.markdown("• ChromaDB (local)")
        st.markdown("• FAISS (fast)")
        st.markdown("• Pinecone (cloud)")
        st.markdown("• Weaviate (hybrid)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔢 Embeddings**")
        st.markdown("• OpenAI Ada-002")
        st.markdown("• Sentence-BERT")
        st.markdown("• Cohere Embed")
        st.markdown("• HuggingFace")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🤖 LLMs**")
        st.markdown("• GPT-4 / GPT-3.5")
        st.markdown("• Claude 2")
        st.markdown("• Llama 2")
        st.markdown("• Mistral")
        st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.header("📊 Knowledge Base Analysis")
    
    df = load_reviews_data()
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        avg_length = df['review'].str.len().mean()
        st.metric("Avg Length", f"{avg_length:.0f} chars")
    with col3:
        st.metric("Physicians", df['physician_name'].nunique())
    with col4:
        st.metric("Hospitals", df['hospital_name'].nunique())
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Review length distribution
        df['review_length'] = df['review'].str.len()
        fig1 = px.histogram(df, x='review_length', nbins=30,
                           title='Review Length Distribution',
                           labels={'review_length': 'Characters'},
                           color_discrete_sequence=['#11998e'])
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Word cloud simulation - top words
        all_text = ' '.join(df['review'].tolist()).lower()
        words = re.findall(r'\w+', all_text)
        word_freq = Counter(words).most_common(20)
        
        word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
        fig3 = px.bar(word_df.head(15), x='Frequency', y='Word',
                     title='Most Frequent Words in Reviews',
                     orientation='h',
                     color='Frequency',
                     color_continuous_scale='Teal')
        fig3.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Reviews by physician
        physician_counts = df['physician_name'].value_counts().head(10).reset_index()
        physician_counts.columns = ['Physician', 'Reviews']
        
        fig2 = px.bar(physician_counts, x='Reviews', y='Physician',
                     title='Top 10 Physicians by Review Count',
                     orientation='h',
                     color='Reviews',
                     color_continuous_scale='Greens')
        fig2.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)
        
        # Sample reviews
        st.markdown("#### 📝 Sample Reviews")
        sample_reviews = df.sample(min(5, len(df)))
        
        for idx, row in sample_reviews.iterrows():
            st.markdown(f'<div class="context-box">', unsafe_allow_html=True)
            st.markdown(f"**Patient:** {row['patient_name']}")
            st.markdown(f"**Physician:** {row['physician_name']}")
            st.markdown(f"**Review:** {row['review'][:200]}...")
            st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.header("💡 RAG Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ✅ RAG Best Practices")
        st.markdown("""
        **1. Document Chunking** 📄
        - Optimal chunk size: 500-1000 tokens
        - Use overlap (50-200 tokens)
        - Preserve context boundaries
        - Consider document structure
        
        **2. Embedding Selection** 🔢
        - Match domain (general vs specific)
        - Consider dimensionality (384-1536)
        - Balance cost vs accuracy
        - Test different models
        
        **3. Retrieval Strategy** 🔍
        - k=3-10 documents typical
        - Use hybrid search (dense + sparse)
        - Apply metadata filtering
        - Re-rank results
        
        **4. Prompt Engineering** 📝
        - Clear instructions to LLM
        - Include retrieved context
        - Request citations
        - Set temperature low (0-0.3)
        
        **5. Evaluation** 📊
        - Track relevance metrics
        - Monitor hallucinations
        - Measure latency
        - A/B test configurations
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Use Cases")
        st.markdown("""
        - **Customer Support**: Knowledge base Q&A
        - **Research**: Scientific paper search
        - **Legal**: Contract analysis
        - **Healthcare**: Medical record retrieval
        - **E-learning**: Course content Q&A
        - **Enterprise**: Internal documentation
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Challenges & Solutions")
        st.markdown("""
        **Challenge 1: Chunking Quality** 📄
        - **Problem**: Lost context across chunks
        - **Solution**: Use semantic chunking, maintain overlap
        
        **Challenge 2: Retrieval Accuracy** 🎯
        - **Problem**: Wrong documents retrieved
        - **Solution**: Hybrid search, re-ranking, better embeddings
        
        **Challenge 3: Hallucinations** 🌀
        - **Problem**: LLM invents information
        - **Solution**: Strict prompts, citation requirements, low temp
        
        **Challenge 4: Latency** ⏱️
        - **Problem**: Slow response times
        - **Solution**: Cache embeddings, faster vector DB, async
        
        **Challenge 5: Cost** 💰
        - **Problem**: API costs add up
        - **Solution**: Local embeddings, batch processing, caching
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔐 Security & Privacy")
        st.markdown("""
        **Best Practices:**
        - ✅ Encrypt data at rest and in transit
        - ✅ Access control on knowledge base
        - ✅ Audit logging for queries
        - ✅ PII detection and masking
        - ✅ Compliance (GDPR, HIPAA)
        - ✅ Local deployment for sensitive data
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance comparison
    st.markdown("---")
    st.markdown("#### 📊 RAG Performance Metrics")
    
    perf_data = {
        'Metric': ['Setup Time', 'Cost per Query', 'Accuracy', 'Update Speed', 'Maintenance'],
        'Pure LLM': ['Minutes', 'High ($$)', '70-80%', 'Re-train', 'None'],
        'RAG': ['Hours', 'Medium ($)', '85-95%', 'Instant', 'Low'],
        'Fine-tuned LLM': ['Days/Weeks', 'Very High ($$$)', '90-98%', 'Re-train', 'High'],
        'Hybrid (RAG + Fine-tune)': ['Weeks', 'High ($$)', '95-99%', 'Medium', 'Medium']
    }
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True, height=250)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🎓 Day 18 Complete: RAG Chatbot</h3>
    <p>Build intelligent chatbots that chat with your knowledge base!</p>
</div>
""", unsafe_allow_html=True)
