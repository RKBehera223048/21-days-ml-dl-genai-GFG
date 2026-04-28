# Day 18: RAG Chatbot - Chat with Your Knowledge Base

## 💬 Project Overview

A **Retrieval-Augmented Generation (RAG)** chatbot that intelligently answers questions by retrieving relevant information from a knowledge base and generating contextual responses. This project demonstrates how to build production-ready Q&A systems that reduce AI hallucinations and provide cited, accurate answers.

---

## 🎯 Objectives

- Build a RAG-powered chatbot
- Implement vector similarity search
- Create document embeddings
- Design efficient retrieval strategies
- Integrate LLMs for generation
- Reduce hallucinations with grounded responses
- Deploy conversational AI systems

---

## 🏗️ Features

### 1. **Chat Demo** 💬
- Interactive Q&A interface
- Real-time knowledge base search
- Context-aware responses
- Chat history tracking
- Source document citations
- Similarity score display

### 2. **How RAG Works** 🔍
- RAG fundamentals explained
- Vector embeddings overview
- 6-step pipeline with code
- Similarity search visualization
- RAG vs Fine-tuning comparison

### 3. **Architecture** 🏗️
- Complete system design
- Production implementation
- Technology stack
- Vector database options
- LLM integration patterns

### 4. **Knowledge Base** 📊
- Document analytics
- Review length distribution
- Word frequency analysis
- Sample data exploration

### 5. **Insights** 💡
- Best practices
- Common challenges
- Performance comparisons
- Security considerations

---

## 🔧 Technical Implementation

### Basic RAG with LangChain
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load documents
loader = PyPDFLoader("knowledge_base.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 4. Create RAG chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 5. Query
response = qa_chain.run("What are the main findings?")
print(response)
```

### Custom RAG System
```python
class RAGSystem:
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.vectorstore = None
    
    def ingest_documents(self, file_paths):
        '''Load and process documents'''
        documents = []
        
        for path in file_paths:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        
        # Chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return len(chunks)
    
    def query(self, question, k=3):
        '''Query the RAG system'''
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(question, k=k)
        
        # Create context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate response
        prompt = f"""
        Use the following context to answer the question.
        If you don't know, say "I don't know based on the provided context."
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = self.llm(prompt)
        
        return {
            'answer': response,
            'sources': relevant_docs
        }

# Usage
rag = RAGSystem(openai_api_key="your-key")
rag.ingest_documents(["doc1.pdf", "doc2.pdf"])

result = rag.query("What are the key points?")
print(result['answer'])
```

### Vector Similarity Search
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Documents
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural language processing enables text understanding"
]

# Create embeddings
doc_embeddings = model.encode(documents)

# Query
query = "What is machine learning?"
query_embedding = model.encode([query])

# Calculate similarities
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

# Rank documents
ranked_indices = np.argsort(similarities)[::-1]

for idx in ranked_indices:
    print(f"Similarity: {similarities[idx]:.3f} - {documents[idx]}")
```

### Chunking Strategies
```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# 1. Recursive Character Splitter (Best for most cases)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# 2. Token-based Splitter (For LLM token limits)
token_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 3. Semantic Splitter (By meaning, not just length)
from langchain.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)

# Use
chunks = recursive_splitter.split_text(long_text)
```

---

## 🔄 RAG Pipeline (6 Steps)

### Step 1: Document Ingestion
- Load documents (PDF, DOCX, TXT, HTML)
- Extract text content
- Preserve metadata (source, date, author)

### Step 2: Chunking
- Split into manageable pieces (500-1000 tokens)
- Maintain context with overlap (50-200 tokens)
- Respect document structure (paragraphs, sections)

### Step 3: Embedding Generation
- Convert chunks to vectors (384-1536 dimensions)
- Use consistent embedding model
- Batch process for efficiency

### Step 4: Vector Storage
- Store in vector database (Chroma, FAISS, Pinecone)
- Create indexes for fast retrieval (HNSW, IVF)
- Enable metadata filtering

### Step 5: Retrieval
- Embed user query
- Similarity search (cosine, dot product)
- Retrieve top-k documents (k=3-10)
- Optional: Re-rank results

### Step 6: Generation
- Combine query + retrieved context
- Send to LLM with instructions
- Generate grounded response
- Include citations

---

## 📊 RAG vs Other Approaches

| Aspect | RAG | Fine-tuning | Pure LLM |
|--------|-----|-------------|----------|
| **Setup Time** | Hours | Days/Weeks | Minutes |
| **Cost** | Medium ($) | High ($$$) | High ($$) |
| **Accuracy** | 85-95% | 90-98% | 70-80% |
| **Updates** | Instant | Re-train | N/A |
| **Hallucinations** | Low | Very Low | High |
| **Citations** | Yes | No | No |
| **Private Data** | Yes | Yes | Limited |
| **Best For** | Knowledge bases | Domain expertise | General Q&A |

---

## 🛠️ Technology Stack

### Vector Databases
- **ChromaDB**: Open-source, local, easy setup
- **FAISS**: Fast, efficient, Facebook AI
- **Pinecone**: Cloud-hosted, scalable
- **Weaviate**: Hybrid search, GraphQL
- **Milvus**: Distributed, production-ready

### Embedding Models
- **OpenAI**: text-embedding-ada-002 (1536 dim)
- **Sentence-BERT**: all-MiniLM-L6-v2 (384 dim)
- **Cohere**: embed-multilingual-v3.0
- **HuggingFace**: Various open-source models

### LLMs
- **GPT-4**: Best accuracy, expensive
- **GPT-3.5-Turbo**: Balanced cost/performance
- **Claude 2**: Long context (100K tokens)
- **Llama 2**: Open-source, self-hosted
- **Mistral**: Efficient, competitive

### Frameworks
- **LangChain**: Most popular RAG framework
- **LlamaIndex**: Data framework for LLMs
- **Haystack**: Production NLP framework
- **Semantic Kernel**: Microsoft's framework

---

## ✅ Best Practices

### Document Chunking
- ✅ Optimal size: 500-1000 tokens
- ✅ Use overlap: 50-200 tokens
- ✅ Respect boundaries (paragraphs, sections)
- ✅ Test different strategies

### Embedding Selection
- ✅ Match domain (general vs specific)
- ✅ Consider dimensionality tradeoff
- ✅ Test on your data
- ✅ Batch encode for efficiency

### Retrieval Strategy
- ✅ Start with k=3-5 documents
- ✅ Use hybrid search (dense + sparse)
- ✅ Apply metadata filters
- ✅ Implement re-ranking

### Prompt Engineering
- ✅ Clear instructions
- ✅ Include retrieved context
- ✅ Request citations
- ✅ Low temperature (0-0.3)

### Evaluation
- ✅ Track relevance (NDCG, MRR)
- ✅ Monitor hallucinations
- ✅ Measure latency
- ✅ A/B test configurations

---

## 💡 Real-World Applications

### 1. **Customer Support** 🎧
- **Use Case**: Answer customer questions from knowledge base
- **Benefits**: 24/7 availability, instant responses
- **Examples**: Intercom, Zendesk AI

### 2. **Research** 🔬
- **Use Case**: Search scientific papers, find relevant studies
- **Benefits**: Faster literature review, discovery
- **Examples**: Elicit, Consensus

### 3. **Legal** ⚖️
- **Use Case**: Contract analysis, case law search
- **Benefits**: Faster research, better accuracy
- **Examples**: Harvey AI, CoCounsel

### 4. **Healthcare** 🏥
- **Use Case**: Medical record retrieval, clinical guidelines
- **Benefits**: Evidence-based care, reduced errors
- **Examples**: Glass Health, Pieces

### 5. **Enterprise** 🏢
- **Use Case**: Internal documentation, policy Q&A
- **Benefits**: Knowledge democratization, productivity
- **Examples**: Glean, Guru

---

## ⚠️ Challenges & Solutions

### Challenge 1: Poor Retrieval Quality
**Problem**: Wrong documents retrieved
**Solutions**:
- Improve embedding model
- Use hybrid search (BM25 + vector)
- Implement re-ranking
- Add metadata filters

### Challenge 2: Context Window Limits
**Problem**: Too much context for LLM
**Solutions**:
- Smarter chunking
- Reduce k (fewer documents)
- Use LLM with larger context (Claude 100K)
- Implement summarization

### Challenge 3: Hallucinations
**Problem**: LLM invents information
**Solutions**:
- Strict prompt instructions
- Require citations
- Low temperature
- Validate against sources

### Challenge 4: Latency
**Problem**: Slow response times
**Solutions**:
- Cache embeddings
- Faster vector DB (FAISS)
- Async operations
- Smaller embedding model

### Challenge 5: Cost
**Problem**: High API costs
**Solutions**:
- Local embeddings (Sentence-BERT)
- Batch processing
- Cache frequent queries
- Use cheaper LLM (GPT-3.5)

---

## 🚀 How to Run

### Prerequisites
```bash
# Install dependencies
pip install streamlit pandas numpy plotly
pip install langchain openai chromadb
pip install sentence-transformers faiss-cpu

# Set OpenAI API key (optional for production)
export OPENAI_API_KEY="your-key-here"
```

### Launch Application
```bash
streamlit run app_day18.py
```

### Use the Chatbot
1. **Ask Questions**: Type your query about hospital reviews
2. **View Response**: See AI-generated answer
3. **Check Sources**: View retrieved documents with similarity scores
4. **Explore Knowledge Base**: Analyze the underlying data

---

## 📚 Libraries Used

- **Streamlit**: Web interface
- **LangChain**: RAG framework
- **OpenAI**: Embeddings and LLM
- **ChromaDB**: Vector database
- **Sentence-Transformers**: Local embeddings
- **FAISS**: Fast similarity search
- **Pandas**: Data handling
- **Plotly**: Visualizations

---

## 🎓 Educational Value

This project teaches:
- RAG architecture and implementation
- Vector embeddings and similarity search
- Document chunking strategies
- LLM prompt engineering
- Production RAG patterns
- Evaluation and optimization

---

## 🔮 Future Enhancements

- **Multi-modal RAG**: Images, tables, charts
- **Conversation Memory**: Track chat context
- **Hybrid Search**: Combine dense and sparse retrieval
- **Re-ranking**: Improve result quality
- **Streaming**: Real-time response generation
- **Evaluation**: Automatic quality metrics
- **Multi-language**: Support various languages
- **Graph RAG**: Leverage knowledge graphs

---

## 📊 Performance Benchmarks

| Configuration | Latency | Accuracy | Cost/Query |
|--------------|---------|----------|------------|
| **Basic RAG** | 2-5 sec | 80-85% | $0.01 |
| **Optimized RAG** | 1-2 sec | 85-90% | $0.02 |
| **Hybrid RAG** | 1-3 sec | 90-95% | $0.03 |
| **Advanced RAG** | 2-4 sec | 95-98% | $0.05 |

*Based on GPT-3.5-Turbo, 1000 documents, k=3*

---

## Day 18 Summary

| Aspect | Details |
|--------|---------|
| **Topic** | Retrieval-Augmented Generation |
| **Technique** | Vector similarity search |
| **Technology** | LangChain, ChromaDB, OpenAI |
| **Application** | Q&A chatbot for hospital reviews |
| **Key Learning** | Build grounded, factual AI systems |
| **Accuracy** | 85-95% with citations |
| **Use Case** | Knowledge base chatbots |

---

**Day 18 of 21 Projects** | GeeksforGeeks ML, Deep Learning & GenAI Course

*Chat with your data intelligently!* 💬📚
