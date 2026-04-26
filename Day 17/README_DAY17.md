# Day 17: Intelligent Internet Search Engine - Build Your Own Google

## 🔍 Project Overview

An **intelligent search engine** that demonstrates web crawling, indexing, and ranking algorithms used by modern search systems. This project covers everything from building a web crawler to implementing TF-IDF, PageRank, and semantic search.

---

## 🎯 Objectives

- Build a functional web crawler
- Create an inverted index
- Implement ranking algorithms (TF-IDF, BM25, PageRank)
- Design a search interface
- Understand how Google works
- Apply semantic search with embeddings
- Deploy a scalable search system

---

## 🏗️ Features

### 1. **Search Demo** 🔍
- Google-style search interface
- Real-time search results
- TF-IDF relevance scoring
- PageRank integration
- Highlighted search terms
- Popular search suggestions

### 2. **Web Crawler** 🕷️
- Configurable crawler settings
- Depth-based crawling
- robots.txt compliance
- Concurrent crawling
- Progress tracking
- Crawler analytics

### 3. **Search Algorithms** 🧠
- TF-IDF explained with code
- PageRank implementation
- BM25 scoring function
- Semantic search with embeddings
- Algorithm comparison

### 4. **Architecture** 🏗️
- Complete system design
- Full pipeline implementation
- Technology stack overview
- Scalability considerations

### 5. **Insights** 💡
- Best practices
- Real-world applications
- Performance comparisons
- Challenges and solutions

---

## 🔧 Technical Implementation

### Web Crawler
```python
from crawl4ai import WebCrawler
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

class SimpleCrawler:
    def __init__(self, start_url, max_pages=100):
        self.start_url = start_url
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = [start_url]
        self.documents = []
    
    def crawl(self):
        '''Crawl websites and extract content'''
        while self.to_visit and len(self.visited) < self.max_pages:
            url = self.to_visit.pop(0)
            
            if url in self.visited:
                continue
            
            try:
                # Fetch page
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract content
                title = soup.find('title').text if soup.find('title') else ''
                text = soup.get_text(separator=' ', strip=True)
                
                # Extract links
                links = []
                for a in soup.find_all('a', href=True):
                    link = urljoin(url, a['href'])
                    if self.same_domain(link):
                        links.append(link)
                        self.to_visit.append(link)
                
                # Store document
                self.documents.append({
                    'url': url,
                    'title': title,
                    'content': text,
                    'links': links
                })
                
                self.visited.add(url)
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
    
    def same_domain(self, url):
        '''Check if URL is from same domain'''
        return urlparse(url).netloc == urlparse(self.start_url).netloc

# Usage
crawler = SimpleCrawler('https://example.com', max_pages=100)
crawler.crawl()
print(f"Crawled {len(crawler.documents)} pages")
```

### Inverted Index
```python
from collections import defaultdict
import re

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.documents = []
    
    def add_document(self, doc_id, text):
        '''Add document to index'''
        self.documents.append(text)
        
        # Tokenize
        tokens = re.findall(r'\w+', text.lower())
        
        # Build index
        for position, token in enumerate(tokens):
            self.index[token].append((doc_id, position))
    
    def search(self, query):
        '''Search for documents containing query terms'''
        query_terms = re.findall(r'\w+', query.lower())
        
        # Find documents containing all terms
        doc_ids = None
        for term in query_terms:
            term_docs = set([doc_id for doc_id, _ in self.index.get(term, [])])
            if doc_ids is None:
                doc_ids = term_docs
            else:
                doc_ids = doc_ids.intersection(term_docs)
        
        return list(doc_ids) if doc_ids else []

# Usage
index = InvertedIndex()
index.add_document(0, "Machine learning is a subset of AI")
index.add_document(1, "Deep learning uses neural networks")
index.add_document(2, "AI and machine learning are related")

results = index.search("machine learning")
print(f"Found in documents: {results}")
```

### TF-IDF Implementation
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TFIDFSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.documents = []
    
    def fit(self, documents):
        '''Build TF-IDF index'''
        self.documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def search(self, query, top_k=10):
        '''Search using TF-IDF similarity'''
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'document': self.documents[idx],
                    'score': similarities[idx]
                })
        
        return results

# Usage
documents = [
    "Machine learning tutorial for beginners",
    "Deep learning with neural networks",
    "Introduction to artificial intelligence"
]

search = TFIDFSearch()
search.fit(documents)
results = search.search("machine learning")

for r in results:
    print(f"Score: {r['score']:.3f} - {r['document']}")
```

### PageRank Algorithm
```python
import numpy as np

def calculate_pagerank(links, num_pages, damping=0.85, iterations=100):
    '''Calculate PageRank for web pages'''
    
    # Initialize ranks
    ranks = np.ones(num_pages) / num_pages
    
    # Build adjacency matrix
    adjacency = np.zeros((num_pages, num_pages))
    for page, outgoing in links.items():
        for link in outgoing:
            adjacency[link, page] = 1
    
    # Normalize by outgoing links
    outgoing_counts = adjacency.sum(axis=0)
    outgoing_counts[outgoing_counts == 0] = 1  # Avoid division by zero
    adjacency = adjacency / outgoing_counts
    
    # Iteratively calculate PageRank
    for _ in range(iterations):
        ranks = (1 - damping) / num_pages + damping * adjacency.dot(ranks)
    
    return ranks

# Usage
links = {
    0: [1, 2],      # Page 0 links to pages 1 and 2
    1: [2],         # Page 1 links to page 2
    2: [0],         # Page 2 links to page 0
    3: [0, 1, 2]    # Page 3 links to all others
}

ranks = calculate_pagerank(links, num_pages=4)
print("PageRank scores:", ranks)
```

### BM25 Scoring
```python
import math
from collections import Counter

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc.split()) for doc in documents) / len(documents)
        self.doc_freqs = []
        self.idf = {}
        
        # Calculate document frequencies
        for doc in documents:
            self.doc_freqs.append(Counter(doc.lower().split()))
        
        # Calculate IDF
        for term in set(word for doc in documents for word in doc.lower().split()):
            df = sum(1 for doc_freq in self.doc_freqs if term in doc_freq)
            self.idf[term] = math.log((len(documents) - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query, doc_id):
        '''Calculate BM25 score for document'''
        score = 0
        doc_len = len(self.documents[doc_id].split())
        doc_freq = self.doc_freqs[doc_id]
        
        for term in query.lower().split():
            if term not in doc_freq:
                continue
            
            tf = doc_freq[term]
            idf = self.idf.get(term, 0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query, top_k=10):
        '''Search using BM25'''
        scores = [(i, self.score(query, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# Usage
docs = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing enables computers to understand text"
]

bm25 = BM25(docs)
results = bm25.search("machine learning artificial intelligence")
for doc_id, score in results:
    print(f"Score: {score:.2f} - {docs[doc_id]}")
```

### Semantic Search with Embeddings
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def fit(self, documents):
        '''Encode documents into embeddings'''
        self.documents = documents
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
    
    def search(self, query, top_k=10):
        '''Search using semantic similarity'''
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': similarities[idx]
            })
        
        return results

# Usage
documents = [
    "How to train machine learning models",
    "Tutorial on neural networks",
    "Guide to artificial intelligence"
]

search = SemanticSearch()
search.fit(documents)

# This will match even with different wording
results = search.search("learning AI systems")
for r in results:
    print(f"Similarity: {r['score']:.3f} - {r['document']}")
```

---

## 📊 Algorithm Comparison

| Algorithm | Accuracy | Speed | Understands Meaning | Requires Links | Best For |
|-----------|----------|-------|---------------------|----------------|----------|
| **TF-IDF** | 75% | Fast | No | No | Keyword search |
| **BM25** | 82% | Fast | No | No | General search |
| **PageRank** | 70% | Medium | No | Yes | Web page ranking |
| **Semantic Search** | 90% | Slow | Yes | No | Contextual search |
| **Hybrid** | 95% | Medium | Yes | Optional | All-purpose |

---

## 🔄 Complete Search Engine Pipeline

### Step-by-Step Process

**1. Crawling** 🕷️
- Start with seed URLs
- Fetch pages using HTTP requests
- Extract content and links
- Follow links to discover new pages
- Respect robots.txt and rate limits

**2. Parsing & Extraction** 📝
- Parse HTML with BeautifulSoup
- Extract text content
- Extract links and metadata
- Clean and normalize text
- Identify document structure

**3. Indexing** 🗃️
- Build inverted index (term → documents)
- Calculate TF-IDF scores
- Store document metadata
- Create forward index (document → terms)
- Compress and shard index

**4. Ranking** 📊
- Calculate PageRank
- Compute document scores
- Apply machine learning ranking
- Consider freshness and quality
- Personalize results

**5. Query Processing** 🔍
- Parse user query
- Spell correction
- Query expansion with synonyms
- Retrieve matching documents
- Rank by relevance

**6. Result Presentation** 📱
- Generate snippets
- Highlight query terms
- Group related results
- Add rich features (images, videos)
- Enable pagination

---

## 💡 Real-World Applications

### 1. **General Web Search** 🌐
- **Examples**: Google, Bing, DuckDuckGo
- **Scale**: Billions of pages
- **Features**: Universal search, instant answers
- **Technology**: Distributed crawling, machine learning ranking

### 2. **E-commerce Search** 🛒
- **Examples**: Amazon, eBay, Shopify
- **Features**: Product filters, recommendations
- **Ranking**: Popularity, reviews, sales
- **Technology**: Faceted search, collaborative filtering

### 3. **Enterprise Search** 🏢
- **Examples**: Google Workspace, Microsoft 365
- **Features**: Document search, people search
- **Security**: Access control, encryption
- **Technology**: Connectors, content enrichment

### 4. **Academic Search** 📚
- **Examples**: Google Scholar, PubMed, arXiv
- **Features**: Citation analysis, paper recommendations
- **Ranking**: Citations, journal impact
- **Technology**: Citation graphs, bibliometrics

### 5. **Code Search** 💻
- **Examples**: GitHub Search, Sourcegraph
- **Features**: Code syntax awareness, regex search
- **Ranking**: Repository stars, recency
- **Technology**: AST parsing, symbol indexing

---

## ✅ Best Practices

### Crawling
- ✅ Respect robots.txt
- ✅ Implement rate limiting (1-5 requests/sec)
- ✅ Use polite user agent
- ✅ Handle errors gracefully
- ✅ Store crawl metadata (timestamp, status)
- ✅ Implement URL deduplication
- ✅ Use distributed architecture for scale

### Indexing
- ✅ Build inverted index for fast lookup
- ✅ Use compression (delta encoding, Huffman)
- ✅ Shard across multiple servers
- ✅ Update index incrementally
- ✅ Store document checksums
- ✅ Implement rollback mechanism

### Ranking
- ✅ Combine multiple ranking signals
- ✅ Use machine learning for ranking (LambdaMART, RankNet)
- ✅ Personalize based on user history
- ✅ Consider document freshness
- ✅ A/B test algorithm changes
- ✅ Monitor relevance metrics (NDCG, MRR)

### Performance
- ✅ Cache popular queries (Redis, Memcached)
- ✅ Use CDN for static assets
- ✅ Optimize database queries
- ✅ Load balance across servers
- ✅ Monitor latency (target <100ms)
- ✅ Implement query suggestion auto-complete

---

## ⚠️ Challenges & Solutions

### Challenge 1: Scale
**Problem**: Billions of pages to crawl and index
**Solutions**:
- Distributed crawling (thousands of machines)
- Sharded indexing across clusters
- MapReduce for batch processing
- NoSQL databases for horizontal scaling

### Challenge 2: Spam
**Problem**: Spam pages manipulating rankings
**Solutions**:
- Content quality algorithms
- Link spam detection
- Machine learning classifiers
- User behavior signals

### Challenge 3: Freshness
**Problem**: Keeping index up-to-date
**Solutions**:
- Incremental indexing
- Priority crawling for changing pages
- Real-time indexing for important content
- Refresh frequency based on page update patterns

### Challenge 4: Relevance
**Problem**: Understanding user intent
**Solutions**:
- Semantic search with embeddings
- Query understanding (entity recognition)
- Click-through rate analysis
- Personalization based on context

### Challenge 5: Performance
**Problem**: Sub-second response times at scale
**Solutions**:
- Multi-level caching
- Query optimization
- Result pre-computation
- Edge computing for low latency

---

## 🚀 How to Run

### Prerequisites
```bash
# Install dependencies
pip install streamlit pandas numpy plotly
pip install beautifulsoup4 requests
pip install scikit-learn sentence-transformers

# For advanced crawling:
pip install scrapy crawl4ai
```

### Launch Application
```bash
streamlit run app_day17.py
```

### Use the Search Engine
1. **Search**: Enter query and click search
2. **Browse Results**: View ranked pages
3. **Configure Crawler**: Set crawl parameters
4. **Start Crawling**: Begin web crawling
5. **View Analytics**: Explore crawler statistics

---

## 📚 Libraries & Technologies

### Crawling
- **Scrapy**: Production web crawler
- **Crawl4AI**: AI-powered crawler
- **Beautiful Soup**: HTML parsing
- **Selenium**: JavaScript-heavy sites

### Search & Ranking
- **scikit-learn**: TF-IDF, ML algorithms
- **Sentence-Transformers**: Semantic embeddings
- **Whoosh**: Pure Python search
- **Elasticsearch**: Distributed search engine

### Storage
- **MongoDB**: Document storage
- **PostgreSQL**: Relational data
- **Redis**: Caching
- **Cassandra**: Distributed database

---

## 🎓 Educational Value

This project teaches:
- Web crawling fundamentals
- Information retrieval concepts
- Ranking algorithm implementation
- Scalable system design
- Performance optimization
- Modern search techniques (semantic search)

---

## 🔮 Future Enhancements

- **Live Crawling**: Real-time web crawler
- **Advanced Ranking**: ML-based ranking (LambdaMART)
- **Image Search**: Visual search capability
- **Voice Search**: Speech-to-text integration
- **Autocomplete**: Query suggestions
- **Spell Correction**: Did-you-mean
- **Related Searches**: Query expansion
- **Rich Snippets**: Structured data display
- **Filters**: Date, domain, file type
- **API**: RESTful search API

---

## 📊 Performance Benchmarks

| Metric | Small Engine | Medium Engine | Large (Google-scale) |
|--------|-------------|---------------|---------------------|
| **Pages Indexed** | 1M | 100M | 100B+ |
| **Query Latency** | 200ms | 100ms | <100ms |
| **Crawl Rate** | 100 pages/sec | 1K pages/sec | 100K+ pages/sec |
| **Index Size** | 10GB | 1TB | 100PB+ |
| **Daily Queries** | 10K | 1M | 8.5B |
| **Servers** | 1-10 | 100-1,000 | 1M+ |

---

## Day 17 Summary

| Aspect | Details |
|--------|---------|
| **Topic** | Intelligent Search Engine |
| **Techniques** | Web Crawling, TF-IDF, PageRank |
| **Algorithms** | BM25, Semantic Search |
| **Application** | Google-like search system |
| **Key Learning** | How search engines work |
| **Technology** | Scrapy, scikit-learn, BERT |
| **Scale** | Millions to billions of pages |

---

**Day 17 of 21 Projects** | GeeksforGeeks ML, Deep Learning & GenAI Course

*Build the next generation of search!* 🔍🌐
