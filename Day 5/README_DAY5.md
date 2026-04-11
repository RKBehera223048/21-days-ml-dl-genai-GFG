# Day 5: Customer Segmentation with Clustering 🛍️

## 🎯 Project Overview
An interactive retail analytics dashboard for customer segmentation using unsupervised machine learning. Features K-Means and Hierarchical clustering algorithms, optimal cluster detection, and actionable customer personas for targeted marketing.

## ✨ Features

### 📊 Data Overview Tab
- **Customer Statistics:** Total customers, gender breakdown, demographics
- **Sample Data:** View raw customer information
- **Dataset Information:** Data types, non-null counts, unique values
- **Summary Statistics:** Statistical overview of all features

### 🔍 Exploratory Data Analysis Tab
- **Gender Distribution:** Pie chart of customer demographics
- **Age Distribution:** Histogram showing age patterns
- **Income Distribution:** Annual income spread visualization
- **Spending Score Distribution:** Customer spending behavior
- **2D Scatter Plots:** Income vs Spending, Age vs Spending
- **3D Visualization:** Interactive 3D customer distribution

### 🎯 Optimal Clusters Tab
- **Elbow Method:** Finding the "elbow" in inertia curve
- **Silhouette Score Analysis:** Cluster quality evaluation
- **Metrics Comparison Table:** Detailed K analysis (K=2 to K=10)
- **Optimal K Suggestion:** Automatic recommendation based on metrics
- **Interpretation Guides:** How to read each chart

### 📈 Clustering Results Tab
- **Algorithm Selection:** Choose K-Means or Hierarchical
- **Quality Metrics:**
  - Silhouette Score (cluster separation)
  - Davies-Bouldin Index (cluster compactness)
  - Calinski-Harabasz Score (cluster density)
- **2D Cluster Visualization:** With centroids (K-Means)
- **3D Cluster Visualization:** When 3 features selected
- **Cluster Size Distribution:** Bar chart of segment sizes

### 👥 Customer Personas Tab
- **Persona Cards:** Each cluster gets a unique persona:
  - 💎 High Value Customers
  - 🎯 Potential Targets
  - ⚡ Impulsive Buyers
  - 💰 Budget Shoppers
  - ⚖️ Standard Customers
- **Segment Profiles:** Detailed statistics per cluster
- **Marketing Strategies:** Tailored recommendations for each persona
- **Characteristics Heatmap:** Visual comparison across clusters

### 💡 Insights Tab
- Segmentation insights summary
- Customer behavior patterns
- Technical skills learned
- Business recommendations
- Complete segmentation summary table
- Key takeaways for implementation

## 🎨 Design Features
- **Purple Gradient Theme:** Modern retail-inspired styling
- **Interactive Controls:** Dynamic clustering parameters
- **Color-Coded Segments:** Viridis color scale for clusters
- **Persona Cards:** Beautiful gradient cards for each segment
- **Responsive Layout:** Adapts to all screen sizes
- **Professional Charts:** All Plotly interactive visualizations

## 🛠️ Installation

1. **Install required packages:**
```bash
pip install streamlit plotly pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 How to Run

1. **Navigate to Day 5 folder:**
```bash
cd "c:\Users\Rasak\Desktop\coding\GFG course Project\Day 5"
```

2. **Run the Streamlit app:**
```bash
streamlit run app_day5.py
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
**Mall Customers Dataset**
- **Source:** Retail mall customer database
- **Samples:** 200 customers
- **Features:** 5 attributes
  - CustomerID: Unique identifier
  - Gender: Male/Female
  - Age: Customer age (18-70)
  - Annual Income (k$): Income in thousands
  - Spending Score (1-100): Mall-assigned spending score

## 🎯 What I Learned

### Unsupervised Learning Concepts
- ✅ **Clustering fundamentals** - Grouping without labels
- ✅ **K-Means algorithm** - Centroid-based clustering
- ✅ **Hierarchical clustering** - Dendrogram approach
- ✅ **Elbow method** - Finding optimal K
- ✅ **Silhouette analysis** - Cluster quality metrics
- ✅ **Feature scaling** - StandardScaler importance
- ✅ **Customer personas** - Business interpretation

### Technical Skills
- ✅ **Multiple algorithms** - K-Means vs Hierarchical
- ✅ **Cluster evaluation** - Silhouette, Davies-Bouldin, Calinski-Harabasz
- ✅ **Optimal K detection** - Elbow and Silhouette methods
- ✅ **3D visualization** - Multi-dimensional data exploration
- ✅ **Centroid analysis** - Understanding cluster centers
- ✅ **Business translation** - Converting clusters to actionable insights

### Evaluation Metrics
- **Silhouette Score:** Measures how similar an object is to its own cluster vs other clusters (range: -1 to 1, higher is better)
- **Davies-Bouldin Index:** Average similarity between clusters (lower is better)
- **Calinski-Harabasz Score:** Ratio of between-cluster to within-cluster dispersion (higher is better)
- **Inertia:** Sum of squared distances to nearest centroid (lower is better, used for elbow method)

## 🔑 Key Findings

### Customer Segments Identified

#### 💎 High Value Customers (High Income, High Spending)
- **Profile:** Affluent, frequent shoppers
- **Strategy:** Premium products, VIP programs, exclusive experiences
- **Priority:** Highest retention value

#### 🎯 Potential Targets (High Income, Low Spending)
- **Profile:** Have money but don't spend much
- **Strategy:** Targeted promotions, demonstrate value, engagement incentives
- **Priority:** High conversion potential

#### ⚡ Impulsive Buyers (Low Income, High Spending)
- **Profile:** Spend beyond their means
- **Strategy:** Payment plans, flash sales, budget alternatives
- **Priority:** Moderate, watch for credit risks

#### 💰 Budget Shoppers (Low Income, Low Spending)
- **Profile:** Price-sensitive, value seekers
- **Strategy:** Discounts, bulk deals, value products
- **Priority:** Volume business

#### ⚖️ Standard Customers (Moderate Income & Spending)
- **Profile:** Average customer base
- **Strategy:** Balanced approach, standard promotions
- **Priority:** Steady revenue source

### Behavior Insights
- Income doesn't always predict spending (some low-income customers spend a lot!)
- Gender patterns show different shopping behaviors
- Age correlates with certain spending patterns
- Multiple profitable segments exist
- Diverse customer base requires diverse strategies

## 📊 Algorithm Comparison

| Feature | K-Means | Hierarchical |
|---------|---------|--------------|
| **Speed** | Fast (O(n)) | Slower (O(n²)) |
| **Scalability** | Large datasets | Small-medium datasets |
| **Shape** | Spherical clusters | Any shape |
| **K Selection** | Must specify K | Can cut dendrogram |
| **Centroids** | Yes, clear centers | No centroids |
| **Result** | May vary (random init) | Deterministic |
| **Best For** | Production, large data | Exploratory analysis |

## 🎬 Visualizations Included
1. **Pie Charts:** Gender distribution
2. **Histograms:** Age, income, spending distributions
3. **2D Scatter Plots:** Feature relationships
4. **3D Scatter Plot:** Customer distribution in 3D space
5. **Line Charts:** Elbow method, Silhouette scores
6. **Bar Charts:** Cluster sizes, metrics comparison
7. **Heatmap:** Feature values across clusters
8. **Cluster Plots:** Segmentation with centroids

## 🚀 Business Applications

### Marketing Strategy
- **Personalization:** Tailor messages to each segment
- **Campaign Targeting:** Focus on right audience
- **Product Development:** Design for specific segments
- **Pricing Strategy:** Segment-based pricing
- **Channel Selection:** Different channels for different groups

### Retail Operations
- **Store Layout:** Optimize for key segments
- **Inventory Management:** Stock for target customers
- **Promotions:** Segment-specific offers
- **Loyalty Programs:** Reward valuable customers
- **Staff Training:** Understand customer types

### Strategic Planning
- **Market Positioning:** Know your customer base
- **Growth Strategy:** Identify expansion opportunities
- **Competitive Analysis:** Segment comparison
- **Resource Allocation:** Invest in profitable segments
- **Risk Management:** Identify vulnerable segments

## 💡 Marketing Recommendations by Segment

### For High Value Customers (💎)
```
✓ Premium product line
✓ VIP membership program
✓ Exclusive events and previews
✓ Personalized concierge service
✓ Luxury brand partnerships
✓ Priority customer service
```

### For Potential Targets (🎯)
```
✓ Educational marketing (show value)
✓ Quality demonstration events
✓ Targeted email campaigns
✓ Introductory offers
✓ Testimonial and social proof
✓ Risk-free trial programs
```

### For Impulsive Buyers (⚡)
```
✓ Limited-time flash sales
✓ Impulse buy product placement
✓ Easy payment options
✓ "Buy now, pay later" programs
✓ Social media engagement
✓ Gamification and rewards
```

### For Budget Shoppers (💰)
```
✓ Clearance sales and discounts
✓ Bulk purchase deals
✓ Value product lines
✓ Loyalty card programs
✓ Price match guarantees
✓ Generic brand options
```

## 📈 Success Metrics

### KPIs to Track
- **Segment Growth:** Monitor each segment's size over time
- **Revenue per Segment:** Track profitability
- **Conversion Rates:** Measure campaign effectiveness
- **Customer Lifetime Value (CLV):** By segment
- **Retention Rates:** Segment-specific retention
- **Cross-Sell Success:** Product affinity by segment

### Implementation Checklist
- [ ] Validate segments with business stakeholders
- [ ] Create detailed persona documents
- [ ] Develop segment-specific marketing materials
- [ ] Train sales and marketing teams
- [ ] Set up tracking and analytics
- [ ] Launch targeted campaigns
- [ ] Monitor and iterate

## 🔬 Advanced Techniques (Future)
- **DBSCAN:** Density-based clustering for irregular shapes
- **Gaussian Mixture Models:** Probabilistic clustering
- **Mini-Batch K-Means:** For very large datasets
- **Feature Engineering:** Creating new relevant features
- **Temporal Clustering:** Analyzing segment evolution
- **Multi-dimensional Clustering:** Using more features
- **Ensemble Clustering:** Combining multiple algorithms

## 📚 Real-World Use Cases
- **Retail:** Mall customer segmentation (this project!)
- **E-commerce:** Online shopper behavior groups
- **Banking:** Customer financial profile segmentation
- **Telecom:** Usage pattern-based customer groups
- **Healthcare:** Patient cohort analysis
- **Insurance:** Risk profile segmentation
- **Streaming:** Content preference clusters
- **Social Media:** User engagement segments

---

**Part of:** 21 Projects, 21 Days: ML, Deep Learning & GenAI - GeeksforGeeks  
**Day:** 5 of 21  
**Topic:** Customer Segmentation with Unsupervised Learning

**🎯 Remember:** Good segmentation is actionable - always translate clusters into business strategies!
