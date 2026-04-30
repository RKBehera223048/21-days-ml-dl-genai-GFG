import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Day 20: Browser Automation Agent",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with browser automation cyan-blue gradient theme
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
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
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
        border-left: 5px solid #06b6d4;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .automation-log {
        background: #2d3748;
        color: #68d391;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
        font-size: 13px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .result-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
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
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=400&fit=crop", 
             use_container_width=True)
    
    st.markdown("### 🌐 Day 20: Browser Automation")
    st.markdown("**AI-Powered Web Agent**")
    
    st.markdown("---")
    st.markdown("### 🎯 What I Learned")
    st.markdown("""
    - 🌐 Browser automation with AI
    - 🤖 Autonomous web navigation
    - 🔍 Intelligent element selection
    - 📊 Data extraction at scale
    - 🎯 Task planning for web tasks
    - 🛠️ Playwright & Selenium
    - ⚡ Headless browser control
    - 🧠 Vision-based web understanding
    """)
    
    st.markdown("---")
    st.info("**Day 20 of 21** - GeeksforGeeks ML & GenAI Course")

# Sample automation tasks
def get_sample_tasks():
    return [
        {
            "id": 1,
            "task": "Search Amazon for 'capture card'",
            "description": "Find products with 4+ stars and 500+ reviews",
            "type": "E-commerce Search",
            "steps": [
                "Navigate to Amazon.com",
                "Search for 'screen capture type c'",
                "Filter by ratings (4+ stars)",
                "Filter by reviews (500+)",
                "Extract product details"
            ],
            "results_count": 6,
            "success": True
        },
        {
            "id": 2,
            "task": "Extract quotes from quotes.toscrape.com",
            "description": "Get first 5 quotes with authors and tags",
            "type": "Web Scraping",
            "steps": [
                "Navigate to quotes.toscrape.com",
                "Identify quote elements",
                "Extract quote text",
                "Extract author names",
                "Extract tags",
                "Format structured output"
            ],
            "results_count": 5,
            "success": True
        },
        {
            "id": 3,
            "task": "LinkedIn job search",
            "description": "Find 'Machine Learning Engineer' jobs in San Francisco",
            "type": "Job Search",
            "steps": [
                "Navigate to LinkedIn Jobs",
                "Enter job title",
                "Set location",
                "Apply filters (remote, full-time)",
                "Extract job listings"
            ],
            "results_count": 25,
            "success": True
        }
    ]

# Sample extracted data
def get_amazon_results():
    return [
        {
            "name": "Guermok Video Capture Card 4K USB3.0",
            "rating": 4.4,
            "reviews": 2400,
            "price": "$29.99",
            "features": "4K, USB-C, Compatible with iPad/Mac/Windows"
        },
        {
            "name": "UGREEN Full HD 1080P Capture Card",
            "rating": 4.1,
            "reviews": 1100,
            "price": "$24.99",
            "features": "1080P 60FPS, Works with Switch 2/PS5/Xbox"
        },
        {
            "name": "4K HDMI to USB 3.0 Capture Card",
            "rating": 4.3,
            "reviews": 818,
            "price": "$19.99",
            "features": "1080P 60FPS, Screen Record Device"
        },
        {
            "name": "Newhope USB 3.0 Capture Card with 100W PD",
            "rating": 4.3,
            "reviews": 1300,
            "price": "$34.99",
            "features": "1080P 60fps, 2K 30fps, Quest 3 Compatible"
        },
        {
            "name": "Capture Card for Nintendo Switch",
            "rating": 4.3,
            "reviews": 1500,
            "price": "$21.99",
            "features": "4K HDMI, 1080P 60FPS, OBS Compatible"
        },
        {
            "name": "Video Capture Card HDMI to USB/Type-C",
            "rating": 4.3,
            "reviews": 2200,
            "price": "$18.99",
            "features": "1080P 60FPS, Game Capture, Live Broadcasting"
        }
    ]

def get_quotes_results():
    return [
        {
            "quote": "The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.",
            "author": "Albert Einstein",
            "tags": ["change", "deep-thoughts", "thinking", "world"]
        },
        {
            "quote": "It is our choices, Harry, that show what we truly are, far more than our abilities.",
            "author": "J.K. Rowling",
            "tags": ["abilities", "choices"]
        },
        {
            "quote": "There are only two ways to live your life. One is as though nothing is a miracle. The other is as though everything is a miracle.",
            "author": "Albert Einstein",
            "tags": ["inspirational", "life", "live", "miracle", "miracles"]
        },
        {
            "quote": "The person, be it gentleman or lady, who has not pleasure in a good novel, must be intolerably stupid.",
            "author": "Jane Austen",
            "tags": ["aliteracy", "books", "classic", "humor"]
        },
        {
            "quote": "Imperfection is beauty, madness is genius and it's better to be absolutely ridiculous than absolutely boring.",
            "author": "Marilyn Monroe",
            "tags": ["be-yourself", "inspirational"]
        }
    ]

# Main app
st.title("🌐 AI-Powered Browser Automation Agent")
st.markdown("### Autonomous Web Navigation, Data Extraction & Task Completion")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Automation Demo", 
    "📊 Results Viewer", 
    "🧠 How It Works",
    "🏗️ Architecture",
    "💡 Insights"
])

with tab1:
    st.header("🤖 Browser Automation Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Select Automation Task")
        
        tasks = get_sample_tasks()
        task_options = [f"{t['id']}. {t['task']}" for t in tasks]
        
        selected_task_str = st.selectbox("Choose a task:", task_options)
        selected_task_id = int(selected_task_str.split(".")[0])
        selected_task = tasks[selected_task_id - 1]
        
        st.markdown(f"**Type:** {selected_task['type']}")
        st.markdown(f"**Description:** {selected_task['description']}")
        
        st.markdown("**Execution Steps:**")
        for i, step in enumerate(selected_task['steps'], 1):
            st.markdown(f"{i}. {step}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Custom task input
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 💬 Or Enter Custom Task")
        
        custom_task = st.text_area(
            "Describe your automation task in plain English:",
            placeholder="e.g., Go to GitHub, search for 'machine learning' repos, find top 5 starred projects, extract names and descriptions",
            height=100
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🚀 Start Automation", type="primary", use_container_width=True):
            st.session_state['automation_running'] = True
            st.session_state['selected_task'] = selected_task
        
        # Show automation execution
        if st.session_state.get('automation_running', False):
            st.markdown("---")
            st.markdown("### 🔄 Automation In Progress...")
            
            task = st.session_state['selected_task']
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate execution
            for i, step in enumerate(task['steps']):
                progress = int((i + 1) / len(task['steps']) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Step {i+1}/{len(task['steps'])}: {step}")
                time.sleep(0.5)
            
            st.success(f"✅ Task completed! Found {task['results_count']} results.")
            
            # Execution log
            st.markdown("#### 📋 Execution Log")
            log_content = f"""
[2024-01-15 10:00:00] Task received: {task['task']}
[2024-01-15 10:00:01] Agent analyzing task requirements...
[2024-01-15 10:00:02] Planning execution steps...
[2024-01-15 10:00:03] Launching headless Chrome browser
[2024-01-15 10:00:04] Executing: {task['steps'][0]}
[2024-01-15 10:00:06] Page loaded successfully
[2024-01-15 10:00:07] Executing: {task['steps'][1]}
[2024-01-15 10:00:09] Element located: search box
[2024-01-15 10:00:10] Text entered: "{task['description'][:30]}..."
[2024-01-15 10:00:11] Executing: {task['steps'][2]}
[2024-01-15 10:00:13] Filter applied successfully
[2024-01-15 10:00:14] Executing: {task['steps'][3]}
[2024-01-15 10:00:16] Data extraction started
[2024-01-15 10:00:20] Found {task['results_count']} matching results
[2024-01-15 10:00:21] Executing: {task['steps'][4]}
[2024-01-15 10:00:25] Data extraction completed
[2024-01-15 10:00:26] Formatting results...
[2024-01-15 10:00:27] ✅ Task completed successfully!
[2024-01-15 10:00:28] Browser closed. Total time: 28 seconds
"""
            st.markdown(f'<div class="automation-log">{log_content}</div>', unsafe_allow_html=True)
            
            st.session_state['automation_running'] = False
            st.session_state['results_ready'] = True
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📊 Agent Capabilities")
        
        st.markdown("**Navigation** 🧭")
        st.markdown("- Open URLs")
        st.markdown("- Click buttons/links")
        st.markdown("- Navigate back/forward")
        
        st.markdown("**Interaction** 🖱️")
        st.markdown("- Fill forms")
        st.markdown("- Select dropdowns")
        st.markdown("- Handle pop-ups")
        
        st.markdown("**Extraction** 📦")
        st.markdown("- Text content")
        st.markdown("- Images & media")
        st.markdown("- Structured data")
        
        st.markdown("**Intelligence** 🧠")
        st.markdown("- Vision understanding")
        st.markdown("- Natural language")
        st.markdown("- Error recovery")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Configuration")
        
        headless = st.checkbox("Headless Mode", value=True)
        screenshots = st.checkbox("Save Screenshots", value=True)
        timeout = st.slider("Timeout (sec):", 10, 120, 30)
        retries = st.slider("Max Retries:", 1, 5, 3)
        
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("📊 Automation Results")
    
    # Task selector
    result_type = st.selectbox(
        "View results for:",
        ["Amazon Product Search", "Quotes Extraction", "LinkedIn Jobs"]
    )
    
    if result_type == "Amazon Product Search":
        st.markdown("### 🛒 Amazon Product Search Results")
        
        products = get_amazon_results()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Products Found", len(products))
        with col2:
            avg_rating = np.mean([p['rating'] for p in products])
            st.metric("Avg Rating", f"{avg_rating:.2f} ⭐")
        with col3:
            total_reviews = sum([p['reviews'] for p in products])
            st.metric("Total Reviews", f"{total_reviews:,}")
        with col4:
            prices = [float(p['price'].replace('$', '')) for p in products]
            st.metric("Avg Price", f"${np.mean(prices):.2f}")
        
        # Results table
        st.markdown("---")
        products_df = pd.DataFrame(products)
        st.dataframe(products_df, use_container_width=True, hide_index=True, height=300)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(products_df, x='name', y='rating',
                         title='Product Ratings Comparison',
                         color='rating',
                         color_continuous_scale='Blues')
            fig1.update_layout(xaxis_tickangle=-45, height=350)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.scatter(products_df, x='reviews', y='rating',
                            size='reviews', color='price',
                            title='Ratings vs Reviews',
                            hover_data=['name'])
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)
    
    elif result_type == "Quotes Extraction":
        st.markdown("### 📚 Extracted Quotes")
        
        quotes = get_quotes_results()
        
        # Display quotes
        for i, quote_data in enumerate(quotes, 1):
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"**Quote {i}:**")
            st.markdown(f'"{quote_data["quote"]}"')
            st.markdown(f"**Author:** {quote_data['author']}")
            st.markdown(f"**Tags:** {', '.join(quote_data['tags'])}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tag analysis
        all_tags = [tag for q in quotes for tag in q['tags']]
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        
        fig = px.bar(x=tag_counts.values, y=tag_counts.index,
                    title='Most Common Tags',
                    orientation='h',
                    color=tag_counts.values,
                    color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download as CSV",
            "sample_data.csv",
            "results.csv",
            "text/csv"
        )
    with col2:
        st.download_button(
            "📥 Download as JSON",
            '{"results": []}',
            "results.json",
            "application/json"
        )

with tab3:
    st.header("🧠 How Browser Automation Agents Work")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 Automation Pipeline")
        st.markdown("""
        **1. Task Understanding** 📝
        - Parse natural language instruction
        - Identify target website/action
        - Extract parameters (search terms, filters)
        - Determine success criteria
        
        **2. Task Planning** 🗺️
        - Break down into atomic steps
        - Determine navigation path
        - Identify required interactions
        - Plan data extraction strategy
        
        **3. Browser Control** 🌐
        - Launch browser (headless/headed)
        - Navigate to target URL
        - Wait for page load
        - Handle dynamic content
        
        **4. Element Location** 🎯
        - Vision-based (screenshot analysis)
        - DOM-based (selectors, XPath)
        - Text-based (OCR, semantic)
        - Hybrid approaches
        
        **5. Interaction Execution** 🖱️
        - Click elements
        - Fill input fields
        - Select options
        - Handle modals/pop-ups
        
        **6. Data Extraction** 📦
        - Parse HTML structure
        - Extract target information
        - Clean and validate data
        - Structure output
        
        **7. Error Handling** 🔧
        - Detect failures
        - Implement retries
        - Fallback strategies
        - Recovery mechanisms
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 💻 Code Example")
        st.code("""
# Playwright Browser Automation
from playwright.async_api import async_playwright

async def automate_search(query):
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(
            headless=True
        )
        page = await browser.new_page()
        
        # Navigate
        await page.goto('https://amazon.com')
        
        # Search
        await page.fill('#twotabsearchtextbox', query)
        await page.click('#nav-search-submit-button')
        
        # Wait for results
        await page.wait_for_selector('.s-result-item')
        
        # Extract products
        products = await page.eval_on_selector_all(
            '.s-result-item',
            '''elements => elements.map(el => ({
                title: el.querySelector('h2')?.innerText,
                price: el.querySelector('.a-price-whole')?.innerText,
                rating: el.querySelector('.a-icon-star-small')?.innerText,
                reviews: el.querySelector('.a-size-base')?.innerText
            }))'''
        )
        
        await browser.close()
        return products

# Usage
results = await automate_search('capture card')
        """, language='python')
        
        st.markdown("#### 🧠 AI-Powered Agent")
        st.code("""
# Browser-use Agent with GPT-4
from browser_use import Agent
from langchain_openai import ChatOpenAI

agent = Agent(
    task="Go to Amazon, search for capture cards with 4+ stars and 500+ reviews",
    llm=ChatOpenAI(model="gpt-4-vision"),
    headless=True
)

result = await agent.run()
print(result)

# Agent automatically:
# 1. Understands task
# 2. Plans steps
# 3. Controls browser
# 4. Extracts data
# 5. Returns results
        """, language='python')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Comparison
    st.markdown("---")
    st.markdown("#### ⚖️ Automation Approaches")
    
    approaches_data = {
        'Approach': ['Selenium', 'Playwright', 'Puppeteer', 'Browser-Use', 'Scrapy'],
        'Ease of Use': ['Medium', 'Easy', 'Easy', 'Very Easy', 'Hard'],
        'Speed': ['Slow', 'Fast', 'Fast', 'Medium', 'Very Fast'],
        'AI-Powered': ['No', 'No', 'No', 'Yes', 'No'],
        'Dynamic Pages': ['Yes', 'Yes', 'Yes', 'Yes', 'Limited'],
        'Best For': ['Legacy support', 'Modern apps', 'Node.js', 'AI agents', 'Static sites']
    }
    
    df = pd.DataFrame(approaches_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab4:
    st.header("🏗️ Browser Agent Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🏛️ System Components")
        st.markdown("""
        ```
        ┌─────────────────────────────────┐
        │    Natural Language Input       │
        │  "Search Amazon for..."         │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      LLM Task Planner           │
        │  - Understand intent            │
        │  - Generate action plan         │
        │  - Identify target elements     │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Browser Controller           │
        │  - Playwright/Selenium          │
        │  - Headless Chrome/Firefox      │
        │  - CDP (Chrome DevTools)        │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │   Vision Agent (Optional)       │
        │  - Screenshot analysis          │
        │  - GPT-4 Vision                 │
        │  - Element localization         │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Action Executor              │
        │  - Navigate, click, type        │
        │  - Scroll, wait                 │
        │  - Handle dynamic content       │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Data Extractor               │
        │  - Parse HTML/DOM               │
        │  - Extract structured data      │
        │  - Clean and validate           │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │   Structured Output             │
        │  JSON, CSV, Database            │
        └─────────────────────────────────┘
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Technology Stack")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Browser Control**")
            st.markdown("• Playwright")
            st.markdown("• Selenium")
            st.markdown("• Puppeteer")
            st.markdown("• CDP")
            
            st.markdown("**AI/LLM**")
            st.markdown("• GPT-4 Vision")
            st.markdown("• Claude")
            st.markdown("• LangChain")
        
        with col_b:
            st.markdown("**Frameworks**")
            st.markdown("• Browser-Use")
            st.markdown("• Skyvern")
            st.markdown("• MultiOn")
            st.markdown("• LaVague")
            
            st.markdown("**Data Handling**")
            st.markdown("• Beautiful Soup")
            st.markdown("• lxml")
            st.markdown("• Scrapy")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Key Features")
        st.markdown("""
        - **Vision-based**: Uses screenshots + GPT-4V
        - **Autonomous**: Plans and executes independently
        - **Error Recovery**: Handles failures gracefully
        - **Multi-page**: Navigates complex workflows
        - **Dynamic Content**: Waits for AJAX/JS
        - **Anti-detection**: Evades bot detection
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Popular tools comparison
    st.markdown("---")
    st.markdown("#### 🛠️ Browser Automation Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🎭 Playwright**")
        st.markdown("Modern, fast, reliable")
        st.markdown("Multi-browser support")
        st.markdown("Best for: Web apps")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🤖 Browser-Use**")
        st.markdown("AI-powered automation")
        st.markdown("Natural language tasks")
        st.markdown("Best for: AI agents")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🕷️ Scrapy**")
        st.markdown("High-performance scraping")
        st.markdown("Async architecture")
        st.markdown("Best for: Data mining")
        st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.header("💡 Browser Automation - Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ✅ Benefits")
        st.markdown("""
        **1. Time Savings** ⏰
        - 100-1000x faster than manual
        - 24/7 operation
        - Parallel execution
        
        **2. Accuracy** 🎯
        - No human errors
        - Consistent results
        - Reliable data extraction
        
        **3. Scalability** 📈
        - Handle thousands of pages
        - Distributed architecture
        - Cloud deployment
        
        **4. Cost Reduction** 💰
        - Eliminate manual labor
        - Reduce staffing needs
        - ROI in weeks
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Use Cases")
        st.markdown("""
        - **E-commerce**: Price monitoring, inventory tracking
        - **Research**: Data collection, competitive analysis
        - **Testing**: UI/UX testing, regression tests
        - **Lead Generation**: Contact scraping, enrichment
        - **Monitoring**: Website uptime, content changes
        - **Social Media**: Post scheduling, analytics
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Challenges & Solutions")
        st.markdown("""
        **Challenge 1: Bot Detection** 🚫
        - **Problem**: Websites block automated browsers
        - **Solution**: Stealth mode, proxies, human patterns
        
        **Challenge 2: Dynamic Content** 🔄
        - **Problem**: AJAX/JavaScript loaded content
        - **Solution**: Explicit waits, retry logic
        
        **Challenge 3: Changing Layouts** 📐
        - **Problem**: Website redesigns break selectors
        - **Solution**: Vision-based (GPT-4V), semantic selectors
        
        **Challenge 4: CAPTCHAs** 🔐
        - **Problem**: Human verification required
        - **Solution**: CAPTCHA-solving services, avoid triggers
        
        **Challenge 5: Legal/Ethical** ⚖️
        - **Problem**: Terms of service violations
        - **Solution**: Respect robots.txt, rate limits, permissions
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔐 Best Practices")
        st.markdown("""
        - ✅ Respect robots.txt
        - ✅ Use rate limiting
        - ✅ Add random delays
        - ✅ Rotate user agents
        - ✅ Handle errors gracefully
        - ✅ Cache when possible
        - ✅ Get permission when needed
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("---")
    st.markdown("#### 📊 Performance Comparison")
    
    perf_data = {
        'Task': ['Single Page Scrape', 'Multi-page Navigation', 'Form Submission', 'Data Extraction (100 items)'],
        'Manual (min)': [2, 10, 5, 60],
        'Selenium (sec)': [5, 30, 10, 120],
        'Playwright (sec)': [3, 15, 5, 60],
        'AI Agent (sec)': [10, 45, 15, 90]
    }
    
    perf_df = pd.DataFrame(perf_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Manual', x=perf_df['Task'], y=perf_df['Manual (min)'] * 60))
    fig.add_trace(go.Bar(name='Selenium', x=perf_df['Task'], y=perf_df['Selenium (sec)']))
    fig.add_trace(go.Bar(name='Playwright', x=perf_df['Task'], y=perf_df['Playwright (sec)']))
    fig.add_trace(go.Bar(name='AI Agent', x=perf_df['Task'], y=perf_df['AI Agent (sec)']))
    
    fig.update_layout(
        title='Time Comparison (seconds, log scale)',
        barmode='group',
        height=400,
        yaxis_type="log"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🎓 Day 20 Complete: Browser Automation Agent</h3>
    <p>Automate the web with AI-powered browser agents!</p>
</div>
""", unsafe_allow_html=True)
