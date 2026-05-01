import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Day 21: AI Newsletter Pipeline",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with newsletter automation gradient theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 25%, #c44569 50%, #556270 100%);
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
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
        border-left: 5px solid #f5576c;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .workflow-node {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .newsletter-preview {
        background: white;
        padding: 25px;
        border-radius: 10px;
        border: 2px solid #f5576c;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #ff6b6b 0%, #c44569 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .workflow-log {
        background: #2d3748;
        color: #68d391;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        max-height: 400px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1557804506-669a67965ba0?w=800&h=400&fit=crop", 
             use_container_width=True)
    
    st.markdown("### 📧 Day 21: AI Newsletter Pipeline")
    st.markdown("**Automated Content Curation & Distribution**")
    
    st.markdown("---")
    st.markdown("### 🎯 What I Learned")
    st.markdown("""
    - 📧 Newsletter automation workflows
    - 🔄 n8n workflow orchestration
    - 🤖 AI-powered content curation
    - 📊 Data aggregation from multiple sources
    - ✍️ Automated email generation
    - 🎯 Personalization at scale
    - ⚡ Scheduled execution
    - 🔗 API integrations (RSS, Twitter, etc.)
    """)
    
    st.markdown("---")
    st.success("**Day 21 of 21** - Final Project! 🎉")

# Sample newsletter data
def get_curated_content():
    return [
        {
            "title": "GPT-4 Turbo Released with 128K Context Window",
            "source": "OpenAI Blog",
            "category": "AI/ML",
            "url": "https://openai.com/blog/gpt-4-turbo",
            "summary": "OpenAI announces GPT-4 Turbo with extended context window, better instruction following, and 3x lower pricing.",
            "date": "2024-01-15",
            "engagement": 15000
        },
        {
            "title": "Google Releases Gemini: Multimodal AI Model",
            "source": "Google AI Blog",
            "category": "AI/ML",
            "url": "https://blog.google/ai/gemini",
            "summary": "Gemini Ultra achieves state-of-the-art performance across text, image, video, and audio understanding.",
            "date": "2024-01-14",
            "engagement": 12000
        },
        {
            "title": "Microsoft Copilot Reaches 100M Users",
            "source": "TechCrunch",
            "category": "Business",
            "url": "https://techcrunch.com/microsoft-copilot",
            "summary": "Microsoft's AI assistant crosses major milestone, driving productivity gains across enterprises.",
            "date": "2024-01-13",
            "engagement": 8500
        },
        {
            "title": "AI Act Approved by European Parliament",
            "source": "EU News",
            "category": "Policy",
            "url": "https://eu.europa.eu/ai-act",
            "summary": "Landmark legislation establishes first comprehensive regulatory framework for artificial intelligence.",
            "date": "2024-01-12",
            "engagement": 9200
        },
        {
            "title": "LangChain Raises $25M Series A",
            "source": "VentureBeat",
            "category": "Funding",
            "url": "https://venturebeat.com/langchain",
            "summary": "Popular LLM framework secures funding to expand developer tools and enterprise offerings.",
            "date": "2024-01-11",
            "engagement": 5400
        }
    ]

def get_newsletter_stats():
    return {
        "total_subscribers": 15420,
        "open_rate": 42.5,
        "click_rate": 18.3,
        "newsletters_sent": 48,
        "avg_reading_time": "4 min 32 sec"
    }

# Generate newsletter content
def generate_newsletter_html(content_items, topic="AI & Technology"):
    current_date = datetime.now().strftime("%B %d, %Y")
    
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white;">
            <h1 style="margin: 0;">🚀 {topic} Weekly</h1>
            <p style="margin: 10px 0 0 0; font-size: 14px;">Your AI-Curated Newsletter | {current_date}</p>
        </div>
        
        <div style="padding: 30px; background: #f9fafb;">
            <p style="font-size: 16px; color: #374151;">
                👋 Hello! Here's your personalized digest of the week's most important {topic.lower()} news, 
                curated by AI and delivered automatically.
            </p>
            
            <h2 style="color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
                📰 Top Stories This Week
            </h2>
    """
    
    for i, item in enumerate(content_items, 1):
        html += f"""
        <div style="background: white; padding: 20px; margin: 20px 0; border-left: 4px solid #667eea; border-radius: 5px;">
            <h3 style="margin: 0 0 10px 0; color: #1f2937;">
                {i}. {item['title']}
            </h3>
            <p style="color: #6b7280; margin: 5px 0; font-size: 14px;">
                📍 {item['source']} | 🏷️ {item['category']} | 📅 {item['date']}
            </p>
            <p style="color: #374151; line-height: 1.6; margin: 10px 0;">
                {item['summary']}
            </p>
            <a href="{item['url']}" style="color: #667eea; text-decoration: none; font-weight: 600;">
                Read more →
            </a>
        </div>
        """
    
    html += """
        <div style="background: #eff6ff; padding: 20px; margin: 20px 0; border-radius: 5px;">
            <h3 style="color: #1e40af; margin: 0 0 10px 0;">💡 Why This Matters</h3>
            <p style="color: #374151; margin: 0; line-height: 1.6;">
                This week highlighted major advancements in AI capabilities, regulatory frameworks, 
                and enterprise adoption. The trend toward multimodal AI and responsible governance 
                continues to accelerate.
            </p>
        </div>
        
        <div style="text-align: center; padding: 20px; background: white; border-radius: 5px;">
            <p style="color: #6b7280; margin: 0 0 10px 0;">
                Enjoying this newsletter? Share it with colleagues!
            </p>
            <a href="#" style="display: inline-block; background: #667eea; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; font-weight: 600;">
                Share Newsletter
            </a>
        </div>
        
        </div>
        
        <div style="background: #1f2937; padding: 20px; text-align: center; color: #9ca3af; font-size: 12px;">
            <p>Powered by AI Newsletter Pipeline | Built with n8n + OpenAI</p>
            <p>You're receiving this because you subscribed to our newsletter.</p>
            <a href="#" style="color: #60a5fa; text-decoration: none;">Unsubscribe</a>
        </div>
    </div>
    """
    
    return html

# Main app
st.title("📧 AI-Powered Newsletter Pipeline")
st.markdown("### Automated Content Curation & Distribution with n8n")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔄 Workflow Demo", 
    "📧 Newsletter Preview", 
    "🏗️ Architecture",
    "📊 Analytics",
    "💡 Insights"
])

with tab1:
    st.header("🔄 Newsletter Automation Workflow")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Workflow Configuration")
        
        newsletter_topic = st.text_input("Newsletter Topic:", value="AI & Technology")
        
        col_a, col_b = st.columns(2)
        with col_a:
            schedule = st.selectbox("Schedule:", ["Daily", "Weekly", "Monthly"])
        with col_b:
            num_articles = st.slider("Articles per issue:", 3, 10, 5)
        
        content_sources = st.multiselect(
            "Content Sources:",
            ["RSS Feeds", "Twitter/X", "Reddit", "Hacker News", "Medium", "Dev.to"],
            default=["RSS Feeds", "Twitter/X", "Hacker News"]
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🚀 Execute Workflow", type="primary", use_container_width=True):
            st.session_state['workflow_running'] = True
        
        if st.session_state.get('workflow_running', False):
            st.markdown("---")
            st.markdown("### 🔄 Workflow Execution")
            
            # Workflow steps
            workflow_steps = [
                ("🕐 Trigger", "Cron schedule activated", "Weekly on Monday 9 AM"),
                ("🔍 Fetch Content", f"Aggregating from {len(content_sources)} sources", "50 articles found"),
                ("🤖 AI Filtering", "Using GPT-4 to rank relevance", f"Top {num_articles} selected"),
                ("✍️ Generate Summary", "AI-generated summaries", "All summaries created"),
                ("🎨 Design Email", "HTML template generation", "Newsletter designed"),
                ("📧 Send Email", f"Distributing to subscribers", "15,420 emails queued"),
                ("📊 Track Metrics", "Recording open/click rates", "Analytics updated")
            ]
            
            progress_bar = st.progress(0)
            
            for i, (step, action, result) in enumerate(workflow_steps):
                progress = int((i + 1) / len(workflow_steps) * 100)
                progress_bar.progress(progress)
                
                st.markdown(f'<div class="workflow-node">', unsafe_allow_html=True)
                st.markdown(f"**{step}**")
                st.markdown(f"{action}")
                st.markdown(f"✅ {result}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                time.sleep(0.3)
            
            st.success("✅ Workflow completed successfully!")
            
            # Execution log
            st.markdown("#### 📋 Execution Log")
            log_content = """
[2024-01-15 09:00:00] Workflow triggered: Weekly Newsletter
[2024-01-15 09:00:01] Fetching RSS feeds...
[2024-01-15 09:00:05] Retrieved 25 articles from TechCrunch
[2024-01-15 09:00:08] Retrieved 15 articles from VentureBeat
[2024-01-15 09:00:10] Retrieved 10 articles from AI blogs
[2024-01-15 09:00:12] Total articles collected: 50
[2024-01-15 09:00:15] Calling OpenAI API for relevance scoring...
[2024-01-15 09:00:22] Articles ranked by relevance
[2024-01-15 09:00:23] Top 5 articles selected
[2024-01-15 09:00:25] Generating AI summaries...
[2024-01-15 09:00:35] All summaries generated
[2024-01-15 09:00:36] Building HTML email template
[2024-01-15 09:00:38] Template rendered successfully
[2024-01-15 09:00:40] Connecting to SendGrid API...
[2024-01-15 09:00:42] Sending batch emails (15,420 recipients)
[2024-01-15 09:05:15] All emails sent successfully
[2024-01-15 09:05:16] Tracking pixels configured
[2024-01-15 09:05:17] ✅ Workflow completed! Total time: 5m 17s
"""
            st.markdown(f'<div class="workflow-log">{log_content}</div>', unsafe_allow_html=True)
            
            st.session_state['workflow_running'] = False
            st.session_state['newsletter_ready'] = True
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Workflow Nodes")
        
        nodes = [
            ("⏰", "Schedule Trigger"),
            ("🔍", "RSS Feed Reader"),
            ("🐦", "Twitter API"),
            ("🤖", "OpenAI GPT-4"),
            ("✍️", "Content Summarizer"),
            ("🎨", "Email Designer"),
            ("📧", "SendGrid/SMTP"),
            ("📊", "Analytics Tracker")
        ]
        
        for icon, name in nodes:
            st.markdown(f"{icon} **{name}**")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📈 Quick Stats")
        
        stats = get_newsletter_stats()
        st.metric("Subscribers", f"{stats['total_subscribers']:,}")
        st.metric("Open Rate", f"{stats['open_rate']}%")
        st.metric("Click Rate", f"{stats['click_rate']}%")
        
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("📧 Newsletter Preview")
    
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("#### 🎨 Generated Newsletter")
    st.markdown("Below is the AI-generated newsletter that would be sent to subscribers:")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate and display newsletter
    content = get_curated_content()
    newsletter_html = generate_newsletter_html(content)
    
    st.markdown('<div class="newsletter-preview">', unsafe_allow_html=True)
    st.markdown(newsletter_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download HTML",
            newsletter_html,
            "newsletter.html",
            "text/html"
        )
    with col2:
        st.download_button(
            "📥 Download Template",
            newsletter_html,
            "newsletter_template.html",
            "text/html"
        )

with tab3:
    st.header("🏗️ n8n Workflow Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 Complete Workflow")
        st.markdown("""
        ```
        ┌─────────────────────────────────┐
        │    Schedule Trigger (Cron)      │
        │  Weekly: Monday 9:00 AM         │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Content Aggregation          │
        │                                 │
        │  ┌──────────┐  ┌──────────┐    │
        │  │RSS Feeds │  │Twitter/X │    │
        │  └────┬─────┘  └────┬─────┘    │
        │       │             │          │
        │  ┌────▼─────┐  ┌────▼─────┐    │
        │  │HN API    │  │Reddit API│    │
        │  └────┬─────┘  └────┬─────┘    │
        └───────┴──────────────┴──────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      Data Transformation         │
        │  - Parse articles               │
        │  - Extract metadata             │
        │  - Deduplicate content          │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      AI Content Ranking         │
        │  - OpenAI GPT-4 API             │
        │  - Relevance scoring            │
        │  - Topic classification         │
        │  - Select top N articles        │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    AI Summary Generation        │
        │  - OpenAI API (GPT-4)           │
        │  - Generate summaries           │
        │  - Extract key points           │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │     Email Template Builder      │
        │  - HTML/CSS generation          │
        │  - Personalization              │
        │  - Dynamic content insertion    │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │      Email Distribution         │
        │  - SendGrid/Mailchimp API       │
        │  - Batch sending                │
        │  - Tracking pixels              │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │     Analytics & Tracking        │
        │  - Open rates                   │
        │  - Click rates                  │
        │  - Store in database            │
        └─────────────────────────────────┘
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 💻 n8n Configuration")
        st.code("""
// n8n Workflow JSON (simplified)
{
  "name": "AI Newsletter Pipeline",
  "nodes": [
    {
      "type": "n8n-nodes-base.cron",
      "name": "Schedule",
      "parameters": {
        "cronExpression": "0 9 * * 1"
      }
    },
    {
      "type": "n8n-nodes-base.rssFeed",
      "name": "RSS Aggregator",
      "parameters": {
        "url": "https://feeds.feedburner.com/TechCrunch"
      }
    },
    {
      "type": "n8n-nodes-base.openAi",
      "name": "GPT-4 Ranker",
      "parameters": {
        "model": "gpt-4",
        "prompt": "Rank these articles by relevance..."
      }
    },
    {
      "type": "n8n-nodes-base.openAi",
      "name": "Summarizer",
      "parameters": {
        "model": "gpt-4",
        "prompt": "Generate 2-sentence summary..."
      }
    },
    {
      "type": "n8n-nodes-base.emailSend",
      "name": "SendGrid",
      "parameters": {
        "toEmail": "={{$json.email}}",
        "subject": "Your Weekly AI Digest"
      }
    }
  ],
  "connections": {
    "Schedule": {"main": [[{"node": "RSS Aggregator"}]]},
    "RSS Aggregator": {"main": [[{"node": "GPT-4 Ranker"}]]},
    "GPT-4 Ranker": {"main": [[{"node": "Summarizer"}]]},
    "Summarizer": {"main": [[{"node": "SendGrid"}]]}
  }
}
        """, language='json')
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🚀 Setup Commands")
        st.code("""
# Docker setup for n8n
docker volume create n8n_data

docker run -it --rm \\
  --name n8n \\
  -p 5678:5678 \\
  -v n8n_data:/home/node/.n8n \\
  docker.n8n.io/n8nio/n8n

# Access: http://localhost:5678

# With environment variables
docker run -it --rm \\
  --name n8n \\
  -p 5678:5678 \\
  -e N8N_BASIC_AUTH_ACTIVE=true \\
  -e N8N_BASIC_AUTH_USER=admin \\
  -e N8N_BASIC_AUTH_PASSWORD=password \\
  -e OPENAI_API_KEY=your_key \\
  -v n8n_data:/home/node/.n8n \\
  docker.n8n.io/n8nio/n8n
        """, language='bash')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tool ecosystem
    st.markdown("---")
    st.markdown("#### 🛠️ Newsletter Automation Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔄 n8n**")
        st.markdown("Workflow automation")
        st.markdown("200+ integrations")
        st.markdown("Self-hosted option")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**⚡ Zapier**")
        st.markdown("No-code automation")
        st.markdown("5000+ apps")
        st.markdown("Easy to use")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔗 Make**")
        st.markdown("Visual automation")
        st.markdown("Complex workflows")
        st.markdown("API integrations")
        st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.header("📊 Newsletter Analytics")
    
    stats = get_newsletter_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Subscribers", f"{stats['total_subscribers']:,}", "+245")
    with col2:
        st.metric("Open Rate", f"{stats['open_rate']}%", "+2.3%")
    with col3:
        st.metric("Click Rate", f"{stats['click_rate']}%", "+1.5%")
    with col4:
        st.metric("Newsletters Sent", stats['newsletters_sent'], "+1")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Open rate over time
        dates = pd.date_range(end=datetime.now(), periods=12, freq='W')
        open_rates = [38.2, 39.5, 41.0, 40.5, 42.1, 43.5, 41.8, 42.5, 43.2, 44.1, 42.8, 42.5]
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=dates,
            y=open_rates,
            mode='lines+markers',
            name='Open Rate',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        fig1.update_layout(
            title='Open Rate Trend (Last 12 Weeks)',
            xaxis_title='Date',
            yaxis_title='Open Rate (%)',
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Content performance
        content = get_curated_content()
        content_df = pd.DataFrame(content)
        
        fig3 = px.bar(content_df, x='title', y='engagement',
                     title='Article Engagement',
                     color='engagement',
                     color_continuous_scale='Blues')
        fig3.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Click rate by category
        categories = ['AI/ML', 'Business', 'Policy', 'Funding', 'Tools']
        click_rates = [22.5, 18.3, 15.2, 19.8, 16.7]
        
        fig2 = px.pie(
            values=click_rates,
            names=categories,
            title='Click Rate by Category',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Subscriber growth
        months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        subscribers = [12500, 13200, 13800, 14400, 14900, 15420]
        
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=months,
            y=subscribers,
            marker_color='#667eea',
            text=subscribers,
            textposition='auto'
        ))
        fig4.update_layout(
            title='Subscriber Growth',
            xaxis_title='Month',
            yaxis_title='Subscribers',
            height=350
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab5:
    st.header("💡 Newsletter Automation - Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ✅ Benefits of Automation")
        st.markdown("""
        **1. Time Savings** ⏰
        - Manual curation: 8-10 hours/week
        - Automated: 30 minutes setup
        - **Ongoing:** 95% time reduction
        
        **2. Consistency** 📅
        - Never miss a send date
        - Consistent quality
        - Predictable delivery
        
        **3. Scalability** 📈
        - Handle 10K+ subscribers
        - Multiple newsletters
        - No additional overhead
        
        **4. Personalization** 🎯
        - AI-powered recommendations
        - Segment-specific content
        - Dynamic content blocks
        
        **5. Cost Efficiency** 💰
        - Reduce staff hours
        - Lower cost per subscriber
        - Better ROI
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Best Practices")
        st.markdown("""
        **Content Curation:**
        - ✅ Use multiple sources (RSS, APIs, social)
        - ✅ AI-powered relevance filtering
        - ✅ Avoid duplicate content
        - ✅ Include trending topics
        
        **Email Design:**
        - ✅ Mobile-responsive templates
        - ✅ Clear hierarchy and CTAs
        - ✅ Consistent branding
        - ✅ Fast load times
        
        **Scheduling:**
        - ✅ Test different send times
        - ✅ Consider time zones
        - ✅ Avoid Mondays & Fridays
        - ✅ Weekly > Daily for engagement
        
        **Analytics:**
        - ✅ Track open/click rates
        - ✅ A/B test subject lines
        - ✅ Monitor unsubscribes
        - ✅ Segment performance
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 💡 Use Cases")
        st.markdown("""
        **1. Industry Newsletters** 📰
        - Tech news digests
        - Weekly market updates
        - Research summaries
        
        **2. Company Updates** 🏢
        - Internal communications
        - Product launches
        - Team announcements
        
        **3. Educational Content** 📚
        - Course updates
        - Learning resources
        - Tips & tutorials
        
        **4. Marketing Campaigns** 🎯
        - Lead nurturing
        - Product promotions
        - Event invitations
        
        **5. Community Newsletters** 👥
        - Member highlights
        - Event recaps
        - Community news
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Common Pitfalls")
        st.markdown("""
        **Avoid:**
        - ❌ Over-automating (lose personal touch)
        - ❌ Ignoring analytics
        - ❌ Sending too frequently
        - ❌ Generic content
        - ❌ Poor mobile experience
        - ❌ Broken links
        - ❌ Spam triggers
        
        **Solutions:**
        - ✅ Review before sending
        - ✅ Regular performance checks
        - ✅ Respect preferences
        - ✅ Quality > Quantity
        - ✅ Test on devices
        - ✅ Validate links
        - ✅ Follow CAN-SPAM
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ROI comparison
    st.markdown("---")
    st.markdown("#### 📊 Manual vs Automated Newsletter")
    
    comparison_data = {
        'Metric': ['Setup Time', 'Weekly Effort', 'Cost/Month', 'Subscribers Supported', 'Personalization'],
        'Manual': ['1 hour', '8-10 hours', '$0 (labor)', '500-1000', 'Low'],
        'Semi-Automated': ['4 hours', '2-3 hours', '$50', '5000-10K', 'Medium'],
        'Fully Automated': ['8 hours', '30 min', '$100', '50K+', 'High']
    }
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# Footer - Final Project!
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;'>
    <h2 style='margin: 0;'>🎉 Day 21 Complete - Final Project!</h2>
    <h1 style='margin: 10px 0; font-size: 48px;'>🏆 CONGRATULATIONS! 🏆</h1>
    <p style='font-size: 20px; margin: 10px 0;'>You've completed all 21 projects!</p>
    <p style='font-size: 18px; margin: 20px 0;'>
        From Titanic EDA to AI Newsletter Pipelines<br/>
        From Basic ML to Advanced GenAI Systems<br/>
        You've mastered the full AI/ML stack!
    </p>
    <p style='font-size: 16px; margin: 20px 0; opacity: 0.9;'>
        🚀 21 Projects | 🧠 21 Days | 💪 Unlimited Potential
    </p>
</div>
""", unsafe_allow_html=True)
