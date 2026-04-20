import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Note: HuggingFace transformers would be imported here for actual implementation
# For demo purposes, we'll simulate the functionality

st.set_page_config(page_title="Day 11: HuggingFace Pipelines", page_icon="🤗", layout="wide")

st.markdown("""<style>
.main {background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);}
.stMetric {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white;}
.stMetric label {color: #ffffff !important; font-weight: 600;}
h1 {color: #ffffff; font-weight: 700; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);}
h2, h3 {color: #f0f0f0; font-weight: 600;}
.highlight-box {background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; border-left: 6px solid #667eea; margin: 10px 0;}
.pipeline-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 20px; 
                text-align: center; color: white; margin: 10px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);}
.stTabs [data-baseweb="tab-list"] {gap: 10px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;}
.stTabs [data-baseweb="tab"] {background: rgba(255,255,255,0.2); color: white; border-radius: 8px; padding: 10px 20px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
.demo-box {background: rgba(255,255,255,0.98); padding: 25px; border-radius: 15px; margin: 15px 0; border: 2px solid #667eea;}
</style>""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🤗 The AI Swiss Army Knife</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Day 11: One-Line Solutions with HuggingFace Pipelines</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🤗 HuggingFace Transformers</li>
            <li>🔧 Pre-built Pipelines</li>
            <li>💬 Sentiment Analysis</li>
            <li>📝 Text Summarization</li>
            <li>🔍 Question Answering</li>
            <li>🎨 Image Generation</li>
            <li>🏷️ Named Entity Recognition</li>
            <li>⚡ One-Line AI Solutions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 11 of 21</strong></p>
        <p>HuggingFace Pipelines</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📖 Overview",
    "💬 Sentiment Analysis",
    "📝 Summarization",
    "❓ Q&A",
    "🖼️ Image Tasks",
    "💡 Insights"
])

with tab1:
    st.markdown("## 📖 HuggingFace Pipelines Overview")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🤗 What is HuggingFace?</h3>
        <p><strong>HuggingFace</strong> is the leading platform for pre-trained NLP and ML models. 
        Their <strong>transformers</strong> library provides simple pipelines for complex AI tasks.</p>
        
        <h4>✨ Key Features:</h4>
        <ul>
            <li><strong>One-Line Solutions:</strong> Complex AI with minimal code</li>
            <li><strong>Pre-trained Models:</strong> State-of-the-art models ready to use</li>
            <li><strong>30+ Tasks:</strong> NLP, Vision, Audio, Multimodal</li>
            <li><strong>Model Hub:</strong> 200,000+ models to choose from</li>
            <li><strong>Production Ready:</strong> Easy deployment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔧 Available Pipelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="pipeline-card">
            <h3>📝 Text Pipelines</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ul style='text-align: left; list-style-position: inside;'>
                <li>sentiment-analysis</li>
                <li>text-generation</li>
                <li>summarization</li>
                <li>question-answering</li>
                <li>fill-mask</li>
                <li>ner (Named Entity Recognition)</li>
                <li>translation</li>
                <li>text-classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pipeline-card">
            <h3>🔊 Audio Pipelines</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ul style='text-align: left; list-style-position: inside;'>
                <li>automatic-speech-recognition</li>
                <li>audio-classification</li>
                <li>text-to-speech</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-card">
            <h3>🖼️ Vision Pipelines</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ul style='text-align: left; list-style-position: inside;'>
                <li>image-classification</li>
                <li>object-detection</li>
                <li>image-segmentation</li>
                <li>image-to-text</li>
                <li>text-to-image</li>
                <li>zero-shot-image-classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pipeline-card">
            <h3>🎭 Multimodal</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ul style='text-align: left; list-style-position: inside;'>
                <li>visual-question-answering</li>
                <li>document-question-answering</li>
                <li>feature-extraction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 💻 Basic Usage")
    
    st.code("""
from transformers import pipeline

# Step 1: Create a pipeline (one line!)
sentiment_analyzer = pipeline('sentiment-analysis')

# Step 2: Use it!
result = sentiment_analyzer("I love this product!")

# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
""", language='python')
    
    st.markdown("""
    <div class="highlight-box">
        <h4>🎯 That's it! Just 2 lines of code.</h4>
        <p>Behind the scenes:</p>
        <ul>
            <li>✅ Model automatically downloaded</li>
            <li>✅ Tokenizer configured</li>
            <li>✅ Text preprocessed</li>
            <li>✅ Inference performed</li>
            <li>✅ Results postprocessed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Pipeline Statistics")
    
    pipeline_data = pd.DataFrame({
        'Task Category': ['Text', 'Vision', 'Audio', 'Multimodal'],
        'Number of Pipelines': [15, 8, 5, 4],
        'Popular Models': ['BERT, GPT', 'ViT, DETR', 'Wav2Vec2', 'CLIP, LayoutLM'],
        'Avg Speed': ['50ms', '100ms', '200ms', '150ms']
    })
    
    st.dataframe(pipeline_data, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("## 💬 Sentiment Analysis")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>What is Sentiment Analysis?</h3>
        <p>Determine if text expresses <strong>positive, negative, or neutral</strong> sentiment. 
        Perfect for analyzing reviews, social media, customer feedback.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Try It Yourself!")
    
    sentiment_text = st.text_area(
        "Enter text to analyze:",
        "This product exceeded my expectations! The quality is amazing and delivery was fast.",
        height=100
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing sentiment..."):
            # Simulate sentiment analysis (in real app, use: pipeline('sentiment-analysis'))
            import time
            time.sleep(1)
            
            # Simulated results
            positive_keywords = ['amazing', 'exceeded', 'fast', 'great', 'excellent', 'love', 'perfect', 'good']
            negative_keywords = ['bad', 'terrible', 'horrible', 'awful', 'poor', 'hate', 'worst']
            
            text_lower = sentiment_text.lower()
            positive_count = sum(1 for word in positive_keywords if word in text_lower)
            negative_count = sum(1 for word in negative_keywords if word in text_lower)
            
            if positive_count > negative_count:
                label = "POSITIVE"
                score = 0.95 + (positive_count * 0.01)
                score = min(score, 0.9999)
                emoji = "😊"
                color = "#4CAF50"
            elif negative_count > positive_count:
                label = "NEGATIVE"
                score = 0.92 + (negative_count * 0.01)
                score = min(score, 0.9999)
                emoji = "😞"
                color = "#F44336"
            else:
                label = "NEUTRAL"
                score = 0.88
                emoji = "😐"
                color = "#FF9800"
        
        st.markdown(f"""
        <div class="demo-box" style="border-color: {color};">
            <h2 style="text-align: center; color: {color};">{emoji} {label}</h2>
            <h3 style="text-align: center; color: #333;">Confidence: {score:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Score", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffebee'},
                    {'range': [50, 75], 'color': '#fff3e0'},
                    {'range': [75, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=300,
            font={'color': "#333", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Example Use Cases")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Customer Reviews**
        - Product feedback
        - Service ratings
        - App reviews
        """)
    
    with col2:
        st.info("""
        **Social Media**
        - Brand monitoring
        - Crisis detection
        - Trend analysis
        """)
    
    with col3:
        st.info("""
        **Business**
        - Survey analysis
        - Employee feedback
        - Market research
        """)
    
    st.markdown("### 💻 Code Example")
    
    st.code("""
from transformers import pipeline

# Create sentiment analyzer
sentiment = pipeline('sentiment-analysis', 
                    model='FacebookAI/roberta-large-mnli')

# Analyze text
text = "This product is amazing!"
result = sentiment(text)

print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
""", language='python')

with tab3:
    st.markdown("## 📝 Text Summarization")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>What is Summarization?</h3>
        <p>Automatically generate <strong>concise summaries</strong> of long documents. 
        Save time reading articles, reports, and documents.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📄 Try It Yourself!")
    
    long_text = st.text_area(
        "Enter text to summarize:",
        """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".

As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine learning capabilities are typically classified as weak AI, as they are designed to perform specific tasks, rather than to possess general intelligence.""",
        height=200
    )
    
    max_length = st.slider("Maximum summary length (words):", 30, 200, 100)
    min_length = st.slider("Minimum summary length (words):", 10, 100, 30)
    
    if st.button("✨ Generate Summary", type="primary", use_container_width=True):
        with st.spinner("🤖 Creating summary..."):
            import time
            time.sleep(1.5)
            
            # Simulated summary (in real app, use: pipeline('summarization'))
            summary = """Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence. 
            AI is the study of intelligent agents that perceive their environment and take actions to achieve goals. 
            As machines become more capable, tasks requiring intelligence are often removed from AI's definition."""
        
        st.markdown("""
        <div class="demo-box">
            <h3 style="color: #667eea;">📝 Summary</h3>
            <p style="font-size: 16px; line-height: 1.6; color: #333;">""" + summary + """</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        original_words = len(long_text.split())
        summary_words = len(summary.split())
        compression_ratio = (1 - summary_words/original_words) * 100
        
        with col1:
            st.metric("Original Words", f"{original_words}")
        with col2:
            st.metric("Summary Words", f"{summary_words}")
        with col3:
            st.metric("Compression", f"{compression_ratio:.1f}%", delta=f"-{original_words-summary_words} words")
        with col4:
            st.metric("Time Saved", "~2 min", delta="Reading time")
    
    st.markdown("### 🎯 Summarization Types")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h4>📄 Extractive Summarization</h4>
            <ul>
                <li>Selects key sentences from original text</li>
                <li>Preserves exact wording</li>
                <li>Faster and more reliable</li>
                <li>Example: TextRank, LexRank</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h4>✍️ Abstractive Summarization</h4>
            <ul>
                <li>Generates new sentences</li>
                <li>Paraphrases content</li>
                <li>More human-like</li>
                <li>Example: BART, T5, Pegasus</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 💻 Code Example")
    
    st.code("""
from transformers import pipeline

# Create summarizer
summarizer = pipeline('summarization', 
                     model='sshleifer/distilbart-cnn-12-6')

# Summarize text
long_text = "Your long article here..."
summary = summarizer(long_text, 
                    max_length=130, 
                    min_length=30, 
                    do_sample=False)

print(summary[0]['summary_text'])
""", language='python')

with tab4:
    st.markdown("## ❓ Question Answering")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>What is Question Answering?</h3>
        <p>Extract <strong>answers</strong> from given context. The model reads a passage 
        and answers questions based on the information within it.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Try It Yourself!")
    
    context = st.text_area(
        "📖 Context (Information):",
        """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world. The tower is 330 meters (1,083 ft) tall, about the same height as an 81-story building. It was the first structure in the world to surpass both the 200-meter and 300-meter mark in height.""",
        height=150
    )
    
    question = st.text_input(
        "❓ Your Question:",
        "How tall is the Eiffel Tower?"
    )
    
    if st.button("🔍 Find Answer", type="primary", use_container_width=True):
        with st.spinner("🤖 Searching for answer..."):
            import time
            time.sleep(1)
            
            # Simulated QA (in real app, use: pipeline('question-answering'))
            # Simple keyword matching for demo
            if "tall" in question.lower() or "height" in question.lower():
                answer = "330 meters (1,083 ft)"
                score = 0.97
            elif "when" in question.lower() or "built" in question.lower():
                answer = "1887 to 1889"
                score = 0.95
            elif "who" in question.lower() or "name" in question.lower():
                answer = "Gustave Eiffel"
                score = 0.96
            elif "where" in question.lower() or "location" in question.lower():
                answer = "Champ de Mars in Paris, France"
                score = 0.94
            else:
                answer = "wrought-iron lattice tower"
                score = 0.85
        
        st.markdown(f"""
        <div class="demo-box">
            <h3 style="color: #667eea;">💡 Answer</h3>
            <h2 style="color: #333; margin: 20px 0;">{answer}</h2>
            <p style="color: #666;">Confidence: {score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence visualization
        fig = go.Figure(go.Bar(
            x=[score],
            y=['Confidence'],
            orientation='h',
            marker=dict(
                color='#667eea',
                line=dict(color='#764ba2', width=2)
            ),
            text=[f'{score:.1%}'],
            textposition='auto',
        ))
        
        fig.update_layout(
            xaxis=dict(range=[0, 1], showticklabels=False),
            height=150,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(255,255,255,0.9)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Sample Questions to Try")
    
    sample_questions = [
        "How tall is the Eiffel Tower?",
        "When was it built?",
        "Who designed the Eiffel Tower?",
        "Where is it located?",
        "What is it made of?"
    ]
    
    cols = st.columns(len(sample_questions))
    for i, q in enumerate(sample_questions):
        with cols[i]:
            if st.button(f"❓ {q}", key=f"q{i}"):
                st.info(f"Try: {q}")
    
    st.markdown("### 💻 Code Example")
    
    st.code("""
from transformers import pipeline

# Create QA pipeline
qa = pipeline('question-answering')

# Ask question
result = qa(question="How tall is the Eiffel Tower?",
            context='''The Eiffel Tower is 330 meters tall...''')

print(result['answer'])  # "330 meters"
print(result['score'])   # 0.97
""", language='python')

with tab5:
    st.markdown("## 🖼️ Image & Vision Tasks")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🎨 Vision Pipelines</h3>
        <p>HuggingFace also provides powerful pipelines for <strong>computer vision</strong> tasks,
        including image classification, object detection, and even text-to-image generation!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="pipeline-card">
            <h3>🏷️ Image Classification</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p>Identify what's in an image</p>
            <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: left; font-size: 12px;">
classifier = pipeline(
    'image-classification'
)
result = classifier(image)
# [{'label': 'cat', 
#   'score': 0.98}]
            </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pipeline-card">
            <h3>🎯 Object Detection</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p>Detect and locate objects</p>
            <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: left; font-size: 12px;">
detector = pipeline(
    'object-detection'
)
result = detector(image)
# [{'label': 'person',
#   'box': {...}}]
            </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pipeline-card">
            <h3>✂️ Image Segmentation</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p>Segment image by pixels</p>
            <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: left; font-size: 12px;">
segmenter = pipeline(
    'image-segmentation'
)
result = segmenter(image)
            </pre>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-card">
            <h3>📸 Image-to-Text</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p>Generate captions for images</p>
            <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: left; font-size: 12px;">
captioner = pipeline(
    'image-to-text'
)
result = captioner(image)
# "A cat sitting on a sofa"
            </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pipeline-card">
            <h3>🎨 Text-to-Image</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p>Generate images from text!</p>
            <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: left; font-size: 12px;">
generator = pipeline(
    'text-to-image'
)
image = generator(
    "A sunset over mountains"
)
            </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pipeline-card">
            <h3>🔍 Zero-Shot Classification</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p>Classify without training</p>
            <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: left; font-size: 12px;">
classifier = pipeline(
    'zero-shot-classification'
)
result = classifier(image, 
    candidate_labels=[...])
            </pre>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🌟 Featured Models")
    
    models_data = pd.DataFrame({
        'Task': ['Image Classification', 'Object Detection', 'Segmentation', 'Image Captioning', 'Text-to-Image'],
        'Popular Model': ['ViT', 'DETR', 'SegFormer', 'BLIP', 'Stable Diffusion'],
        'Parameters': ['86M', '41M', '85M', '224M', '1B+'],
        'Speed': ['Fast', 'Medium', 'Medium', 'Fast', 'Slow'],
        'Accuracy': ['Very High', 'High', 'High', 'Good', 'Creative']
    })
    
    st.dataframe(models_data, use_container_width=True, hide_index=True)

with tab6:
    st.markdown("## 💡 Key Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 Why Use Pipelines?</h3>
            <ul>
                <li><strong>Simplicity:</strong> One line of code for complex tasks</li>
                <li><strong>Consistency:</strong> Standardized API across all tasks</li>
                <li><strong>Speed:</strong> Optimized for performance</li>
                <li><strong>Reliability:</strong> Battle-tested in production</li>
                <li><strong>Flexibility:</strong> Easy model swapping</li>
                <li><strong>Documentation:</strong> Excellent community support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>⚙️ Customization Options</h3>
            <ul>
                <li><strong>Model Selection:</strong> Choose specific models</li>
                <li><strong>Device:</strong> CPU, GPU, or TPU</li>
                <li><strong>Batch Processing:</strong> Process multiple inputs</li>
                <li><strong>Parameters:</strong> Temperature, max_length, etc.</li>
                <li><strong>Tokenizer:</strong> Custom preprocessing</li>
                <li><strong>Post-processing:</strong> Custom output formatting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🚀 Performance Tips</h3>
            <ul>
                <li>Use GPU for faster inference</li>
                <li>Enable batch processing for multiple inputs</li>
                <li>Cache pipelines to avoid reloading</li>
                <li>Use smaller models for real-time apps</li>
                <li>Quantize models for edge devices</li>
                <li>Monitor memory usage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>🌟 Real-World Applications</h3>
            <ul>
                <li><strong>Customer Service:</strong> Sentiment analysis, chatbots</li>
                <li><strong>Content Moderation:</strong> Text classification</li>
                <li><strong>News:</strong> Summarization, NER</li>
                <li><strong>E-commerce:</strong> Product classification, Q&A</li>
                <li><strong>Healthcare:</strong> Document analysis</li>
                <li><strong>Finance:</strong> Sentiment from reports</li>
                <li><strong>Social Media:</strong> Trend analysis</li>
                <li><strong>Research:</strong> Literature review automation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>📊 Model Hub Statistics</h3>
            <ul>
                <li>200,000+ models available</li>
                <li>10,000+ datasets</li>
                <li>30+ supported tasks</li>
                <li>100+ languages supported</li>
                <li>1M+ downloads per day</li>
                <li>Active community of researchers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>⚠️ Considerations</h3>
            <ul>
                <li>Model size and memory requirements</li>
                <li>Inference latency for real-time apps</li>
                <li>API rate limits (if using hosted models)</li>
                <li>Data privacy and security</li>
                <li>Model bias and fairness</li>
                <li>Licensing restrictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 💻 Complete Example: Multi-Task Analysis")
    
    st.code("""
from transformers import pipeline

# Initialize multiple pipelines
sentiment = pipeline('sentiment-analysis')
summarizer = pipeline('summarization')
qa = pipeline('question-answering')
ner = pipeline('ner')

# Analyze text
text = "Your document here..."

# Get sentiment
sentiment_result = sentiment(text)

# Generate summary
summary = summarizer(text, max_length=100)

# Extract entities
entities = ner(text)

# Answer questions
answer = qa(question="What is this about?", context=text)

# All in just a few lines of code!
""", language='python')
    
    st.markdown("### 🎓 Key Takeaways")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>✅ What We Learned:</h4>
        <ul>
            <li>HuggingFace provides <strong>one-line solutions</strong> for complex AI tasks</li>
            <li>Pipelines handle all the <strong>complexity</strong> behind the scenes</li>
            <li><strong>30+ tasks</strong> available across NLP, Vision, and Audio</li>
            <li>Easy to <strong>swap models</strong> and <strong>customize parameters</strong></li>
            <li>Perfect for <strong>rapid prototyping</strong> and <strong>production deployment</strong></li>
            <li>Extensive <strong>model hub</strong> with pre-trained models ready to use</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🤗 HuggingFace Pipelines Mastered!</h3>
    <p><strong>Day 11 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>The Swiss Army Knife of AI - One Line Solutions</p>
</div>
""", unsafe_allow_html=True)
