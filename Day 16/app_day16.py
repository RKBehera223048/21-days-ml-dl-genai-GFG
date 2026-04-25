import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import re

# Page configuration
st.set_page_config(
    page_title="Day 16: Intelligent Document Automation",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with document processing orange gradient theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #fd8451 100%);
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
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
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
    
    .ocr-output {
        background: #2d3748;
        color: #90cdf4;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f5576c;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .resume-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1586281380349-632531db7ed4?w=800&h=400&fit=crop", 
             use_container_width=True)
    
    st.markdown("### 📄 Day 16: Intelligent OCR Bot")
    st.markdown("**Smart Document Processing & Automation**")
    
    st.markdown("---")
    st.markdown("### 🎯 What I Learned")
    st.markdown("""
    - 📸 Optical Character Recognition (OCR)
    - 📄 Document structure analysis
    - 🧠 Information extraction from PDFs
    - 🔍 Entity recognition in documents
    - 📊 Resume parsing & analysis
    - 🏗️ Document classification
    - ✨ Text cleaning & preprocessing
    - 🤖 Automated document workflows
    """)
    
    st.markdown("---")
    st.info("**Day 16 of 21** - GeeksforGeeks ML & GenAI Course")

# Sample resume data
@st.cache_data
def get_sample_resumes():
    return [
        {
            "id": 1,
            "name": "John Smith",
            "email": "john.smith@email.com",
            "phone": "+1-555-0101",
            "experience_years": 5,
            "education": "B.S. Computer Science, MIT (2018)",
            "skills": ["Python", "Machine Learning", "TensorFlow", "AWS", "Docker", "SQL"],
            "current_role": "Senior ML Engineer",
            "company": "Tech Corp",
            "summary": "Experienced ML engineer with 5 years in production ML systems. Led team of 4 engineers building recommendation engines. Deployed 10+ models to production.",
            "location": "San Francisco, CA"
        },
        {
            "id": 2,
            "name": "Sarah Johnson",
            "email": "sarah.j@email.com",
            "phone": "+1-555-0102",
            "experience_years": 3,
            "education": "M.S. Data Science, Stanford (2020)",
            "skills": ["Python", "Deep Learning", "PyTorch", "NLP", "LangChain", "FastAPI"],
            "current_role": "Data Scientist",
            "company": "AI Innovations",
            "summary": "Data scientist specializing in NLP and GenAI. Built chatbots serving 100K+ users. Published 3 research papers on transformer models.",
            "location": "New York, NY"
        },
        {
            "id": 3,
            "name": "Michael Chen",
            "email": "m.chen@email.com",
            "phone": "+1-555-0103",
            "experience_years": 7,
            "education": "Ph.D. AI, Carnegie Mellon (2016)",
            "skills": ["Python", "Computer Vision", "Keras", "OpenCV", "GCP", "Kubernetes"],
            "current_role": "Lead AI Researcher",
            "company": "Research Labs Inc",
            "summary": "PhD in AI with 7 years research experience. Expert in computer vision and object detection. 15+ publications in top conferences.",
            "location": "Boston, MA"
        },
        {
            "id": 4,
            "name": "Emily Davis",
            "email": "emily.davis@email.com",
            "phone": "+1-555-0104",
            "experience_years": 4,
            "education": "B.S. Statistics, UC Berkeley (2019)",
            "skills": ["Python", "R", "Statistical Modeling", "A/B Testing", "Tableau", "Spark"],
            "current_role": "Analytics Manager",
            "company": "DataCorp",
            "summary": "Analytics manager with strong statistical background. Led 20+ A/B tests driving $5M revenue. Built dashboards for C-level executives.",
            "location": "Seattle, WA"
        },
        {
            "id": 5,
            "name": "David Wilson",
            "email": "d.wilson@email.com",
            "phone": "+1-555-0105",
            "experience_years": 6,
            "education": "M.S. Computer Engineering, Georgia Tech (2017)",
            "skills": ["Python", "MLOps", "Airflow", "Jenkins", "Terraform", "Prometheus"],
            "current_role": "MLOps Engineer",
            "company": "CloudTech",
            "summary": "MLOps specialist building scalable ML infrastructure. Reduced model deployment time by 80%. Managed 50+ production models.",
            "location": "Austin, TX"
        }
    ]

# Simulate OCR processing
def simulate_ocr_extraction(resume_data):
    """Simulate OCR text extraction from resume"""
    return f"""
RESUME

Name: {resume_data['name']}
Email: {resume_data['email']}
Phone: {resume_data['phone']}
Location: {resume_data['location']}

EDUCATION
{resume_data['education']}

PROFESSIONAL SUMMARY
{resume_data['summary']}

WORK EXPERIENCE
{resume_data['current_role']} at {resume_data['company']}
{resume_data['experience_years']} years of experience

SKILLS
{', '.join(resume_data['skills'])}
"""

# Main app
st.title("📄 Intelligent Document Automation Bot")
st.markdown("### Extract, Analyze & Automate Document Processing with OCR")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 OCR Demo", 
    "📊 Resume Analysis", 
    "🧠 How OCR Works",
    "🏗️ Architecture",
    "💡 Insights"
])

with tab1:
    st.header("📸 Optical Character Recognition Demo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📄 Select Resume to Process")
        
        resumes = get_sample_resumes()
        
        resume_options = [f"Resume {r['id']}: {r['name']} - {r['current_role']}" for r in resumes]
        selected = st.selectbox("Choose a resume:", resume_options)
        
        selected_idx = int(selected.split(":")[0].replace("Resume ", "")) - 1
        selected_resume = resumes[selected_idx]
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display resume info card
        st.markdown('<div class="resume-card">', unsafe_allow_html=True)
        st.markdown(f"### {selected_resume['name']}")
        st.markdown(f"**{selected_resume['current_role']}** @ {selected_resume['company']}")
        st.markdown(f"📍 {selected_resume['location']}")
        st.markdown(f"📧 {selected_resume['email']}")
        st.markdown(f"📞 {selected_resume['phone']}")
        st.markdown(f"🎓 {selected_resume['education']}")
        st.markdown(f"⏱️ **Experience:** {selected_resume['experience_years']} years")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🔍 Process with OCR", type="primary", use_container_width=True):
            st.session_state['ocr_processed'] = True
            st.session_state['processed_resume'] = selected_resume
    
    with col2:
        if st.session_state.get('ocr_processed', False):
            resume = st.session_state['processed_resume']
            
            with st.spinner("Processing document with OCR..."):
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                st.success("✅ OCR Processing Complete!")
            
            # Show extracted text
            st.markdown("#### 📝 Extracted Text")
            ocr_text = simulate_ocr_extraction(resume)
            st.markdown(f'<div class="ocr-output">{ocr_text}</div>', unsafe_allow_html=True)
            
            # Show extracted entities
            st.markdown("#### 🎯 Extracted Entities")
            
            entities_df = pd.DataFrame({
                'Entity Type': ['Name', 'Email', 'Phone', 'Education', 'Experience', 'Skills'],
                'Extracted Value': [
                    resume['name'],
                    resume['email'],
                    resume['phone'],
                    resume['education'],
                    f"{resume['experience_years']} years",
                    f"{len(resume['skills'])} skills identified"
                ],
                'Confidence': ['99%', '98%', '97%', '95%', '99%', '96%']
            })
            
            st.dataframe(entities_df, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Select a resume and click 'Process with OCR' to see results")
            
            # Show example OCR process
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.markdown("#### 🔄 OCR Processing Steps")
            st.markdown("""
            **1. Image Preprocessing** 📸
            - Grayscale conversion
            - Noise reduction
            - Contrast enhancement
            - Deskewing
            
            **2. Text Detection** 🔍
            - Locate text regions
            - Identify text blocks
            - Separate text from images
            
            **3. Character Recognition** 🔤
            - OCR engine (Tesseract/EasyOCR)
            - Character segmentation
            - Pattern matching
            
            **4. Post-processing** ✨
            - Spell checking
            - Format correction
            - Structure analysis
            
            **5. Entity Extraction** 🎯
            - Named Entity Recognition
            - Information categorization
            - Data structuring
            """)
            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("📊 Resume Analysis Dashboard")
    
    resumes = get_sample_resumes()
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", len(resumes))
    with col2:
        avg_exp = np.mean([r['experience_years'] for r in resumes])
        st.metric("Avg Experience", f"{avg_exp:.1f} years")
    with col3:
        total_skills = sum([len(r['skills']) for r in resumes])
        st.metric("Total Skills", total_skills)
    with col4:
        unique_skills = len(set([skill for r in resumes for skill in r['skills']]))
        st.metric("Unique Skills", unique_skills)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Experience distribution
        exp_data = pd.DataFrame([
            {'Name': r['name'], 'Experience': r['experience_years']} 
            for r in resumes
        ])
        
        fig1 = px.bar(exp_data, x='Name', y='Experience',
                      title='Experience Distribution (Years)',
                      color='Experience',
                      color_continuous_scale='Reds')
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Skills frequency
        all_skills = {}
        for r in resumes:
            for skill in r['skills']:
                all_skills[skill] = all_skills.get(skill, 0) + 1
        
        skills_df = pd.DataFrame([
            {'Skill': k, 'Count': v} 
            for k, v in sorted(all_skills.items(), key=lambda x: x[1], reverse=True)
        ])
        
        fig3 = px.bar(skills_df.head(10), x='Count', y='Skill',
                      title='Top 10 Skills Across Resumes',
                      orientation='h',
                      color='Count',
                      color_continuous_scale='Oranges')
        fig3.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Location distribution
        locations = [r['location'].split(',')[-1].strip() for r in resumes]
        loc_df = pd.DataFrame({'Location': locations})
        loc_count = loc_df['Location'].value_counts().reset_index()
        loc_count.columns = ['Location', 'Count']
        
        fig2 = px.pie(loc_count, values='Count', names='Location',
                      title='Candidate Distribution by State',
                      color_discrete_sequence=px.colors.sequential.RdBu)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Role distribution
        roles = [r['current_role'] for r in resumes]
        role_df = pd.DataFrame({'Role': roles})
        role_count = role_df['Role'].value_counts().reset_index()
        role_count.columns = ['Role', 'Count']
        
        fig4 = px.pie(role_count, values='Count', names='Role',
                      title='Current Roles Distribution',
                      color_discrete_sequence=px.colors.sequential.Sunset)
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Detailed table
    st.markdown("---")
    st.markdown("#### 📋 Detailed Resume Comparison")
    
    comparison_df = pd.DataFrame([{
        'Name': r['name'],
        'Role': r['current_role'],
        'Company': r['company'],
        'Experience': f"{r['experience_years']} years",
        'Education': r['education'].split(',')[0],
        'Skills': len(r['skills']),
        'Location': r['location']
    } for r in resumes])
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=250)

with tab3:
    st.header("🧠 How OCR Technology Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📚 OCR Fundamentals")
        st.markdown("""
        **Optical Character Recognition (OCR)** converts different types of documents 
        (scanned paper, PDFs, images) into editable and searchable data.
        
        **Key Technologies:**
        
        **1. Traditional OCR (Tesseract)**
        - Rule-based pattern matching
        - Template matching
        - Feature extraction
        - Character segmentation
        
        **2. Deep Learning OCR**
        - CNN for image processing
        - RNN/LSTM for sequence recognition
        - Attention mechanisms
        - End-to-end learning
        
        **3. Modern Approaches**
        - Vision Transformers (ViT)
        - TrOCR (Transformer-based OCR)
        - DocFormer
        - LayoutLM
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔧 Implementation Example")
        st.code("""
# Using Tesseract OCR
import pytesseract
from PIL import Image

# Basic OCR
image = Image.open('document.jpg')
text = pytesseract.image_to_string(image)

# With configuration
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(
    image, 
    config=custom_config
)

# Extract detailed data
data = pytesseract.image_to_data(
    image, 
    output_type='dict'
)
        """, language='python')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📄 Document Processing Pipeline")
        st.markdown("""
        **Step 1: Preprocessing** 🖼️
        ```python
        # Image enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.threshold(denoised, 0, 255, 
                               cv2.THRESH_BINARY + 
                               cv2.THRESH_OTSU)[1]
        ```
        
        **Step 2: Text Detection** 🔍
        ```python
        # Using EAST detector
        detector = cv2.dnn.readNet('east_detector.pb')
        blob = cv2.dnn.blobFromImage(image)
        detector.setInput(blob)
        boxes = detector.forward()
        ```
        
        **Step 3: Recognition** 🔤
        ```python
        # Using EasyOCR
        import easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image)
        ```
        
        **Step 4: Entity Extraction** 🎯
        ```python
        # Using spaCy NER
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(extracted_text)
        
        entities = {
            'persons': [ent.text for ent in doc.ents 
                       if ent.label_ == 'PERSON'],
            'emails': re.findall(r'\\S+@\\S+', text),
            'phones': re.findall(r'\\+?\\d[\\d -]{8,}', text)
        }
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Comparison table
    st.markdown("---")
    st.markdown("#### ⚖️ OCR Engine Comparison")
    
    comparison_data = {
        'Engine': ['Tesseract', 'EasyOCR', 'PaddleOCR', 'AWS Textract', 'Google Vision', 'Azure OCR'],
        'Accuracy': ['85-90%', '90-95%', '92-96%', '95-98%', '96-99%', '95-98%'],
        'Speed': ['Fast', 'Medium', 'Medium', 'Fast', 'Fast', 'Fast'],
        'Languages': [100, 80, 80, 75, 100, 70],
        'Cost': ['Free', 'Free', 'Free', '$$', '$$$', '$$'],
        'Best For': ['General', 'Handwriting', 'Asian langs', 'Forms', 'Photos', 'Business docs']
    }
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

with tab4:
    st.header("🏗️ OCR System Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🏛️ System Architecture")
        st.markdown("""
        ```
        ┌─────────────────────────────────┐
        │      Document Input             │
        │  (PDF, Image, Scan)             │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │   Image Preprocessing           │
        │  - Grayscale conversion         │
        │  - Noise reduction              │
        │  - Contrast enhancement          │
        │  - Deskewing & rotation         │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Text Region Detection        │
        │  - Layout analysis              │
        │  - Text block identification    │
        │  - Line & word segmentation     │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │     OCR Engine                  │
        │  - Character recognition        │
        │  - Deep learning models         │
        │  - Language detection           │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Post-processing              │
        │  - Spell correction             │
        │  - Format preservation          │
        │  - Confidence scoring           │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    NLP Analysis                 │
        │  - Named Entity Recognition     │
        │  - Information extraction       │
        │  - Classification               │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │    Structured Output            │
        │  (JSON, Database, API)          │
        └─────────────────────────────────┘
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔧 Complete Implementation")
        st.code("""
# Full OCR Pipeline
import cv2
import pytesseract
import spacy
import re

class DocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
    
    def preprocess(self, image_path):
        '''Image preprocessing'''
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Threshold
        thresh = cv2.threshold(
            denoised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        
        return thresh
    
    def extract_text(self, image):
        '''OCR extraction'''
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(
            image, 
            config=custom_config
        )
        return text
    
    def extract_entities(self, text):
        '''NER extraction'''
        doc = self.nlp(text)
        
        entities = {
            'name': None,
            'email': None,
            'phone': None,
            'skills': [],
            'experience': None
        }
        
        # Extract person name
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['name'] = ent.text
                break
        
        # Extract email
        emails = re.findall(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[A-Z|a-z]{2,}',
            text
        )
        if emails:
            entities['email'] = emails[0]
        
        # Extract phone
        phones = re.findall(
            r'\\+?\\d[\\d -]{8,}\\d',
            text
        )
        if phones:
            entities['phone'] = phones[0]
        
        return entities
    
    def process(self, image_path):
        '''Full pipeline'''
        # Preprocess
        processed_img = self.preprocess(image_path)
        
        # Extract text
        text = self.extract_text(processed_img)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        return {
            'raw_text': text,
            'entities': entities
        }

# Usage
processor = DocumentProcessor()
result = processor.process('resume.pdf')
        """, language='python')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Popular tools
    st.markdown("---")
    st.markdown("#### 🛠️ Popular OCR Tools & Libraries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📦 Tesseract**")
        st.markdown("Google's open-source OCR")
        st.markdown("100+ languages")
        st.markdown("Free & widely used")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📦 EasyOCR**")
        st.markdown("Deep learning-based")
        st.markdown("80+ languages")
        st.markdown("Good for handwriting")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📦 AWS Textract**")
        st.markdown("Cloud-based OCR")
        st.markdown("Form & table extraction")
        st.markdown("High accuracy")
        st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.header("💡 Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ✅ Benefits of OCR Automation")
        st.markdown("""
        **1. Time Savings** ⏰
        - Manual entry: 10-15 min/document
        - OCR: 5-10 seconds/document
        - **100x faster processing**
        
        **2. Cost Reduction** 💰
        - Eliminate manual data entry
        - Reduce staffing costs
        - **Save 70-90% on processing costs**
        
        **3. Accuracy** 🎯
        - Human error rate: 1-5%
        - Modern OCR: 95-99% accuracy
        - **Fewer mistakes & corrections**
        
        **4. Scalability** 📈
        - Process thousands of documents
        - 24/7 automated operations
        - **Handle peak loads easily**
        
        **5. Accessibility** 🌍
        - Digital archives
        - Searchable documents
        - **Better information retrieval**
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Real-World Applications")
        st.markdown("""
        - **HR**: Resume screening automation
        - **Finance**: Invoice processing
        - **Healthcare**: Medical record digitization
        - **Legal**: Contract analysis
        - **Retail**: Receipt processing
        - **Government**: ID verification
        - **Education**: Test grading
        - **Logistics**: Package label reading
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Challenges & Solutions")
        st.markdown("""
        **Challenge 1: Poor Image Quality** 📷
        - **Solution**: Preprocessing (denoise, enhance)
        - Use adaptive thresholding
        - Apply image enhancement techniques
        
        **Challenge 2: Complex Layouts** 📄
        - **Solution**: Layout analysis algorithms
        - Use LayoutLM or DocFormer
        - Multi-stage processing
        
        **Challenge 3: Handwriting** ✍️
        - **Solution**: Deep learning OCR
        - Use EasyOCR or Google Vision
        - Fine-tune on specific handwriting
        
        **Challenge 4: Multiple Languages** 🌍
        - **Solution**: Language detection first
        - Use multilingual models
        - Tesseract supports 100+ languages
        
        **Challenge 5: Low Accuracy** ❌
        - **Solution**: Combine multiple OCR engines
        - Post-processing with spell check
        - Use confidence scores
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔐 Security & Privacy")
        st.markdown("""
        **Best Practices:**
        - ✅ Encrypt documents in transit
        - ✅ Use secure cloud storage
        - ✅ Implement access controls
        - ✅ Audit logging
        - ✅ GDPR/HIPAA compliance
        - ✅ Data retention policies
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance comparison
    st.markdown("---")
    st.markdown("#### 📊 OCR Performance Metrics")
    
    perf_data = {
        'Metric': ['Processing Speed', 'Accuracy', 'Cost per 1000 pages', 'Setup Time', 'Maintenance'],
        'Manual Entry': ['10-15 min', '95-99%', '$500-1000', 'Immediate', 'High (training)'],
        'Traditional OCR': ['30-60 sec', '85-90%', '$50-100', '1-2 weeks', 'Medium'],
        'Modern AI OCR': ['5-10 sec', '95-98%', '$100-200', '1-2 days', 'Low'],
        'Cloud OCR': ['2-5 sec', '96-99%', '$150-300', 'Minutes', 'None']
    }
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True, height=250)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🎓 Day 16 Complete: Intelligent Document Automation</h3>
    <p>Transform documents into actionable data with OCR & AI!</p>
</div>
""", unsafe_allow_html=True)
