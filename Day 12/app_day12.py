import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Day 12: Computer Vision", page_icon="👁️", layout="wide")

st.markdown("""<style>
.main {background: linear-gradient(135deg, #11998e 0%, #38ef7d 50%, #06beb6 100%);}
.stMetric {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 15px; color: white;}
.stMetric label {color: #ffffff !important; font-weight: 600;}
h1 {color: #ffffff; font-weight: 700; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);}
h2, h3 {color: #f0f0f0; font-weight: 600;}
.highlight-box {background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; border-left: 6px solid #11998e; margin: 10px 0;}
.detection-card {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 25px; border-radius: 20px; 
                  text-align: center; color: white; margin: 10px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);}
.stTabs [data-baseweb="tab-list"] {gap: 10px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;}
.stTabs [data-baseweb="tab"] {background: rgba(255,255,255,0.2); color: white; border-radius: 8px; padding: 10px 20px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);}
</style>""", unsafe_allow_html=True)

def create_sample_image(width=640, height=480, objects=True):
    """Create a sample image with simple shapes"""
    img = Image.new('RGB', (width, height), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    
    # Draw ground
    draw.rectangle([(0, height*0.6), (width, height)], fill=(34, 139, 34))
    
    if objects:
        # Draw a "car" (rectangle + circles)
        car_x, car_y = width*0.3, height*0.7
        draw.rectangle([(car_x, car_y), (car_x+100, car_y+40)], fill=(255, 0, 0))
        draw.ellipse([(car_x+10, car_y+30), (car_x+30, car_y+50)], fill=(0, 0, 0))
        draw.ellipse([(car_x+70, car_y+30), (car_x+90, car_y+50)], fill=(0, 0, 0))
        
        # Draw a "person" (stick figure)
        person_x, person_y = width*0.6, height*0.7
        draw.ellipse([(person_x, person_y), (person_x+20, person_y+20)], fill=(255, 220, 177))
        draw.line([(person_x+10, person_y+20), (person_x+10, person_y+60)], fill=(0, 0, 0), width=3)
        draw.line([(person_x+10, person_y+30), (person_x-5, person_y+45)], fill=(0, 0, 0), width=3)
        draw.line([(person_x+10, person_y+30), (person_x+25, person_y+45)], fill=(0, 0, 0), width=3)
        
        # Draw a "tree"
        tree_x, tree_y = width*0.8, height*0.6
        draw.rectangle([(tree_x, tree_y), (tree_x+15, tree_y+60)], fill=(101, 67, 33))
        draw.ellipse([(tree_x-20, tree_y-40), (tree_x+35, tree_y+10)], fill=(0, 128, 0))
    
    return img

def draw_detection_box(img, bbox, label, confidence, color=(0, 255, 0)):
    """Draw bounding box on image"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    text = f"{label} ({confidence:.0%})"
    text_bbox = draw.textbbox((x1, y1), text, font=font)
    draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
    draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
    
    return img_copy

def enhance_resolution(img, scale_factor=2):
    """Simple upscaling simulation"""
    new_size = (img.width * scale_factor, img.height * scale_factor)
    # Use LANCZOS for high-quality upscaling
    enhanced = img.resize(new_size, Image.LANCZOS)
    
    # Apply sharpening
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.2)
    
    return enhanced

st.markdown("<h1 style='text-align: center;'>👁️ Real-World Computer Vision</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Day 12: Object Detection & Image Enhancement</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1518098268026-4e89f1a2cd8e?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>👁️ Object Detection</li>
            <li>📦 Bounding Boxes</li>
            <li>🎯 YOLO & DETR</li>
            <li>🔍 Image Enhancement</li>
            <li>📸 Super Resolution</li>
            <li>🎨 Image Upscaling</li>
            <li>💎 Quality Improvement</li>
            <li>🔬 CV Applications</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 12 of 21</strong></p>
        <p>Computer Vision Tasks</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 CV Overview",
    "🎯 Object Detection",
    "🔍 Image Enhancement",
    "🏗️ Architectures",
    "💡 Insights"
])

with tab1:
    st.markdown("## 📖 Computer Vision Overview")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>👁️ What is Computer Vision?</h3>
        <p><strong>Computer Vision (CV)</strong> enables computers to understand and interpret visual information
        from the world. It's how machines "see" and make sense of images and videos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Core Computer Vision Tasks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="detection-card">
            <h3>🏷️ Image Classification</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p><strong>Task:</strong> What is this?</p>
            <p><strong>Input:</strong> Image</p>
            <p><strong>Output:</strong> Class label (e.g., "cat")</p>
            <p><strong>Models:</strong> ResNet, VGG, EfficientNet</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detection-card">
            <h3>🔍 Object Detection</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p><strong>Task:</strong> What and where?</p>
            <p><strong>Input:</strong> Image</p>
            <p><strong>Output:</strong> Boxes + labels</p>
            <p><strong>Models:</strong> YOLO, DETR, Faster R-CNN</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detection-card">
            <h3>✂️ Segmentation</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p><strong>Task:</strong> Pixel-level classification</p>
            <p><strong>Input:</strong> Image</p>
            <p><strong>Output:</strong> Mask per pixel</p>
            <p><strong>Models:</strong> U-Net, Mask R-CNN</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="detection-card">
            <h3>📸 Image Enhancement</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p><strong>Task:</strong> Improve image quality</p>
            <p><strong>Input:</strong> Low-quality image</p>
            <p><strong>Output:</strong> Enhanced image</p>
            <p><strong>Models:</strong> ESRGAN, Real-ESRGAN</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detection-card">
            <h3>🎭 Face Recognition</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p><strong>Task:</strong> Identify people</p>
            <p><strong>Input:</strong> Face image</p>
            <p><strong>Output:</strong> Person ID</p>
            <p><strong>Models:</strong> FaceNet, ArcFace</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detection-card">
            <h3>📝 OCR</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p><strong>Task:</strong> Read text from images</p>
            <p><strong>Input:</strong> Image with text</p>
            <p><strong>Output:</strong> Text string</p>
            <p><strong>Models:</strong> Tesseract, EasyOCR</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 CV Task Comparison")
    
    cv_comparison = pd.DataFrame({
        'Task': ['Classification', 'Object Detection', 'Segmentation', 'Enhancement', 'Face Recognition'],
        'Complexity': ['Low', 'Medium', 'High', 'Medium', 'Medium'],
        'Output Type': ['Label', 'Boxes+Labels', 'Pixel Mask', 'Enhanced Image', 'Identity'],
        'Speed': ['⚡Fast', '⚡Medium', '🐢Slow', '⚡Medium', '⚡Fast'],
        'Common Use': ['Product ID', 'Surveillance', 'Medical', 'Photo Editing', 'Security']
    })
    
    st.dataframe(cv_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("### 🔬 Applications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Autonomous Vehicles**
        - Pedestrian detection
        - Traffic sign recognition
        - Lane detection
        - Obstacle avoidance
        """)
    
    with col2:
        st.info("""
        **Healthcare**
        - Disease diagnosis
        - Tumor detection
        - X-ray analysis
        - Medical imaging
        """)
    
    with col3:
        st.info("""
        **Retail & E-commerce**
        - Product search
        - Inventory tracking
        - Visual recommendation
        - Quality control
        """)

with tab2:
    st.markdown("## 🎯 Object Detection")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>What is Object Detection?</h3>
        <p><strong>Object Detection</strong> locates and identifies multiple objects in an image.
        It outputs <strong>bounding boxes</strong> (coordinates) and <strong>class labels</strong> for each detected object.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎨 Demo: Object Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📷 Original Image")
        sample_img = create_sample_image(640, 480, objects=True)
        st.image(sample_img, use_container_width=True, caption="Sample scene with objects")
    
    with col2:
        st.markdown("#### 🎯 Detection Results")
        
        # Simulate detections
        detections = [
            {'label': 'car', 'bbox': (192, 336, 292, 396), 'confidence': 0.95},
            {'label': 'person', 'bbox': (380, 336, 404, 396), 'confidence': 0.92},
            {'label': 'tree', 'bbox': (492, 252, 527, 348), 'confidence': 0.88}
        ]
        
        # Draw all detections
        detected_img = sample_img.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        for i, det in enumerate(detections):
            detected_img = draw_detection_box(
                detected_img, 
                det['bbox'], 
                det['label'], 
                det['confidence'],
                colors[i % len(colors)]
            )
        
        st.image(detected_img, use_container_width=True, caption="Detected objects with bounding boxes")
    
    st.markdown("### 📋 Detection Details")
    
    detection_df = pd.DataFrame([
        {
            'Object': det['label'].capitalize(),
            'Confidence': f"{det['confidence']:.1%}",
            'Location (x1,y1,x2,y2)': str(det['bbox']),
            'Area': (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
        }
        for det in detections
    ])
    
    st.dataframe(detection_df, use_container_width=True, hide_index=True)
    
    # Confidence chart
    fig = go.Figure(data=[
        go.Bar(
            x=[d['label'].capitalize() for d in detections],
            y=[d['confidence'] for d in detections],
            marker=dict(
                color=['#11998e', '#38ef7d', '#06beb6'],
                line=dict(color='white', width=2)
            ),
            text=[f"{d['confidence']:.1%}" for d in detections],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Detection Confidence Scores",
        yaxis=dict(title="Confidence", range=[0, 1], tickformat='.0%'),
        xaxis=dict(title="Object"),
        height=350,
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🎯 How Object Detection Works")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>Detection Pipeline</h4>
        <ol>
            <li><strong>Feature Extraction:</strong> CNN extracts visual features</li>
            <li><strong>Region Proposal:</strong> Identify potential object locations</li>
            <li><strong>Classification:</strong> Classify what's in each region</li>
            <li><strong>Box Regression:</strong> Refine bounding box coordinates</li>
            <li><strong>Non-Max Suppression:</strong> Remove duplicate detections</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🏆 Popular Detection Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h4>⚡ YOLO (You Only Look Once)</h4>
            <ul>
                <li><strong>Speed:</strong> Real-time (30-150 FPS)</li>
                <li><strong>Approach:</strong> Single-stage detector</li>
                <li><strong>Versions:</strong> YOLOv3, YOLOv5, YOLOv8</li>
                <li><strong>Use Case:</strong> Video surveillance, autonomous driving</li>
                <li><strong>Advantage:</strong> Extremely fast</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>🎯 Faster R-CNN</h4>
            <ul>
                <li><strong>Speed:</strong> 5-10 FPS</li>
                <li><strong>Approach:</strong> Two-stage detector</li>
                <li><strong>Components:</strong> Region Proposal Network + Classifier</li>
                <li><strong>Use Case:</strong> High-accuracy requirements</li>
                <li><strong>Advantage:</strong> Very accurate</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h4>🤖 DETR (Detection Transformer)</h4>
            <ul>
                <li><strong>Speed:</strong> 10-20 FPS</li>
                <li><strong>Approach:</strong> Transformer-based</li>
                <li><strong>Innovation:</strong> End-to-end, no NMS needed</li>
                <li><strong>Use Case:</strong> Research, complex scenes</li>
                <li><strong>Advantage:</strong> Simple architecture</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>📱 SSD (Single Shot Detector)</h4>
            <ul>
                <li><strong>Speed:</strong> 20-30 FPS</li>
                <li><strong>Approach:</strong> Single-stage with multi-scale</li>
                <li><strong>Features:</strong> Multiple feature maps</li>
                <li><strong>Use Case:</strong> Mobile and embedded devices</li>
                <li><strong>Advantage:</strong> Good speed/accuracy balance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("## 🔍 Image Enhancement & Super Resolution")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>What is Image Enhancement?</h3>
        <p><strong>Image Enhancement</strong> improves visual quality by increasing resolution, removing noise, 
        sharpening details, or improving color/contrast. <strong>Super Resolution</strong> specifically 
        increases image resolution while preserving details.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📸 Demo: Resolution Enhancement")
    
    # Create low-res sample
    base_img = create_sample_image(160, 120, objects=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📉 Low Resolution (160x120)")
        st.image(base_img, use_container_width=True)
        st.metric("Size", "160 x 120 px")
        st.metric("File Size", "~5 KB")
    
    with col2:
        st.markdown("#### 📊 Standard Upscale (320x240)")
        simple_upscale = base_img.resize((320, 240), Image.BILINEAR)
        st.image(simple_upscale, use_container_width=True)
        st.metric("Size", "320 x 240 px")
        st.metric("Method", "Bilinear")
    
    with col3:
        st.markdown("#### 📈 Enhanced (320x240)")
        enhanced_img = enhance_resolution(base_img, scale_factor=2)
        st.image(enhanced_img, use_container_width=True)
        st.metric("Size", "320 x 240 px")
        st.metric("Method", "LANCZOS + Sharpen")
    
    st.markdown("### 🎨 Enhancement Techniques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h4>🔧 Traditional Methods</h4>
            <ul>
                <li><strong>Bicubic Interpolation:</strong> Smooth upscaling</li>
                <li><strong>Lanczos Resampling:</strong> High-quality resize</li>
                <li><strong>Sharpening Filters:</strong> Edge enhancement</li>
                <li><strong>Denoise:</strong> Noise reduction</li>
                <li><strong>Contrast Adjustment:</strong> Improve visibility</li>
                <li><strong>Histogram Equalization:</strong> Balance brightness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h4>🤖 Deep Learning Methods</h4>
            <ul>
                <li><strong>ESRGAN:</strong> Enhanced Super-Resolution GAN</li>
                <li><strong>Real-ESRGAN:</strong> Practical face restoration</li>
                <li><strong>SRGAN:</strong> Photo-realistic upscaling</li>
                <li><strong>SwinIR:</strong> Transformer-based SR</li>
                <li><strong>EDSR:</strong> Enhanced Deep SR</li>
                <li><strong>RCAN:</strong> Residual Channel Attention</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Quality Metrics")
    
    metrics_data = pd.DataFrame({
        'Metric': ['PSNR', 'SSIM', 'LPIPS', 'Perceptual Quality'],
        'Low Res': ['22.5 dB', '0.65', '0.45', 'Poor'],
        'Standard Upscale': ['25.8 dB', '0.78', '0.32', 'Fair'],
        'Enhanced': ['28.4 dB', '0.89', '0.18', 'Good'],
        'Description': [
            'Peak Signal-to-Noise Ratio',
            'Structural Similarity Index',
            'Learned Perceptual Similarity',
            'Human perception'
        ]
    })
    
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)
    
    st.markdown("### 🎯 Applications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Photography**
        - Old photo restoration
        - Low-light enhancement
        - Upscale for printing
        - Face beautification
        """)
    
    with col2:
        st.info("""
        **Medical Imaging**
        - X-ray enhancement
        - MRI super-resolution
        - CT scan improvement
        - Diagnostic quality
        """)
    
    with col3:
        st.info("""
        **Video Processing**
        - 4K/8K upscaling
        - Streaming quality
        - Surveillance footage
        - Video restoration
        """)
    
    st.markdown("### 💻 Code Example")
    
    st.code("""
from PIL import Image, ImageFilter, ImageEnhance

# Load low-resolution image
img = Image.open('low_res.jpg')

# Upscale with high-quality resampling
upscaled = img.resize((img.width*2, img.height*2), Image.LANCZOS)

# Apply sharpening
enhanced = upscaled.filter(ImageFilter.SHARPEN)

# Enhance contrast
enhancer = ImageEnhance.Contrast(enhanced)
final = enhancer.enhance(1.2)

final.save('enhanced.jpg')
""", language='python')

with tab4:
    st.markdown("## 🏗️ Model Architectures")
    
    st.markdown("### 🎯 YOLO Architecture")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>You Only Look Once (YOLO)</h4>
        <p>YOLO treats object detection as a <strong>single regression problem</strong>, 
        going directly from image pixels to bounding box coordinates and class probabilities.</p>
        
        <h5>Key Components:</h5>
        <ul>
            <li><strong>Backbone:</strong> CSPDarknet53 (feature extraction)</li>
            <li><strong>Neck:</strong> PANet (feature pyramid)</li>
            <li><strong>Head:</strong> Detection layers at multiple scales</li>
            <li><strong>Output:</strong> Grid cells with boxes + classes</li>
        </ul>
        
        <h5>Advantages:</h5>
        <ul>
            <li>⚡ Real-time performance (30-150 FPS)</li>
            <li>🎯 Good accuracy for speed</li>
            <li>📦 Single network (end-to-end)</li>
            <li>🌍 Strong contextual understanding</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 DETR Architecture")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>Detection Transformer (DETR)</h4>
        <p>DETR uses <strong>transformers</strong> (like BERT/GPT) for object detection, 
        eliminating the need for hand-crafted components like NMS and anchor boxes.</p>
        
        <h5>Key Components:</h5>
        <ul>
            <li><strong>CNN Backbone:</strong> ResNet-50/101</li>
            <li><strong>Transformer Encoder:</strong> Process image features</li>
            <li><strong>Transformer Decoder:</strong> Generate object queries</li>
            <li><strong>Feed-Forward Networks:</strong> Predict boxes and classes</li>
        </ul>
        
        <h5>Innovations:</h5>
        <ul>
            <li>🔄 Set-based prediction (no NMS needed)</li>
            <li>🎓 Bipartite matching loss</li>
            <li>🎯 Direct set prediction</li>
            <li>📝 Simpler pipeline</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📸 ESRGAN Architecture")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>Enhanced Super-Resolution GAN</h4>
        <p>ESRGAN uses <strong>generative adversarial networks</strong> to produce 
        photo-realistic high-resolution images from low-resolution inputs.</p>
        
        <h5>Key Components:</h5>
        <ul>
            <li><strong>Generator:</strong> Residual-in-Residual Dense Blocks (RRDB)</li>
            <li><strong>Discriminator:</strong> VGG-style network for realism</li>
            <li><strong>Perceptual Loss:</strong> VGG feature matching</li>
            <li><strong>Adversarial Loss:</strong> Relativistic average</li>
        </ul>
        
        <h5>Improvements over SRGAN:</h5>
        <ul>
            <li>🎨 More realistic textures</li>
            <li>🔍 Better perceptual quality</li>
            <li>⚙️ Improved network capacity</li>
            <li>📈 Higher PSNR and SSIM</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Architecture Comparison")
    
    arch_comparison = pd.DataFrame({
        'Model': ['YOLOv8', 'DETR', 'Faster R-CNN', 'ESRGAN'],
        'Task': ['Object Detection', 'Object Detection', 'Object Detection', 'Super Resolution'],
        'Speed (FPS)': ['150', '20', '10', '5-10'],
        'Parameters': ['25M', '41M', '137M', '16M'],
        'Accuracy': ['High', 'Very High', 'Very High', 'N/A'],
        'Best For': ['Real-time', 'Research', 'Accuracy', 'Photo quality']
    })
    
    st.dataframe(arch_comparison, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("## 💡 Key Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 Object Detection Insights</h3>
            <ul>
                <li><strong>Choose by Use Case:</strong> YOLO for speed, R-CNN for accuracy</li>
                <li><strong>Anchor Boxes:</strong> Pre-defined box shapes help detection</li>
                <li><strong>NMS:</strong> Removes overlapping duplicate detections</li>
                <li><strong>Multi-Scale:</strong> Detect objects of different sizes</li>
                <li><strong>Data Augmentation:</strong> Essential for good performance</li>
                <li><strong>IoU Threshold:</strong> Balance precision vs recall</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>📸 Enhancement Best Practices</h3>
            <ul>
                <li><strong>Quality vs Speed:</strong> Deep learning slow but better</li>
                <li><strong>Artifacts:</strong> Watch for over-sharpening</li>
                <li><strong>Realistic Textures:</strong> GANs produce best results</li>
                <li><strong>Upscale Limit:</strong> 4x is practical maximum</li>
                <li><strong>Face Enhancement:</strong> Specialized models work best</li>
                <li><strong>Batch Processing:</strong> More efficient for videos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>⚠️ Common Challenges</h3>
            <ul>
                <li><strong>Small Objects:</strong> Hard to detect accurately</li>
                <li><strong>Occlusion:</strong> Partially hidden objects</li>
                <li><strong>Class Imbalance:</strong> Some classes rare in training</li>
                <li><strong>Domain Shift:</strong> Different from training data</li>
                <li><strong>Computational Cost:</strong> Real-time on edge devices</li>
                <li><strong>Hallucination:</strong> SR may add fake details</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>🚀 Real-World Applications</h3>
            <ul>
                <li><strong>Autonomous Vehicles:</strong> Real-time object detection</li>
                <li><strong>Surveillance:</strong> Threat detection, crowd monitoring</li>
                <li><strong>Retail:</strong> Checkout-free stores, inventory</li>
                <li><strong>Manufacturing:</strong> Defect detection, quality control</li>
                <li><strong>Agriculture:</strong> Crop disease, yield estimation</li>
                <li><strong>Healthcare:</strong> Medical image analysis</li>
                <li><strong>Sports:</strong> Player tracking, performance analysis</li>
                <li><strong>Media:</strong> Content moderation, copyright</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>📊 Performance Metrics</h3>
            <ul>
                <li><strong>mAP:</strong> Mean Average Precision (detection)</li>
                <li><strong>IoU:</strong> Intersection over Union (box overlap)</li>
                <li><strong>FPS:</strong> Frames Per Second (speed)</li>
                <li><strong>PSNR:</strong> Peak Signal-to-Noise Ratio (quality)</li>
                <li><strong>SSIM:</strong> Structural Similarity Index</li>
                <li><strong>LPIPS:</strong> Learned Perceptual Image Patch Similarity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🔮 Future Trends</h3>
            <ul>
                <li><strong>Transformers:</strong> Replacing CNNs everywhere</li>
                <li><strong>Diffusion Models:</strong> New approach to enhancement</li>
                <li><strong>Efficient Models:</strong> Mobile and edge deployment</li>
                <li><strong>Few-Shot Learning:</strong> Less training data needed</li>
                <li><strong>Multi-Modal:</strong> Combining vision with text/audio</li>
                <li><strong>Explainability:</strong> Understanding model decisions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🎓 Key Takeaways")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>✅ What We Learned:</h4>
        <ul>
            <li><strong>Object Detection</strong> identifies and locates multiple objects in images</li>
            <li><strong>YOLO</strong> provides real-time detection, <strong>DETR</strong> uses transformers</li>
            <li><strong>Image Enhancement</strong> improves quality through upscaling and restoration</li>
            <li><strong>Super Resolution</strong> uses GANs to create realistic high-res images</li>
            <li><strong>Trade-offs</strong> exist between speed, accuracy, and quality</li>
            <li><strong>Applications</strong> span from autonomous vehicles to medical imaging</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>👁️ Computer Vision Mastered!</h3>
    <p><strong>Day 12 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>Object Detection & Image Enhancement - Real-World CV</p>
</div>
""", unsafe_allow_html=True)
