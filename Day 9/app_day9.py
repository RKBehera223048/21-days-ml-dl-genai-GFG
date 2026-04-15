import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Day 9: Transfer Learning", page_icon="🎨", layout="wide")

st.markdown("""<style>
.main {background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);}
.stMetric {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 15px; color: white;}
.stMetric label {color: #ffffff !important; font-weight: 600;}
h1 {color: #ffffff; font-weight: 700; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);}
h2, h3 {color: #f0f0f0; font-weight: 600;}
.highlight-box {background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; border-left: 6px solid #667eea; margin: 10px 0;}
.model-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 20px; 
             text-align: center; color: white; margin: 10px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);}
.stTabs [data-baseweb="tab-list"] {gap: 10px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;}
.stTabs [data-baseweb="tab"] {background: rgba(255,255,255,0.2); color: white; border-radius: 8px; padding: 10px 20px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
</style>""", unsafe_allow_html=True)

# CIFAR-100 class names
cifar100_fine_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

@st.cache_data
def load_cifar100():
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return (X_train, y_train), (X_test, y_test)

@st.cache_resource
def build_transfer_model(base_model_name, input_shape=(32, 32, 3), num_classes=100):
    if base_model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    else:  # MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

st.markdown("<h1 style='text-align: center;'>🎨 Advanced Vision AI with Transfer Learning</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Day 9: Pre-trained Models for Image Classification</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🔄 Transfer Learning</li>
            <li>🏗️ Pre-trained Models (ResNet, VGG, MobileNet)</li>
            <li>🎯 Feature Extraction</li>
            <li>🔧 Fine-tuning Techniques</li>
            <li>📊 CIFAR-100 Dataset</li>
            <li>⚡ Model Comparison</li>
            <li>🎨 Computer Vision</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    
    model_choice = st.selectbox("Select Pre-trained Model:", 
                                ["ResNet50", "VGG16", "MobileNetV2"])
    
    epochs = st.slider("Training Epochs:", 1, 10, 3)
    batch_size = st.selectbox("Batch Size:", [32, 64, 128], index=1)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 9 of 21</strong></p>
        <p>Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)

# Load data
with st.spinner('📥 Loading CIFAR-100 dataset...'):
    (X_train, y_train), (X_test, y_test) = load_cifar100()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset", 
    "🏗️ Model Architecture", 
    "🎯 Training", 
    "📈 Evaluation",
    "💡 Insights"
])

with tab1:
    st.markdown("## 📊 CIFAR-100 Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Train Images", f"{len(X_train):,}")
    with col2:
        st.metric("Test Images", f"{len(X_test):,}")
    with col3:
        st.metric("Image Size", "32x32x3")
    with col4:
        st.metric("Classes", "100")
    with col5:
        st.metric("Color", "RGB")
    
    st.markdown("### 🖼️ Sample Images from CIFAR-100")
    
    # Show random samples
    num_samples = 20
    sample_indices = np.random.choice(len(X_train), num_samples, replace=False)
    
    fig = make_subplots(
        rows=4, cols=5,
        subplot_titles=[cifar100_fine_labels[y_train[idx]] for idx in sample_indices],
        vertical_spacing=0.12,
        horizontal_spacing=0.05
    )
    
    for i, idx in enumerate(sample_indices):
        row = i // 5 + 1
        col = i % 5 + 1
        img = X_train[idx]
        fig.add_trace(go.Image(z=img), row=row, col=col)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        paper_bgcolor='rgba(255,255,255,0.9)',
        title_text="Random Samples from CIFAR-100"
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_dist = np.bincount(y_train, minlength=100)
        fig = px.bar(
            x=list(range(100)),
            y=train_dist,
            title='Training Set - Samples per Class',
            labels={'x': 'Class ID', 'y': 'Count'},
            color=train_dist,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        test_dist = np.bincount(y_test, minlength=100)
        fig = px.bar(
            x=list(range(100)),
            y=test_dist,
            title='Test Set - Samples per Class',
            labels={'x': 'Class ID', 'y': 'Count'},
            color=test_dist,
            color_continuous_scale='Plasma'
        )
        fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🏷️ Class Categories (Sample)")
    
    categories = {
        'Animals': ['bear', 'bee', 'butterfly', 'camel', 'cattle', 'dolphin', 'elephant', 'fox', 'lion', 'tiger'],
        'Vehicles': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'tractor', 'rocket', 'tank'],
        'Nature': ['cloud', 'forest', 'mountain', 'sea', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree'],
        'Objects': ['bottle', 'chair', 'clock', 'cup', 'lamp', 'table', 'telephone', 'keyboard'],
        'Food': ['apple', 'orange', 'pear', 'mushroom', 'sweet_pepper']
    }
    
    for category, items in categories.items():
        with st.expander(f"📂 {category} ({len(items)} classes shown)"):
            st.write(", ".join(items))

with tab2:
    st.markdown("## 🏗️ Transfer Learning Architecture")
    
    st.markdown(f"### 🎯 Selected Model: **{model_choice}**")
    
    # Model comparison
    model_info = {
        'ResNet50': {
            'params': '25.6M',
            'depth': '50 layers',
            'strength': 'Deep residual connections, excellent for complex features',
            'use_case': 'High accuracy tasks, image classification',
            'speed': '⭐⭐⭐'
        },
        'VGG16': {
            'params': '138M',
            'depth': '16 layers',
            'strength': 'Simple architecture, good feature extraction',
            'use_case': 'Feature extraction, transfer learning',
            'speed': '⭐⭐'
        },
        'MobileNetV2': {
            'params': '3.5M',
            'depth': 'Inverted residuals',
            'strength': 'Lightweight, fast, mobile-friendly',
            'use_case': 'Mobile deployment, edge devices',
            'speed': '⭐⭐⭐⭐⭐'
        }
    }
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (name, info) in enumerate(model_info.items()):
        col = [col1, col2, col3][idx]
        with col:
            is_selected = (name == model_choice)
            card_style = "model-card" if is_selected else "model-card" 
            opacity = "1.0" if is_selected else "0.6"
            
            st.markdown(f"""
            <div class="{card_style}" style="opacity: {opacity};">
                <h3 style='margin:0; color: white;'>{'✓ ' if is_selected else ''}{name}</h3>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <p style='margin:5px 0;'><strong>Parameters:</strong> {info['params']}</p>
                <p style='margin:5px 0;'><strong>Depth:</strong> {info['depth']}</p>
                <p style='margin:5px 0;'><strong>Speed:</strong> {info['speed']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 📐 Architecture Overview")
    
    st.info(f"""
    **Transfer Learning Strategy:**
    
    1. **Base Model:** {model_choice} (pre-trained on ImageNet)
    2. **Feature Extraction:** Frozen base layers (weights not updated)
    3. **Custom Top Layers:**
       - GlobalAveragePooling2D
       - Dense(512, activation='relu') + Dropout(0.5)
       - Dense(256, activation='relu') + Dropout(0.3)
       - Dense(100, activation='softmax') - Output layer
    
    4. **Training:** Only custom layers are trained on CIFAR-100
    """)
    
    # Build model for visualization
    with st.spinner(f'🔨 Building {model_choice} model...'):
        model = build_transfer_model(model_choice)
    
    st.markdown("### 📋 Model Summary")
    
    # Get model summary
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    summary_text = '\n'.join(summary_list)
    
    with st.expander("Click to view full model architecture"):
        st.code(summary_text, language='text')
    
    # Model statistics
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", f"{total_params:,}")
    with col2:
        st.metric("Trainable", f"{trainable_params:,}")
    with col3:
        st.metric("Frozen", f"{non_trainable_params:,}")
    
    st.success(f"✓ Only {(trainable_params/total_params*100):.1f}% of parameters will be trained!")

with tab3:
    st.markdown("## 🎯 Model Training")
    
    st.markdown(f"### 📊 Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model_choice)
    with col2:
        st.metric("Epochs", epochs)
    with col3:
        st.metric("Batch Size", batch_size)
    
    # Preprocess data based on model
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        
        # Preprocess
        with st.spinner(f'⚙️ Preprocessing images for {model_choice}...'):
            if model_choice == "ResNet50":
                X_train_prep = preprocess_resnet(X_train.astype('float32'))
                X_test_prep = preprocess_resnet(X_test.astype('float32'))
            elif model_choice == "VGG16":
                X_train_prep = preprocess_vgg(X_train.astype('float32'))
                X_test_prep = preprocess_vgg(X_test.astype('float32'))
            else:
                X_train_prep = preprocess_mobilenet(X_train.astype('float32'))
                X_test_prep = preprocess_mobilenet(X_test.astype('float32'))
        
        # Use subset for demo (faster training)
        train_size = 5000
        X_train_subset = X_train_prep[:train_size]
        y_train_subset = y_train[:train_size]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training
        with st.spinner(f'🔄 Training {model_choice}...'):
            history = model.fit(
                X_train_subset, y_train_subset,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
        
        progress_bar.progress(100)
        status_text.success(f'✓ Training completed! Final accuracy: {history.history["val_accuracy"][-1]:.4f}')
        
        # Save to session state
        st.session_state['model'] = model
        st.session_state['history'] = history
        st.session_state['X_test_prep'] = X_test_prep
        st.session_state['model_name'] = model_choice
        
        # Plot training history
        st.markdown("### 📈 Training Progress")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history.history['accuracy'],
                mode='lines+markers',
                name='Train Accuracy',
                line=dict(color='#667eea', width=3)
            ))
            fig.add_trace(go.Scatter(
                y=history.history['val_accuracy'],
                mode='lines+markers',
                name='Val Accuracy',
                line=dict(color='#f5576c', width=3)
            ))
            fig.update_layout(
                title='Model Accuracy',
                xaxis_title='Epoch',
                yaxis_title='Accuracy',
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history.history['loss'],
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='#667eea', width=3)
            ))
            fig.add_trace(go.Scatter(
                y=history.history['val_loss'],
                mode='lines+markers',
                name='Val Loss',
                line=dict(color='#f5576c', width=3)
            ))
            fig.update_layout(
                title='Model Loss',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"🎉 Model trained successfully with {model_choice}!")
    
    else:
        st.info("👆 Click the button above to start training the model")
        
        st.markdown("### ⏱️ Expected Training Time")
        st.warning(f"""
        **Note:** Training on full dataset may take considerable time:
        - **ResNet50:** ~15-20 min per epoch
        - **VGG16:** ~20-25 min per epoch  
        - **MobileNetV2:** ~10-15 min per epoch
        
        Demo uses subset (5,000 images) for faster demonstration.
        """)

with tab4:
    st.markdown("## 📈 Model Evaluation")
    
    if 'model' in st.session_state:
        model = st.session_state['model']
        X_test_prep = st.session_state['X_test_prep']
        model_name = st.session_state['model_name']
        
        # Evaluate on test set (use subset for demo)
        test_subset = 1000
        X_test_subset = X_test_prep[:test_subset]
        y_test_subset = y_test[:test_subset]
        
        with st.spinner('📊 Evaluating model on test set...'):
            test_loss, test_acc = model.evaluate(X_test_subset, y_test_subset, verbose=0)
            predictions = model.predict(X_test_subset, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{test_acc:.4f}")
        with col2:
            st.metric("Test Loss", f"{test_loss:.4f}")
        with col3:
            correct = np.sum(pred_classes == y_test_subset)
            st.metric("Correct", f"{correct}/{test_subset}")
        with col4:
            st.metric("Error Rate", f"{((1-test_acc)*100):.2f}%")
        
        st.markdown("### 🎯 Sample Predictions")
        
        if st.button("🔄 Show Random Predictions"):
            num_samples = 10
            random_indices = np.random.choice(test_subset, num_samples, replace=False)
            
            fig = make_subplots(
                rows=2, cols=5,
                subplot_titles=[f'Sample {i+1}' for i in range(num_samples)],
                vertical_spacing=0.15
            )
            
            for idx, test_idx in enumerate(random_indices):
                img = X_test[test_idx]
                true_label = y_test_subset[test_idx]
                pred_label = pred_classes[test_idx]
                confidence = predictions[test_idx][pred_label] * 100
                
                row = idx // 5 + 1
                col = idx % 5 + 1
                
                fig.add_trace(go.Image(z=img), row=row, col=col)
                
                is_correct = (pred_label == true_label)
                color = 'green' if is_correct else 'red'
                
                fig.layout.annotations[idx].text = (
                    f"True: {cifar100_fine_labels[true_label]}<br>"
                    f"Pred: {cifar100_fine_labels[pred_label]}<br>"
                    f"{confidence:.1f}%"
                )
                fig.layout.annotations[idx].font.color = color
                fig.layout.annotations[idx].font.size = 9
            
            fig.update_layout(
                height=500,
                showlegend=False,
                paper_bgcolor='rgba(255,255,255,0.9)',
                title_text=f"Random Predictions - {model_name}"
            )
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Top-5 Accuracy")
        
        top5_correct = 0
        for i in range(len(pred_classes)):
            top5_preds = np.argsort(predictions[i])[-5:]
            if y_test_subset[i] in top5_preds:
                top5_correct += 1
        
        top5_accuracy = top5_correct / len(pred_classes)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top-1 Accuracy", f"{test_acc:.4f}")
        with col2:
            st.metric("Top-5 Accuracy", f"{top5_accuracy:.4f}")
        
        st.info("""
        **Top-5 Accuracy:** Measures if the correct class is in the top 5 predictions.
        This is often higher than top-1 accuracy and useful for recommendation systems.
        """)
        
    else:
        st.warning("⚠️ Please train the model first in the 'Training' tab!")

with tab5:
    st.markdown("## 💡 Key Insights & Learnings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 Transfer Learning Benefits</h3>
            <ul>
                <li><strong>Pre-trained Knowledge:</strong> Models trained on ImageNet (1.2M images)</li>
                <li><strong>Faster Training:</strong> Only train top layers, not entire network</li>
                <li><strong>Better Performance:</strong> Leverages learned features from large datasets</li>
                <li><strong>Less Data Needed:</strong> Works well even with smaller datasets</li>
                <li><strong>Resource Efficient:</strong> Saves compute time and energy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🏗️ Model Comparison</h3>
            <ul>
                <li><strong>ResNet50:</strong> Best accuracy, moderate speed, 25.6M params</li>
                <li><strong>VGG16:</strong> Good features, slower, largest (138M params)</li>
                <li><strong>MobileNetV2:</strong> Fastest, mobile-friendly, smallest (3.5M)</li>
                <li><strong>Choice Depends On:</strong> Accuracy vs speed vs deployment target</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>📚 Technical Concepts Learned</h3>
            <ul>
                <li><strong>Transfer Learning:</strong> Using pre-trained models for new tasks</li>
                <li><strong>Feature Extraction:</strong> Freezing base layers, training top</li>
                <li><strong>Fine-tuning:</strong> Gradually unfreezing layers for better fit</li>
                <li><strong>Global Average Pooling:</strong> Better than Flatten for CNNs</li>
                <li><strong>Dropout:</strong> Regularization to prevent overfitting</li>
                <li><strong>Data Preprocessing:</strong> Model-specific normalization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🚀 Real-World Applications</h3>
            <ul>
                <li><strong>Medical Imaging:</strong> Disease detection from X-rays/MRIs</li>
                <li><strong>Autonomous Vehicles:</strong> Object detection and recognition</li>
                <li><strong>Retail:</strong> Product classification and search</li>
                <li><strong>Security:</strong> Face recognition, surveillance</li>
                <li><strong>Agriculture:</strong> Crop disease identification</li>
                <li><strong>Wildlife:</strong> Species identification from camera traps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 When to Use Transfer Learning")
    
    use_cases = pd.DataFrame({
        'Scenario': [
            'Small Dataset (<10K images)',
            'Medium Dataset (10K-100K)',
            'Large Dataset (>100K)',
            'Limited Compute',
            'Mobile Deployment',
            'High Accuracy Required'
        ],
        'Recommendation': [
            'Always use transfer learning',
            'Transfer learning highly recommended',
            'Can train from scratch or use transfer learning',
            'Use MobileNetV2 with transfer learning',
            'MobileNetV2 or EfficientNet',
            'ResNet50 or deeper variants'
        ],
        'Benefit': [
            'Prevents overfitting',
            'Faster convergence',
            'Competitive performance',
            'Fast training',
            'Lightweight inference',
            'State-of-the-art results'
        ]
    })
    
    st.dataframe(use_cases, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🎓 Key Takeaways</h3>
        <ul>
            <li><strong>Don't Reinvent the Wheel:</strong> Use pre-trained models when possible</li>
            <li><strong>Start Simple:</strong> Begin with feature extraction before fine-tuning</li>
            <li><strong>Monitor Overfitting:</strong> Use dropout and validation sets</li>
            <li><strong>Choose Wisely:</strong> Match model to your constraints (speed/accuracy/size)</li>
            <li><strong>Preprocess Correctly:</strong> Use model-specific preprocessing functions</li>
            <li><strong>Experiment:</strong> Try different models to find best fit</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🎨 Advanced Vision AI Complete!</h3>
    <p><strong>Day 9 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>Transfer Learning with Pre-trained Models on CIFAR-100</p>
</div>
""", unsafe_allow_html=True)
