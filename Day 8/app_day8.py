import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Day 8: Fashion MNIST", page_icon="👕", layout="wide")

st.markdown("""<style>
.main {background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ff6b6b 100%);}
.stMetric {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white;}
h1 {color: #ffffff; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);}
h2, h3 {color: #f0f0f0;}
.highlight-box {background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; border-left: 6px solid #f5576c;}
</style>""", unsafe_allow_html=True)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@st.cache_data
def load_fashion_mnist():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

@st.cache_resource
def build_ann():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def build_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

st.markdown("<h1 style='text-align: center;'>👕 Fashion MNIST Classification</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Day 8: Vision AI - Building a Digit Recognizer from Scratch</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1441984904996-e0b6ba687e04?w=400", use_container_width=True)
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
        <h4 style='color: #ffffff;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🧠 Neural Networks (ANN)</li>
            <li>🖼️ Convolutional Neural Networks (CNN)</li>
            <li>📊 Image classification</li>
            <li>🎯 Model architecture design</li>
            <li>📈 Training & validation</li>
            <li>🔍 Model evaluation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    model_type = st.selectbox("Select Model:", ["ANN (Dense)", "CNN (Convolutional)"])
    epochs = st.slider("Training Epochs:", 1, 10, 5)
    batch_size = st.slider("Batch Size:", 32, 256, 128)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #f0f0f0;'><strong>Day 8 of 21</strong></p>", unsafe_allow_html=True)

(train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🧠 Model Training", "📈 Evaluation", "🔮 Predictions"])

with tab1:
    st.markdown("## 📊 Fashion MNIST Dataset")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Train Images", f"{len(train_images):,}")
    with col2:
        st.metric("Test Images", f"{len(test_images):,}")
    with col3:
        st.metric("Image Size", "28x28")
    with col4:
        st.metric("Classes", "10")
    with col5:
        st.metric("Grayscale", "0-255")
    
    st.markdown("### 👕 Sample Images from Each Class")
    
    fig = make_subplots(rows=2, cols=5, subplot_titles=class_names)
    
    for i in range(10):
        idx = np.where(train_labels == i)[0][0]
        row = i // 5 + 1
        col = i % 5 + 1
        fig.add_trace(go.Heatmap(z=train_images[idx], colorscale='gray', showscale=False), row=row, col=col)
    
    fig.update_layout(height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_dist = np.bincount(train_labels)
        fig = px.bar(x=class_names, y=train_dist, title='Training Set Distribution',
                    labels={'x': 'Class', 'y': 'Count'}, color=train_dist, color_continuous_scale='Blues')
        fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        test_dist = np.bincount(test_labels)
        fig = px.bar(x=class_names, y=test_dist, title='Test Set Distribution',
                    labels={'x': 'Class', 'y': 'Count'}, color=test_dist, color_continuous_scale='Reds')
        fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## 🧠 Model Architecture & Training")
    
    if model_type == "ANN (Dense)":
        model = build_ann()
        train_data = train_images
        test_data = test_images
    else:
        model = build_cnn()
        train_data = train_images.reshape(-1, 28, 28, 1)
        test_data = test_images.reshape(-1, 28, 28, 1)
    
    st.markdown(f"### 🏗️ {model_type} Architecture")
    
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.code('\n'.join(model_summary))
    
    st.markdown(f"### 📈 Training {model_type}")
    
    if st.button("🚀 Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner('Training model...'):
            history = model.fit(
                train_data, train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            progress_bar.progress(100)
            status_text.success(f'✓ Training completed! Final accuracy: {history.history["val_accuracy"][-1]:.4f}')
        
        st.session_state['model'] = model
        st.session_state['history'] = history
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines+markers', name='Train Accuracy', line=dict(color='#667eea', width=3)))
            fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines+markers', name='Val Accuracy', line=dict(color='#f5576c', width=3)))
            fig.update_layout(title='Model Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy',
                            paper_bgcolor='rgba(255,255,255,0.9)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines+markers', name='Train Loss', line=dict(color='#667eea', width=3)))
            fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines+markers', name='Val Loss', line=dict(color='#f5576c', width=3)))
            fig.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss',
                            paper_bgcolor='rgba(255,255,255,0.9)', height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## 📈 Model Evaluation")
    
    if 'model' in st.session_state:
        model = st.session_state['model']
        test_data_eval = test_images if model_type == "ANN (Dense)" else test_images.reshape(-1, 28, 28, 1)
        
        test_loss, test_acc = model.evaluate(test_data_eval, test_labels, verbose=0)
        predictions = model.predict(test_data_eval, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{test_acc:.4f}")
        with col2:
            st.metric("Test Loss", f"{test_loss:.4f}")
        with col3:
            correct = np.sum(pred_classes == test_labels)
            st.metric("Correct Predictions", f"{correct:,}")
        with col4:
            st.metric("Incorrect", f"{len(test_labels) - correct:,}")
        
        st.markdown("### 🔥 Confusion Matrix")
        
        cm = confusion_matrix(test_labels, pred_classes)
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=class_names, y=class_names,
                       text_auto=True, color_continuous_scale='Reds')
        fig.update_layout(title='Confusion Matrix', paper_bgcolor='rgba(255,255,255,0.9)', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Per-Class Accuracy")
        
        class_correct = []
        for i in range(10):
            mask = test_labels == i
            class_acc = np.mean(pred_classes[mask] == test_labels[mask])
            class_correct.append(class_acc * 100)
        
        fig = px.bar(x=class_names, y=class_correct, title='Accuracy by Class',
                    labels={'x': 'Class', 'y': 'Accuracy (%)'}, color=class_correct,
                    color_continuous_scale='Viridis')
        fig.update_layout(paper_bgcolor='rgba(255,255,255,0.9)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please train the model first in the 'Model Training' tab!")

with tab4:
    st.markdown("## 🔮 Make Predictions")
    
    if 'model' in st.session_state:
        model = st.session_state['model']
        
        st.markdown("### 🎲 Random Test Samples")
        
        if st.button("🔄 Show Random Predictions"):
            fig = make_subplots(rows=2, cols=5, subplot_titles=[f'Sample {i+1}' for i in range(10)])
            
            random_indices = np.random.choice(len(test_images), 10, replace=False)
            
            for idx, test_idx in enumerate(random_indices):
                img = test_images[test_idx]
                true_label = test_labels[test_idx]
                
                test_img = img if model_type == "ANN (Dense)" else img.reshape(1, 28, 28, 1)
                if model_type == "ANN (Dense)":
                    test_img = test_img.reshape(1, 28, 28)
                
                pred = model.predict(test_img, verbose=0)
                pred_class = np.argmax(pred)
                confidence = np.max(pred) * 100
                
                row = idx // 5 + 1
                col = idx % 5 + 1
                
                fig.add_trace(go.Heatmap(z=img, colorscale='gray', showscale=False), row=row, col=col)
                
                color = 'green' if pred_class == true_label else 'red'
                fig.layout.annotations[idx].text = f"True: {class_names[true_label]}<br>Pred: {class_names[pred_class]}<br>{confidence:.1f}%"
                fig.layout.annotations[idx].font.color = color
            
            fig.update_layout(height=600, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)')
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### ✏️ Draw Your Own (Coming Soon)")
        st.info("Feature to draw and classify your own fashion items will be added!")
    else:
        st.warning("⚠️ Please train the model first!")

st.markdown("---")
st.markdown("<div style='text-align: center; color: white;'><h3>👕 Fashion MNIST Classification Complete!</h3><p>Day 8 of 21 | Deep Learning with CNNs</p></div>", unsafe_allow_html=True)
