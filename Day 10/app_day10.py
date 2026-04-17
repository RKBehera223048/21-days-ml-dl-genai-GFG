import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Day 10: GANs", page_icon="🎨", layout="wide")

st.markdown("""<style>
.main {background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 50%, #ff6b6b 100%);}
.stMetric {background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 15px; color: white;}
.stMetric label {color: #ffffff !important; font-weight: 600;}
h1 {color: #ffffff; font-weight: 700; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);}
h2, h3 {color: #f0f0f0; font-weight: 600;}
.highlight-box {background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; border-left: 6px solid #8e2de2; margin: 10px 0;}
.gan-card {background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%); padding: 25px; border-radius: 20px; 
            text-align: center; color: white; margin: 10px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);}
.stTabs [data-baseweb="tab-list"] {gap: 10px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;}
.stTabs [data-baseweb="tab"] {background: rgba(255,255,255,0.2); color: white; border-radius: 8px; padding: 10px 20px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%);}
</style>""", unsafe_allow_html=True)

def build_generator(latent_dim=100):
    model = keras.Sequential([
        layers.Dense(7*7*256, input_dim=latent_dim),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh')
    ], name='generator')
    return model

def build_discriminator(img_shape=(28,28,1)):
    model = keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=img_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ], name='discriminator')
    return model

@st.cache_data
def load_mnist_data():
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_train = (X_train - 127.5) / 127.5  # Normalize to [-1, 1]
    return X_train

def generate_latent_points(latent_dim, n_samples):
    return np.random.randn(n_samples * latent_dim).reshape(n_samples, latent_dim)

st.markdown("<h1 style='text-align: center;'>🎨 Creative AI with Generative Adversarial Networks</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Day 10: Generating Art & Images from Random Noise</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1561214115-f2f134cc4912?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🎭 Generative Adversarial Networks</li>
            <li>⚔️ Generator vs Discriminator</li>
            <li>🎲 Latent Space Exploration</li>
            <li>🎨 Image Generation</li>
            <li>🔄 Adversarial Training</li>
            <li>📊 Mode Collapse</li>
            <li>🖼️ Neural Style Transfer</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Generation Settings")
    
    num_samples = st.slider("Samples to Generate:", 4, 16, 9)
    latent_dim = st.slider("Latent Dimension:", 50, 200, 100)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 10 of 21</strong></p>
        <p>Creative AI with GANs</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 GAN Basics",
    "🏗️ Architecture",
    "🎯 Training Demo",
    "🎨 Generation",
    "💡 Insights"
])

with tab1:
    st.markdown("## 📖 Understanding GANs")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🎭 What are GANs?</h3>
        <p><strong>Generative Adversarial Networks (GANs)</strong> are a class of machine learning frameworks 
        where two neural networks compete with each other in a game-theoretic scenario.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="gan-card">
            <h3 style='margin:0; color: white;'>🎨 Generator</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p style='margin:10px 0;'><strong>Role:</strong> The Artist</p>
            <p style='margin:10px 0;'><strong>Goal:</strong> Create fake images that look real</p>
            <p style='margin:10px 0;'><strong>Input:</strong> Random noise (latent vector)</p>
            <p style='margin:10px 0;'><strong>Output:</strong> Generated image</p>
            <p style='margin:10px 0;'><strong>Objective:</strong> Fool the discriminator</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ✨ Generator Process")
        st.info("""
        1. **Random Noise:** Start with random numbers (latent vector)
        2. **Upsampling:** Gradually increase spatial dimensions
        3. **Deconvolution:** Conv2DTranspose layers expand features
        4. **Activation:** Tanh for final output (-1 to 1 range)
        5. **Output:** Generated image (e.g., 28x28 digit)
        """)
    
    with col2:
        st.markdown("""
        <div class="gan-card">
            <h3 style='margin:0; color: white;'>🕵️ Discriminator</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <p style='margin:10px 0;'><strong>Role:</strong> The Critic</p>
            <p style='margin:10px 0;'><strong>Goal:</strong> Distinguish real from fake</p>
            <p style='margin:10px 0;'><strong>Input:</strong> Real or generated image</p>
            <p style='margin:10px 0;'><strong>Output:</strong> Probability (real or fake)</p>
            <p style='margin:10px 0;'><strong>Objective:</strong> Catch the generator's fakes</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Discriminator Process")
        st.info("""
        1. **Input Image:** Real or fake image
        2. **Convolution:** Extract features with Conv2D
        3. **Downsampling:** Reduce spatial dimensions
        4. **Classification:** Dense layer with sigmoid
        5. **Output:** Probability (0=fake, 1=real)
        """)
    
    st.markdown("### ⚔️ The Adversarial Game")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>Game Theory in Action</h4>
        <ul>
            <li><strong>Generator tries to maximize:</strong> Probability of fooling discriminator</li>
            <li><strong>Discriminator tries to maximize:</strong> Accuracy in detecting fakes</li>
            <li><strong>Min-Max Game:</strong> Generator minimizes what discriminator maximizes</li>
            <li><strong>Nash Equilibrium:</strong> When generator creates perfect fakes</li>
            <li><strong>Training Process:</strong> Alternating updates - discriminator, then generator</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Training flow visualization
    st.markdown("### 🔄 Training Flow")
    
    flow_data = {
        'Step': ['1. Real Data', '2. Random Noise', '3. Generate Fakes', '4. Train Discriminator', '5. Train Generator', '6. Repeat'],
        'Process': [
            'Load real images from dataset',
            'Sample from latent space',
            'Generator creates fake images',
            'Update D to classify real/fake',
            'Update G to fool D',
            'Continue until convergence'
        ],
        'Goal': [
            'Ground truth examples',
            'Starting point for generation',
            'Create realistic images',
            'Improve detection ability',
            'Improve generation quality',
            'Reach Nash equilibrium'
        ]
    }
    
    st.dataframe(pd.DataFrame(flow_data), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("## 🏗️ GAN Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Generator Architecture")
        
        generator = build_generator(latent_dim)
        
        gen_summary = []
        generator.summary(print_fn=lambda x: gen_summary.append(x))
        
        with st.expander("📋 View Generator Layers"):
            st.code('\n'.join(gen_summary[:30]), language='text')
        
        st.markdown("#### 🔧 Key Components")
        st.markdown("""
        - **Input:** Random noise vector (latent_dim)
        - **Dense Layer:** Projects to 7x7x256
        - **Reshape:** Form initial spatial structure
        - **Conv2DTranspose:** Upsample (deconvolution)
        - **BatchNormalization:** Stabilize training
        - **LeakyReLU:** Allow negative gradients
        - **Output:** 28x28x1 image with tanh
        """)
        
        gen_params = generator.count_params()
        st.metric("Generator Parameters", f"{gen_params:,}")
    
    with col2:
        st.markdown("### 🕵️ Discriminator Architecture")
        
        discriminator = build_discriminator()
        
        disc_summary = []
        discriminator.summary(print_fn=lambda x: disc_summary.append(x))
        
        with st.expander("📋 View Discriminator Layers"):
            st.code('\n'.join(disc_summary[:30]), language='text')
        
        st.markdown("#### 🔧 Key Components")
        st.markdown("""
        - **Input:** Image (28x28x1)
        - **Conv2D:** Feature extraction
        - **LeakyReLU:** Prevent dying ReLU
        - **Dropout:** Regularization
        - **Strided Convolutions:** Downsampling
        - **Flatten:** Convert to 1D
        - **Output:** Sigmoid probability
        """)
        
        disc_params = discriminator.count_params()
        st.metric("Discriminator Parameters", f"{disc_params:,}")
    
    st.markdown("### 📊 Architecture Comparison")
    
    arch_comparison = pd.DataFrame({
        'Component': ['Input', 'Main Operation', 'Layers', 'Parameters', 'Output', 'Activation'],
        'Generator': [
            f'{latent_dim}D noise',
            'Upsampling (Conv2DTranspose)',
            '8 layers',
            f'{gen_params:,}',
            '28x28x1 image',
            'Tanh'
        ],
        'Discriminator': [
            '28x28x1 image',
            'Downsampling (Conv2D)',
            '7 layers',
            f'{disc_params:,}',
            'Probability (0-1)',
            'Sigmoid'
        ]
    })
    
    st.dataframe(arch_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("### 🎯 Loss Functions")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>Binary Cross-Entropy Loss</h4>
        <ul>
            <li><strong>Discriminator Loss:</strong> -[log(D(x)) + log(1 - D(G(z)))]</li>
            <li><strong>Generator Loss:</strong> -log(D(G(z)))</li>
        </ul>
        <p>Where:</p>
        <ul>
            <li>x = real image</li>
            <li>z = random noise</li>
            <li>G(z) = generated image</li>
            <li>D(·) = discriminator output</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("## 🎯 GAN Training Demonstration")
    
    st.markdown("### 📊 MNIST Dataset")
    
    with st.spinner('📥 Loading MNIST dataset...'):
        X_train = load_mnist_data()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Training Images", f"{len(X_train):,}")
    with col2:
        st.metric("Image Size", "28x28")
    with col3:
        st.metric("Channels", "1 (Gray)")
    with col4:
        st.metric("Normalization", "[-1, 1]")
    with col5:
        st.metric("Classes", "0-9 digits")
    
    st.markdown("### 🖼️ Real MNIST Samples")
    
    sample_indices = np.random.choice(len(X_train), 10, replace=False)
    
    fig = make_subplots(rows=2, cols=5, subplot_titles=[f'Sample {i+1}' for i in range(10)])
    
    for idx, sample_idx in enumerate(sample_indices):
        img = X_train[sample_idx].squeeze()
        row = idx // 5 + 1
        col = idx % 5 + 1
        fig.add_trace(go.Heatmap(z=img, colorscale='gray', showscale=False), row=row, col=col)
    
    fig.update_layout(height=400, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🚀 Training Process")
    
    st.info("""
    **GAN Training Steps:**
    
    1. **Load Real Images:** Batch from MNIST dataset
    2. **Generate Fake Images:** Generator creates from random noise
    3. **Train Discriminator:**
       - Train on real images (label=1)
       - Train on fake images (label=0)
       - Update discriminator weights
    4. **Train Generator:**
       - Generate new fakes
       - Try to fool discriminator (want label=1)
       - Update generator weights via discriminator gradients
    5. **Repeat:** Alternate between D and G training
    """)
    
    st.markdown("### 📈 Training Challenges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h4>⚠️ Common Issues</h4>
            <ul>
                <li><strong>Mode Collapse:</strong> Generator produces limited variety</li>
                <li><strong>Training Instability:</strong> Oscillating losses</li>
                <li><strong>Vanishing Gradients:</strong> Generator can't learn</li>
                <li><strong>Discriminator Too Strong:</strong> Perfect classification blocks learning</li>
                <li><strong>Non-Convergence:</strong> Models don't stabilize</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h4>✅ Solutions</h4>
            <ul>
                <li><strong>LeakyReLU:</strong> Prevent dying gradients</li>
                <li><strong>Batch Normalization:</strong> Stabilize training</li>
                <li><strong>Dropout:</strong> Regularize discriminator</li>
                <li><strong>Label Smoothing:</strong> Soften real=1 labels</li>
                <li><strong>Learning Rate Tuning:</strong> Balance G and D</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.warning("""
    **Note:** Training GANs from scratch requires significant time (hours to days). 
    This demonstration shows the architecture and process. Pre-trained models can 
    generate images instantly.
    """)

with tab4:
    st.markdown("## 🎨 Image Generation")
    
    st.markdown("### 🎲 Generate Images from Random Noise")
    
    if st.button("✨ Generate Images", type="primary", use_container_width=True):
        with st.spinner('🎨 Generating images...'):
            # Generate random latent vectors
            noise = generate_latent_points(latent_dim, num_samples)
            
            # Use generator to create images
            generated_images = generator.predict(noise, verbose=0)
            
            # Denormalize from [-1, 1] to [0, 1]
            generated_images = (generated_images + 1) / 2.0
        
        st.success(f"✓ Generated {num_samples} images!")
        
        # Display generated images
        cols_per_row = 4 if num_samples >= 16 else 3
        rows = (num_samples + cols_per_row - 1) // cols_per_row
        
        fig = make_subplots(
            rows=rows, 
            cols=cols_per_row,
            subplot_titles=[f'Generated #{i+1}' for i in range(num_samples)]
        )
        
        for i in range(num_samples):
            img = generated_images[i].squeeze()
            row = i // cols_per_row + 1
            col = i % cols_per_row + 1
            fig.add_trace(go.Heatmap(z=img, colorscale='gray', showscale=False), row=row, col=col)
        
        fig.update_layout(
            height=150 * rows,
            showlegend=False,
            paper_bgcolor='rgba(255,255,255,0.9)',
            title_text=f"Generated Images (Latent Dim: {latent_dim})"
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Note:** These images are generated by an **untrained** generator, so they appear as random noise.
        A trained GAN would produce realistic digit images. The architecture is ready - training would take 
        several hours on MNIST dataset.
        """)
    
    st.markdown("### 🔍 Latent Space Exploration")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>What is Latent Space?</h4>
        <p>The <strong>latent space</strong> is the multi-dimensional space of random inputs to the generator.
        Each point in this space maps to a unique generated image.</p>
        
        <h4>🎯 Key Properties:</h4>
        <ul>
            <li><strong>Dimensionality:</strong> Typically 50-200 dimensions</li>
            <li><strong>Continuity:</strong> Nearby points create similar images</li>
            <li><strong>Interpolation:</strong> Can smoothly transition between images</li>
            <li><strong>Arithmetic:</strong> Vector math can manipulate attributes</li>
            <li><strong>Disentanglement:</strong> Different dimensions control different features</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎭 Style Transfer & Manipulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Vector Arithmetic Examples:**
        - Man with glasses - Man + Woman = Woman with glasses
        - Smiling face - Neutral face + Sad face = Sad face
        - Young - Old + Middle-aged = Different age expression
        """)
    
    with col2:
        st.markdown("""
        **Applications:**
        - Face aging/de-aging
        - Gender transformation
        - Attribute manipulation (smile, hair, glasses)
        - Style transfer between images
        """)

with tab5:
    st.markdown("## 💡 Key Insights & Applications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 GAN Variants & Evolution</h3>
            <ul>
                <li><strong>DCGAN:</strong> Deep Convolutional GAN (this implementation)</li>
                <li><strong>WGAN:</strong> Wasserstein GAN (better training stability)</li>
                <li><strong>StyleGAN:</strong> High-resolution, photorealistic faces</li>
                <li><strong>CycleGAN:</strong> Unpaired image-to-image translation</li>
                <li><strong>Pix2Pix:</strong> Paired image-to-image translation</li>
                <li><strong>BigGAN:</strong> Large-scale, high-fidelity generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>📚 Technical Concepts</h3>
            <ul>
                <li><strong>Adversarial Training:</strong> Two networks compete</li>
                <li><strong>Nash Equilibrium:</strong> Optimal balance point</li>
                <li><strong>Mode Collapse:</strong> Loss of diversity</li>
                <li><strong>Latent Space:</strong> Continuous representation</li>
                <li><strong>Transposed Convolution:</strong> Upsampling operation</li>
                <li><strong>Batch Normalization:</strong> Training stabilization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>🚀 Real-World Applications</h3>
            <ul>
                <li><strong>Art Generation:</strong> Creating unique artworks</li>
                <li><strong>Face Generation:</strong> This Person Does Not Exist</li>
                <li><strong>Data Augmentation:</strong> Synthetic training data</li>
                <li><strong>Super Resolution:</strong> Enhance image quality</li>
                <li><strong>Style Transfer:</strong> Apply artistic styles</li>
                <li><strong>Video Game Assets:</strong> Generate textures, levels</li>
                <li><strong>Fashion:</strong> Virtual try-on, design generation</li>
                <li><strong>Medicine:</strong> Synthetic medical images</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>⚠️ Ethical Considerations</h3>
            <ul>
                <li><strong>Deepfakes:</strong> Potential for misuse</li>
                <li><strong>Authenticity:</strong> Distinguishing real from fake</li>
                <li><strong>Copyright:</strong> Generated content ownership</li>
                <li><strong>Bias:</strong> Training data biases in output</li>
                <li><strong>Misinformation:</strong> Fake news generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 GAN vs Other Generative Models")
    
    comparison = pd.DataFrame({
        'Model': ['GAN', 'VAE', 'Diffusion Models', 'Autoregressive'],
        'Quality': ['⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐'],
        'Training Stability': ['⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐'],
        'Speed': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐', '⭐⭐⭐'],
        'Diversity': ['⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐'],
        'Use Case': ['Faces, Art', 'Compression', 'High Quality', 'Text, Audio']
    })
    
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🎓 Key Takeaways</h3>
        <ul>
            <li><strong>Revolutionary Technique:</strong> GANs opened new possibilities in AI</li>
            <li><strong>Adversarial Learning:</strong> Competition drives improvement</li>
            <li><strong>Latent Space Magic:</strong> Continuous representation enables manipulation</li>
            <li><strong>Training Challenges:</strong> Requires careful tuning and patience</li>
            <li><strong>Endless Applications:</strong> From art to science to entertainment</li>
            <li><strong>Ethical Responsibility:</strong> With great power comes great responsibility</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🎨 Creative AI with GANs Complete!</h3>
    <p><strong>Day 10 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>Generative Adversarial Networks - Creating Art from Noise</p>
</div>
""", unsafe_allow_html=True)
