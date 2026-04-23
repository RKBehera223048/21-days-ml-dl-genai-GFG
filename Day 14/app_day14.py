import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Day 14: Build Your GPT", page_icon="🤖", layout="wide")

st.markdown("""<style>
.main {background: linear-gradient(135deg, #134e5e 0%, #71b280 50%, #4a00e0 100%);}
.stMetric {background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); padding: 20px; border-radius: 15px; color: white;}
.stMetric label {color: #ffffff !important; font-weight: 600;}
h1 {color: #ffffff; font-weight: 700; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);}
h2, h3 {color: #f0f0f0; font-weight: 600;}
.highlight-box {background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; border-left: 6px solid #134e5e; margin: 10px 0;}
.gpt-card {background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); padding: 25px; border-radius: 20px; 
            text-align: center; color: white; margin: 10px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);}
.stTabs [data-baseweb="tab-list"] {gap: 10px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;}
.stTabs [data-baseweb="tab"] {background: rgba(255,255,255,0.2); color: white; border-radius: 8px; padding: 10px 20px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);}
.chat-box {background: rgba(255,255,255,0.98); padding: 20px; border-radius: 15px; margin: 10px 0; border-left: 4px solid #134e5e;}
.code-box {background: #1e293b; color: #e2e8f0; padding: 15px; border-radius: 10px; font-family: monospace; margin: 10px 0;}
</style>""", unsafe_allow_html=True)

def is_python_question(text):
    """Check if question is related to Python coding"""
    python_keywords = [
        'python', 'code', 'function', 'class', 'def', 'import', 'list', 
        'dict', 'loop', 'if', 'else', 'for', 'while', 'variable',
        'syntax', 'error', 'debug', 'pandas', 'numpy', 'lambda',
        'exception', 'module', 'package', 'pip', 'script'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in python_keywords)

def generate_code_response(prompt):
    """Simulate GPT-2 response for Python questions"""
    responses = {
        'function': """Here's a simple Python function example:

```python
def greet(name):
    '''
    Greets a person by name.
    
    Args:
        name (str): The person's name
        
    Returns:
        str: A greeting message
    '''
    return f"Hello, {name}!"

# Usage
message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

Key points about Python functions:
- Use 'def' keyword to define
- Include docstrings for documentation
- Return values with 'return' statement
- Can have default parameters""",
        
        'list': """Python lists are versatile data structures:

```python
# Creating lists
my_list = [1, 2, 3, 4, 5]
mixed_list = [1, "hello", 3.14, True]

# Common operations
my_list.append(6)        # Add to end
my_list.insert(0, 0)     # Insert at position
my_list.remove(3)        # Remove value
popped = my_list.pop()   # Remove and return last

# List comprehension
squares = [x**2 for x in range(5)]
# Output: [0, 1, 4, 9, 16]

# Slicing
subset = my_list[1:4]    # Elements 1-3
reversed_list = my_list[::-1]  # Reverse
```""",
        
        'class': """Object-oriented programming in Python:

```python
class Person:
    def __init__(self, name, age):
        '''Initialize a Person object'''
        self.name = name
        self.age = age
    
    def greet(self):
        '''Return a greeting'''
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    def birthday(self):
        '''Increment age by 1'''
        self.age += 1
        return f"Happy birthday! Now {self.age} years old"

# Usage
person = Person("Alice", 25)
print(person.greet())
person.birthday()
```

Classes provide:
- Encapsulation of data and methods
- Code reusability through inheritance
- Better organization""",
        
        'loop': """Python loops for iteration:

```python
# For loop
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# Iterate over list
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit)

# Enumerate for index + value
for idx, fruit in enumerate(fruits):
    print(f"{idx}: {fruit}")

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# List comprehension (faster)
squares = [x**2 for x in range(5)]
```""",
        
        'default': """Here's a general Python coding example:

```python
# Python basics
def process_data(data, multiplier=2):
    '''
    Process a list of numbers
    
    Args:
        data (list): List of numbers
        multiplier (int): Value to multiply by
        
    Returns:
        list: Processed numbers
    '''
    result = []
    for num in data:
        processed = num * multiplier
        result.append(processed)
    return result

# Usage
numbers = [1, 2, 3, 4, 5]
output = process_data(numbers, multiplier=3)
print(output)  # [3, 6, 9, 12, 15]

# More Pythonic way
output = [x * 3 for x in numbers]
```

Remember:
- Python is readable and concise
- Use meaningful variable names
- Follow PEP 8 style guide
- Document your code"""
    }
    
    prompt_lower = prompt.lower()
    for key in responses:
        if key in prompt_lower:
            return responses[key]
    
    return responses['default']

st.markdown("<h1 style='text-align: center;'>🤖 Build Your Own GPT</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Day 14: LLM Fine-tuning & Custom AI Assistants</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4 style='color: #ffffff; margin-bottom: 10px;'>📚 What I Learned</h4>
        <ul style='color: #f0f0f0;'>
            <li>🤖 GPT Architecture</li>
            <li>🎯 Transformer Models</li>
            <li>🔧 Model Fine-tuning</li>
            <li>⚡ Quantization</li>
            <li>💬 Text Generation</li>
            <li>🎓 Transfer Learning</li>
            <li>📝 Prompt Engineering</li>
            <li>🚀 LLM Deployment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    
    temperature = st.slider("Temperature:", 0.0, 2.0, 0.7, 0.1, 
                           help="Higher = more creative, Lower = more focused")
    max_length = st.slider("Max Length:", 50, 500, 200, 50,
                          help="Maximum tokens to generate")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #f0f0f0;'>
        <p><strong>Day 14 of 21</strong></p>
        <p>Build Your Own GPT</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 GPT Basics",
    "💬 Code Assistant",
    "🏗️ Architecture",
    "🔧 Fine-tuning",
    "💡 Insights"
])

with tab1:
    st.markdown("## 🤖 Understanding GPT")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>What is GPT?</h3>
        <p><strong>GPT (Generative Pre-trained Transformer)</strong> is a language model that can 
        understand and generate human-like text. It's trained on vast amounts of text data and can 
        perform various language tasks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="gpt-card">
            <h3>📚 GPT Family</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ul style='text-align: left; padding-left: 20px;'>
                <li><strong>GPT-1:</strong> 117M parameters (2018)</li>
                <li><strong>GPT-2:</strong> 1.5B parameters (2019)</li>
                <li><strong>GPT-3:</strong> 175B parameters (2020)</li>
                <li><strong>GPT-3.5:</strong> ChatGPT base (2022)</li>
                <li><strong>GPT-4:</strong> Multimodal (2023)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>🎯 Key Capabilities</h4>
            <ul>
                <li><strong>Text Generation:</strong> Write stories, code, emails</li>
                <li><strong>Question Answering:</strong> Answer based on context</li>
                <li><strong>Summarization:</strong> Condense long texts</li>
                <li><strong>Translation:</strong> Between languages</li>
                <li><strong>Code Generation:</strong> Write programming code</li>
                <li><strong>Reasoning:</strong> Solve logical problems</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="gpt-card">
            <h3>⚙️ How GPT Works</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ol style='text-align: left; padding-left: 20px;'>
                <li><strong>Input:</strong> Text prompt/question</li>
                <li><strong>Tokenization:</strong> Split into tokens</li>
                <li><strong>Embedding:</strong> Convert to vectors</li>
                <li><strong>Transformer:</strong> Process with attention</li>
                <li><strong>Prediction:</strong> Generate next token</li>
                <li><strong>Output:</strong> Complete response</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>🔑 Key Components</h4>
            <ul>
                <li><strong>Tokenizer:</strong> Text → numbers</li>
                <li><strong>Embedding Layer:</strong> Dense representations</li>
                <li><strong>Transformer Blocks:</strong> Self-attention layers</li>
                <li><strong>Position Encoding:</strong> Sequence order</li>
                <li><strong>Language Head:</strong> Final prediction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Model Comparison")
    
    model_comparison = pd.DataFrame({
        'Model': ['GPT-2 Small', 'GPT-2 Medium', 'GPT-2 Large', 'GPT-2 XL', 'GPT-3'],
        'Parameters': ['117M', '345M', '762M', '1.5B', '175B'],
        'Layers': [12, 24, 36, 48, 96],
        'Hidden Size': [768, 1024, 1280, 1600, 12288],
        'Speed': ['⚡Fast', '⚡Medium', '🐢Slow', '🐢Very Slow', '🐢Extremely Slow'],
        'Quality': ['Good', 'Better', 'Great', 'Excellent', 'Outstanding']
    })
    
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("### 🎯 GPT vs Other Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **BERT**
        - Bidirectional encoding
        - Understanding tasks
        - Classification, NER
        - Can't generate text
        """)
    
    with col2:
        st.success("""
        **GPT**
        - Unidirectional (left-to-right)
        - Generation tasks
        - Creative writing
        - Can generate text
        """)
    
    with col3:
        st.info("""
        **T5**
        - Encoder-decoder
        - Text-to-text format
        - Versatile tasks
        - Unified framework
        """)

with tab2:
    st.markdown("## 💬 Python Code Assistant")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🤖 Custom GPT for Python Coding</h3>
        <p>This assistant is fine-tuned to answer <strong>only Python coding questions</strong>.
        It will filter out non-coding queries and provide code examples and explanations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💭 Ask Your Python Question")
    
    user_prompt = st.text_area(
        "Enter your Python coding question:",
        "How do I create a function in Python?",
        height=100,
        help="Ask anything about Python programming!"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        generate_button = st.button("🚀 Generate Response", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if generate_button and user_prompt:
        # Check if Python-related
        if is_python_question(user_prompt):
            with st.spinner('🤖 Generating response...'):
                import time
                time.sleep(1)  # Simulate processing
                
                response = generate_code_response(user_prompt)
            
            st.markdown("### 🤖 Assistant Response:")
            
            st.markdown(f"""
            <div class="chat-box">
                {response}
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✓ Response generated successfully!")
            
        else:
            st.error("""
            ⚠️ **Non-Coding Question Detected**
            
            I'm a specialized Python coding assistant. I can only answer questions related to:
            - Python programming
            - Code examples
            - Syntax and functions
            - Debugging
            - Libraries and packages
            
            Please ask a Python-related question!
            """)
    
    st.markdown("### 📝 Example Questions to Try")
    
    example_questions = [
        "How do I create a function in Python?",
        "What is a list comprehension?",
        "How to define a class in Python?",
        "Explain Python loops",
        "How to handle exceptions in Python?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"💡 {question}", key=f"example_{i}"):
                st.info(f"Try this: {question}")
    
    st.markdown("### 🎯 How the Filtering Works")
    
    st.code("""
def is_python_question(text):
    '''Check if question is related to Python coding'''
    python_keywords = [
        'python', 'code', 'function', 'class', 'def', 
        'import', 'list', 'dict', 'loop', 'if', 'else'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in python_keywords)

# Usage
if is_python_question(user_prompt):
    response = model.generate(user_prompt)
else:
    response = "I can only answer Python coding questions"
""", language='python')

with tab3:
    st.markdown("## 🏗️ GPT Architecture")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🧠 Transformer Architecture</h3>
        <p>GPT is built on the <strong>Transformer architecture</strong>, which uses self-attention 
        mechanisms to process sequences efficiently and capture long-range dependencies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="gpt-card">
            <h3>📥 Input Processing</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ol style='text-align: left; padding-left: 20px;'>
                <li><strong>Tokenization:</strong> "Hello world" → [15496, 995]</li>
                <li><strong>Embedding:</strong> Tokens → 768-dim vectors</li>
                <li><strong>Position Encoding:</strong> Add position info</li>
                <li><strong>Input Vector:</strong> Ready for transformer</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>🔍 Self-Attention</h4>
            <p>The key innovation in Transformers:</p>
            <ul>
                <li><strong>Query (Q):</strong> What am I looking for?</li>
                <li><strong>Key (K):</strong> What do I contain?</li>
                <li><strong>Value (V):</strong> What do I actually have?</li>
            </ul>
            <p><code>Attention(Q,K,V) = softmax(QK^T/√d)V</code></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="gpt-card">
            <h3>🔄 Multi-Head Attention</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ul style='text-align: left; padding-left: 20px;'>
                <li>Multiple attention heads (8-16)</li>
                <li>Each head learns different patterns</li>
                <li>Concatenate all heads</li>
                <li>Linear projection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="gpt-card">
            <h3>⚙️ Transformer Block</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ol style='text-align: left; padding-left: 20px;'>
                <li><strong>Multi-Head Attention</strong></li>
                <li><strong>Add & Normalize</strong></li>
                <li><strong>Feed-Forward Network</strong></li>
                <li><strong>Add & Normalize</strong></li>
            </ol>
            <p style='text-align: left;'>Repeat this block N times (12-96)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>📊 Feed-Forward Network</h4>
            <pre style='background: #1e293b; color: #e2e8f0; padding: 10px; border-radius: 5px;'>
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

Where:
- W₁: [d_model, d_ff]
- W₂: [d_ff, d_model]
- d_ff = 4 × d_model
            </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="gpt-card">
            <h3>📤 Output Generation</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <ol style='text-align: left; padding-left: 20px;'>
                <li><strong>Final Layer:</strong> Last transformer output</li>
                <li><strong>Language Head:</strong> Linear layer</li>
                <li><strong>Softmax:</strong> Probability distribution</li>
                <li><strong>Sample:</strong> Pick next token</li>
                <li><strong>Repeat:</strong> Until end token</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 💻 Code Implementation")
    
    st.code("""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model
model_name = "gpt2"  # or "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Generate text
def generate_text(prompt, max_length=100):
    # Encode input
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode output
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Usage
prompt = "How to create a Python function:"
response = generate_text(prompt)
print(response)
""", language='python')
    
    st.markdown("### 📊 Architecture Parameters")
    
    arch_params = pd.DataFrame({
        'Component': ['Embedding Dim', 'Num Layers', 'Num Heads', 'FFN Dim', 'Vocab Size', 'Context Length'],
        'GPT-2 Small': ['768', '12', '12', '3072', '50257', '1024'],
        'GPT-2 Medium': ['1024', '24', '16', '4096', '50257', '1024'],
        'GPT-2 Large': ['1280', '36', '20', '5120', '50257', '1024'],
        'GPT-2 XL': ['1600', '48', '25', '6400', '50257', '1024']
    })
    
    st.dataframe(arch_params, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("## 🔧 Fine-tuning & Quantization")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🎯 What is Fine-tuning?</h3>
        <p><strong>Fine-tuning</strong> adapts a pre-trained model to your specific task by training 
        on domain-specific data. It's faster and requires less data than training from scratch.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h4>📚 Fine-tuning Process</h4>
            <ol>
                <li><strong>Load Pre-trained Model:</strong> GPT-2, GPT-3.5, etc.</li>
                <li><strong>Prepare Dataset:</strong> Task-specific examples</li>
                <li><strong>Format Data:</strong> Prompt + completion pairs</li>
                <li><strong>Train:</strong> Small learning rate, few epochs</li>
                <li><strong>Validate:</strong> Monitor performance</li>
                <li><strong>Save:</strong> Fine-tuned model</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>🎓 Transfer Learning Benefits</h4>
            <ul>
                <li><strong>Less Data:</strong> Few thousand examples</li>
                <li><strong>Faster Training:</strong> Hours instead of weeks</li>
                <li><strong>Better Performance:</strong> Leverage pre-training</li>
                <li><strong>Lower Cost:</strong> Less compute needed</li>
                <li><strong>Domain Adaptation:</strong> Specialize for your task</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h4>⚡ Quantization</h4>
            <p><strong>Quantization</strong> reduces model size and speeds up inference by using 
            lower precision (e.g., int8 instead of float32).</p>
            
            <h5>Precision Levels:</h5>
            <ul>
                <li><strong>FP32:</strong> Full precision (32-bit)</li>
                <li><strong>FP16:</strong> Half precision (16-bit)</li>
                <li><strong>INT8:</strong> 8-bit integers</li>
                <li><strong>INT4:</strong> 4-bit (extreme compression)</li>
            </ul>
            
            <h5>Benefits:</h5>
            <ul>
                <li>4x smaller model size (FP32 → INT8)</li>
                <li>2-4x faster inference</li>
                <li>Less memory usage</li>
                <li>Edge device deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 💻 Fine-tuning Code")
    
    st.code("""
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
dataset = load_dataset("your_dataset")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    learning_rate=5e-5,
    save_steps=1000,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train!
trainer.train()

# Save fine-tuned model
model.save_pretrained("./my_finetuned_gpt2")
tokenizer.save_pretrained("./my_finetuned_gpt2")
""", language='python')
    
    st.markdown("### 📊 Fine-tuning vs Training from Scratch")
    
    comparison_df = pd.DataFrame({
        'Aspect': ['Training Time', 'Data Required', 'Compute Cost', 'Quality', 'Use Case'],
        'From Scratch': ['Weeks/Months', 'Millions', '$$$$$', 'Excellent (if enough data)', 'Unique domain'],
        'Fine-tuning': ['Hours/Days', 'Thousands', '$', 'Very Good', 'Specific task'],
        'Prompt Engineering': ['Minutes', 'None', '$', 'Good', 'Quick solution']
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("## 💡 Key Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 When to Use GPT</h3>
            <ul>
                <li><strong>Text Generation:</strong> Creative writing, code</li>
                <li><strong>Chatbots:</strong> Conversational AI</li>
                <li><strong>Content Creation:</strong> Articles, emails</li>
                <li><strong>Code Assistance:</strong> GitHub Copilot-style</li>
                <li><strong>Summarization:</strong> Condense documents</li>
                <li><strong>Translation:</strong> Language conversion</li>
                <li><strong>Q&A Systems:</strong> Knowledge retrieval</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>⚙️ Prompt Engineering Tips</h3>
            <ul>
                <li><strong>Be Specific:</strong> Clear instructions</li>
                <li><strong>Provide Context:</strong> Background information</li>
                <li><strong>Use Examples:</strong> Few-shot learning</li>
                <li><strong>Set Tone:</strong> Formal, casual, technical</li>
                <li><strong>Iterate:</strong> Refine based on output</li>
                <li><strong>Control Length:</strong> Specify desired length</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🚀 Deployment Options</h3>
            <ul>
                <li><strong>Cloud APIs:</strong> OpenAI, Azure, AWS</li>
                <li><strong>Self-hosted:</strong> HuggingFace Inference</li>
                <li><strong>Edge Devices:</strong> Quantized models</li>
                <li><strong>On-Premise:</strong> Private deployment</li>
                <li><strong>Serverless:</strong> AWS Lambda, Cloud Functions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>⚠️ Limitations & Challenges</h3>
            <ul>
                <li><strong>Hallucinations:</strong> May generate false information</li>
                <li><strong>Bias:</strong> Inherits training data biases</li>
                <li><strong>Context Length:</strong> Limited memory (1024-32k tokens)</li>
                <li><strong>Computational Cost:</strong> Expensive to run</li>
                <li><strong>No Real-time Knowledge:</strong> Training cutoff date</li>
                <li><strong>Inconsistency:</strong> Different outputs for same input</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>💰 Cost Optimization</h3>
            <ul>
                <li><strong>Use Smaller Models:</strong> GPT-2 vs GPT-3</li>
                <li><strong>Quantization:</strong> Reduce precision</li>
                <li><strong>Caching:</strong> Store common responses</li>
                <li><strong>Batching:</strong> Process multiple requests</li>
                <li><strong>Prompt Compression:</strong> Reduce input tokens</li>
                <li><strong>Local Hosting:</strong> Avoid API costs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <h3>🔮 Future Trends</h3>
            <ul>
                <li><strong>Multimodal:</strong> Text + images + audio</li>
                <li><strong>Longer Context:</strong> 100k+ tokens</li>
                <li><strong>Smaller Models:</strong> Efficient architectures</li>
                <li><strong>Specialization:</strong> Domain-specific LLMs</li>
                <li><strong>Tool Use:</strong> API calling, browsing</li>
                <li><strong>Reasoning:</strong> Better logical thinking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Model Selection Guide")
    
    selection_guide = pd.DataFrame({
        'Use Case': ['Quick Prototyping', 'Production Chatbot', 'Code Generation', 'Creative Writing', 'Resource Constrained'],
        'Recommended Model': ['GPT-2', 'GPT-3.5', 'Codex/GPT-4', 'GPT-3/4', 'Quantized GPT-2'],
        'Reason': [
            'Fast, free, easy to deploy',
            'Good balance of quality and cost',
            'Specialized for code',
            'Best creative capabilities',
            'Small size, fast inference'
        ]
    })
    
    st.dataframe(selection_guide, use_container_width=True, hide_index=True)
    
    st.markdown("### 🎓 Key Takeaways")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>✅ What We Learned:</h4>
        <ul>
            <li><strong>GPT</strong> uses Transformer architecture with self-attention</li>
            <li><strong>Fine-tuning</strong> adapts pre-trained models to specific tasks</li>
            <li><strong>Quantization</strong> reduces model size for deployment</li>
            <li><strong>Prompt engineering</strong> is crucial for good results</li>
            <li><strong>Trade-offs</strong> exist between model size, quality, and cost</li>
            <li><strong>Custom AI assistants</strong> can be built by filtering and fine-tuning</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🤖 Build Your Own GPT Complete!</h3>
    <p><strong>Day 14 of 21 Projects</strong> | GeeksforGeeks ML, Deep Learning & GenAI Course</p>
    <p>LLM Fine-tuning & Custom AI Assistants - The Future of AI</p>
</div>
""", unsafe_allow_html=True)
