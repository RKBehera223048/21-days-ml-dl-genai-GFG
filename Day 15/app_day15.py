import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import re

# Page configuration
st.set_page_config(
    page_title="Day 15: NL to SQL Generator",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with database blue gradient theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
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
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
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
        border-left: 5px solid #4facfe;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .sql-box {
        background: #2d3748;
        color: #68d391;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4facfe;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1544383835-bda2bc66a55d?w=800&h=400&fit=crop", 
             use_container_width=True)
    
    st.markdown("### 💬 Day 15: NL to SQL Generator")
    st.markdown("**Talk to Your Data with Natural Language**")
    
    st.markdown("---")
    st.markdown("### 🎯 What I Learned")
    st.markdown("""
    - 💬 Natural Language Understanding
    - 🔄 Text to SQL conversion
    - 🧠 Query intent recognition
    - 📊 Database schema analysis
    - 🎯 Entity & relationship extraction
    - 🔍 SQL query generation
    - ✅ Query validation & optimization
    - 🤖 LLM integration for NL2SQL
    """)
    
    st.markdown("---")
    st.info("**Day 15 of 21** - GeeksforGeeks ML & GenAI Course")

# Load data
@st.cache_data
def load_employee_data():
    try:
        df = pd.read_csv(r"Employees.csv")
    except:
        try:
            df = pd.read_csv(r"../Day 15/Employees.csv")
        except:
            # Create sample data
            np.random.seed(42)
            df = pd.DataFrame({
                'employee_id': range(1, 101),
                'first_name': [f'Employee{i}' for i in range(1, 101)],
                'last_name': [f'Last{i}' for i in range(1, 101)],
                'salary': np.random.randint(30000, 150000, 100),
                'hire_date': pd.date_range('2010-01-01', periods=100, freq='30D'),
                'department_id': np.random.randint(1, 6, 100),
                'department': np.random.choice(['IT', 'Sales', 'Marketing', 'Finance', 'HR'], 100),
                'residence_city': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Berlin'], 100),
                'age': np.random.randint(22, 65, 100),
                'job_level': np.random.choice(['Entry Level', 'Mid Level', 'Senior', 'Executive'], 100)
            })
    return df

# Create SQLite database
@st.cache_resource
def create_database():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    df = load_employee_data()
    df.to_sql('employees', conn, index=False, if_exists='replace')
    return conn

# Natural Language to SQL mapping
def parse_natural_language(query):
    """Convert natural language to SQL query"""
    query_lower = query.lower()
    
    # Pattern matching for common queries
    patterns = {
        r'(average|avg|mean) salary': {
            'sql': 'SELECT AVG(salary) as average_salary FROM employees',
            'description': 'Calculate average salary of all employees'
        },
        r'total (number of )?employees': {
            'sql': 'SELECT COUNT(*) as total_employees FROM employees',
            'description': 'Count total number of employees'
        },
        r'highest (paid|salary)': {
            'sql': 'SELECT * FROM employees ORDER BY salary DESC LIMIT 5',
            'description': 'Find highest paid employees'
        },
        r'lowest (paid|salary)': {
            'sql': 'SELECT * FROM employees ORDER BY salary ASC LIMIT 5',
            'description': 'Find lowest paid employees'
        },
        r'department.*average salary|average salary.*department': {
            'sql': 'SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC',
            'description': 'Average salary by department'
        },
        r'employees in (\w+)': {
            'sql': 'SELECT * FROM employees WHERE department LIKE "%{match}%" LIMIT 20',
            'description': 'Find employees in specific department',
            'extract': True
        },
        r'salary (greater|more) than (\d+)': {
            'sql': 'SELECT * FROM employees WHERE salary > {match} LIMIT 20',
            'description': 'Employees with salary greater than specified amount',
            'extract': True
        },
        r'youngest employees': {
            'sql': 'SELECT * FROM employees ORDER BY age ASC LIMIT 10',
            'description': 'Find youngest employees'
        },
        r'oldest employees': {
            'sql': 'SELECT * FROM employees ORDER BY age DESC LIMIT 10',
            'description': 'Find oldest employees'
        },
        r'(executive|senior|mid level|entry level)': {
            'sql': 'SELECT * FROM employees WHERE job_level LIKE "%{match}%" LIMIT 20',
            'description': 'Filter by job level',
            'extract': True
        },
        r'all departments': {
            'sql': 'SELECT DISTINCT department FROM employees',
            'description': 'List all unique departments'
        },
        r'department count': {
            'sql': 'SELECT department, COUNT(*) as employee_count FROM employees GROUP BY department ORDER BY employee_count DESC',
            'description': 'Count employees by department'
        },
    }
    
    for pattern, info in patterns.items():
        match = re.search(pattern, query_lower)
        if match:
            sql = info['sql']
            if info.get('extract') and match.groups():
                # Extract the matched value
                value = match.group(len(match.groups()))
                sql = sql.format(match=value)
            return sql, info['description']
    
    # Default query if no pattern matches
    return None, "Sorry, I couldn't understand your query. Try asking about salaries, departments, or employee counts."

# Main app
st.title("💬 Natural Language to SQL Generator")
st.markdown("### Talk to Your Data - No SQL Knowledge Required!")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 NL2SQL Demo", 
    "📊 Database Explorer", 
    "🧠 How It Works",
    "🔧 Architecture",
    "💡 Insights"
])

with tab1:
    st.header("🎯 Natural Language to SQL Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### Ask Questions About Your Data")
        
        # Example questions
        st.markdown("**Try these example questions:**")
        examples = [
            "What is the average salary?",
            "Show me highest paid employees",
            "How many employees in IT department?",
            "Average salary by department",
            "Show employees with salary greater than 100000",
            "Who are the youngest employees?",
            "List all departments"
        ]
        
        example_cols = st.columns(3)
        selected_example = None
        for i, example in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(example, key=f"ex_{i}", use_container_width=True):
                    selected_example = example
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Query input
        user_query = st.text_area(
            "Enter your question in plain English:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="e.g., What is the average salary of employees in IT department?"
        )
        
        if st.button("🔍 Generate SQL & Execute", type="primary", use_container_width=True):
            if user_query:
                with st.spinner("Parsing your question..."):
                    sql_query, description = parse_natural_language(user_query)
                    
                    if sql_query:
                        st.success(f"✅ **Understood:** {description}")
                        
                        # Display generated SQL
                        st.markdown("**Generated SQL Query:**")
                        st.markdown(f'<div class="sql-box">{sql_query}</div>', unsafe_allow_html=True)
                        
                        # Execute query
                        try:
                            conn = create_database()
                            result_df = pd.read_sql_query(sql_query, conn)
                            
                            st.markdown("**Query Results:**")
                            st.dataframe(result_df, use_container_width=True, height=400)
                            
                            # Download button
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results as CSV",
                                csv,
                                "query_results.csv",
                                "text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
                    else:
                        st.warning(description)
                        st.info("💡 **Tip:** Try rephrasing your question or use one of the example queries above.")
            else:
                st.warning("Please enter a question!")
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 📈 Quick Stats")
        
        df = load_employee_data()
        
        st.metric("Total Employees", f"{len(df):,}")
        st.metric("Avg Salary", f"${df['salary'].mean():,.2f}")
        st.metric("Departments", df['department'].nunique())
        st.metric("Avg Age", f"{df['age'].mean():.1f} years")
        
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("📊 Database Explorer")
    
    df = load_employee_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Salary Range", f"${df['salary'].min():,.0f} - ${df['salary'].max():,.0f}")
    with col4:
        st.metric("Age Range", f"{df['age'].min()} - {df['age'].max()}")
    
    st.markdown("---")
    
    # Schema information
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🗄️ Database Schema")
        st.markdown("**Table: employees**")
        st.markdown("""
        - `employee_id` (INTEGER)
        - `first_name` (TEXT)
        - `last_name` (TEXT)
        - `salary` (REAL)
        - `hire_date` (DATE)
        - `department_id` (INTEGER)
        - `department` (TEXT)
        - `residence_city` (TEXT)
        - `age` (INTEGER)
        - `job_level` (TEXT)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Visualizations
        st.markdown("#### 📊 Data Distribution")
        
        # Salary by department
        fig1 = px.box(df, x='department', y='salary', color='department',
                      title='Salary Distribution by Department')
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    # Sample data
    st.markdown("#### 📋 Sample Data")
    st.dataframe(df.head(20), use_container_width=True, height=400)

with tab3:
    st.header("🧠 How Natural Language to SQL Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 NL2SQL Pipeline")
        st.markdown("""
        **1. Natural Language Input** 💬
        - User asks question in plain English
        - "What is the average salary by department?"
        
        **2. Intent Recognition** 🎯
        - Identify query type (SELECT, aggregate, filter)
        - Extract entities (table, columns, conditions)
        - Recognize aggregation functions
        
        **3. Entity Extraction** 📝
        - Tables: employees, departments
        - Columns: salary, department, age
        - Operators: AVG, SUM, COUNT, MAX, MIN
        - Conditions: WHERE, GROUP BY, ORDER BY
        
        **4. SQL Generation** 🔧
        - Map entities to SQL syntax
        - Construct proper query structure
        - Add necessary JOINs and clauses
        
        **5. Query Validation** ✅
        - Check syntax correctness
        - Verify table/column existence
        - Validate data types
        
        **6. Execution** ▶️
        - Run query on database
        - Return formatted results
        - Handle errors gracefully
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🧩 Key Components")
        
        st.markdown("**1. Tokenization**")
        st.code("""
# Break sentence into tokens
"average salary" → ["average", "salary"]
→ Recognize: AVG function + salary column
        """)
        
        st.markdown("**2. Pattern Matching**")
        st.code("""
# Regular expressions for common patterns
r'average salary' → SELECT AVG(salary)
r'employees in IT' → WHERE department='IT'
r'salary > 100000' → WHERE salary > 100000
        """)
        
        st.markdown("**3. Schema Mapping**")
        st.code("""
# Map natural language to schema
"department" → employees.department
"salary" → employees.salary
"count employees" → COUNT(*)
        """)
        
        st.markdown("**4. LLM Integration**")
        st.code("""
# Using LLMs for complex queries
prompt = f'''
Convert to SQL:
Question: {user_question}
Schema: {database_schema}
SQL:
'''
sql = llm.generate(prompt)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Example transformations
    st.markdown("---")
    st.markdown("#### 📚 Example Transformations")
    
    examples_data = {
        'Natural Language': [
            'What is the average salary?',
            'Show highest paid employees',
            'Count employees in IT',
            'Employees older than 50',
            'Salary by department'
        ],
        'Generated SQL': [
            'SELECT AVG(salary) FROM employees',
            'SELECT * FROM employees ORDER BY salary DESC LIMIT 5',
            'SELECT COUNT(*) FROM employees WHERE department="IT"',
            'SELECT * FROM employees WHERE age > 50',
            'SELECT department, AVG(salary) FROM employees GROUP BY department'
        ],
        'Query Type': [
            'Aggregation',
            'Sorting',
            'Filtering + Aggregation',
            'Filtering',
            'Grouping + Aggregation'
        ]
    }
    
    examples_df = pd.DataFrame(examples_data)
    st.dataframe(examples_df, use_container_width=True, height=250)

with tab4:
    st.header("🔧 NL2SQL Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🏗️ System Architecture")
        
        st.markdown("""
        ```
        ┌─────────────────────────────────┐
        │   User Natural Language Input   │
        └───────────────┬─────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │    Text Preprocessing Layer     │
        │  - Tokenization                 │
        │  - Normalization                │
        │  - Stop word removal            │
        └───────────────┬─────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │    Intent Classification        │
        │  - Query type detection         │
        │  - Entity recognition (NER)     │
        │  - Relation extraction          │
        └───────────────┬─────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │     Schema Mapping Layer        │
        │  - Table identification         │
        │  - Column mapping               │
        │  - Relationship inference       │
        └───────────────┬─────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │    SQL Generation Engine        │
        │  - Syntax construction          │
        │  - JOIN logic                   │
        │  - WHERE clause building        │
        └───────────────┬─────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │    Query Validator              │
        │  - Syntax check                 │
        │  - Security validation          │
        │  - Performance optimization     │
        └───────────────┬─────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │    Database Execution           │
        │  - Query execution              │
        │  - Result formatting            │
        │  - Error handling               │
        └───────────────┬─────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │   Natural Language Response     │
        └─────────────────────────────────┘
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🤖 LLM-Based Approach")
        
        st.markdown("**Modern NL2SQL using Large Language Models:**")
        
        st.code("""
# Using OpenAI GPT for NL2SQL
import openai

def nl_to_sql_with_llm(question, schema):
    prompt = f'''
    You are a SQL expert. Convert natural language 
    to SQL query.
    
    Database Schema:
    {schema}
    
    Question: {question}
    
    Generate SQL query:
    '''
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a SQL expert"},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Example usage
schema = '''
Table: employees
Columns: id, name, salary, department, age
'''

question = "What is the average salary by department?"
sql = nl_to_sql_with_llm(question, schema)
# Output: SELECT department, AVG(salary) 
#         FROM employees 
#         GROUP BY department
        """, language='python')
        
        st.markdown("**Alternative: Fine-tuned Models**")
        st.code("""
# Using specialized NL2SQL models
from transformers import pipeline

# Load pre-trained NL2SQL model
nl2sql = pipeline(
    "text2text-generation",
    model="t5-base-finetuned-sql"
)

# Convert to SQL
result = nl2sql(
    "Show me employees earning more than 100k"
)
# Output: SELECT * FROM employees WHERE salary > 100000
        """, language='python')
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tools and frameworks
    st.markdown("---")
    st.markdown("#### 🛠️ Popular NL2SQL Tools & Frameworks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔷 LangChain SQL**")
        st.markdown("Framework for building NL2SQL with LLMs")
        st.markdown("✅ Easy integration")
        st.markdown("✅ Multiple DB support")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔷 SQLCoder**")
        st.markdown("Open-source NL2SQL model")
        st.markdown("✅ 7B/15B parameters")
        st.markdown("✅ High accuracy")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔷 Vanna AI**")
        st.markdown("AI-powered SQL generation")
        st.markdown("✅ Self-learning")
        st.markdown("✅ Context-aware")
        st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.header("💡 Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ✅ Benefits of NL2SQL")
        st.markdown("""
        **1. Accessibility** 🌍
        - Non-technical users can query databases
        - No SQL knowledge required
        - Democratizes data access
        
        **2. Speed** ⚡
        - Faster than writing SQL manually
        - Instant query generation
        - Reduced development time
        
        **3. Accuracy** 🎯
        - Fewer syntax errors
        - Consistent query patterns
        - LLMs understand context
        
        **4. Business Impact** 💼
        - Self-service analytics
        - Faster decision making
        - Reduced dependency on data teams
        
        **5. Scalability** 📈
        - Works with any database schema
        - Handles complex queries
        - Learns from user patterns
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Use Cases")
        st.markdown("""
        - **Business Intelligence**: Ad-hoc reporting
        - **Customer Support**: Query customer data
        - **Data Science**: Exploratory analysis
        - **E-commerce**: Product queries
        - **Healthcare**: Patient record search
        - **Finance**: Transaction analysis
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Challenges & Limitations")
        st.markdown("""
        **1. Ambiguity** 🤔
        - Natural language can be vague
        - Multiple interpretations possible
        - Context dependencies
        
        **2. Complex Queries** 🧩
        - Nested subqueries are hard
        - Multiple JOIN operations
        - Advanced SQL features
        
        **3. Schema Understanding** 📚
        - Requires knowledge of database structure
        - Table relationships must be clear
        - Column naming conventions matter
        
        **4. Security Concerns** 🔒
        - SQL injection risks
        - Unauthorized data access
        - Query validation needed
        
        **5. Performance** ⚡
        - Generated queries may be inefficient
        - No query optimization
        - Index usage not guaranteed
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔐 Security Best Practices")
        st.markdown("""
        1. **Input Validation**: Sanitize all inputs
        2. **Parameterized Queries**: Use prepared statements
        3. **Access Control**: Role-based permissions
        4. **Query Whitelisting**: Allow only safe operations
        5. **Rate Limiting**: Prevent abuse
        6. **Audit Logging**: Track all queries
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("---")
    st.markdown("#### 📊 NL2SQL Performance Comparison")
    
    comparison_data = {
        'Approach': ['Rule-Based', 'ML Models (LSTM)', 'Transformer (BERT)', 'GPT-3.5', 'GPT-4', 'Specialized (SQLCoder)'],
        'Accuracy': [60, 70, 78, 85, 92, 88],
        'Speed (ms)': [50, 200, 300, 1000, 2000, 500],
        'Cost': ['Free', 'Free', 'Free', 'Low', 'High', 'Free'],
        'Complexity': ['Simple', 'Medium', 'Medium', 'All', 'All', 'High']
    }
    
    comp_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(comp_df, x='Approach', y='Accuracy', 
                 title='NL2SQL Accuracy by Approach',
                 color='Accuracy',
                 color_continuous_scale='Blues')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(comp_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3>🎓 Day 15 Complete: Natural Language to SQL Generator</h3>
    <p>Making databases accessible to everyone through conversational AI!</p>
</div>
""", unsafe_allow_html=True)
