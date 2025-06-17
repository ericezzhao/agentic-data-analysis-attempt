"""
Streamlit Data Analysis UI - Enhanced for Task 5.2

A modern, user-friendly interface for the Agentic AI Data Analysis system.
Enhanced with real-time feedback, smart suggestions, and advanced UX features.
"""

import asyncio
import base64
import io
import os
import tempfile
import time
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st
from pathlib import Path

# Import our improved agent
import sys
sys.path.append('.')
from agent.improved_data_analysis_agent import create_improved_data_analysis_agent

# Import Task 5.3 export and visualization components
from ui.export_manager import ExportManager, InteractiveVisualizer

# Page configuration
st.set_page_config(
    page_title="Agentic AI Data Analysis - Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations and better UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-in;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in;
    }
    .query-box {
        background: linear-gradient(135deg, #e8f4fd, #f0f8ff);
        padding: 1.2rem;
        border-left: 4px solid #1f77b4;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-box {
        background: linear-gradient(135deg, #f0f8f0, #f8fff8);
        padding: 1.2rem;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #fff5f5, #ffefef);
        padding: 1.2rem;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-processing {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "dataset_loaded" not in st.session_state:
        st.session_state.dataset_loaded = False
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = "idle"
    if "suggested_queries" not in st.session_state:
        st.session_state.suggested_queries = []
    if "last_query_time" not in st.session_state:
        st.session_state.last_query_time = None
    # Task 5.3 additions
    if "export_manager" not in st.session_state:
        st.session_state.export_manager = ExportManager()
    if "interactive_visualizer" not in st.session_state:
        st.session_state.interactive_visualizer = InteractiveVisualizer()
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_insights" not in st.session_state:
        st.session_state.last_insights = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""


def display_header():
    """Display the enhanced application header."""
    st.markdown('<h1 class="main-header">🚀 Agentic AI Data Analysis - Enhanced</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced AI-powered data analysis with real-time insights</p>', 
        unsafe_allow_html=True
    )


def display_agent_status():
    """Display current agent status with visual indicators."""
    status = st.session_state.agent_status
    
    if status == "idle":
        st.markdown(
            '<div class="status-indicator">🟢 Agent Ready - Upload data or ask questions</div>',
            unsafe_allow_html=True
        )
    elif status == "processing":
        st.markdown(
            '<div class="status-indicator status-processing">🟡 Processing - Analyzing your query...</div>',
            unsafe_allow_html=True
        )
    elif status == "loading":
        st.markdown(
            '<div class="status-indicator status-processing">🔄 Loading - Processing dataset...</div>',
            unsafe_allow_html=True
        )
    elif status == "error":
        st.markdown(
            '<div class="status-indicator status-error">🔴 Error - Check details below</div>',
            unsafe_allow_html=True
        )
    elif status == "success":
        st.markdown(
            '<div class="status-indicator status-success">✅ Success - Query completed</div>',
            unsafe_allow_html=True
        )


def setup_sidebar():
    """Setup enhanced sidebar with configuration options."""
    with st.sidebar:
        st.header("🔧 Enhanced Configuration")
        
        # API Key input with validation
        openai_key = st.text_input("OpenAI API Key:", type="password", key="openai_key")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success("✅ API key configured!")
        else:
            st.warning("⚠️ Please enter your OpenAI API key to proceed.")
            return False
        
        st.divider()
        
        # Enhanced settings
        st.subheader("🎯 Query Settings")
        max_results = st.slider("Max Results:", 1, 50, 10, key="max_results")
        enable_debug = st.checkbox("Debug Mode", key="enable_debug")
        
        st.divider()
        
        # Enhanced dataset info
        st.subheader("📊 Dataset Status")
        if st.session_state.dataset_loaded:
            st.success(f"✅ **{st.session_state.dataset_name}**")
            if st.session_state.current_df is not None:
                df = st.session_state.current_df
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Rows", f"{len(df):,}")
                    st.metric("🔢 Numeric", len(df.select_dtypes(include=['number']).columns))
                with col2:
                    st.metric("📋 Columns", len(df.columns))
                    st.metric("📝 Text", len(df.select_dtypes(include=['object']).columns))
        else:
            st.info("📂 No dataset loaded")
        
        # Performance metrics
        if st.session_state.query_history:
            st.divider()
            st.subheader("📈 Performance")
            successful_queries = sum(1 for entry in st.session_state.query_history 
                                   if entry['result'].get('success', False))
            success_rate = (successful_queries / len(st.session_state.query_history)) * 100
            
            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.metric("Total Queries", len(st.session_state.query_history))
            
            if st.session_state.last_query_time:
                st.metric("Last Query", 
                         f"{(time.time() - st.session_state.last_query_time):.0f}s ago")
        
        return True


def generate_smart_suggestions(df: pd.DataFrame) -> List[str]:
    """Generate smart query suggestions based on dataset characteristics."""
    suggestions = []
    
    # Analyze dataset characteristics
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Statistical suggestions
    if len(numeric_cols) > 0:
        first_numeric = numeric_cols[0]
        suggestions.extend([
            f"What's the average {first_numeric}?",
            f"Show me the range of {first_numeric}",
            f"Create a histogram of {first_numeric}"
        ])
    
    # Categorical analysis suggestions
    if len(categorical_cols) > 0:
        first_categorical = categorical_cols[0]
        suggestions.extend([
            f"How many unique {first_categorical} are there?",
            f"Show distribution of {first_categorical}"
        ])
    
    # Advanced analysis suggestions
    if len(numeric_cols) >= 2:
        suggestions.append(f"Correlation between {numeric_cols[0]} and {numeric_cols[1]}")
    
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append(f"Average {numeric_cols[0]} by {categorical_cols[0]}")
    
    # Business intelligence suggestions
    suggestions.extend([
        "Show me summary statistics",
        "What are the key insights?",
        "Find outliers in the data"
    ])
    
    return suggestions[:8]  # Limit to 8 suggestions


def display_smart_suggestions():
    """Display smart query suggestions as clickable buttons."""
    if not st.session_state.dataset_loaded or st.session_state.current_df is None:
        return
    
    if not st.session_state.suggested_queries:
        st.session_state.suggested_queries = generate_smart_suggestions(st.session_state.current_df)
    
    st.subheader("💡 Smart Suggestions")
    st.markdown("Click on any suggestion to run the query:")
    
    # Create suggestion buttons in columns
    cols = st.columns(2)
    for i, suggestion in enumerate(st.session_state.suggested_queries):
        col = cols[i % 2]
        with col:
            if st.button(f"📝 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                st.session_state.current_query = suggestion
                st.rerun()


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to temporary location."""
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


async def load_dataset(file_path: str, file_name: str) -> bool:
    """Enhanced dataset loading with progress updates."""
    try:
        st.session_state.agent_status = "loading"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize agent
        status_text.text("🚀 Initializing AI agent...")
        progress_bar.progress(25)
        
        if st.session_state.agent is None:
            st.session_state.agent = create_improved_data_analysis_agent(
                session_id="enhanced_streamlit_session",
                debug_mode=st.session_state.get("enable_debug", False)
            )
        
        # Load DataFrame for preview
        status_text.text("📊 Loading dataset...")
        progress_bar.progress(50)
        
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            df = None
        
        if df is not None:
            st.session_state.current_df = df
            st.session_state.suggested_queries = generate_smart_suggestions(df)
        
        # Process with agent
        status_text.text("🧠 Processing with AI agent...")
        progress_bar.progress(75)
        
        result = await st.session_state.agent.analyze_dataset(
            file_path=file_path,
            file_name=file_name,
            description=f"Enhanced dataset uploaded via Streamlit: {file_name}"
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Dataset loading complete!")
        
        if result.get("success", False):
            st.session_state.dataset_loaded = True
            st.session_state.dataset_name = file_name
            st.session_state.agent_status = "success"
            
            # Clear progress indicators
            await asyncio.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return True
        else:
            st.session_state.agent_status = "error"
            st.error(f"Failed to load dataset: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        st.session_state.agent_status = "error"
        st.error(f"Error loading dataset: {e}")
        return False


def display_file_upload():
    """Enhanced file upload interface."""
    st.subheader("📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, Excel (.xlsx, .xls). Maximum size: 100MB"
    )
    
    if uploaded_file is not None:
        # Enhanced file details
        file_size_mb = uploaded_file.size / (1024*1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Filename", uploaded_file.name)
        with col2:
            st.metric("📊 Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("📋 Type", uploaded_file.type.split('/')[-1].upper())
        
        # File validation
        if file_size_mb > 100:
            st.error("⚠️ File too large! Please upload files smaller than 100MB.")
            return
        
        # Enhanced load button
        if st.button("🚀 Load Dataset", type="primary", use_container_width=True):
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                success = asyncio.run(load_dataset(file_path, uploaded_file.name))
                if success:
                    st.success("🎉 Dataset loaded successfully!")
                    st.balloons()  # Celebration effect
                    st.rerun()
                # Clean up temp file
                try:
                    os.unlink(file_path)
                except:
                    pass


def display_dataset_preview():
    """Enhanced dataset preview with interactive features."""
    if not st.session_state.dataset_loaded or st.session_state.current_df is None:
        return
    
    df = st.session_state.current_df
    
    st.subheader("📊 Dataset Overview")
    
    # Enhanced statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        st.metric("🔢 Numeric", numeric_cols)
    with col4:
        text_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("📝 Text", text_cols)
    
    # Interactive tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Column Info", "🧠 AI Insights"])
    
    with tab1:
        num_rows = st.slider("Rows to display:", 5, min(20, len(df)), 10, key="preview_rows")
        st.dataframe(df.head(num_rows), use_container_width=True)
        
        if len(df) > num_rows:
            st.info(f"Showing {num_rows} of {len(df):,} total rows")
    
    with tab2:
        # Enhanced column analysis
        col_info = []
        for col in df.columns:
            col_data = {
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-null": f"{df[col].count():,}",
                "Null": f"{df[col].isnull().sum():,}",
                "Unique": f"{df[col].nunique():,}"
            }
            col_info.append(col_data)
        
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    with tab3:
        # AI-powered insights
        st.markdown("### Dataset Analysis")
        insights = [
            f"📊 Your dataset contains {len(df):,} records across {len(df.columns)} variables",
            f"🔢 Found {len(df.select_dtypes(include=['number']).columns)} numeric columns for statistical analysis",
            f"📝 Found {len(df.select_dtypes(include=['object']).columns)} text columns for grouping and filtering",
        ]
        
        # Add data quality insights
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if null_percentage > 10:
            insights.append(f"⚠️ Data has {null_percentage:.1f}% missing values - consider data cleaning")
        else:
            insights.append(f"✅ Good data quality - only {null_percentage:.1f}% missing values")
        
        for insight in insights:
            st.write(f"• {insight}")


async def process_query(query: str) -> Dict[str, Any]:
    """Enhanced query processing with status updates."""
    if not st.session_state.agent or not st.session_state.dataset_loaded:
        return {"success": False, "message": "No dataset loaded or agent not initialized"}
    
    try:
        st.session_state.agent_status = "processing"
        st.session_state.last_query_time = time.time()
        
        result = await st.session_state.agent.query_data(query, st.session_state.dataset_name)
        
        st.session_state.agent_status = "success" if result.get("success") else "error"
        return result
    except Exception as e:
        st.session_state.agent_status = "error"
        return {"success": False, "message": f"Error processing query: {str(e)}"}


def display_query_interface():
    """Enhanced natural language query interface."""
    if not st.session_state.dataset_loaded:
        st.info("👆 Please upload and load a dataset first to start querying.")
        return
    
    st.subheader("💬 Ask Questions About Your Data")
    
    # Display smart suggestions
    display_smart_suggestions()
    
    st.markdown("---")
    
    # Enhanced query input
    query = st.text_area(
        "Enter your question:",
        height=120,
        placeholder="Try asking:\n• What is the average salary by department?\n• Show me employees in New York\n• Create a visualization of the data\n• Who has the highest performance?",
        key="current_query"
    )
    
    # Enhanced query buttons
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        if st.button("🔍 Analyze", type="primary", disabled=not query.strip()):
            process_and_display_query(query)
    with col2:
        if st.button("🔄 Clear"):
            st.session_state.current_query = ""
            st.rerun()
    with col3:
        if st.button("💡 Refresh Suggestions"):
            st.session_state.suggested_queries = generate_smart_suggestions(st.session_state.current_df)
            st.rerun()


def process_and_display_query(query: str):
    """Enhanced query processing with real-time feedback."""
    if not query.strip():
        return
    
    # Show enhanced processing status
    with st.spinner("🧠 AI is analyzing your query..."):
        result = asyncio.run(process_query(query))
    
    # Add to history
    st.session_state.query_history.append({
        "query": query,
        "timestamp": pd.Timestamp.now(),
        "result": result
    })
    
    # Display enhanced results
    display_query_results(query, result)


def display_query_results(query: str, result: Dict[str, Any]):
    """Display enhanced query results."""
    st.markdown(f'<div class="query-box"><strong>🔍 Query:</strong> {query}</div>', unsafe_allow_html=True)
    
    if result.get("success", False):
        # Success results with enhanced formatting
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        
        # Show insights
        insights = result.get("insights", [])
        if insights:
            st.markdown("**💡 Key Insights:**")
            for i, insight in enumerate(insights, 1):
                st.markdown(f"{i}. {insight}")
            st.markdown("---")
        
        # Show results
        results = result.get("results", [])
        if results:
            st.markdown("**📋 Detailed Results:**")
            
            for i, res in enumerate(results, 1):
                result_type = res.get("type", "unknown")
                content = res.get("content", "")
                
                if result_type == "visualization":
                    # Enhanced chart display
                    try:
                        chart_data = res.get("chart_data", "")
                        if chart_data:
                            st.markdown(f"**📊 Visualization {i}:**")
                            image_data = base64.b64decode(chart_data)
                            st.image(image_data, caption=content, use_column_width=True)
                            
                            # Add download option
                            st.download_button(
                                label=f"💾 Download Chart {i}",
                                data=image_data,
                                file_name=f"chart_{i}_{int(time.time())}.png",
                                mime="image/png"
                            )
                    except Exception as e:
                        st.write(f"📊 {content}")
                        st.error(f"Error displaying chart: {e}")
                
                elif result_type == "data_row":
                    # Enhanced data row display
                    full_data = res.get("full_data", {})
                    if full_data:
                        st.markdown(f"**📝 Record {i}:**")
                        with st.expander(f"View details for record {i}"):
                            st.json(full_data)
                    else:
                        st.write(f"**{i}.** {content}")
                
                else:
                    # Regular text result
                    st.markdown(f"**{i}.** {content}")
                
                if i < len(results):
                    st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Enhanced error display
        error_msg = result.get("message", "Unknown error occurred")
        st.markdown(f'<div class="error-box"><strong>❌ Error:</strong> {error_msg}</div>', unsafe_allow_html=True)
        
        # Suggest alternatives
        st.markdown("**💡 Try these alternatives:**")
        st.markdown("• Rephrase your question")
        st.markdown("• Use the smart suggestions above")
        st.markdown("• Try a simpler question first")


def display_query_history():
    """Enhanced query history display."""
    if not st.session_state.query_history:
        return
    
    st.subheader("📜 Query History")
    
    # History controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{len(st.session_state.query_history)} queries** in your session")
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
    
    # Display recent queries
    for i, entry in enumerate(reversed(st.session_state.query_history[-5:]), 1):
        with st.expander(f"Query {i}: {entry['query'][:50]}{'...' if len(entry['query']) > 50 else ''}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Query:** {entry['query']}")
                st.caption(f"**Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                if st.button(f"🔁 Re-run", key=f"rerun_{i}"):
                    st.session_state.current_query = entry['query']
                    st.rerun()
            
            # Show result summary
            if entry['result'].get('success'):
                results = entry['result'].get('results', [])
                if results and len(results) > 0:
                    st.write(f"**Result:** {results[0].get('content', '')[:100]}...")
            else:
                st.write(f"**Error:** {entry['result'].get('message', '')}")


def main():
    """Enhanced main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display agent status
    display_agent_status()
    
    # Setup sidebar
    api_configured = setup_sidebar()
    
    if not api_configured:
        st.stop()
    
    # Main content area
    if not st.session_state.dataset_loaded:
        # File upload interface
        display_file_upload()
    else:
        # Dataset preview
        display_dataset_preview()
        
        st.markdown("---")
        
        # Query interface
        display_query_interface()
        
        # Query history
        if st.session_state.query_history:
            st.markdown("---")
            display_query_history()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🚀 <strong>Enhanced Agentic AI Data Analysis</strong> | 
            Built with Advanced Streamlit, OpenAI GPT-4, and ChromaDB</p>
            <p style="font-size: 0.9rem;">
            ✨ Features: Real-time feedback • Smart suggestions • Enhanced UX • Progress tracking
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main() 