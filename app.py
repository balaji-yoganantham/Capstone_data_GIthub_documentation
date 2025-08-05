import streamlit as st
import logging
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from rag_system import EnhancedRAGSystem, GROQ_API_KEY, GROQ_MODEL_NAME, EMBEDDING_MODEL_NAME, DOCUMENTS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
import numpy as np

# MLflow Integration
try:
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="🗡️ Zoro - GitHub API Assistant",
    page_icon="🗡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1f4037 0%, #99f2c8 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #4ecdc4;
    }
    
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sidebar-metric {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'initialization_attempted' not in st.session_state:
    st.session_state.initialization_attempted = False
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = {}

def create_performance_charts():
    """Create performance visualization charts"""
    if not st.session_state.rag_system or not st.session_state.rag_system.conversation_history:
        st.info("📊 No conversation data available yet. Start chatting to see analytics!")
        return
    
    history = st.session_state.rag_system.conversation_history
    
    # Create dataframe from conversation history
    df = pd.DataFrame([
        {
            'Question': i+1,
            'Confidence': conv['confidence'],
            'Response Time': conv['response_time'],
            'Sources Used': len(conv['sources']),
            'Timestamp': conv['timestamp']
        }
        for i, conv in enumerate(history)
    ])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Confidence Scores Over Time', 'Response Times', 
                       'Sources Distribution', 'Performance Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "indicator"}]]
    )
    
    # Confidence line chart
    fig.add_trace(
        go.Scatter(x=df['Question'], y=df['Confidence'], 
                  mode='lines+markers', name='Confidence',
                  line=dict(color='#4ecdc4', width=3)),
        row=1, col=1
    )
    
    # Response time bar chart
    fig.add_trace(
        go.Bar(x=df['Question'], y=df['Response Time'], 
               name='Response Time', marker_color='#ff6b6b'),
        row=1, col=2
    )
    
    # Sources histogram
    fig.add_trace(
        go.Histogram(x=df['Sources Used'], name='Sources Used',
                    marker_color='#667eea', nbinsx=10),
        row=2, col=1
    )
    
    # Performance indicator
    avg_confidence = df['Confidence'].mean()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=avg_confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Avg Confidence"},
            delta={'reference': 0.8},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.8], 'color': "yellow"},
                    {'range': [0.8, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, 
                     title_text="📊 Real-time Performance Analytics")
    
    return fig

def auto_initialize_system():
    """Auto-initialize the system on app startup"""
    if not st.session_state.initialization_attempted and GROQ_API_KEY:
        st.session_state.initialization_attempted = True
        
        # Create progress bar for initialization
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🚀 Auto-initializing Enhanced RAG system..."):
            try:
                status_text.text("🔧 Initializing components...")
                progress_bar.progress(20)
                
                rag_system = EnhancedRAGSystem()
                
                if not rag_system.initialize_components(GROQ_API_KEY):
                    st.error("Failed to auto-initialize components!")
                    return False
                
                status_text.text("📚 Creating vector store...")
                progress_bar.progress(50)
                
                if not rag_system.create_vectorstore(DOCUMENTS_FOLDER):
                    st.error("Failed to create vectorstore!")
                    return False
                
                status_text.text("🔗 Setting up conversational chain...")
                progress_bar.progress(80)
                
                if not rag_system.setup_conversational_chain():
                    st.error("Failed to setup conversational chain!")
                    return False
                
                progress_bar.progress(100)
                status_text.text("✅ System ready!")
                
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                
                time.sleep(1)  # Show success message briefly
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ System auto-initialized successfully with conversational memory!")
                return True
                
            except Exception as e:
                st.error(f"Auto-initialization failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                return False
    
    return st.session_state.initialized

def display_system_metrics():
    """Display real-time system metrics in sidebar"""
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_conversation_stats()
        
        if stats:
            st.sidebar.markdown("### 📊 Live Metrics")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Questions", stats.get('total_questions', 0))
                st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")
            
            with col2:
                st.metric("Response Time", f"{stats.get('avg_response_time', 0):.2f}s")
                st.metric("Memory Size", stats.get('memory_size', 0))
            
            # Memory summary
            memory_summary = st.session_state.rag_system.get_memory_summary()
            st.sidebar.info(f"🧠 **Memory Status**: {memory_summary}")
        
        else:
            st.sidebar.info("📊 No metrics available yet. Start chatting!")

def main():
    # Header with gradient background
    st.markdown("""
        <div class="main-header">
            <h1>🗡️ Zoro - Enhanced GitHub API Assistant</h1>
            <p>Powered by LangChain ConversationalRetrievalChain with Memory 🧠</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Auto-initialize system
    auto_initialize_system()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key status
        if GROQ_API_KEY:
            st.markdown('<p class="status-good">✅ Groq API Key loaded from .env</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ No API key found in .env file</p>', unsafe_allow_html=True)
            st.code("GROQ_API_KEY=your_key_here", language="bash")
            st.stop()
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🚀 System Status")
        if st.session_state.initialized:
            st.markdown('<p class="status-good">🟢 System Ready with Memory</p>', unsafe_allow_html=True)
            st.markdown('<p class="status-good">🧠 Conversational Memory Active</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">🟡 System Not Initialized</p>', unsafe_allow_html=True)
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reinit", help="Reinitialize system"):
                st.session_state.initialization_attempted = False
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear", help="Clear conversation"):
                if st.session_state.rag_system:
                    st.session_state.rag_system.clear_memory()
                st.session_state.conversation_history = []
                st.success("Memory cleared!")
                st.rerun()
        
        st.markdown("---")
        
        # Display live metrics
        display_system_metrics()
        
        # System configuration info
        st.markdown("### 🔧 Configuration")
        config_data = {
            "Model": GROQ_MODEL_NAME,
            "Embeddings": EMBEDDING_MODEL_NAME.split('/')[-1],
            "Chunk Size": CHUNK_SIZE,
            "Chunk Overlap": CHUNK_OVERLAP,
        }
        
        for key, value in config_data.items():
            st.markdown(f"**{key}:** `{value}`")
    
    # Main content area
    if not st.session_state.initialized:
        st.warning("⚠️ System initialization required. Please wait or check sidebar for errors.")
        
        # Show system overview while initializing
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                ### 🚀 Enhanced Features
                - ✅ **Conversational Memory**
                - ✅ **LangChain Framework**
                - ✅ **Real-time Analytics** 
                - ✅ **Performance Metrics**
                - ✅ **Advanced UI/UX**
            """)
        
        with col2:
            st.markdown("""
                ### 🔧 Technical Stack
                - **LLM:** Groq Llama3-70B
                - **Vector DB:** ChromaDB
                - **Embeddings:** BGE-Large
                - **Memory:** Window Buffer
                - **Chain:** ConversationalRetrieval
            """)
        
        with col3:
            st.markdown("""
                ### 📊 Analytics
                - **Response Times**
                - **Confidence Scores**
                - **Source Utilization**
                - **Memory Usage**
                - **Performance Trends**
            """)
        
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "🧪 Evaluation"])
    
    with tab1:
        # Chat interface
        st.markdown("### 💬 Chat with Zoro")
        
        # Suggested questions
        with st.expander("💡 Suggested Questions", expanded=False):
            suggestions = [
                "How do I authenticate with the GitHub API?",
                "How can I list all repositories for a user?",
                "What are the rate limits for GitHub API?",  
                "How do I create a repository using the API?",
                "What should I do if I get a 404 error from the API?",
                "How do webhooks work in GitHub?",
                "How can I search for repositories?",
                "What are the different types of GitHub tokens?"
            ]
            
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                        # Add to chat and get response
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": suggestion
                        })
                        
                        with st.spinner("🤔 Thinking with conversational memory..."):
                            try:
                                response_data = st.session_state.rag_system.get_response(suggestion)
                                
                                assistant_message = {
                                    "role": "assistant",
                                    "content": response_data["answer"],
                                    "confidence": response_data.get("confidence", 0.0),
                                    "sources": response_data.get("sources", []),
                                    "response_time": response_data.get("response_time", 0.0),
                                    "retrieved_chunks": response_data.get("retrieved_chunks", []),
                                    "memory_context": response_data.get("memory_context", 0)
                                }
                                
                                st.session_state.conversation_history.append(assistant_message)
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
        
        # Display conversation history (top)
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.conversation_history):
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.write(message["content"])
                elif message["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🗡️"):
                        st.write(message["content"])
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            confidence = message.get("confidence", 0.0)
                            color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                            st.metric("Confidence", f"{confidence:.2f}", delta=None)
                        with col2:
                            response_time = message.get("response_time", 0.0)
                            st.metric("Response Time", f"{response_time:.2f}s")
                        with col3:
                            sources_count = len(message.get("sources", []))
                            st.metric("Sources", sources_count)
                        with col4:
                            memory_context = message.get("memory_context", 0)
                            st.metric("Memory Items", memory_context)
                        # Expandable sections
                        if message.get("retrieved_chunks"):
                            with st.expander("📄 Retrieved Chunks", expanded=False):
                                for j, chunk in enumerate(message["retrieved_chunks"]):
                                    st.markdown(f"**Source {j+1}: {chunk['source']}**")
                                    st.write(chunk["content"])
                                    if j < len(message["retrieved_chunks"]) - 1:
                                        st.markdown("---")
                        if message.get("sources"):
                            with st.expander("📚 Sources Used", expanded=False):
                                for source in message["sources"]:
                                    st.write(f"• {source}")

        # Chat input (bottom, like ChatGPT)
        prompt = st.chat_input("Ask me anything about GitHub API... (I remember our conversation! 🧠)")
        if prompt:
            # Add user message
            st.session_state.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            # Display user message immediately
            with st.chat_message("user", avatar="👤"):
                st.write(prompt)
            # Get and display assistant response
            with st.chat_message("assistant", avatar="🗡️"):
                with st.spinner("🤔 Thinking with conversational memory..."):
                    try:
                        response_data = st.session_state.rag_system.get_response(prompt)
                        # Display answer
                        st.write(response_data["answer"])
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            confidence = response_data.get("confidence", 0.0)
                            st.metric("Confidence", f"{confidence:.2f}")
                        with col2:
                            response_time = response_data.get("response_time", 0.0)
                            st.metric("Response Time", f"{response_time:.2f}s")
                        with col3:
                            sources_count = len(response_data.get("sources", []))
                            st.metric("Sources", sources_count)
                        with col4:
                            memory_context = response_data.get("memory_context", 0)
                            st.metric("Memory Items", memory_context)
                        # Store assistant message
                        assistant_message = {
                            "role": "assistant",
                            "content": response_data["answer"],
                            "confidence": response_data.get("confidence", 0.0),
                            "sources": response_data.get("sources", []),
                            "response_time": response_data.get("response_time", 0.0),
                            "retrieved_chunks": response_data.get("retrieved_chunks", []),
                            "memory_context": response_data.get("memory_context", 0)
                        }
                        st.session_state.conversation_history.append(assistant_message)
                        # Display expandable sections
                        if response_data.get("retrieved_chunks"):
                            with st.expander("📄 Retrieved Chunks", expanded=False):
                                for j, chunk in enumerate(response_data["retrieved_chunks"]):
                                    st.markdown(f"**Source {j+1}: {chunk['source']}**")
                                    st.write(chunk["content"])
                                    if j < len(response_data["retrieved_chunks"]) - 1:
                                        st.markdown("---")
                        if response_data.get("sources"):
                            with st.expander("📚 Sources Used", expanded=False):
                                for source in response_data["sources"]:
                                    st.write(f"• {source}")
                    except Exception as e:
                        error_message = f"I apologize, but I encountered an error: {str(e)}"
                        st.error(error_message)
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": error_message
                        })
    
    with tab2:
        # Analytics dashboard
        st.markdown("### 📊 Performance Analytics Dashboard")
        
        if st.session_state.rag_system and st.session_state.rag_system.conversation_history:
            # Display performance charts
            fig = create_performance_charts()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            stats = st.session_state.rag_system.get_conversation_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                    <div class="metric-container">
                        <h3>Total Questions</h3>
                        <h2>{}</h2>
                    </div>
                """.format(stats.get('total_questions', 0)), unsafe_allow_html=True)
            
            with col2:
                avg_conf = stats.get('avg_confidence', 0)
                color = "#28a745" if avg_conf > 0.8 else "#ffc107" if avg_conf > 0.5 else "#dc3545"
                st.markdown("""
                    <div class="metric-container" style="background: linear-gradient(135deg, {}, #764ba2);">
                        <h3>Avg Confidence</h3>
                        <h2>{:.2f}</h2>
                    </div>
                """.format(color, avg_conf), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class="metric-container">
                        <h3>Avg Response Time</h3>
                        <h2>{:.2f}s</h2>
                    </div>
                """.format(stats.get('avg_response_time', 0)), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                    <div class="metric-container">
                        <h3>Unique Sources</h3>
                        <h2>{}</h2>
                    </div>
                """.format(stats.get('unique_sources', 0)), unsafe_allow_html=True)
            
            # Conversation timeline
            st.markdown("### 📈 Conversation Timeline")
            
            if st.session_state.rag_system.conversation_history:
                timeline_data = []
                for i, conv in enumerate(st.session_state.rag_system.conversation_history):
                    timeline_data.append({
                        'Question': i + 1,
                        'Timestamp': pd.to_datetime(conv['timestamp']),
                        'Confidence': conv['confidence'],
                        'Response_Time': conv['response_time'],
                        'Question_Text': conv['question'][:50] + "..." if len(conv['question']) > 50 else conv['question']
                    })
                
                timeline_df = pd.DataFrame(timeline_data)
                
                # Interactive timeline chart
                timeline_fig = px.scatter(
                    timeline_df, 
                    x='Timestamp', 
                    y='Confidence',
                    size='Response_Time',
                    hover_data=['Question_Text', 'Response_Time'],
                    title="Conversation Timeline (Size = Response Time)",
                    color='Confidence',
                    color_continuous_scale='RdYlGn'
                )
                
                timeline_fig.update_layout(height=400)
                st.plotly_chart(timeline_fig, use_container_width=True)
        
        else:
            st.info("📊 No analytics data available yet. Start chatting to see performance metrics!")
    
    with tab3:
        # Evaluation system
        st.markdown("### 🧪 System Evaluation")
        
        # Display existing evaluation files
        import os
        evaluation_folder = "evaluation"
        
        if os.path.exists(evaluation_folder):
            evaluation_files = [f for f in os.listdir(evaluation_folder) if f.endswith('.json')]
            if evaluation_files:
                st.markdown("### 📁 Previous Evaluation Results")
                st.write(f"Found {len(evaluation_files)} evaluation result(s):")
                
                # Sort files by modification time (newest first)
                evaluation_files.sort(key=lambda x: os.path.getmtime(os.path.join(evaluation_folder, x)), reverse=True)
                
                for i, filename in enumerate(evaluation_files[:5]):  # Show last 5 files
                    file_path = os.path.join(evaluation_folder, filename)
                    file_size = os.path.getsize(file_path)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📄 **{filename}**")
                    with col2:
                        st.write(f"📊 {file_size // 1024} KB")
                    with col3:
                        st.write(f"🕒 {file_time.strftime('%Y-%m-%d %H:%M')}")
                
                if len(evaluation_files) > 5:
                    st.info(f"... and {len(evaluation_files) - 5} more files")
                
                # Add option to load previous evaluation
                st.markdown("### 📖 Load Previous Evaluation")
                selected_file = st.selectbox(
                    "Choose an evaluation file to view:",
                    ["Select a file..."] + evaluation_files,
                    key="evaluation_file_selector"
                )
                
                if selected_file and selected_file != "Select a file...":
                    try:
                        file_path = os.path.join(evaluation_folder, selected_file)
                        with open(file_path, 'r') as f:
                            loaded_results = json.load(f)
                        
                        st.success(f"✅ Loaded evaluation results from {selected_file}")
                        
                        # Display loaded results
                        if "aggregate_metrics" in loaded_results:
                            metrics = loaded_results["aggregate_metrics"]
                            
                            # Display metrics in a compact format
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if "error" not in metrics.get('f1_score', {}):
                                    f1_score = metrics['f1_score']['mean']
                                    st.metric("F1 Score", f"{f1_score:.3f}", f"±{metrics['f1_score']['std']:.3f}")
                            
                            with col2:
                                if "error" not in metrics.get('rouge1_f', {}):
                                    rouge_score = metrics['rouge1_f']['mean']
                                    st.metric("ROUGE-1 F1", f"{rouge_score:.3f}", f"±{metrics['rouge1_f']['std']:.3f}")
                            
                            with col3:
                                if "error" not in metrics.get('keyword_coverage', {}):
                                    keyword_cov = metrics['keyword_coverage']['mean']
                                    st.metric("Keyword Coverage", f"{keyword_cov:.3f}", f"±{metrics['keyword_coverage']['std']:.3f}")
                            
                            with col4:
                                success_rate = loaded_results["successful_evaluations"] / loaded_results["total_questions"]
                                st.metric("Success Rate", f"{success_rate:.1%}")
                        
                        # Show evaluation timestamp
                        if "evaluation_timestamp" in loaded_results:
                            st.info(f"📅 Evaluation performed on: {loaded_results['evaluation_timestamp']}")
                        
                    except Exception as e:
                        st.error(f"Error loading evaluation file: {e}")
                
                st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Evaluate the RAG system performance using predefined Q&A pairs with ground truth answers.")
            st.info("💡 This will test the system's ability to provide accurate and relevant responses.")
        
        with col2:
            if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
                if st.session_state.rag_system:
                    with st.spinner("Running comprehensive evaluation..."):
                        try:
                            # Clear memory before evaluation for consistency
                            original_memory = st.session_state.rag_system.memory
                            st.session_state.rag_system.clear_memory()
                            
                            evaluation_results = st.session_state.rag_system.run_evaluation()
                            
                            # Restore original memory
                            st.session_state.rag_system.memory = original_memory
                            
                            # Display results with improved visibility
                            st.markdown("## 📊 Evaluation Results")
                            st.markdown("---")
                            
                            metrics = evaluation_results["aggregate_metrics"]
                            
                            # Summary metrics with improved styling and error handling
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if "error" not in metrics.get('f1_score', {}):
                                    f1_score = metrics['f1_score']['mean']
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                                        <h3 style="margin: 0; font-size: 24px;">F1 Score</h3>
                                        <h2 style="margin: 10px 0; font-size: 36px;">{f1_score:.3f}</h2>
                                        <p style="margin: 0; font-size: 16px;">±{metrics['f1_score']['std']:.3f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.error(f"F1 Score: {metrics['f1_score']['error']}")
                            
                            with col2:
                                if "error" not in metrics.get('rouge1_f', {}):
                                    rouge_score = metrics['rouge1_f']['mean']
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px; color: white;">
                                        <h3 style="margin: 0; font-size: 24px;">ROUGE-1 F1</h3>
                                        <h2 style="margin: 10px 0; font-size: 36px;">{rouge_score:.3f}</h2>
                                        <p style="margin: 0; font-size: 16px;">±{metrics['rouge1_f']['std']:.3f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.error(f"ROUGE-1 F1: {metrics['rouge1_f']['error']}")
                            
                            with col3:
                                if "error" not in metrics.get('keyword_coverage', {}):
                                    keyword_cov = metrics['keyword_coverage']['mean']
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; color: white;">
                                        <h3 style="margin: 0; font-size: 24px;">Keyword Coverage</h3>
                                        <h2 style="margin: 10px 0; font-size: 36px;">{keyword_cov:.3f}</h2>
                                        <p style="margin: 0; font-size: 16px;">±{metrics['keyword_coverage']['std']:.3f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.error(f"Keyword Coverage: {metrics['keyword_coverage']['error']}")
                            
                            with col4:
                                success_rate = evaluation_results["successful_evaluations"] / evaluation_results["total_questions"]
                                st.markdown(f"""
                                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 10px; color: white;">
                                    <h3 style="margin: 0; font-size: 24px;">Success Rate</h3>
                                    <h2 style="margin: 10px 0; font-size: 36px;">{success_rate:.1%}</h2>
                                    <p style="margin: 0; font-size: 16px;">{evaluation_results["successful_evaluations"]}/{evaluation_results["total_questions"]} questions</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add a summary section for better visibility
                            st.markdown("---")
                            st.markdown("### 📈 Evaluation Summary")
                            
                            # Create a summary table
                            summary_data = {
                                "Metric": ["F1 Score", "ROUGE-1 F1", "Keyword Coverage", "Success Rate"],
                                "Value": [
                                    f"{metrics.get('f1_score', {}).get('mean', 0):.3f}" if "error" not in metrics.get('f1_score', {}) else "Error",
                                    f"{metrics.get('rouge1_f', {}).get('mean', 0):.3f}" if "error" not in metrics.get('rouge1_f', {}) else "Error",
                                    f"{metrics.get('keyword_coverage', {}).get('mean', 0):.3f}" if "error" not in metrics.get('keyword_coverage', {}) else "Error",
                                    f"{success_rate:.1%}"
                                ],
                                "Status": [
                                    "✅ Good" if metrics.get('f1_score', {}).get('mean', 0) > 0.5 else "⚠️ Needs Improvement" if "error" not in metrics.get('f1_score', {}) else "❌ Error",
                                    "✅ Good" if metrics.get('rouge1_f', {}).get('mean', 0) > 0.5 else "⚠️ Needs Improvement" if "error" not in metrics.get('rouge1_f', {}) else "❌ Error",
                                    "✅ Good" if metrics.get('keyword_coverage', {}).get('mean', 0) > 0.7 else "⚠️ Needs Improvement" if "error" not in metrics.get('keyword_coverage', {}) else "❌ Error",
                                    "✅ Excellent" if success_rate == 1.0 else "✅ Good" if success_rate > 0.8 else "⚠️ Needs Improvement"
                                ]
                            }
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.table(summary_df)
                            
                            # Add evaluation statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"**Total Questions:** {evaluation_results['total_questions']}")
                            with col2:
                                st.success(f"**Successful:** {evaluation_results['successful_evaluations']}")
                            with col3:
                                st.warning(f"**Failed:** {evaluation_results['failed_evaluations']}")
                            
                            st.markdown("---")
                            
                            # Visualization of results
                            eval_data = []
                            for i, result in enumerate(evaluation_results["individual_results"]):
                                if "error" not in result:
                                    try:
                                        eval_data.append({
                                            'Question': i + 1,
                                            'F1_Score': result.get('f1_metrics', {}).get('f1', 0.0),
                                            'ROUGE_F1': result.get('rouge_scores', {}).get('rouge1_f', 0.0),
                                            'Keyword_Coverage': result.get('keyword_coverage', 0.0),
                                            'Question_Text': result['question'][:30] + "..."
                                        })
                                    except (KeyError, TypeError) as e:
                                        logger.warning(f"Error processing result {i}: {e}")
                                        continue
                            
                            if eval_data:
                                eval_df = pd.DataFrame(eval_data)
                                
                                # Create evaluation charts
                                eval_fig = make_subplots(
                                    rows=1, cols=3,
                                    subplot_titles=('F1 Scores', 'ROUGE-1 F1', 'Keyword Coverage')
                                )
                                
                                eval_fig.add_trace(
                                    go.Bar(x=eval_df['Question'], y=eval_df['F1_Score'], 
                                          name='F1', marker_color='#4ecdc4'),
                                    row=1, col=1
                                )
                                
                                eval_fig.add_trace(
                                    go.Bar(x=eval_df['Question'], y=eval_df['ROUGE_F1'], 
                                          name='ROUGE', marker_color='#ff6b6b'),
                                    row=1, col=2
                                )
                                
                                eval_fig.add_trace(
                                    go.Bar(x=eval_df['Question'], y=eval_df['Keyword_Coverage'], 
                                          name='Keywords', marker_color='#667eea'),
                                    row=1, col=3
                                )
                                
                                eval_fig.update_layout(height=400, showlegend=False,
                                                     title_text="📊 Detailed Evaluation Results")
                                
                                st.plotly_chart(eval_fig, use_container_width=True)
                            
                            # Detailed results
                            with st.expander("📋 Detailed Results", expanded=False):
                                for i, result in enumerate(evaluation_results["individual_results"]):
                                    if "error" not in result:
                                        st.markdown(f"**Question {i+1}:** {result['question']}")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Metrics:**")
                                            try:
                                                f1_score = result.get('f1_metrics', {}).get('f1', 0.0)
                                                rouge_score = result.get('rouge_scores', {}).get('rouge1_f', 0.0)
                                                keyword_coverage = result.get('keyword_coverage', 0.0)
                                                
                                                st.write(f"• F1 Score: {f1_score:.3f}")
                                                st.write(f"• ROUGE-1 F1: {rouge_score:.3f}")
                                                st.write(f"• Keyword Coverage: {keyword_coverage:.3f}")
                                            except (KeyError, TypeError) as e:
                                                st.error(f"Error displaying metrics: {e}")
                                        
                                        with col2:
                                            st.write("**Keywords Found:**")
                                            try:
                                                found_keywords = result.get('found_keywords', [])
                                                expected_keywords = result.get('expected_keywords', [])
                                                found = len(found_keywords)
                                                total = len(expected_keywords)
                                                st.write(f"• Found: {found}/{total}")
                                                if found_keywords:
                                                    st.write("• " + ", ".join(found_keywords))
                                            except (KeyError, TypeError) as e:
                                                st.error(f"Error displaying keywords: {e}")
                                        
                                        st.write("**System Response:**")
                                        try:
                                            st.info(result.get('predicted_response', 'No response available'))
                                        except (KeyError, TypeError) as e:
                                            st.error(f"Error displaying response: {e}")
                                        
                                        st.markdown("---")
                                    else:
                                        st.error(f"**Question {i+1}:** {result.get('question', 'Unknown question')} - {result.get('error', 'Unknown error')}")
                                        st.markdown("---")
                            
                            # Save results to evaluation folder
                            import os
                            evaluation_folder = "evaluation"
                            
                            # Create evaluation folder if it doesn't exist
                            if not os.path.exists(evaluation_folder):
                                os.makedirs(evaluation_folder)
                            
                            results_file = os.path.join(evaluation_folder, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                            try:
                                with open(results_file, 'w') as f:
                                    json.dump(evaluation_results, f, indent=2)
                                st.success(f"📁 Results saved to: {results_file}")
                            except Exception as e:
                                st.warning(f"Could not save results: {e}")
                        
                        except Exception as e:
                            st.error(f"Evaluation failed: {str(e)}")
                
                else:
                    st.error("Please initialize the system first!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em; padding: 1rem;'>
            <p>🗡️ <strong>Zoro - Enhanced GitHub API Assistant</strong></p>
            <p>Powered by LangChain ConversationalRetrievalChain • Groq Llama3-70B • ChromaDB</p>
            <p>Created by <strong>Balaji</strong> • Enhanced with Memory & Analytics 🧠📊</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()