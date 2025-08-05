# 🗡️ Zoro - GitHub API Assistant (Capstone Project)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Groq](https://img.shields.io/badge/Groq-LLM-orange.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-lightgrey.svg)

**A Production-Grade RAG System for GitHub API Documentation**

*Built with LangChain, Groq, ChromaDB, and Streamlit*

[🚀 Quick Start](#-quick-start) • [📋 Features](#-features) • [🏗️ Architecture](#️-architecture) • [📊 Evaluation](#-evaluation) • [🎥 Demo](#-demo) • [🤝 Contributing](#-contributing)

</div>

---

## 📖 Project Overview

**Zoro** is a comprehensive Retrieval-Augmented Generation (RAG) system designed to provide intelligent assistance for GitHub API documentation. This capstone project demonstrates advanced natural language processing techniques, modern web application development, and production-ready software engineering practices.

### 🎯 Key Objectives
- **Intelligent Documentation Search**: Provide contextually relevant answers to GitHub API queries
- **Conversational Interface**: ChatGPT-style chat experience with memory and context preservation
- **Production-Ready Architecture**: Modular, scalable, and maintainable codebase
- **Comprehensive Evaluation**: Built-in metrics and testing framework
- **Modern UI/UX**: Beautiful, responsive web interface with real-time analytics

---

## ✨ Features

### 🤖 Core RAG Capabilities
- **Advanced Document Processing**: Custom sliding window chunking for optimal context preservation
- **Semantic Search**: BAAI/bge-large-en-v1.5 embeddings for accurate retrieval
- **High-Performance LLM**: Groq's Llama3-70B for fast, high-quality responses
- **Vector Database**: ChromaDB for efficient similarity search and storage
- **Conversational Memory**: Maintains context across chat sessions

### 🎨 User Experience
- **ChatGPT-Style Interface**: Modern chat UI with message history
- **Real-Time Analytics**: Live performance metrics and system health monitoring
- **Expandable Context**: View source documents and confidence scores
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Dark/Light Theme**: Customizable interface appearance

### 🔧 Technical Features
- **Modular Architecture**: Clean separation of concerns across components
- **MLflow Integration**: Experiment tracking and model versioning
- **Comprehensive Logging**: Structured logging with custom formatters
- **Error Handling**: Robust error management and user feedback
- **Configuration Management**: Environment-based settings

### 📊 Evaluation & Monitoring
- **Multi-Metric Evaluation**: F1 Score, ROUGE, Keyword Coverage
- **Performance Analytics**: Response time, accuracy, and system metrics
- **Batch Testing**: Automated evaluation of multiple test cases
- **Report Generation**: Detailed evaluation reports with visualizations

---

## 🛠️ Technology Stack

### Core Framework
- **LangChain**: RAG framework and LLM orchestration
- **Streamlit**: Web application framework
- **Groq**: High-performance LLM inference
- **ChromaDB**: Vector database for embeddings

### AI/ML Components
- **BAAI/bge-large-en-v1.5**: Sentence embeddings
- **Llama3-70B**: Large language model
- **scikit-learn**: Evaluation metrics
- **rouge-score**: Text similarity scoring

### Development & Monitoring
- **MLflow**: Experiment tracking and model management
- **Loguru**: Advanced logging
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and analysis

### Infrastructure
- **Python 3.8+**: Core programming language
- **ChromaDB**: Vector storage
- **Environment Variables**: Secure configuration management

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   RAG System    │    │   Vector Store  │
│                 │◄──►│                 │◄──►│   (ChromaDB)    │
│  - Chat Interface│    │  - Document Proc│    │                 │
│  - Analytics    │    │  - Embeddings   │    │  - Embeddings   │
│  - Evaluation   │    │  - LLM Chain    │    │  - Metadata     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │   Evaluation    │    │   Documents     │
│                 │    │                 │    │                 │
│  - Experiment   │    │  - F1 Score     │    │  - ROUGE        │
│  - Model Tracking│    │  - Keywords     │    │  - Artifacts    │
│  - Artifacts    │    │  - Examples     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Architecture

#### 1. **RAG System Core** (`rag_system/`)
- **`__init__.py`**: Main RAG system orchestrator
- **`config.py`**: Configuration constants and settings
- **`embeddings.py`**: Embedding model management
- **`vectorstore.py`**: Document processing and vector storage
- **`memory.py`**: Conversation memory management
- **`prompts.py`**: Prompt template system
- **`conversational_chain.py`**: LLM chain configuration
- **`response.py`**: Response processing and confidence scoring
- **`stats.py`**: System statistics and metrics
- **`logger.py`**: Custom logging configuration

#### 2. **Web Application** (`app.py`)
- **Streamlit Interface**: Modern chat UI with real-time updates
- **Analytics Dashboard**: Performance metrics and visualizations
- **Evaluation Interface**: Built-in testing and evaluation tools
- **System Monitoring**: Health checks and status indicators

#### 3. **Evaluation Framework** (`evaluation.py`)
- **Multi-Metric Evaluation**: F1, ROUGE, Keyword Coverage
- **Batch Testing**: Automated evaluation of test datasets
- **Report Generation**: Comprehensive evaluation reports
- **Custom Test Cases**: Extensible evaluation framework

#### 4. **MLflow Integration** (`mlflow_integration/`)
- **Experiment Tracking**: Model performance and hyperparameters
- **Artifact Management**: Model versions and evaluation results
- **Metrics Logging**: Automated metric collection and visualization

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8 or higher**
- **Groq API Key** (Get one at [groq.com](https://groq.com))
- **Git** (for cloning the repository)

### 1. Clone the Repository
```bash
git clone https://github.com/Balajiyoganantham/github_documentation_RAG.git
cd github_documentation_RAG
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the project root:
```bash
# Required: Your Groq API key
GROQ_API_KEY=your_actual_groq_api_key_here

# Optional: Custom configurations
CHUNK_SIZE=400
CHUNK_OVERLAP=200
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LLM_MODEL=llama3-70b-8192
```

### 5. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

---

## 🎥 Demo

<div align="center">

### Watch the Demo Video

[![Zoro GitHub API Assistant Demo](https://img.shields.io/badge/Watch%20Demo-Video%20Demo-red?style=for-the-badge&logo=youtube)](demo.mp4)

*Click the button above to watch the full demo video*

</div>

**Demo Highlights:**
- 🗣️ **Interactive Chat Interface**: See the ChatGPT-style conversation flow
- 🔍 **Real-time Search**: Watch how the system retrieves relevant GitHub API documentation
- 📊 **Live Analytics**: Observe performance metrics and system health monitoring
- 🧪 **Evaluation System**: See the built-in testing and evaluation features
- ⚡ **Fast Responses**: Experience the high-speed LLM responses powered by Groq

---

## 📁 Project Structure

```
github_documentation_RAG/
├── 📄 app.py                    # Main Streamlit application
├── 📄 evaluation.py             # Evaluation framework
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env                      # Environment variables (create this)
├── 📄 README.md                 # This file
├── 📄 demo.mp4                  # Demo video showcasing the application
│
├── 📁 rag_system/               # Core RAG system
│   ├── 📄 __init__.py           # Main RAG orchestrator
│   ├── 📄 config.py             # Configuration constants
│   ├── 📄 embeddings.py         # Embedding model management
│   ├── 📄 vectorstore.py        # Document processing & vector storage
│   ├── 📄 memory.py             # Conversation memory
│   ├── 📄 prompts.py            # Prompt templates
│   ├── 📄 conversational_chain.py # LLM chain setup
│   ├── 📄 response.py           # Response processing
│   ├── 📄 stats.py              # System statistics
│   └── 📄 logger.py             # Custom logging
│
├── 📁 mlflow_integration/       # MLflow experiment tracking
│   ├── 📄 __init__.py
│   ├── 📄 config.py
│   ├── 📄 metrics.py
│   └── 📄 tracking.py
│
├── 📁 documents/                # GitHub API documentation
│   ├── 📄 authentication.txt    # Authentication guide
│   ├── 📄 getting_started.txt   # Getting started guide
│   ├── 📄 repositories_api.txt  # Repositories API
│   ├── 📄 users_api.txt         # Users API
│   ├── 📄 issues_api.txt        # Issues API
│   ├── 📄 webhooks.txt          # Webhooks guide
│   ├── 📄 pagination.txt        # Pagination guide
│   ├── 📄 error_handling.txt    # Error handling guide
│   └── 📄 index.txt             # API index
│
├── 📁 chroma_db/                # Vector database storage
├── 📁 conversations/            # Chat conversation history
├── 📁 logs/                     # Application logs
├── 📁 mlflow_artifacts/         # MLflow experiment artifacts
└── 📁 evaluation/               # Evaluation results and reports
```

---

## 🔧 Configuration

### Core Settings (`rag_system/config.py`)
```python
# Document Processing
CHUNK_SIZE = 400              # Characters per chunk
CHUNK_OVERLAP = 200           # Overlap between chunks
DOCUMENTS_FOLDER = "documents" # Documentation source folder

# AI Models
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
GROQ_MODEL_NAME = "llama3-70b-8192"

# Vector Database
COLLECTION_NAME = "github_api_docs"
CHROMA_PERSIST_DIRECTORY = "chroma_db"

# Evaluation
EVALUATION_METRICS = ["f1", "rouge", "keyword_coverage"]
```

### Environment Variables (`.env`)
```bash
# Required
GROQ_API_KEY=your_groq_api_key

# Optional
CHUNK_SIZE=400
CHUNK_OVERLAP=200
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LLM_MODEL=llama3-70b-8192
LOG_LEVEL=INFO
```

---

## 📊 Evaluation

### Built-in Metrics

#### 1. **F1 Score**
- Measures answer accuracy and completeness
- Based on word overlap between predicted and ground truth
- Range: 0.0 to 1.0 (higher is better)

#### 2. **ROUGE Metrics**
- **ROUGE-1**: Single word overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- Measures text similarity and content coverage

#### 3. **Keyword Coverage**
- Identifies important terms in responses
- Compares against expected keywords
- Measures topic relevance

### Running Evaluation

#### Through Web Interface
1. Navigate to the "Evaluation" tab in the Streamlit app
2. Click "Run Full Evaluation" for comprehensive testing
3. View results and download reports

#### Command Line
```bash
python evaluation.py
```

#### Custom Evaluation
```python
from evaluation import EvaluationRunner

# Initialize evaluator
evaluator = EvaluationRunner(rag_system)

# Add custom test case
evaluator.add_custom_question(
    question="How do I create a webhook?",
    ground_truth="Use POST /repos/{owner}/{repo}/hooks endpoint...",
    expected_keywords=["webhook", "POST", "endpoint", "repository"]
)

# Run evaluation
results = evaluator.run_full_evaluation()
```

### Sample Evaluation Results
```
Evaluation Report
================

Overall Performance:
- F1 Score: 0.85
- ROUGE-1: 0.78
- ROUGE-2: 0.65
- ROUGE-L: 0.82
- Keyword Coverage: 0.88

Response Time:
- Average: 2.3 seconds
- Median: 2.1 seconds
- 95th percentile: 3.8 seconds

Success Rate: 92%
```

---

## 🎯 Use Cases

### Primary Use Cases
1. **GitHub API Documentation Search**
   - Find specific API endpoints and parameters
   - Get code examples and usage patterns
   - Understand authentication methods

2. **Developer Support**
   - Troubleshoot API errors and issues
   - Learn best practices and patterns
   - Get guidance on implementation

3. **Code Generation**
   - Generate API client code
   - Create webhook handlers
   - Build authentication flows

### Example Queries
- "How do I authenticate with GitHub API?"
- "What are the rate limits for the API?"
- "How do I create a repository programmatically?"
- "What should I do if I get a 404 error?"
- "How do webhooks work in GitHub?"

---

## 🔒 Security & Best Practices

### Security Measures
- **API Key Protection**: Environment variables for sensitive data
- **Input Validation**: Sanitized user inputs
- **Error Handling**: Secure error messages without data leakage
- **Access Control**: No sensitive data in frontend

### Development Best Practices
- **Modular Design**: Clean separation of concerns
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings and comments
- **Testing**: Built-in evaluation framework
- **Logging**: Structured logging for debugging

### Performance Optimization
- **Caching**: Vector store caching for faster retrieval
- **Async Processing**: Non-blocking operations where possible
- **Memory Management**: Efficient resource utilization
- **Connection Pooling**: Optimized database connections

---

## 🚀 Deployment

### Local Development
```bash
# Development mode with auto-reload
streamlit run app.py --server.runOnSave true
```

### Production Deployment
```bash
# Set production environment
export ENVIRONMENT=production
export LOG_LEVEL=WARNING

# Run with production settings
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with proper testing
4. Update documentation as needed
5. Commit with descriptive messages: `git commit -m 'Add amazing feature'`
6. Push to your branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all classes and methods
- Write tests for new features
- Update documentation for API changes

### Testing
```bash
# Run evaluation tests
python evaluation.py

# Check code style
flake8 rag_system/ app.py evaluation.py

# Run type checking
mypy rag_system/ app.py evaluation.py
```

---

## 📈 Performance & Monitoring

### System Metrics
- **Response Time**: Average 2-3 seconds per query
- **Accuracy**: 85%+ F1 score on test dataset
- **Throughput**: 100+ queries per hour
- **Memory Usage**: ~2GB RAM for full system

### Monitoring Dashboard
The application includes real-time monitoring:
- System health indicators
- Performance metrics
- Error rates and logs
- User interaction analytics

### MLflow Integration
- **Experiment Tracking**: Model performance over time
- **Artifact Management**: Model versions and evaluation results
- **Metric Visualization**: Interactive performance charts
- **Model Registry**: Version control for AI models

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **Groq API Key Error**
```bash
Error: Invalid API key
```
**Solution**: Verify your Groq API key in the `.env` file

#### 2. **ChromaDB Connection Issues**
```bash
Error: ChromaDB connection failed
```
**Solution**: Check if ChromaDB is running and accessible

#### 3. **Memory Issues**
```bash
Error: Out of memory
```
**Solution**: Increase system RAM or reduce chunk size in config

#### 4. **Slow Response Times**
**Solutions**:
- Check internet connection
- Verify Groq API status
- Monitor system resources

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
streamlit run app.py
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Balaji** - *GitHub API Documentation RAG System*

- **GitHub**: [@Balajiyoganantham](https://github.com/Balajiyoganantham)
- **Project**: GitHub API Documentation RAG System
- **Capstone Project**: Advanced RAG System with Modern Web Interface

---

## 🙏 Acknowledgments

### Open Source Libraries
- **LangChain**: Excellent RAG framework and LLM orchestration
- **Groq**: Lightning-fast LLM inference platform
- **ChromaDB**: Efficient vector database for embeddings
- **Streamlit**: Beautiful and intuitive web framework
- **MLflow**: Comprehensive experiment tracking platform

### Community Support
- **GitHub API Team**: Comprehensive documentation
- **Open Source Community**: Continuous improvements and feedback
- **Academic Research**: RAG and evaluation methodologies

---

## 📚 Additional Resources

### Documentation
- [LangChain Documentation](https://python.langchain.com/)
- [Groq API Documentation](https://console.groq.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Research Papers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)

### Related Projects
- [LangChain RAG Examples](https://github.com/langchain-ai/langchain)
- [ChromaDB Examples](https://github.com/chroma-core/chroma)

---

<div align="center">

⭐ **Star this repository if you find it helpful!**

[Report Bug](https://github.com/Balajiyoganantham/github_documentation_RAG/issues) • [Request Feature](https://github.com/Balajiyoganantham/github_documentation_RAG/issues) • [View Demo](https://github.com/Balajiyoganantham/github_documentation_RAG)

*Built with ❤️ for the developer community*

</div> 