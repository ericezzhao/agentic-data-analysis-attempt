# 🚀 Agentic AI Data Analysis System

> **⚠️ WARNING:** The AI agent behavior is not guaranteed to recognize and return proper query results. Still under active development and query processing accuracy may vary (a lot). Please verify results independently. I'm trying my best here... But if you have ways to improve the behavior, please reach out as I am happy to learn!

An AI data analysis system built with the Model Context Protocol (MCP), ChromaDB, and Streamlit UI. Features multi-modal query processing, conversation memory, and data visualization capabilities.

## ✨ Key Features and Tools
- **Multi-modal Query Processing**: Statistical, semantic, and SQL-based analysis
- **Conversation Memory**: ChromaDB-powered persistent memory across sessions
- **Streamlit UI**: Modern UI
- **MCP Protocol**: Model Context Protocol implementation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp env.example .env
# Edit .env with your OpenAI API key
# Also add API key on Streamlit
```

### 3. Launch Demo Application
```bash
# Interactive demo with sample datasets
python demo_with_sample_data.py

# Or run Streamlit directly
streamlit run ui/streamlit_app.py
```

### 4. Access Web Interface
Open your browser to: `http://localhost:8503`

## 🎮 Demo Features

The `demo_with_sample_data.py` launcher creates realistic sample datasets:
- **Sales Performance** (2023-2024): Regional sales with seasonal trends
- **Customer Demographics**: 2,000 customer profiles with satisfaction metrics
- **Financial Performance**: Quarterly department budgets and efficiency
- **Website Analytics**: Daily traffic patterns and conversion tracking

**No file upload required** - try the system immediately with comprehensive sample data

## 📁 Structure

```
agentic_ai_analysis/
├── agent/                          # Core AI agent implementation
│   ├── improved_data_analysis_agent.py  # Main agent with MCP protocol
│   └── ...
├── mcp_server/                     # Model Context Protocol server
│   ├── data_analysis_server.py     # MCP server implementation
│   ├── protocol.py                 # Custom MCP protocol layer
│   └── schema_definitions.py       # MCP tools and resources
├── ui/                            # Streamlit user interface
│   ├── streamlit_app.py           # Main UI with Task 5.3 features
│   ├── export_manager.py          # Export and visualization system
│   └── README.md                  # UI documentation
├── tests/                         # Core testing suite
│   ├── test_mcp_server.py         # MCP server tests
│   ├── test_agent.py              # Agent functionality tests
│   └── ...
├── database/                      # ChromaDB configuration
├── config/                        # System configuration
├── utils/                         # Utility functions
├── docs/                          # Documentation
├── data/                          # Sample datasets
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── demo_with_sample_data.py       # Interactive demo launcher with sample data
└── env.example                    # Environment configuration template
```

## 🎯 Usage Guide

### 1. **Upload Data**
- Drag & drop CSV or Excel files
- Automatic data validation and preview
- Smart column detection and analysis

### 2. **Query Your Data**
- Ask questions pertaining to the data
- AI suggests relevant queries based on your data
- Support for complex statistical and business intelligence queries

### 3. **Create Custom Visualizations**
- Interactive chart builder with multiple themes
- 7+ chart types: histograms, scatter plots, heatmaps, etc.

### 4. **Export Results**
- Variety of formats

## 🔧 Technical Features

### Multi-Modal Query Processing
- **Statistical Queries**: Direct calculations for aggregations, statistics
- **Semantic Search**: Vector-based similarity search using ChromaDB
- **SQL Analysis**: Complex data filtering and transformations
- **AI Reasoning**: GPT-4 powered insights and explanations

### Export Capabilities
- **CSV Export**: Clean data extraction with proper formatting
- **Excel Export**: Multi-sheet workbooks with analysis summaries
- **JSON Export**: Structured data with metadata and timestamps
- **ZIP Packages**: Complete analysis packages for sharing
- **Chart Export**: High-resolution PNG downloads

### Memory System
- **Conversation History**: Persistent across sessions
- **Context Retention**: Remembers previous queries and datasets
- **Smart Suggestions**: Based on conversation and data characteristics

## 📊 Sample Queries

Try these example queries with the demo sample data:

```
Sales Performance Dataset:
• "What are our top performing regions by revenue?"
• "Show me sales trends by season"
• "Which sales reps have the highest total revenue?"
• "What's the average profit margin by product?"

Customer Demographics Dataset:
• "What's the age distribution of our customers?"
• "How does income correlate with lifetime value?"
• "Which customer segments have highest satisfaction scores?"
• "Show me customers by subscription type"

Financial Performance Dataset:
• "Which departments exceed their budgets most often?"
• "Show budget variance trends by year"
• "What's the efficiency score by department?"
• "Compare actual vs budgeted spending"

Website Analytics Dataset:
• "Which pages have the highest conversion rates?"
• "Show traffic patterns by day of week"
• "What's the bounce rate by traffic source?"
• "Display page views over time"
```

## 🧪 Testing

Run the complete test suite:
```bash
python -m pytest tests/ -v
```

Core tests included:
- MCP server functionality
- Agent query processing
- ChromaDB integration
- Export system validation

## 🛠️ Development

### Environment Setup
```bash
pip install -r requirements-dev.txt
```

### Running Development Mode
```bash
streamlit run ui/streamlit_app.py --server.runOnSave true
```

## 📖 Documentation

- **UI Guide**: `ui/README.md`
- **Agent Documentation**: `agent/README.md`
- **MCP Protocol**: `mcp_server/README.md`
- **API Reference**: `docs/api.md`