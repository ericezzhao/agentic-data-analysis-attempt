# 📊 Agentic AI Data Analysis - Streamlit UI

A modern, user-friendly web interface for the Agentic AI Data Analysis system. Upload your datasets and ask questions in natural language to get instant insights, statistics, and visualizations.

## 🚀 Quick Start

### 1. Launch the Application

```bash
# From the agentic_ai_analysis directory
python run_streamlit.py
```

The app will open in your browser at: `http://localhost:8501`

### 2. Configure API Key

- Enter your OpenAI API key in the sidebar
- The key is required for the AI agent to process your queries

### 3. Upload Your Dataset

- Click "Choose a CSV or Excel file" 
- Select your data file (supports `.csv`, `.xlsx`, `.xls`)
- Click "🚀 Load Dataset" to process the file

### 4. Start Analyzing

- Once loaded, you'll see a preview of your data
- Use the query interface to ask questions in natural language
- Examples:
  - "What is the average salary by department?"
  - "Show me employees in New York"
  - "Create a visualization of age distribution"
  - "Who has the highest performance rating?"

## ✨ Features

### 📁 **File Upload & Preview**
- Drag-and-drop or click to upload CSV/Excel files
- Automatic data preview with statistics
- Column information and data types

### 💬 **Natural Language Queries**
- Ask questions in plain English
- No SQL or technical knowledge required
- Supports statistical analysis, filtering, and aggregations

### 📈 **Smart Visualizations**
- Automatic chart generation based on queries
- Histograms, scatter plots, bar charts
- Interactive charts with zoom and pan

### 🔍 **Semantic Search**
- Find relevant data rows using natural language
- "Find wealthy customers in tech industry"
- "Show me high-performing sales people"

### 📜 **Query History**
- Track all your questions and results
- Easily re-run previous queries
- Export query results

## 🎯 Example Queries

### Statistical Analysis
```
- "What is the average salary?"
- "Show me the range of ages"
- "Calculate total revenue by quarter"
- "What's the median income by department?"
```

### Data Filtering & Search
```
- "Find employees in California"
- "Show me customers with income > 50k"
- "List all products with rating > 4.5"
- "Who joined the company in 2023?"
```

### Visualizations
```
- "Create a histogram of salaries"
- "Plot age vs salary scatter chart"
- "Show distribution of departments"
- "Visualize sales trends over time"
```

### Business Intelligence
```
- "Who is the highest paid employee?"
- "Which department has the best performance?"
- "Find our top 10 customers by revenue"
- "What's our customer retention rate?"
```

## 🛠️ Technical Requirements

### System Requirements
- Python 3.9+
- OpenAI API key
- 4GB+ RAM (for large datasets)

### Dependencies
- Streamlit 1.45+
- Pandas 2.0+
- OpenAI Python client
- ChromaDB for vector operations

### Supported File Formats
- **CSV**: Comma-separated values with headers
- **Excel**: .xlsx and .xls formats
- **File Size**: Up to 100MB recommended

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set in .env file
OPENAI_API_KEY=your_openai_api_key_here
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Streamlit Configuration
The app uses these default settings:
- Host: `localhost`
- Port: `8501`
- Browser stats collection: Disabled

## 📊 Data Processing

### Automatic Data Cleaning
- Handles missing values gracefully
- Detects and converts data types
- Processes date/time columns automatically

### Performance Optimization
- Caches datasets for faster querying
- Efficient vector embeddings for semantic search
- Progressive loading for large files

### Privacy & Security
- Data processed locally (not sent to external services except OpenAI)
- API keys stored in session only
- No persistent data storage

## 🐛 Troubleshooting

### Common Issues

**"No module named 'streamlit'"**
```bash
pip install streamlit pandas
```

**"OpenAI API key required"**
- Get your API key from: https://platform.openai.com/api-keys
- Enter it in the sidebar of the app

**"Dataset failed to load"**
- Check file format (CSV/Excel only)
- Ensure file has column headers
- Try a smaller file first

**"Query returned no results"**
- Try rephrasing your question
- Check if the dataset contains relevant data
- Use more specific column names

### Getting Help
1. Check the query history for successful examples
2. Try simpler queries first
3. Verify your dataset has the expected columns
4. Make sure your OpenAI API key has credits

## 🚀 Advanced Usage

### Custom Queries
The AI agent supports complex queries combining multiple operations:
```
"Show me employees in engineering with salary > 80k, 
 sorted by performance rating, and create a visualization"
```

### Batch Analysis
- Upload multiple related datasets
- Cross-reference data between files
- Generate comprehensive reports

### Export Results
- Query results can be copied
- Visualizations are displayed inline
- History tracking for reproducibility

---

**Built with ❤️ using Streamlit, OpenAI, and ChromaDB** 