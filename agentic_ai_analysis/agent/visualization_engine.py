"""Visualization Engine for Data Analysis Agent.

Provides chart generation and visualization suggestions based on data queries.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
import io
import base64
import numpy as np

logger = logging.getLogger(__name__)

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

class VisualizationEngine:
    """Generate visualizations and chart suggestions for data analysis."""
    
    def __init__(self, dataset_cache: Dict[str, pd.DataFrame]):
        """Initialize with reference to agent's dataset cache."""
        self._cache = dataset_cache
        
        # Chart type mappings based on data patterns
        self._chart_mappings = {
            "histogram": {"single_numeric": True, "description": "Distribution of values"},
            "bar": {"categorical_vs_numeric": True, "description": "Compare categories"},
            "scatter": {"two_numeric": True, "description": "Relationship between variables"},
            "line": {"time_series": True, "description": "Trends over time"},
            "pie": {"categorical_count": True, "description": "Proportions of categories"},
            "box": {"categorical_vs_numeric": True, "description": "Distribution by category"}
        }
        
        # Visualization keywords for natural language detection
        self._viz_keywords = {
            "show", "plot", "chart", "graph", "visualize", "display",
            "histogram", "bar", "scatter", "line", "pie", "distribution"
        }
    
    def suggest_visualizations(self, dataset_name: str, columns: List[str] = None) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations for given dataset/columns."""
        if dataset_name not in self._cache:
            return [{"error": f"Dataset '{dataset_name}' not found"}]
        
        df = self._cache[dataset_name]
        target_columns = columns or list(df.columns)
        suggestions = []
        
        # Analyze each column for visualization opportunities
        for col in target_columns:
            if col not in df.columns:
                continue
                
            col_data = df[col]
            col_type = str(col_data.dtype)
            
            # Single column visualizations
            if pd.api.types.is_numeric_dtype(col_data):
                suggestions.append({
                    "chart_type": "histogram",
                    "columns": [col],
                    "title": f"Distribution of {col}",
                    "description": f"Shows the frequency distribution of {col} values"
                })
                
                suggestions.append({
                    "chart_type": "box",
                    "columns": [col],
                    "title": f"Box Plot of {col}",
                    "description": f"Shows quartiles and outliers for {col}"
                })
            
            elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                suggestions.append({
                    "chart_type": "bar",
                    "columns": [col],
                    "title": f"Count of {col} Categories",
                    "description": f"Shows frequency of each {col} category"
                })
                
                suggestions.append({
                    "chart_type": "pie",
                    "columns": [col],
                    "title": f"Proportion of {col} Categories",
                    "description": f"Shows percentage breakdown of {col} categories"
                })
        
        # Two-column relationships
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric vs Numeric (scatter plots)
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                suggestions.append({
                    "chart_type": "scatter",
                    "columns": [col1, col2],
                    "title": f"{col1} vs {col2}",
                    "description": f"Shows relationship between {col1} and {col2}"
                })
        
        # Categorical vs Numeric (grouped bar/box plots)
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                suggestions.append({
                    "chart_type": "box",
                    "columns": [cat_col, num_col],
                    "title": f"{num_col} by {cat_col}",
                    "description": f"Shows {num_col} distribution across {cat_col} categories"
                })
        
        return suggestions[:8]  # Limit to top 8 suggestions
    
    def create_visualization(self, dataset_name: str, chart_type: str, columns: List[str], 
                           title: str = None, **kwargs) -> Dict[str, Any]:
        """Create a visualization and return it as base64 encoded image."""
        try:
            if dataset_name not in self._cache:
                return {"success": False, "error": f"Dataset '{dataset_name}' not found"}
            
            df = self._cache[dataset_name]
            
            # Validate columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {"success": False, "error": f"Columns not found: {missing_cols}"}
            
            # Set up the plot
            plt.figure(figsize=(10, 6))
            
            # Generate chart based on type
            if chart_type == "histogram":
                return self._create_histogram(df, columns[0], title, **kwargs)
            elif chart_type == "bar":
                return self._create_bar_chart(df, columns[0], title, **kwargs)
            elif chart_type == "scatter":
                if len(columns) < 2:
                    return {"success": False, "error": "Scatter plot requires 2 columns"}
                return self._create_scatter_plot(df, columns[0], columns[1], title, **kwargs)
            elif chart_type == "line":
                return self._create_line_chart(df, columns, title, **kwargs)
            elif chart_type == "pie":
                return self._create_pie_chart(df, columns[0], title, **kwargs)
            elif chart_type == "box":
                return self._create_box_plot(df, columns, title, **kwargs)
            else:
                return {"success": False, "error": f"Unsupported chart type: {chart_type}"}
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return {"success": False, "error": f"Visualization failed: {str(e)}"}
    
    def _create_histogram(self, df: pd.DataFrame, column: str, title: str = None, **kwargs) -> Dict[str, Any]:
        """Create histogram for numeric column."""
        data = df[column].dropna()
        
        plt.hist(data, bins=kwargs.get('bins', 20), alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(title or f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        return self._save_plot_as_base64()
    
    def _create_bar_chart(self, df: pd.DataFrame, column: str, title: str = None, **kwargs) -> Dict[str, Any]:
        """Create bar chart for categorical column."""
        value_counts = df[column].value_counts().head(10)  # Top 10 categories
        
        bars = plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.7)
        plt.title(title or f"Count of {column} Categories")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, value_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(value_counts),
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        return self._save_plot_as_base64()
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, title: str = None, **kwargs) -> Dict[str, Any]:
        """Create scatter plot for two numeric columns."""
        clean_data = df[[x_col, y_col]].dropna()
        
        plt.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, color='green')
        plt.title(title or f"{x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, alpha=0.3)
        
        # Add trend line if requested
        if kwargs.get('trend_line', True):
            z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
            p = np.poly1d(z)
            plt.plot(clean_data[x_col], p(clean_data[x_col]), "r--", alpha=0.8, linewidth=1)
        
        return self._save_plot_as_base64()
    
    def _create_line_chart(self, df: pd.DataFrame, columns: List[str], title: str = None, **kwargs) -> Dict[str, Any]:
        """Create line chart for time series or sequential data."""
        for i, col in enumerate(columns):
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                plt.plot(df.index, df[col], label=col, marker='o', markersize=3)
        
        plt.title(title or f"Line Chart: {', '.join(columns)}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return self._save_plot_as_base64()
    
    def _create_pie_chart(self, df: pd.DataFrame, column: str, title: str = None, **kwargs) -> Dict[str, Any]:
        """Create pie chart for categorical column."""
        value_counts = df[column].value_counts().head(8)  # Top 8 categories
        
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(title or f"Distribution of {column}")
        plt.axis('equal')
        
        return self._save_plot_as_base64()
    
    def _create_box_plot(self, df: pd.DataFrame, columns: List[str], title: str = None, **kwargs) -> Dict[str, Any]:
        """Create box plot for numeric data, optionally grouped by category."""
        if len(columns) == 1:
            # Single numeric column
            data = df[columns[0]].dropna()
            plt.boxplot([data], labels=[columns[0]])
        elif len(columns) == 2:
            # Categorical vs Numeric
            cat_col, num_col = columns
            if pd.api.types.is_numeric_dtype(df[cat_col]):
                cat_col, num_col = num_col, cat_col  # Swap if needed
            
            categories = df[cat_col].unique()[:10]  # Max 10 categories
            data_by_category = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
            
            plt.boxplot(data_by_category, labels=categories)
            plt.xticks(rotation=45, ha='right')
        
        plt.title(title or f"Box Plot: {', '.join(columns)}")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return self._save_plot_as_base64()
    
    def _save_plot_as_base64(self) -> Dict[str, Any]:
        """Save current matplotlib plot as base64 encoded string."""
        try:
            # Save plot to BytesIO object
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Encode as base64
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Clear the current plot
            plt.clf()
            plt.close()
            
            return {
                "success": True,
                "image_base64": img_base64,
                "format": "png"
            }
        except Exception as e:
            plt.clf()  # Clean up even on error
            plt.close()
            logger.error(f"Error saving plot: {e}")
            return {"success": False, "error": f"Failed to save plot: {str(e)}"}
    
    def detect_visualization_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect if query is asking for visualization and parse intent."""
        query_lower = query.lower().strip()
        
        # Check for visualization keywords
        has_viz_keyword = any(keyword in query_lower for keyword in self._viz_keywords)
        if not has_viz_keyword:
            return None
        
        # Parse chart type
        chart_type = None
        for chart in self._chart_mappings.keys():
            if chart in query_lower:
                chart_type = chart
                break
        
        # Parse column names (simple approach - look for words that might be column names)
        words = query_lower.split()
        potential_columns = [word.strip('().,!?') for word in words 
                           if word not in self._viz_keywords and len(word) > 2]
        
        return {
            "is_visualization": True,
            "chart_type": chart_type,
            "potential_columns": potential_columns,
            "original_query": query
        } 