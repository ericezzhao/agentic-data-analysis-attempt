"""
Export Manager for Task 5.3 - Advanced Result Visualization and Export Features

Handles comprehensive export capabilities including:
- Multi-format data exports (CSV, Excel, JSON, PDF)
- Chart exports with high-quality rendering
- Interactive visualization options
- Batch export capabilities
- Report generation
"""

import io
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import zipfile
import tempfile
import os
from pathlib import Path

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ExportManager:
    """Advanced export management for data analysis results."""
    
    def __init__(self):
        self.supported_formats = {
            'csv': 'Comma-Separated Values',
            'excel': 'Microsoft Excel',
            'json': 'JSON Format',
            'pdf': 'PDF Report' if REPORTLAB_AVAILABLE else 'PDF (Not Available)',
            'png': 'PNG Image',
            'svg': 'SVG Vector',
            'zip': 'Complete Package'
        }
    
    def export_data_results(self, results: List[Dict[str, Any]], format_type: str = 'csv') -> bytes:
        """Export data results in specified format."""
        if format_type == 'csv':
            return self._export_to_csv(results)
        elif format_type == 'excel':
            return self._export_to_excel(results)
        elif format_type == 'json':
            return self._export_to_json(results)
        elif format_type == 'pdf' and REPORTLAB_AVAILABLE:
            return self._export_to_pdf(results)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_to_csv(self, results: List[Dict[str, Any]]) -> bytes:
        """Export results to CSV format."""
        # Extract data rows and create DataFrame
        data_rows = []
        for result in results:
            if result.get('type') == 'data_row':
                full_data = result.get('full_data', {})
                if full_data:
                    data_rows.append(full_data)
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue().encode('utf-8')
        else:
            # Export text results
            text_results = [result.get('content', '') for result in results]
            csv_data = '\n'.join(text_results)
            return csv_data.encode('utf-8')
    
    def _export_to_excel(self, results: List[Dict[str, Any]]) -> bytes:
        """Export results to Excel format with multiple sheets."""
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Sheet 1: Data Results
            data_rows = []
            for result in results:
                if result.get('type') == 'data_row':
                    full_data = result.get('full_data', {})
                    if full_data:
                        data_rows.append(full_data)
            
            if data_rows:
                df_data = pd.DataFrame(data_rows)
                df_data.to_excel(writer, sheet_name='Data Results', index=False)
            
            # Sheet 2: Analysis Summary
            summary_data = []
            for i, result in enumerate(results, 1):
                summary_data.append({
                    'Result #': i,
                    'Type': result.get('type', 'unknown'),
                    'Content': result.get('content', '')[:100] + '...' if len(result.get('content', '')) > 100 else result.get('content', ''),
                    'Has Chart': 'Yes' if result.get('chart_data') else 'No'
                })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Analysis Summary', index=False)
        
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    
    def _export_to_json(self, results: List[Dict[str, Any]]) -> bytes:
        """Export results to JSON format."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'results': results
        }
        
        # Remove binary chart data for JSON export (too large)
        for result in export_data['results']:
            if 'chart_data' in result:
                result['chart_data'] = '<base64_image_data_removed>'
        
        return json.dumps(export_data, indent=2, ensure_ascii=False).encode('utf-8')
    
    def _export_to_pdf(self, results: List[Dict[str, Any]]) -> bytes:
        """Export results to PDF report format."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF export")
        
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("Data Analysis Report", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Results
        for i, result in enumerate(results, 1):
            # Result header
            story.append(Paragraph(f"Result {i}: {result.get('type', 'Unknown').title()}", styles['Heading2']))
            
            # Content
            content = result.get('content', '')
            if content:
                story.append(Paragraph(content, styles['Normal']))
            
            # Chart (if available)
            if result.get('chart_data'):
                try:
                    chart_bytes = base64.b64decode(result['chart_data'])
                    chart_image = Image(io.BytesIO(chart_bytes))
                    chart_image.drawHeight = 3 * inch
                    chart_image.drawWidth = 4 * inch
                    story.append(chart_image)
                except Exception:
                    story.append(Paragraph("Chart could not be embedded", styles['Italic']))
            
            story.append(Spacer(1, 20))
        
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    
    def export_visualization(self, chart_data: str, format_type: str = 'png') -> bytes:
        """Export individual visualization in specified format."""
        if not chart_data:
            raise ValueError("No chart data provided")
        
        try:
            image_bytes = base64.b64decode(chart_data)
            
            if format_type == 'png':
                return image_bytes
            elif format_type == 'svg':
                # For SVG, we'd need to regenerate the chart
                # This is a simplified implementation
                return image_bytes
            else:
                raise ValueError(f"Unsupported visualization format: {format_type}")
        
        except Exception as e:
            raise ValueError(f"Error processing chart data: {e}")
    
    def create_comprehensive_export(self, 
                                  query: str,
                                  results: List[Dict[str, Any]], 
                                  insights: List[str],
                                  dataset_name: str = "dataset") -> bytes:
        """Create a comprehensive ZIP export with all formats."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export data in multiple formats
            try:
                csv_data = self._export_to_csv(results)
                zip_file.writestr(f"{dataset_name}_results_{timestamp}.csv", csv_data)
            except Exception as e:
                print(f"CSV export failed: {e}")
            
            try:
                excel_data = self._export_to_excel(results)
                zip_file.writestr(f"{dataset_name}_results_{timestamp}.xlsx", excel_data)
            except Exception as e:
                print(f"Excel export failed: {e}")
            
            try:
                json_data = self._export_to_json(results)
                zip_file.writestr(f"{dataset_name}_results_{timestamp}.json", json_data)
            except Exception as e:
                print(f"JSON export failed: {e}")
            
            # Export individual charts
            chart_count = 0
            for i, result in enumerate(results):
                if result.get('chart_data'):
                    try:
                        chart_bytes = self.export_visualization(result['chart_data'], 'png')
                        zip_file.writestr(f"chart_{i+1}_{timestamp}.png", chart_bytes)
                        chart_count += 1
                    except Exception as e:
                        print(f"Chart export failed for result {i+1}: {e}")
            
            # Export summary report
            summary_content = self._create_summary_report(query, results, insights, chart_count)
            zip_file.writestr(f"{dataset_name}_summary_{timestamp}.txt", summary_content.encode('utf-8'))
            
            # Export PDF if available
            if REPORTLAB_AVAILABLE:
                try:
                    pdf_data = self._export_to_pdf(results)
                    zip_file.writestr(f"{dataset_name}_report_{timestamp}.pdf", pdf_data)
                except Exception as e:
                    print(f"PDF export failed: {e}")
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def _create_summary_report(self, query: str, results: List[Dict[str, Any]], 
                             insights: List[str], chart_count: int) -> str:
        """Create a text summary report."""
        report_lines = [
            "=" * 60,
            "DATA ANALYSIS SUMMARY REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Query: {query}",
            "",
            "INSIGHTS:",
            "-" * 20
        ]
        
        for i, insight in enumerate(insights, 1):
            report_lines.append(f"{i}. {insight}")
        
        report_lines.extend([
            "",
            "RESULTS SUMMARY:",
            "-" * 20,
            f"Total Results: {len(results)}",
            f"Visualizations: {chart_count}",
            ""
        ])
        
        # Add detailed results
        for i, result in enumerate(results, 1):
            result_type = result.get('type', 'unknown')
            content = result.get('content', '')
            
            report_lines.extend([
                f"Result {i} ({result_type.upper()}):",
                content,
                ""
            ])
        
        report_lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])
        
        return '\n'.join(report_lines)
    
    def get_export_options(self) -> Dict[str, Dict[str, str]]:
        """Get available export options with descriptions."""
        options = {}
        
        for format_type, description in self.supported_formats.items():
            if format_type == 'pdf' and not REPORTLAB_AVAILABLE:
                continue
            
            options[format_type] = {
                'name': description,
                'extension': self._get_file_extension(format_type),
                'mime_type': self._get_mime_type(format_type)
            }
        
        return options
    
    def _get_file_extension(self, format_type: str) -> str:
        """Get file extension for format type."""
        extensions = {
            'csv': '.csv',
            'excel': '.xlsx', 
            'json': '.json',
            'pdf': '.pdf',
            'png': '.png',
            'svg': '.svg',
            'zip': '.zip'
        }
        return extensions.get(format_type, '.txt')
    
    def _get_mime_type(self, format_type: str) -> str:
        """Get MIME type for format type."""
        mime_types = {
            'csv': 'text/csv',
            'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'json': 'application/json',
            'pdf': 'application/pdf',
            'png': 'image/png',
            'svg': 'image/svg+xml',
            'zip': 'application/zip'
        }
        return mime_types.get(format_type, 'text/plain')


class InteractiveVisualizer:
    """Enhanced interactive visualization capabilities."""
    
    def __init__(self):
        self.chart_themes = {
            'default': 'whitegrid',
            'professional': 'whitegrid',
            'minimal': 'white',
            'dark': 'darkgrid'
        }
    
    def create_interactive_chart_options(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate interactive chart configuration options."""
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        chart_options = {
            'chart_types': [
                'histogram', 'scatter', 'bar', 'line', 'box', 'violin', 'heatmap'
            ],
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'themes': list(self.chart_themes.keys()),
            'color_palettes': ['Set1', 'Set2', 'viridis', 'plasma', 'coolwarm']
        }
        
        # Suggest appropriate charts based on data
        suggested_charts = []
        if len(numeric_cols) >= 1:
            suggested_charts.extend(['histogram', 'box'])
        if len(numeric_cols) >= 2:
            suggested_charts.extend(['scatter', 'heatmap'])
        if len(categorical_cols) >= 1:
            suggested_charts.append('bar')
        
        chart_options['suggested_charts'] = suggested_charts
        
        return chart_options
    
    def generate_custom_visualization(self, 
                                    data: pd.DataFrame,
                                    chart_type: str,
                                    x_column: str = None,
                                    y_column: str = None,
                                    color_column: str = None,
                                    theme: str = 'default',
                                    color_palette: str = 'Set1') -> str:
        """Generate custom visualization based on user parameters."""
        
        # Set style
        sns.set_style(self.chart_themes.get(theme, 'default'))
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        try:
            if chart_type == 'histogram' and x_column:
                sns.histplot(data=data, x=x_column, kde=True)
                plt.title(f'Distribution of {x_column}')
                
            elif chart_type == 'scatter' and x_column and y_column:
                sns.scatterplot(data=data, x=x_column, y=y_column, hue=color_column, palette=color_palette)
                plt.title(f'{y_column} vs {x_column}')
                
            elif chart_type == 'bar' and x_column:
                if data[x_column].dtype == 'object':
                    value_counts = data[x_column].value_counts()
                    sns.barplot(x=value_counts.index, y=value_counts.values, palette=color_palette)
                    plt.title(f'Count of {x_column}')
                    plt.xticks(rotation=45)
                else:
                    sns.histplot(data=data, x=x_column, kde=False)
                    plt.title(f'Distribution of {x_column}')
                    
            elif chart_type == 'box' and x_column:
                if color_column:
                    sns.boxplot(data=data, x=color_column, y=x_column, palette=color_palette)
                    plt.title(f'{x_column} by {color_column}')
                else:
                    sns.boxplot(data=data, y=x_column)
                    plt.title(f'{x_column} Distribution')
                    
            elif chart_type == 'heatmap':
                numeric_data = data.select_dtypes(include=['number'])
                if len(numeric_data.columns) > 1:
                    correlation = numeric_data.corr()
                    sns.heatmap(correlation, annot=True, cmap=color_palette, center=0)
                    plt.title('Correlation Heatmap')
                else:
                    plt.text(0.5, 0.5, 'Need at least 2 numeric columns for correlation', 
                            ha='center', va='center', transform=plt.gca().transAxes)
            
            else:
                plt.text(0.5, 0.5, f'Chart type "{chart_type}" not supported or missing parameters', 
                        ha='center', va='center', transform=plt.gca().transAxes)
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            chart_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            plt.close()
            # Create error visualization
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, f'Error creating visualization:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            plt.axis('off')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            error_chart = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return error_chart 