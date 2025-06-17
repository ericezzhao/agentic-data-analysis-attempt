"""
File utility functions for data analysis project.

This module provides utilities for file validation, type checking,
and basic file information extraction.
"""

import os
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_file_type(file_path: Union[str, Path], allowed_types: Optional[List[str]] = None) -> bool:
    """
    Validate if a file type is allowed.
    
    Args:
        file_path: Path to the file
        allowed_types: List of allowed file extensions (defaults to csv, xlsx, xls)
        
    Returns:
        True if file type is allowed, False otherwise
    """
    if allowed_types is None:
        allowed_types = ['csv', 'xlsx', 'xls']
    
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower().lstrip('.')
    
    return file_extension in allowed_types


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Union[str, int, float]]:
    """
    Get basic information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = file_path.stat()
    
    return {
        "name": file_path.name,
        "extension": file_path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified_time": stat.st_mtime,
        "mime_type": mimetypes.guess_type(str(file_path))[0],
        "is_readable": os.access(file_path, os.R_OK)
    }


def estimate_file_rows(file_path: Union[str, Path]) -> int:
    """
    Estimate the number of rows in a CSV/Excel file without loading it entirely.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Estimated number of rows
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.csv':
            # Quick row count for CSV
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f) - 1  # Subtract header
        elif file_extension in ['.xlsx', '.xls']:
            # For Excel files, we need to load to get accurate count
            df = pd.read_excel(file_path, nrows=1)  # Just to check if readable
            df_full = pd.read_excel(file_path)
            return len(df_full)
        else:
            return 0
    except Exception as e:
        logger.warning(f"Could not estimate rows for {file_path}: {e}")
        return 0


def validate_file_size(file_path: Union[str, Path], max_size_mb: int = 100) -> bool:
    """
    Validate if file size is within allowed limits.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if file size is acceptable, False otherwise
    """
    file_info = get_file_info(file_path)
    return file_info["size_mb"] <= max_size_mb


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = (
        df_clean.columns
        .str.strip()  # Remove leading/trailing whitespace
        .str.replace(r'[^\w\s]', '_', regex=True)  # Replace special chars with underscore
        .str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscore
        .str.lower()  # Convert to lowercase
    )
    
    # Handle duplicate column names
    cols = df_clean.columns.tolist()
    seen = set()
    new_cols = []
    
    for col in cols:
        if col in seen:
            counter = 1
            new_col = f"{col}_{counter}"
            while new_col in seen:
                counter += 1
                new_col = f"{col}_{counter}"
            new_cols.append(new_col)
            seen.add(new_col)
        else:
            new_cols.append(col)
            seen.add(col)
    
    df_clean.columns = new_cols
    return df_clean


def detect_encoding(file_path: Union[str, Path]) -> str:
    """
    Detect the encoding of a text file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding string
    """
    try:
        import chardet
        
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    except ImportError:
        logger.warning("chardet not available, defaulting to utf-8")
        return 'utf-8'
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}, defaulting to utf-8")
        return 'utf-8'


def safe_read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Safely read a CSV file with encoding detection and error handling.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame loaded from CSV
    """
    file_path = Path(file_path)
    
    # Try to detect encoding if not specified
    if 'encoding' not in kwargs:
        kwargs['encoding'] = detect_encoding(file_path)
    
    # Common encodings to try
    encodings = [kwargs.get('encoding', 'utf-8'), 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            kwargs['encoding'] = encoding
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully read CSV with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading CSV with encoding {encoding}: {e}")
            if encoding == encodings[-1]:  # Last encoding attempt
                raise
    
    raise ValueError(f"Unable to read CSV file {file_path} with any supported encoding")


def preview_file_data(file_path: Union[str, Path], n_rows: int = 5) -> Dict[str, Union[pd.DataFrame, str]]:
    """
    Preview the first few rows of a data file.
    
    Args:
        file_path: Path to the file
        n_rows: Number of rows to preview
        
    Returns:
        Dictionary with preview data and information
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = safe_read_csv(file_path, nrows=n_rows)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=n_rows)
        else:
            return {"error": f"Unsupported file type: {file_extension}"}
        
        return {
            "success": True,
            "preview": df,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 