"""Statistics Engine for Data Analysis Agent.

Provides fast statistical calculations over cached pandas DataFrames.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StatisticsEngine:
    """Compute basic statistics over cached datasets."""

    def __init__(self, dataset_cache: Dict[str, pd.DataFrame]):
        """Initialize with reference to agent's dataset cache."""
        self._cache = dataset_cache
        
        # Map of supported aliases to canonical function names
        self._stat_aliases = {
            "mean": "mean",
            "average": "mean", 
            "avg": "mean",
            "median": "median",
            "sum": "sum",
            "total": "sum",
            "min": "min",
            "max": "max",
            "count": "count",
            "std": "std",
            "variance": "var",
            "var": "var"
        }

    def compute(self, dataset_name: str, column: str, stat: str) -> Tuple[Optional[float], Optional[str]]:
        """Compute a statistic for a column in a dataset.
        
        Args:
            dataset_name: Name of the dataset in cache
            column: Column name to compute statistic for
            stat: Statistic type (mean, sum, etc.)
            
        Returns:
            Tuple of (value, error_message). If successful, error_message is None.
        """
        try:
            # Check if dataset exists in cache
            if dataset_name not in self._cache:
                return None, f"Dataset '{dataset_name}' not found in cache"
            
            df = self._cache[dataset_name]
            
            # Check if column exists
            if column not in df.columns:
                available_cols = ", ".join(df.columns)
                return None, f"Column '{column}' not found. Available columns: {available_cols}"
            
            # Check if statistic is supported
            canonical_stat = self._stat_aliases.get(stat.lower())
            if not canonical_stat:
                supported_stats = ", ".join(self._stat_aliases.keys())
                return None, f"Statistic '{stat}' not supported. Available: {supported_stats}"
            
            # Get the column data
            column_data = df[column]
            
            # Apply the statistic
            if canonical_stat == "mean":
                result = column_data.mean()
            elif canonical_stat == "median":
                result = column_data.median()
            elif canonical_stat == "sum":
                result = column_data.sum()
            elif canonical_stat == "min":
                result = column_data.min()
            elif canonical_stat == "max":
                result = column_data.max()
            elif canonical_stat == "count":
                result = column_data.count()  # Excludes NaN values
            elif canonical_stat == "std":
                result = column_data.std()
            elif canonical_stat == "var":
                result = column_data.var()
            else:
                return None, f"Internal error: unknown canonical stat '{canonical_stat}'"
            
            # Handle NaN results
            if pd.isna(result):
                return None, f"Result is NaN - column '{column}' may contain non-numeric data"
            
            logger.info(f"Computed {stat} of '{column}' in '{dataset_name}': {result}")
            return float(result), None
            
        except Exception as e:
            logger.error(f"Error computing {stat} of {column}: {e}")
            return None, f"Error computing statistic: {str(e)}"

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get basic information about a cached dataset."""
        if dataset_name not in self._cache:
            return {"error": f"Dataset '{dataset_name}' not found"}
        
        df = self._cache[dataset_name]
        return {
            "name": dataset_name,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns)
        }
    
    def list_cached_datasets(self) -> list:
        """List all datasets currently in cache."""
        return list(self._cache.keys()) 