"""
File reading utilities for the language detection module.
"""

import pandas as pd
from typing import Union
import logging

class FileReader:
    """
    Handles reading of CSV and Excel files.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV or Excel file into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            pd.DataFrame: The loaded data
            
        Raises:
            ValueError: If file format is not supported
        """
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise