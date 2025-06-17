"""
Utility functions for file handling and data processing.
"""

import pandas as pd
import os
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class FileReader:
    """
    Handles reading CSV and Excel files.
    """
    
    def __init__(self):
        self.supported_extensions = {'.csv', '.xlsx', '.xls'}
    
    def read_file(self, file_path: str, sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Read a CSV or Excel file.
        
        Args:
            file_path (str): Path to the file
            sheet_name (str, optional): Sheet name for Excel files
            
        Returns:
            pd.DataFrame or None if error
        """
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_extensions:
            logger.error(f"Unsupported file extension: {file_ext}")
            return None
        
        try:
            if file_ext == '.csv':
                return self._read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return self._read_excel(file_path, sheet_name)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def _read_csv(self, file_path: str) -> pd.DataFrame:
        """Read CSV file with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully read CSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading CSV with {encoding}: {str(e)}")
                continue
        
        raise ValueError(f"Could not read CSV file with any of the attempted encodings: {encodings}")
    
    def _read_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Read Excel file."""
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        logger.info(f"Successfully read Excel file, sheet: {sheet_name or 'default'}")
        return df
    
    def get_excel_sheet_names(self, file_path: str) -> list:
        """Get list of sheet names in an Excel file."""
        try:
            xl = pd.ExcelFile(file_path)
            return xl.sheet_names
        except Exception as e:
            logger.error(f"Error getting sheet names: {str(e)}")
            return []

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_file_path(file_path: str) -> bool:
    """Validate if file path exists and has supported extension."""
    if not os.path.exists(file_path):
        return False
    
    supported_extensions = {'.csv', '.xlsx', '.xls'}
    file_ext = os.path.splitext(file_path)[1].lower()
    
    return file_ext in supported_extensions