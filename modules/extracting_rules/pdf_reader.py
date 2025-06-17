"""
PDF reading utilities for extracting text from ISP documents.
"""

import logging
from typing import Optional, List, Dict
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFReader:
    """
    Handles reading and text extraction from PDF files.
    """
    
    def __init__(self):
        """Initialize PDF reader with available libraries."""
        self.available_libraries = self._check_available_libraries()
        
    def _check_available_libraries(self) -> List[str]:
        """Check which PDF libraries are available."""
        available = []
        
        try:
            import pdfplumber
            available.append('pdfplumber')
        except ImportError:
            pass
            
        try:
            import PyPDF2
            available.append('PyPDF2')
        except ImportError:
            pass
            
        try:
            import pymupdf  # fitz
            available.append('pymupdf')
        except ImportError:
            pass
            
        if not available:
            logger.warning("No PDF libraries found. Please install: pip install pdfplumber PyPDF2 pymupdf")
            
        return available
    
    def extract_text(self, pdf_path: str, method: str = 'auto') -> Optional[str]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            method (str): Extraction method ('auto', 'pdfplumber', 'PyPDF2', 'pymupdf')
            
        Returns:
            str: Extracted text or None if failed
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
            
        if method == 'auto':
            # Try methods in order of preference
            for lib in ['pdfplumber', 'pymupdf', 'PyPDF2']:
                if lib in self.available_libraries:
                    try:
                        text = self._extract_with_method(pdf_path, lib)
                        if text and text.strip():
                            logger.info(f"Successfully extracted text using {lib}")
                            return text
                    except Exception as e:
                        logger.warning(f"Failed to extract with {lib}: {str(e)}")
                        continue
        else:
            if method in self.available_libraries:
                return self._extract_with_method(pdf_path, method)
            else:
                logger.error(f"Method {method} not available. Available: {self.available_libraries}")
                
        return None
    
    def _extract_with_method(self, pdf_path: str, method: str) -> Optional[str]:
        """Extract text using specific method."""
        
        if method == 'pdfplumber':
            return self._extract_with_pdfplumber(pdf_path)
        elif method == 'PyPDF2':
            return self._extract_with_pypdf2(pdf_path)
        elif method == 'pymupdf':
            return self._extract_with_pymupdf(pdf_path)
        else:
            logger.error(f"Unknown extraction method: {method}")
            return None
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Optional[str]:
        """Extract text using pdfplumber."""
        try:
            import pdfplumber
            
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                        
                        # Also try to extract tables
                        tables = page.extract_tables()
                        for table_num, table in enumerate(tables):
                            if table:
                                table_text = self._format_table(table, page_num + 1, table_num + 1)
                                text_content.append(table_text)
                                
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                        continue
                        
            return '\n\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            return None
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Optional[str]:
        """Extract text using PyPDF2."""
        try:
            import PyPDF2
            
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                        continue
                        
            return '\n\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            return None
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Optional[str]:
        """Extract text using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
            
            text_content = []
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    text = page.get_text()
                    if text:
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                        
                    # Also try to extract tables
                    tables = page.find_tables()
                    for table_num, table in enumerate(tables):
                        try:
                            table_data = table.extract()
                            if table_data:
                                table_text = self._format_table(table_data, page_num + 1, table_num + 1)
                                text_content.append(table_text)
                        except Exception as e:
                            logger.warning(f"Error extracting table {table_num + 1} on page {page_num + 1}: {str(e)}")
                            
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
                    
            pdf_document.close()
            return '\n\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {str(e)}")
            return None
    
    def _format_table(self, table_data: List[List], page_num: int, table_num: int) -> str:
        """Format table data as text."""
        if not table_data:
            return ""
            
        table_text = [f"--- Table {table_num} on Page {page_num} ---"]
        
        for row in table_data:
            if row:  # Skip empty rows
                # Clean and join cells
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                table_text.append(" | ".join(cleaned_row))
                
        return "\n".join(table_text)
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """Get basic information about the PDF."""
        if not os.path.exists(pdf_path):
            return {'error': 'File not found'}
            
        try:
            info = {
                'file_path': pdf_path,
                'file_size': os.path.getsize(pdf_path),
                'available_methods': self.available_libraries
            }
            
            # Try to get page count with available library
            if 'pdfplumber' in self.available_libraries:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    info['page_count'] = len(pdf.pages)
                    
            elif 'PyPDF2' in self.available_libraries:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    info['page_count'] = len(pdf_reader.pages)
                    
            elif 'pymupdf' in self.available_libraries:
                import fitz
                pdf_document = fitz.open(pdf_path)
                info['page_count'] = pdf_document.page_count
                pdf_document.close()
                
            return info
            
        except Exception as e:
            return {'error': str(e)}