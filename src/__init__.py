"""
Credit Risk Assessment Python Package
"""

__version__ = "1.0.0"
__author__ = "Banking Analytics Team"
__description__ = "Predictive Analytics-Based Decision Support Framework for Credit Risk Assessment"

from .config import *
from .preprocessing import DataPreprocessor

__all__ = ['DataPreprocessor', 'RAW_DATA_PATH', 'PROCESSED_DATA_PATH']
