"""
Main entry point for Credit Risk Assessment Project
Executes the complete data preprocessing pipeline
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor
from config import RAW_DATA_PATH, DATA_PROCESSED

def main():
    """Execute the complete preprocessing pipeline"""
    
    # Create preprocessor instance
    preprocessor = DataPreprocessor()
    
    # Run full pipeline
    try:
        processed_data = preprocessor.run_full_pipeline(RAW_DATA_PATH)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY")
        print("=" * 80)
        print(f"\nInput file: {RAW_DATA_PATH}")
        print(f"Output directory: {DATA_PROCESSED}")
        print(f"\nFinal dataset shape: {processed_data.shape}")
        print(f"Features: {list(processed_data.columns)}")
        
        return processed_data
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
