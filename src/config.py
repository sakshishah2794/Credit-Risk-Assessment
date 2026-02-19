"""
Configuration and constants for Credit Risk Assessment Project
"""

import os

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, '..', 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, '..', 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, '..', 'results')

# Dataset paths
RAW_DATA_PATH = os.path.join(DATA_RAW, 'LC_loans_granting_model_dataset.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_PROCESSED, 'preprocessed_data.csv')
PROCESSED_DATA_TRAIN = os.path.join(DATA_PROCESSED, 'train_data.csv')
PROCESSED_DATA_TEST = os.path.join(DATA_PROCESSED, 'test_data.csv')

# Target variable
TARGET_COLUMN = 'Default'
BINARY_MAPPING = {
    'Charged Off': 1,
    'Default': 1,
    'Fully Paid': 0,
    1: 1,
    0: 0
}

# Feature categories
NUMERIC_FEATURES = [
    'revenue',
    'dti_n',
    'loan_amnt',
    'fico_n',
    'experience_c'
]

CATEGORICAL_FEATURES = [
    'emp_length',
    'purpose',
    'home_ownership_n',
    'addr_state'
]

# Columns to drop (irrelevant/high missing values)
DROP_COLUMNS = [
    'id',                    # Just an identifier
    'zip_code',             # Too sparse, similar to addr_state
    'title',                # High missing values and redundant with purpose
    'desc',                 # Very high missing values (90%+)
    'issue_d'               # Date feature - can be handled separately if needed
]

# SMOTE parameters
SMOTE_RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5

# Feature scaling
SCALER_TYPE = 'StandardScaler'  # Options: 'StandardScaler', 'MinMaxScaler'

# Random seeds
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Data quality thresholds
MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with > 50% missing values

# Categorical encoding
ENCODING_STRATEGY = 'label'  # Options: 'label', 'onehot'
