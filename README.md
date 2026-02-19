# Credit Risk Assessment - Predictive Analytics Decision Support Framework

## Project Overview

This is a complete structured Python project for **credit risk prediction** using machine learning. The project implements a comprehensive data preprocessing pipeline for banking sector loan default prediction.

**Objective:** Predict loan default risk (Classification: Default vs. Fully Paid)

**Dataset:** LC_loans_granting_model_dataset.csv (1,347,681 records)

---

## Project Structure

```
Credit-Risk-Assessment/
├── data/
│   ├── raw/
│   │   └── LC_loans_granting_model_dataset.csv    # Original dataset
│   └── processed/
│       ├── preprocessed_data.csv                  # Full processed data
│       ├── train_data.csv                         # Training set
│       └── test_data.csv                          # Test set
├── src/
│   ├── __init__.py
│   ├── config.py                                  # Configuration & constants
│   └── preprocessing.py                           # Preprocessing pipeline
├── notebooks/
│   └── 01_data_preprocessing.ipynb               # Jupyter notebook
├── results/
│   └── (Model results, plots, reports)
├── main.py                                        # Entry point
├── requirements.txt                               # Dependencies
└── README.md                                      # This file
```

---

## Phase 1: Data Preprocessing

The preprocessing pipeline includes **8 key steps**:

### Step 1: Load Dataset

- Load CSV file using pandas
- Dataset Info: 1,347,681 rows × 15 columns

### Step 2: Display Dataset Structure

- Show shape, data types, and statistical summary
- Display target variable distribution

### Step 3: Clean Missing Values

- Identify variables with missing values
- Drop columns with high missing percentage (>50%)
- Forward-fill numeric columns with median
- Forward-fill categorical columns with mode

### Step 4: Convert Target Variable

- Convert `Default` column to binary classification
- **Target Encoding:**
  - 1 = Default / Charged Off
  - 0 = Fully Paid

### Step 5: Remove Irrelevant Columns

- Drop ID columns (no predictive value)
- Remove high-missing columns (>90%):
  - `zip_code` (redundant with state)
  - `title` (high missing + redundant)
  - `desc` (90%+ missing)
- Drop temporal columns if needed

### Step 6: Encode Categorical Variables

- Apply Label Encoding for categorical features
- Categorical features: `emp_length`, `purpose`, `home_ownership_n`, `addr_state`
- One-Hot Encoding available as alternative

### Step 7: Feature Scaling (Standardization)

- StandardScaler: Zero mean, unit variance
- MinMaxScaler: Range [0, 1] available
- Applied to all numeric features

### Step 8: Handle Class Imbalance (SMOTE)

- Apply SMOTE (Synthetic Minority Over-sampling Technique)
- Balance classes for better model training
- Parameters: k_neighbors=5, random_state=42

---

## Installation & Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing Pipeline

```bash
python main.py
```

---

## Dataset Features

### Numeric Features:

- `revenue` - Annual income
- `dti_n` - Debt-to-income ratio
- `loan_amnt` - Loan amount
- `fico_n` - Credit score (FICO)
- `experience_c` - Years of credit history

### Categorical Features:

- `emp_length` - Employment length
- `purpose` - Loan purpose
- `home_ownership_n` - Home ownership status
- `addr_state` - State of residence

### Target Variable:

- `Default` - Binary classification (0=Fully Paid, 1=Default)

---

## Configuration

Edit `src/config.py` to customize:

```python
# Feature selection
NUMERIC_FEATURES = ['revenue', 'dti_n', 'loan_amnt', 'fico_n', 'experience_c']
CATEGORICAL_FEATURES = ['emp_length', 'purpose', 'home_ownership_n', 'addr_state']

# Drop columns
DROP_COLUMNS = ['id', 'zip_code', 'title', 'desc', 'issue_d']

# SMOTE parameters
SMOTE_RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5

# Encoding strategy: 'label' or 'onehot'
ENCODING_STRATEGY = 'label'

# Scaler type: 'StandardScaler' or 'MinMaxScaler'
SCALER_TYPE = 'StandardScaler'

# Train-Test split ratio
TEST_SIZE = 0.2

# Missing value threshold (drop columns)
MISSING_VALUE_THRESHOLD = 0.5
```

---

## Output Files

After running the pipeline, the following files are generated:

1. **preprocessed_data.csv** - Complete processed dataset (balanced)
2. **train_data.csv** - Training set (80%)
3. **test_data.csv** - Test set (20%)

---

## Usage Example

### Using the Python Module

```python
from src.preprocessing import DataPreprocessor
from src.config import RAW_DATA_PATH

# Create preprocessor
preprocessor = DataPreprocessor()

# Run full pipeline
processed_data = preprocessor.run_full_pipeline(RAW_DATA_PATH)

# Access results
print(processed_data.shape)
print(processed_data.head())
```

### Command Line Execution

```bash
python main.py
```

---

## Data Statistics

### Before Preprocessing

- Rows: 1,347,681
- Columns: 15
- Missing values: Yes (zip_code, title, desc especially)
- Class imbalance: Yes (Majority: Fully Paid, Minority: Default)

### After Preprocessing

- Balanced classes using SMOTE
- Scaled features for ML models
- Encoded categorical variables
- Removed irrelevant features
- Ready for model training

---

## Dependencies

- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning & preprocessing
- **imbalanced-learn** - SMOTE implementation
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **jupyter** - Notebook environment

---

## Next Phases

### Phase 2: Exploratory Data Analysis (EDA)

- Distribution analysis
- Correlation analysis
- Feature importance analysis

### Phase 3: Feature Engineering

- Create new features
- Feature selection
- Dimensionality reduction

### Phase 4: Model Development

- Train multiple models (Logistic Regression, Random Forest, XGBoost, etc.)
- Hyperparameter tuning
- Model evaluation and comparison

### Phase 5: Model Validation & Deployment

- Cross-validation
- Performance metrics
- Business insights
- Deployment strategy

---

## Key Metrics

The model will be evaluated using:

- **Accuracy** - Overall correctness
- **Precision** - True positives among predicted positives
- **Recall** - True positives among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under the ROC curve
- **Confusion Matrix** - Detailed classification breakdown

---

## Author Notes

- This project uses **SMOTE** to handle class imbalance, critical for credit risk prediction
- Feature scaling is important for distance-based and regularized algorithms
- Categorical encoding strategy can be adjusted based on downstream model type
- Cross-validation will be implemented in Phase 4

---

## License

This project is for educational and commercial use.

---

## Contact & Support

For questions or improvements, please refer to the project structure and documentation.

---

**Last Updated:** February 2026
