"""
Data Preprocessing Module for Credit Risk Assessment
Includes data cleaning, feature engineering, encoding, scaling, and SMOTE balancing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from config import *


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for credit risk assessment
    """
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.scaler = None
        self.label_encoders = {}
        self.original_df = None
        self.processed_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, file_path):
        """Step 1: Load dataset using pandas"""
        print("=" * 80)
        print("PHASE 1: DATA PREPROCESSING")
        print("=" * 80)
        print("\n[STEP 1] Loading dataset...")
        
        df = pd.read_csv(file_path, low_memory=False)
        self.original_df = df.copy()
        self.processed_df = df.copy()
        
        print(f"✓ Dataset loaded successfully")
        return df
    
    def display_dataset_structure(self):
        """Step 2: Display dataset structure and summary"""
        print("\n[STEP 2] Displaying dataset structure and summary...")
        
        df = self.processed_df
        print(f"\n📊 DATASET STRUCTURE")
        print(f"  • Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\n📋 DATA TYPES:")
        print(df.dtypes)
        print(f"\n📈 STATISTICAL SUMMARY:")
        print(df.describe())
        print(f"\n🎯 TARGET VARIABLE DISTRIBUTION (Before encoding):")
        print(df[TARGET_COLUMN].value_counts())
        
        return df
    
    def handle_missing_values(self):
        """Step 3: Clean missing values"""
        print("\n[STEP 3] Handling missing values...")
        
        df = self.processed_df
        missing_before = df.isnull().sum().sum()
        
        print(f"\n  Missing values before cleaning:")
        missing_info = df.isnull().sum()
        missing_info = missing_info[missing_info > 0]
        if len(missing_info) > 0:
            for col, count in missing_info.items():
                pct = (count / len(df)) * 100
                print(f"    {col}: {count} ({pct:.2f}%)")
        else:
            print("    No missing values found")
        
        # Drop columns with high missing values
        high_missing_cols = []
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > MISSING_VALUE_THRESHOLD:
                high_missing_cols.append(col)
        
        if high_missing_cols:
            print(f"\n  Dropping columns with >{MISSING_VALUE_THRESHOLD*100}% missing values:")
            for col in high_missing_cols:
                pct = (df[col].isnull().sum() / len(df)) * 100
                print(f"    • {col} ({pct:.2f}% missing)")
            df = df.drop(columns=high_missing_cols)
        
        # Fill remaining missing values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"    • Filled {col} with median")
        
        # Fill remaining missing values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
                print(f"    • Filled {col} with mode")
        
        missing_after = df.isnull().sum().sum()
        print(f"\n✓ Missing values after cleaning: {missing_after}")
        self.processed_df = df
        return df
    
    def convert_target_variable(self):
        """Step 4: Convert loan_status into binary target (1=Default/Charged Off, 0=Fully Paid)"""
        print("\n[STEP 4] Converting target variable to binary classification...")
        
        df = self.processed_df
        
        print(f"\n  Original target distribution:")
        print(df[TARGET_COLUMN].value_counts())
        print(f"  Unique values: {df[TARGET_COLUMN].unique()}")
        
        # Binary conversion: 1 = Default/Charged Off, 0 = Fully Paid
        # Assuming target is already binary (0, 1) or needs mapping
        if df[TARGET_COLUMN].dtype == 'object':
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map(BINARY_MAPPING)
        else:
            # Already numeric, ensure it's binary
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        
        print(f"\n  Converted binary target distribution:")
        print(df[TARGET_COLUMN].value_counts())
        print(f"  • Class 0 (Fully Paid): {(df[TARGET_COLUMN] == 0).sum()}")
        print(f"  • Class 1 (Default): {(df[TARGET_COLUMN] == 1).sum()}")
        print(f"  • Imbalance ratio: {(df[TARGET_COLUMN] == 1).sum() / (df[TARGET_COLUMN] == 0).sum():.4f}")
        
        self.processed_df = df
        return df
    
    def remove_irrelevant_columns(self):
        """Step 5: Remove irrelevant columns"""
        print("\n[STEP 5] Removing irrelevant columns...")
        
        df = self.processed_df
        cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
        
        if cols_to_drop:
            print(f"  Dropping columns:")
            for col in cols_to_drop:
                print(f"    • {col}")
            df = df.drop(columns=cols_to_drop)
        
        print(f"✓ Remaining features: {df.shape[1]}")
        print(f"  {list(df.columns)}")
        
        self.processed_df = df
        return df
    
    def encode_categorical_variables(self):
        """Step 6: Encode categorical variables"""
        print("\n[STEP 6] Encoding categorical variables...")
        
        df = self.processed_df
        categorical_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]
        
        print(f"  Categorical columns to encode: {categorical_cols}")
        
        if ENCODING_STRATEGY == 'label':
            print("  Using Label Encoding...")
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"    • {col}: {len(le.classes_)} unique values encoded")
        
        elif ENCODING_STRATEGY == 'onehot':
            print("  Using One-Hot Encoding...")
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            print(f"    • Features after one-hot encoding: {df.shape[1]}")
        
        print(f"✓ Data shape after encoding: {df.shape}")
        self.processed_df = df
        return df
    
    def apply_feature_scaling(self):
        """Step 7: Feature scaling (standardization)"""
        print("\n[STEP 7] Applying feature scaling (Standardization)...")
        
        df = self.processed_df
        numeric_cols = [col for col in df.columns 
                       if col != TARGET_COLUMN and df[col].dtype in [np.number]]
        
        print(f"  Numeric features to scale: {numeric_cols}")
        
        if SCALER_TYPE == 'StandardScaler':
            self.scaler = StandardScaler()
            print("  Using StandardScaler (Zero mean, Unit variance)")
        else:
            self.scaler = MinMaxScaler()
            print("  Using MinMaxScaler (Range 0-1)")
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        print(f"✓ Feature scaling complete")
        print(f"  • Scaled mean (sample): {df[numeric_cols].mean().mean():.6f}")
        print(f"  • Scaled std (sample): {df[numeric_cols].std().mean():.6f}")
        
        self.processed_df = df
        return df
    
    def handle_class_imbalance(self):
        """Step 8: Handle class imbalance using SMOTE"""
        print("\n[STEP 8] Handling class imbalance using SMOTE...")
        
        df = self.processed_df
        
        # Separate features and target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        print(f"\n  Before SMOTE:")
        print(f"    • Class 0: {(y == 0).sum()}")
        print(f"    • Class 1: {(y == 1).sum()}")
        print(f"    • Imbalance ratio: {(y == 1).sum() / (y == 0).sum():.4f}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=SMOTE_RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
        X_smote, y_smote = smote.fit_resample(X, y)
        
        print(f"\n  After SMOTE:")
        print(f"    • Class 0: {(y_smote == 0).sum()}")
        print(f"    • Class 1: {(y_smote == 1).sum()}")
        print(f"    • Imbalance ratio: {(y_smote == 1).sum() / (y_smote == 0).sum():.4f}")
        print(f"    • New dataset size: {X_smote.shape[0]}")
        
        # Combine balanced data
        balanced_df = pd.DataFrame(X_smote, columns=X.columns)
        balanced_df[TARGET_COLUMN] = y_smote
        
        print(f"\n✓ Class imbalance handled successfully")
        self.processed_df = balanced_df
        return balanced_df
    
    def split_train_test(self):
        """Split data into train and test sets"""
        print("\n[SPLITTING DATA] Train-Test split...")
        
        df = self.processed_df
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"  • Training set: {self.X_train.shape[0]} samples")
        print(f"  • Test set: {self.X_test.shape[0]} samples")
        print(f"  • Feature count: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self):
        """Save processed data to files"""
        print("\n[SAVING DATA]...")
        
        # Save full processed data
        self.processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"✓ Full processed data saved: {PROCESSED_DATA_PATH}")
        
        # Save train/test sets
        if self.X_train is not None and self.X_test is not None:
            train_df = self.X_train.copy()
            train_df[TARGET_COLUMN] = self.y_train
            train_df.to_csv(PROCESSED_DATA_TRAIN, index=False)
            
            test_df = self.X_test.copy()
            test_df[TARGET_COLUMN] = self.y_test
            test_df.to_csv(PROCESSED_DATA_TEST, index=False)
            
            print(f"✓ Training set saved: {PROCESSED_DATA_TRAIN}")
            print(f"✓ Test set saved: {PROCESSED_DATA_TEST}")
    
    def run_full_pipeline(self, file_path):
        """Run complete preprocessing pipeline"""
        print("\n")
        print("█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "  PREDICTIVE ANALYTICS-BASED DECISION SUPPORT FRAMEWORK".center(78) + "█")
        print("█" + "  CREDIT RISK ASSESSMENT IN THE BANKING SECTOR".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        
        # Execute all steps
        self.load_data(file_path)
        self.display_dataset_structure()
        self.handle_missing_values()
        self.convert_target_variable()
        self.remove_irrelevant_columns()
        self.encode_categorical_variables()
        self.apply_feature_scaling()
        self.handle_class_imbalance()
        self.split_train_test()
        self.save_processed_data()
        
        print("\n" + "=" * 80)
        print("✓ PHASE 1: DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nProcessed data shape: {self.processed_df.shape}")
        print(f"Files saved in: {DATA_PROCESSED}")
        
        return self.processed_df
