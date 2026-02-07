
import pandas as pd
import numpy as np
import pickle
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class LoanDataPreprocessor:
    """
    Preprocessor for loan data.
    Handles missing values, scaling of numerical features, and encoding of categorical features.
    """
    
    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        """
        Initialize preprocessor.
        
        Args:
            numerical_features (List[str]): List of numerical feature names
            categorical_features (List[str]): List of categorical feature names
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.num_imputers: Dict[str, float] = {}  # Median values for numerical imputation
        self.cat_imputers: Dict[str, Any] = {}    # Mode values for categorical imputation
        
    def fit(self, df: pd.DataFrame) -> 'LoanDataPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            self
        """
        # 1. Fit numerical features
        # Calculate medians for imputation
        for feat in self.numerical_features:
            self.num_imputers[feat] = df[feat].median()
            
        # Fill missing values for scaling fitting
        X_num = df[self.numerical_features].fillna(self.num_imputers)
        self.scaler.fit(X_num)
        
        # 2. Fit categorical features
        for feat in self.categorical_features:
            # Calculate mode for imputation
            self.cat_imputers[feat] = df[feat].mode()[0]
            
            # Fill missing
            series = df[feat].fillna(self.cat_imputers[feat]).astype(str)
            
            # Initialize and fit encoder
            le = LabelEncoder()
            le.fit(series)
            self.encoders[feat] = le
            
        return self
        
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the data.
        
        Args:
            df (pd.DataFrame): Data to transform
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (numerical_features, categorical_features)
        """
        # 1. Transform numerical features
        X_num = df[self.numerical_features].fillna(self.num_imputers)
        X_num = self.scaler.transform(X_num)
        
        # 2. Transform categorical features
        X_cat_list = []
        for feat in self.categorical_features:
            # Impute
            series = df[feat].fillna(self.cat_imputers[feat]).astype(str)
            
            # Transform
            le = self.encoders[feat]
            
            # Handle unknown categories by mapping them to the most frequent class (mode)
            # or raising an error? For robustness, we'll map unknowns to the first class (usually 0)
            # or use a custom robust transform.
            # Here we implement a simple safe transform: 
            # if value not in classes, use mode (which we know is in classes).
            
            # Check for unknown values
            mask = ~series.isin(le.classes_)
            if mask.any():
                # Replace unknown with mode (stored in cat_imputers, need to ensure it's string)
                mode_val = str(self.cat_imputers[feat])
                series[mask] = mode_val
                
            encoded = le.transform(series)
            X_cat_list.append(encoded)
            
        X_cat = np.stack(X_cat_list, axis=1)
        
        return X_num.astype(np.float32), X_cat.astype(np.int64)
    
    def get_cardinalities(self) -> Dict[str, int]:
        """
        Get cardinality (number of unique values) for each categorical feature.
        
        Returns:
            Dict[str, int]: Dictionary mapping feature name to cardinality
        """
        return {feat: len(enc.classes_) for feat, enc in self.encoders.items()}
    
    def save(self, path: str):
        """Save preprocessor to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path: str) -> 'LoanDataPreprocessor':
        """Load preprocessor from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_and_preprocess_data(
    train_path: str, 
    test_path: str,
    numerical_features: List[str],
    categorical_features: List[str]
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], LoanDataPreprocessor]:
    """
    Load data from CSVs and preprocess it.
    
    Args:
        train_path (str): Path to training CSV
        test_path (str): Path to test CSV
        numerical_features (List[str]): List of numerical feature names
        categorical_features (List[str]): List of categorical feature names
        
    Returns:
        (train_num, train_cat, train_target), (test_num, test_cat), preprocessor
    """
    # 1. Load data
    print(f"Loading data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Check features
    # Ensure all features exist
    for f in numerical_features + categorical_features:
        if f not in train_df.columns:
            raise ValueError(f"Feature {f} not found in training data")
    
    # 3. Initialize and fit preprocessor
    preprocessor = LoanDataPreprocessor(numerical_features, categorical_features)
    preprocessor.fit(train_df)
    
    # 4. Transform data
    train_num, train_cat = preprocessor.transform(train_df)
    test_num, test_cat = preprocessor.transform(test_df)
    
    # 5. Extract target
    if 'loan_status' in train_df.columns:
        train_target = train_df['loan_status'].values.astype(np.int64)
    else:
        raise ValueError("Target column 'loan_status' not found in training data")
        
    return (train_num, train_cat, train_target), (test_num, test_cat), preprocessor
