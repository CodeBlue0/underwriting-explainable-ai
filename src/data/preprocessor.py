"""
Preprocessor for ICR (Identify Age-Related Conditions) Dataset.
Handles missing values, scaling, and encoding.
"""
import pandas as pd
import numpy as np
import pickle
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class ICRPreprocessor:
    """
    Preprocessor for ICR dataset.
    - Removes BQ, EL columns (high missing rate with target correlation)
    - Imputes numerical with median, categorical with mode
    - Scales numerical with StandardScaler
    - Encodes categorical with LabelEncoder
    """
    
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        exclude_columns: List[str] = None
    ):
        """
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            exclude_columns: Columns to exclude (e.g., ['Id', 'Class', 'BQ', 'EL'])
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.exclude_columns = exclude_columns or []
        
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.num_imputers: Dict[str, float] = {}  # Median for numerical
        self.cat_imputers: Dict[str, Any] = {}    # Mode for categorical
        self._fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'ICRPreprocessor':
        """
        Fit the preprocessor to the training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            self
        """
        # 1. Fit numerical features
        for feat in self.numerical_features:
            if feat in df.columns:
                self.num_imputers[feat] = df[feat].median()
        
        # Fill missing for scaling fit
        X_num = df[self.numerical_features].copy()
        for feat in self.numerical_features:
            if feat in X_num.columns:
                X_num[feat] = X_num[feat].fillna(self.num_imputers.get(feat, 0))
        
        self.scaler.fit(X_num)
        
        # 2. Fit categorical features
        for feat in self.categorical_features:
            if feat in df.columns:
                # Calculate mode for imputation
                self.cat_imputers[feat] = df[feat].mode()[0]
                
                # Fit encoder
                series = df[feat].fillna(self.cat_imputers[feat]).astype(str)
                le = LabelEncoder()
                le.fit(series)
                self.encoders[feat] = le
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the data.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            (numerical_features, categorical_features) as numpy arrays
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # 1. Transform numerical features
        X_num = df[self.numerical_features].copy()
        for feat in self.numerical_features:
            if feat in X_num.columns:
                X_num[feat] = X_num[feat].fillna(self.num_imputers.get(feat, 0))
        
        X_num_scaled = self.scaler.transform(X_num)
        
        # 2. Transform categorical features
        X_cat_list = []
        for feat in self.categorical_features:
            if feat in df.columns:
                series = df[feat].fillna(self.cat_imputers[feat]).astype(str)
                le = self.encoders[feat]
                
                # Handle unknown categories
                mask = ~series.isin(le.classes_)
                if mask.any():
                    series[mask] = str(self.cat_imputers[feat])
                
                encoded = le.transform(series)
                X_cat_list.append(encoded)
        
        X_cat = np.stack(X_cat_list, axis=1) if X_cat_list else np.zeros((len(df), 0), dtype=np.int64)
        
        return X_num_scaled.astype(np.float32), X_cat.astype(np.int64)
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def get_cardinalities(self) -> Dict[str, int]:
        """Get cardinality for each categorical feature."""
        return {feat: len(enc.classes_) for feat, enc in self.encoders.items()}
    
    def inverse_transform_numerical(self, X_num: np.ndarray) -> np.ndarray:
        """Inverse transform scaled numerical features."""
        return self.scaler.inverse_transform(X_num)
    
    def save(self, path: str):
        """Save preprocessor to file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> 'ICRPreprocessor':
        """Load preprocessor from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_icr_data(
    config,
    train_path: str = None,
    test_path: str = None
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], ICRPreprocessor]:
    """
    Load and preprocess ICR dataset.
    
    Args:
        config: ICRConfig object
        train_path: Path to training CSV (defaults to config.train_path)
        test_path: Path to test CSV (defaults to config.test_path)
        
    Returns:
        (train_num, train_cat, train_target), (test_num, test_cat), preprocessor
    """
    train_path = train_path or config.train_path
    test_path = test_path or config.test_path
    
    print(f"Loading ICR data from {train_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Initialize preprocessor
    preprocessor = ICRPreprocessor(
        numerical_features=config.numerical_features,
        categorical_features=config.categorical_features,
        exclude_columns=config.exclude_columns
    )
    
    # Fit on training data
    preprocessor.fit(train_df)
    
    # Transform
    train_num, train_cat = preprocessor.transform(train_df)
    test_num, test_cat = preprocessor.transform(test_df)
    
    # Extract target
    train_target = train_df[config.target_column].values.astype(np.int64)
    
    print(f"  Numerical features: {train_num.shape[1]}")
    print(f"  Categorical features: {train_cat.shape[1]}")
    print(f"  Target distribution: {dict(pd.Series(train_target).value_counts())}")
    
    return (train_num, train_cat, train_target), (test_num, test_cat), preprocessor
