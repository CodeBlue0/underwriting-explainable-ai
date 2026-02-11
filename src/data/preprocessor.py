"""
Preprocessor for MNIST Dataset.
Handles loading via torchvision and flattening.
"""
import torch
import numpy as np
import pickle
from typing import List, Tuple, Dict, Any, Optional
import os
from torchvision import datasets, transforms


class MNISTPreprocessor:
    """
    Preprocessor for MNIST dataset.
    - Loads using torchvision
    - Flattens 28x28 images to 784 vectors
    - Scales pixel values to [0, 1]
    - Handles empty categorical features
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        self.numerical_features = [f'pixel_{i}' for i in range(784)]
        self.categorical_features = []
        self._fitted = False
        self.transform_pipeline = transforms.Compose([
            transforms.ToTensor(),  # Scales [0, 255] -> [0.0, 1.0]
            transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
        ])
        
    def fit(self, *args, **kwargs) -> 'MNISTPreprocessor':
        """
        No-op for MNIST (stateless scaling).
        """
        self._fitted = True
        return self
    
    def transform(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the dataset.
        
        Args:
            dataset: torchvision dataset
            
        Returns:
            (numerical_features, categorical_features) as numpy arrays
        """
        # Load all data into memory
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, targets = next(iter(loader))
        
        # Flatten: (N, 1, 28, 28) -> (N, 784)
        X_num = data.view(data.size(0), -1).numpy()
        
        # Empty categorical
        X_cat = np.zeros((len(dataset), 0), dtype=np.int64)
        
        return X_num.astype(np.float32), X_cat.astype(np.int64), targets.numpy().astype(np.int64)
    
    def get_cardinalities(self) -> Dict[str, int]:
        """Get cardinality for categorical features (empty)."""
        return {}
    
    def inverse_transform_numerical(self, X_num: np.ndarray) -> np.ndarray:
        """
        Inverse transform not fully supported due to normalization, 
        returning un-normalized roughly to [0, 1] range for visualization.
        """
        # Undo normalization: x * std + mean
        return X_num * 0.3081 + 0.1307
    
    def inverse_transform_categorical(self, X_cat: np.ndarray) -> Dict[str, List[Any]]:
        """Empty dict for MNIST."""
        return {}
    
    def save(self, path: str):
        """Save preprocessor to file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> 'MNISTPreprocessor':
        """Load preprocessor from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_mnist_data(
    config,
    train_path: str = None, # Root dir for download
    test_path: str = None   # Root dir for download
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], MNISTPreprocessor]:
    """
    Load and preprocess MNIST dataset.
    
    Args:
        config: ModelConfig object
        train_path: Root directory for data
        
    Returns:
        (train_num, train_cat, train_target), (test_num, test_cat, test_target), preprocessor
    """
    root = train_path or config.train_path
    os.makedirs(root, exist_ok=True)
    
    print(f"Loading MNIST data from {root}...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root, train=False, download=True, transform=transform)
    
    preprocessor = MNISTPreprocessor()
    preprocessor.fit()
    
    print("Transforming training data...")
    train_num, train_cat, train_target = preprocessor.transform(train_dataset)
    # Transform labels: 1 if multiple of 3, else 0
    train_target = (train_target % 4 == 0).astype(np.int64)
    
    print("Transforming test data...")
    test_num, test_cat, test_target = preprocessor.transform(test_dataset)
    test_target = (test_target % 4 == 0).astype(np.int64)
    
    print(f"  Train shape: {train_num.shape}")
    print(f"  Train targets (binary): {np.bincount(train_target)}")
    print(f"  Test shape: {test_num.shape}")
    
    return (train_num, train_cat, train_target), (test_num, test_cat, test_target), preprocessor
