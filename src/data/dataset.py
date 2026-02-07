"""
Dataset utilities for creating PyTorch DataLoaders.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, Optional


class LoanDataset(Dataset):
    """
    PyTorch Dataset for loan data.
    """
    
    def __init__(
        self,
        numerical_features: np.ndarray,
        categorical_features: np.ndarray,
        targets: Optional[np.ndarray] = None
    ):
        """
        Args:
            numerical_features: (N, n_numerical) array of scaled numerical features
            categorical_features: (N, n_categorical) array of encoded categorical features
            targets: (N,) array of binary labels (optional for test set)
        """
        self.numerical = torch.tensor(numerical_features, dtype=torch.float32)
        self.categorical = torch.tensor(categorical_features, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        
    def __len__(self) -> int:
        return len(self.numerical)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.targets is not None:
            return self.numerical[idx], self.categorical[idx], self.targets[idx]
        else:
            return self.numerical[idx], self.categorical[idx]


def create_data_loaders(
    numerical_features: np.ndarray,
    categorical_features: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 256,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        numerical_features: (N, n_numerical) scaled numerical features
        categorical_features: (N, n_categorical) encoded categorical features
        targets: (N,) binary labels
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = LoanDataset(numerical_features, categorical_features, targets)
    
    # Split into train and validation
    n_samples = len(full_dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [n_train, n_val],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_loader(
    numerical_features: np.ndarray,
    categorical_features: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0
) -> DataLoader:
    """
    Create test data loader (without targets).
    
    Args:
        numerical_features: (N, n_numerical) scaled numerical features
        categorical_features: (N, n_categorical) encoded categorical features
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        
    Returns:
        test_loader
    """
    test_dataset = LoanDataset(numerical_features, categorical_features)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader
