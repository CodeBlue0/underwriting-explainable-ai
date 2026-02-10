"""
Configuration module for ICR (Identify Age-Related Conditions) Dataset.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class ICRConfig:
    """Configuration for ICR dataset with PTaRL model."""
    
    # Feature definitions (excluding Id, Class, BQ, EL)
    numerical_features: List[str] = field(default_factory=lambda: [
        'AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ',
        'BN', 'BP', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL',
        'CR', 'CS', 'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL',
        'DN', 'DU', 'DV', 'DY', 'EB', 'EE', 'EG', 'EH', 'EP', 'EU',
        'FC', 'FD ', 'FE', 'FI', 'FL', 'FR', 'FS', 'GB', 'GE', 'GF',
        'GH', 'GI', 'GL'
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: ['EJ'])
    
    # Columns to exclude
    exclude_columns: List[str] = field(default_factory=lambda: ['Id', 'Class', 'BQ', 'EL'])
    
    # Target column
    target_column: str = 'Class'
    
    # Categorical cardinalities (will be updated from data)
    categorical_cardinalities: Dict[str, int] = field(default_factory=lambda: {
        'EJ': 2  # A, B
    })
    
    # FT-Transformer architecture
    d_model: int = 128  # Smaller for small dataset
    n_heads: int = 4
    n_layers: int = 4
    d_ffn: int = 512
    dropout: float = 0.3  # Higher dropout for small dataset
    attention_dropout: float = 0.2
    ffn_dropout: float = 0.3
    
    # Decoder architecture
    decoder_hidden_dim: int = 256
    
    # Training hyperparameters
    batch_size: int = 32  # Small batch for small dataset
    learning_rate: float = 3e-4  # Reduced from 1e-3 for stability
    weight_decay: float = 1e-4
    epochs: int = 150  # Increased from 50 (not used directly, but good for reference)
    early_stopping_patience: int = 30  # Increased from 15
    
    # Loss weights (Phase 1)
    lambda_reconstruction: float = 0.1
    
    # Paths
    train_path: str = '/workspace/data/icr/train.csv'
    test_path: str = '/workspace/data/icr/test.csv'
    greeks_path: str = '/workspace/data/icr/greeks.csv'
    model_save_path: str = '/workspace/checkpoints/icr'
    
    # PTaRL Settings
    use_ptarl: bool = True
    n_global_prototypes: int = 6  # ceil(log2(54)) ≈ 6
    phase1_epochs: int = 150  # Increased from 50
    phase2_epochs: int = 150  # Increased from 50
    
    # PTaRL Loss Weights (per paper)
    ptarl_weights: Dict[str, float] = field(default_factory=lambda: {
        'task_weight': 1.0,
        'projection_weight': 0.25,
        'diversifying_weight': 0.25,
        'orthogonalization_weight': 0.25,
        'reconstruction_weight': 0.1
    })
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Random seed
    seed: int = 42
    
    @property
    def n_numerical(self) -> int:
        return len(self.numerical_features)
    
    @property
    def n_categorical(self) -> int:
        return len(self.categorical_features)
    
    @property
    def n_features(self) -> int:
        return self.n_numerical + self.n_categorical
    
    def get_cardinality_list(self) -> List[int]:
        """Get list of cardinalities in order of categorical_features."""
        return [self.categorical_cardinalities[f] for f in self.categorical_features]


def get_icr_config() -> ICRConfig:
    """Returns the ICR configuration."""
    return ICRConfig()
