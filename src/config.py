"""
Configuration module for Prototype-based Neuro-Symbolic Model (MNIST).
Contains all hyperparameters and feature definitions.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class ModelConfig:
    """Complete configuration for the prototype network on MNIST."""
    
    # Feature definitions
    # MNIST images are 28x28 = 784 pixels
    numerical_features: List[str] = field(default_factory=lambda: [f'pixel_{i}' for i in range(784)])
    
    # MNIST has no categorical features
    categorical_features: List[str] = field(default_factory=list)
    
    # Target column
    target_column: str = 'label'
    n_classes: int = 1

    
    # Categorical feature cardinalities (empty for MNIST)
    categorical_cardinalities: Dict[str, int] = field(default_factory=dict)
    
    # FT-Transformer architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ffn: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    
    # Prototype layer
    n_prototypes: int = 20  # 2 per class * 10 classes
    prototype_dim: int = 128  # Same as d_model
    similarity_type: str = 'rbf'  # 'rbf' or 'cosine'
    rbf_sigma: float = 1.0  # Normalized inputs, so 1.0 is reasonable
    
    # Class imbalance handling (MNIST is balanced)
    pos_weight: float = 1.0
    
    # Decoder architecture
    decoder_hidden_dim: int = 256
    
    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    early_stopping_patience: int = 5
    
    # Loss weights
    lambda_reconstruction: float = 0.5  # Higher weight for reconstruction on images
    lambda_diversity: float = 0.1
    lambda_clustering: float = 0.1
    
    # Paths
    train_path: str = '/workspace/data/mnist'  # Root for download
    test_path: str = '/workspace/data/mnist'   # Root for download
    model_save_path: str = '/workspace/checkpoints/mnist'
    
    # PTaRL Settings
    use_ptarl: bool = True  # Enable PTaRL two-phase training
    n_global_prototypes: int = 10  # One global prototype per digit class (0-9)
    phase1_epochs: int = 10  # Epochs for Phase 1
    phase2_epochs: int = 10  # Epochs for Phase 2
    
    # PTaRL Loss Weights
    ptarl_weights: Dict[str, float] = field(default_factory=lambda: {
        'task_weight': 1.0,
        'projection_weight': 0.1,
        'diversifying_weight': 0.1,
        'orthogonalization_weight': 0.1,
        'reconstruction_weight': 0.5
    })
    
    # Sinkhorn Distance Parameters
    sinkhorn_eps: float = 0.1
    sinkhorn_max_iter: int = 50
    
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


# Prototype descriptions (0-9 digits)
PROTOTYPE_DESCRIPTIONS = {
    i: {
        'ko': f'숫자 {i}의 전형적인 형태',
        'en': f'Typical form of digit {i}',
        'description_ko': f'{i}를 나타내는 이미지 프로토타입',
        'description_en': f'Image prototype representing digit {i}'
    } for i in range(10)
}


def get_default_config() -> ModelConfig:
    """Returns the default configuration."""
    return ModelConfig()
