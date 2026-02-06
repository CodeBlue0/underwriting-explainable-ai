"""
Feature Decoder Network.

Reconstructs original features from latent representation for interpretability
and regularization.
"""
import torch
import torch.nn as nn
from typing import List, Tuple


class FeatureDecoder(nn.Module):
    """
    Decoder network that reconstructs original features from latent space.
    
    This serves two purposes:
    1. Regularization: ensures latent space preserves useful information
    2. Interpretability: allows decoding prototypes to feature space
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_numerical: int,
        categorical_cardinalities: List[int],
        hidden_dim: int = 128
    ):
        """
        Args:
            latent_dim: Dimension of latent vectors (d_model from encoder)
            n_numerical: Number of numerical features to reconstruct
            categorical_cardinalities: List of cardinality for each categorical feature
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_numerical = n_numerical
        self.categorical_cardinalities = categorical_cardinalities
        self.n_categorical = len(categorical_cardinalities)
        
        # Shared hidden layers
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Numerical feature decoder (outputs continuous values)
        self.numerical_decoder = nn.Linear(hidden_dim, n_numerical)
        
        # Categorical feature decoders (outputs logits for each category)
        self.categorical_decoders = nn.ModuleList([
            nn.Linear(hidden_dim, cardinality) 
            for cardinality in categorical_cardinalities
        ])
    
    def forward(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Reconstruct features from latent representation.
        
        Args:
            z: (batch_size, latent_dim) latent vectors
            
        Returns:
            num_recon: (batch_size, n_numerical) reconstructed numerical features (scaled)
            cat_recons: List of (batch_size, cardinality) logits for each categorical
        """
        # Shared hidden representation
        hidden = self.shared_layers(z)
        
        # Decode numerical features
        num_recon = self.numerical_decoder(hidden)
        
        # Decode categorical features (output logits)
        cat_recons = [decoder(hidden) for decoder in self.categorical_decoders]
        
        return num_recon, cat_recons
    
    def decode_to_predictions(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Decode and return predicted categorical indices (argmax).
        
        Args:
            z: (batch_size, latent_dim) latent vectors
            
        Returns:
            num_recon: (batch_size, n_numerical) reconstructed numerical features
            cat_preds: List of (batch_size,) predicted categorical indices
        """
        num_recon, cat_logits = self.forward(z)
        cat_preds = [logits.argmax(dim=1) for logits in cat_logits]
        return num_recon, cat_preds


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for numerical and categorical features.
    """
    
    def __init__(self, num_weight: float = 1.0, cat_weight: float = 1.0):
        super().__init__()
        self.num_weight = num_weight
        self.cat_weight = cat_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        num_recon: torch.Tensor,
        num_target: torch.Tensor,
        cat_recons: List[torch.Tensor],
        cat_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction losses.
        
        Args:
            num_recon: (batch_size, n_numerical) reconstructed numerical
            num_target: (batch_size, n_numerical) target numerical (scaled)
            cat_recons: List of (batch_size, cardinality) logits
            cat_target: (batch_size, n_categorical) target indices
            
        Returns:
            total_loss: Combined reconstruction loss
            num_loss: Numerical reconstruction loss (MSE)
            cat_loss: Categorical reconstruction loss (CE)
        """
        # Numerical reconstruction loss (MSE)
        num_loss = self.mse_loss(num_recon, num_target)
        
        # Categorical reconstruction loss (mean CE across all categorical features)
        cat_losses = []
        for i, cat_logits in enumerate(cat_recons):
            cat_losses.append(self.ce_loss(cat_logits, cat_target[:, i]))
        cat_loss = torch.stack(cat_losses).mean() if cat_losses else torch.tensor(0.0)
        
        # Combined loss
        total_loss = self.num_weight * num_loss + self.cat_weight * cat_loss
        
        return total_loss, num_loss, cat_loss
