"""
Prototype Layer for case-based reasoning.

This module implements learnable prototype vectors with similarity computation
for interpretable predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PrototypeLayer(nn.Module):
    """
    Learnable prototype layer for prototype-based classification.
    
    Each prototype represents a typical "case" or "archetype" in the data.
    Predictions are made based on similarity to these prototypes, enabling
    case-based reasoning and interpretability.
    
    Attributes:
        prototypes: (n_prototypes, prototype_dim) learnable prototype vectors
    """
    
    def __init__(
        self,
        n_prototypes: int,
        prototype_dim: int,
        similarity_type: str = 'rbf',
        rbf_sigma: float = 1.0
    ):
        """
        Args:
            n_prototypes: Number of prototype vectors
            prototype_dim: Dimension of each prototype (should match encoder output)
            similarity_type: 'rbf' (Gaussian kernel) or 'cosine' similarity
            rbf_sigma: Sigma parameter for RBF similarity
        """
        super().__init__()
        self.n_prototypes = n_prototypes
        self.prototype_dim = prototype_dim
        self.similarity_type = similarity_type
        
        # Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # Learnable sigma for RBF (allows model to learn optimal temperature)
        if similarity_type == 'rbf':
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(rbf_sigma)))
    
    @property
    def sigma(self) -> torch.Tensor:
        """Get sigma value (learned or fixed)."""
        if self.similarity_type == 'rbf':
            return torch.exp(self.log_sigma)
        return torch.tensor(1.0)
    
    def compute_similarity(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between latent representations and prototypes.
        
        Args:
            z: (batch_size, prototype_dim) latent vectors from encoder
            
        Returns:
            similarities: (batch_size, n_prototypes) similarity scores
        """
        if self.similarity_type == 'rbf':
            # RBF (Gaussian) similarity: s(z, p) = exp(-||z - p||^2 / (2 * sigma^2))
            # Compute squared distances
            # z: (B, D), prototypes: (P, D)
            distances_sq = torch.cdist(z, self.prototypes, p=2).pow(2)  # (B, P)
            sigma = self.sigma
            similarities = torch.exp(-distances_sq / (2 * sigma ** 2))
        
        elif self.similarity_type == 'cosine':
            # Cosine similarity: s(z, p) = (z · p) / (||z|| * ||p||)
            z_norm = F.normalize(z, p=2, dim=1)  # (B, D)
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)  # (P, D)
            similarities = torch.mm(z_norm, proto_norm.t())  # (B, P)
            # Scale to [0, 1] range
            similarities = (similarities + 1) / 2
        
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        
        return similarities
    
    def compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 distances between latent representations and prototypes.
        
        Args:
            z: (batch_size, prototype_dim) latent vectors
            
        Returns:
            distances: (batch_size, n_prototypes) L2 distances
        """
        return torch.cdist(z, self.prototypes, p=2)
    
    def diversity_loss(self) -> torch.Tensor:
        """
        Compute prototype diversity loss to encourage separation.
        
        Penalizes prototypes that are too similar to each other,
        encouraging them to represent diverse cases.
        
        Returns:
            loss: scalar diversity loss (minimize to increase diversity)
        """
        # Compute pairwise cosine similarity
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarity_matrix = torch.mm(proto_norm, proto_norm.t())  # (P, P)
        
        # Create mask to exclude self-similarity (diagonal)
        mask = 1.0 - torch.eye(
            self.n_prototypes, 
            device=self.prototypes.device
        )
        
        # Mean of off-diagonal similarities (we want to minimize this)
        diversity_loss = (similarity_matrix * mask).sum() / mask.sum()
        
        return diversity_loss
    
    def clustering_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute clustering loss to encourage latent vectors to be close to prototypes.
        
        This ensures that prototypes actually represent the data distribution.
        
        Args:
            z: (batch_size, prototype_dim) latent vectors
            
        Returns:
            loss: scalar clustering loss
        """
        # Distance from each z to its nearest prototype
        distances = self.compute_distances(z)  # (B, P)
        min_distances = distances.min(dim=1).values  # (B,)
        
        return min_distances.mean()
    
    def get_nearest_prototype(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the nearest prototype for each latent vector.
        
        Args:
            z: (batch_size, prototype_dim) latent vectors
            
        Returns:
            indices: (batch_size,) index of nearest prototype
            distances: (batch_size,) distance to nearest prototype
        """
        distances = self.compute_distances(z)
        min_distances, indices = distances.min(dim=1)
        return indices, min_distances
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute similarities to all prototypes.
        
        Args:
            z: (batch_size, prototype_dim) latent vectors from encoder
            
        Returns:
            similarities: (batch_size, n_prototypes) similarity scores
        """
        return self.compute_similarity(z)
    
    def decode_prototypes(self, decoder: nn.Module) -> torch.Tensor:
        """
        Decode prototypes to feature space using the decoder network.
        
        Args:
            decoder: Decoder network that maps prototype_dim -> features
            
        Returns:
            Decoded prototype representations
        """
        return decoder(self.prototypes)


class ClassificationHead(nn.Module):
    """
    Classification head with non-negative weights for interpretability.
    
    Uses non-negative weights to ensure clear interpretation:
    - Positive weight: higher similarity to prototype increases probability
    - Non-negative constraint: monotonic relationship between similarity and prediction
    """
    
    def __init__(self, n_prototypes: int, n_classes: int = 1):
        """
        Args:
            n_prototypes: Number of prototype similarity scores as input
            n_classes: Number of output classes (1 for binary classification)
        """
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        
        # Raw weights (will be transformed to non-negative)
        self.weight_raw = nn.Parameter(torch.randn(n_prototypes, n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight_raw)
    
    @property
    def weight(self) -> torch.Tensor:
        """Non-negative weights using softplus transformation."""
        return F.softplus(self.weight_raw)
    
    def forward(self, similarities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarities: (batch_size, n_prototypes) prototype similarities
            
        Returns:
            logits: (batch_size, n_classes) classification logits
        """
        # Linear combination with non-negative weights
        logits = torch.mm(similarities, self.weight) + self.bias
        
        if self.n_classes == 1:
            logits = logits.squeeze(-1)
        
        return logits
    
    def get_prototype_contributions(self, similarities: torch.Tensor) -> torch.Tensor:
        """
        Get contribution of each prototype to the final prediction.
        
        Args:
            similarities: (batch_size, n_prototypes) prototype similarities
            
        Returns:
            contributions: (batch_size, n_prototypes) weighted contributions
        """
        # Element-wise multiplication with weights
        weights = self.weight.squeeze(-1)  # (P,)
        contributions = similarities * weights  # (B, P)
        return contributions
