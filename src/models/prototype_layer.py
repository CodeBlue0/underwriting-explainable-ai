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
        # Initialize with std=1.0 to match encoder output (LayerNorm) scale
        # This prevents vanishing similarities (exp(-large_dist)) at initialization
        nn.init.normal_(self.prototypes, std=1.0)
        
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


class ClassBalancedPrototypeLayer(nn.Module):
    """
    Class-Balanced Prototype Layer that ensures prototypes for both classes.
    
    Solves the problem of all prototypes clustering around majority class by:
    1. Pre-assigning prototypes to each class (e.g., 5 for Default, 5 for Non-Default)
    2. Using class-stratified KMeans initialization
    3. Adding class separation loss to maintain class-specific prototypes
    """
    
    def __init__(
        self,
        n_prototypes: int,
        prototype_dim: int,
        n_prototypes_per_class: Optional[int] = None,
        similarity_type: str = 'rbf',
        rbf_sigma: float = 1.0,
        class_separation_weight: float = 0.1,
        random_seed: int = 42
    ):
        """
        Args:
            n_prototypes: Total number of prototype vectors
            prototype_dim: Dimension of each prototype
            n_prototypes_per_class: Prototypes per class (default: n_prototypes // 2)
            similarity_type: 'rbf' or 'cosine' similarity
            rbf_sigma: Sigma for RBF kernel
            class_separation_weight: Weight for class separation loss
            random_seed: Random seed for KMeans initialization
        """
        super().__init__()
        self.n_prototypes = n_prototypes
        self.prototype_dim = prototype_dim
        self.similarity_type = similarity_type
        self.class_separation_weight = class_separation_weight
        self.random_seed = random_seed
        
        # Split prototypes between classes
        self.n_per_class = n_prototypes_per_class or (n_prototypes // 2)
        self.n_class0 = self.n_per_class  # Non-Default (negative class)
        self.n_class1 = n_prototypes - self.n_per_class  # Default (positive class)
        
        # Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, prototype_dim))
        nn.init.normal_(self.prototypes, std=1.0)
        
        # Register which prototypes belong to which class
        # First n_class0 prototypes -> Class 0 (Non-Default)
        # Remaining prototypes -> Class 1 (Default)
        self.register_buffer(
            'prototype_classes',
            torch.cat([
                torch.zeros(self.n_class0),
                torch.ones(self.n_class1)
            ]).long()
        )
        
        # Learnable sigma for RBF
        if similarity_type == 'rbf':
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(rbf_sigma)))
        
        self._initialized = False
    
    @property
    def sigma(self) -> torch.Tensor:
        if self.similarity_type == 'rbf':
            return torch.exp(self.log_sigma)
        return torch.tensor(1.0)
    
    def initialize_from_data(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Initialize prototypes using class-stratified KMeans.
        
        Args:
            embeddings: (N, D) embeddings from training data
            labels: (N,) class labels (0 or 1)
        """
        from sklearn.cluster import KMeans
        
        device = self.prototypes.device
        embeddings_np = embeddings.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Class 0 (Non-Default) prototypes
        mask_class0 = labels_np == 0
        if mask_class0.sum() >= self.n_class0:
            kmeans0 = KMeans(n_clusters=self.n_class0, random_state=self.random_seed, n_init='auto')
            kmeans0.fit(embeddings_np[mask_class0])
            centers0 = kmeans0.cluster_centers_
        else:
            # If not enough samples, use available samples
            centers0 = embeddings_np[mask_class0][:self.n_class0]
            if len(centers0) < self.n_class0:
                # Pad with random noise if needed
                padding = self.n_class0 - len(centers0)
                centers0 = np.vstack([
                    centers0,
                    embeddings_np[mask_class0].mean(axis=0, keepdims=True) + 
                    np.random.randn(padding, embeddings_np.shape[1]) * 0.1
                ])
        
        # Class 1 (Default) prototypes
        mask_class1 = labels_np == 1
        if mask_class1.sum() >= self.n_class1:
            kmeans1 = KMeans(n_clusters=self.n_class1, random_state=self.random_seed, n_init='auto')
            kmeans1.fit(embeddings_np[mask_class1])
            centers1 = kmeans1.cluster_centers_
        else:
            centers1 = embeddings_np[mask_class1][:self.n_class1]
            if len(centers1) < self.n_class1:
                padding = self.n_class1 - len(centers1)
                centers1 = np.vstack([
                    centers1,
                    embeddings_np[mask_class1].mean(axis=0, keepdims=True) + 
                    np.random.randn(padding, embeddings_np.shape[1]) * 0.1
                ])
        
        # Combine and set prototypes
        import numpy as np
        all_centers = np.vstack([centers0, centers1])
        self.prototypes.data = torch.tensor(all_centers, dtype=torch.float32, device=device)
        self._initialized = True
        
        print(f"  Initialized {self.n_class0} Non-Default prototypes, {self.n_class1} Default prototypes")
    
    def compute_similarity(self, z: torch.Tensor) -> torch.Tensor:
        """Compute similarity between latent representations and prototypes."""
        if self.similarity_type == 'rbf':
            distances_sq = torch.cdist(z, self.prototypes, p=2).pow(2)
            sigma = self.sigma
            similarities = torch.exp(-distances_sq / (2 * sigma ** 2))
        elif self.similarity_type == 'cosine':
            z_norm = F.normalize(z, p=2, dim=1)
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)
            similarities = torch.mm(z_norm, proto_norm.t())
            similarities = (similarities + 1) / 2
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        return similarities
    
    def compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Compute L2 distances to prototypes."""
        return torch.cdist(z, self.prototypes, p=2)
    
    def diversity_loss(self) -> torch.Tensor:
        """Encourage prototypes to be different from each other."""
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarity_matrix = torch.mm(proto_norm, proto_norm.t())
        mask = 1.0 - torch.eye(self.n_prototypes, device=self.prototypes.device)
        return (similarity_matrix * mask).sum() / mask.sum()
    
    def class_separation_loss(self) -> torch.Tensor:
        """
        Encourage prototypes of different classes to be far apart.
        
        Penalizes similarity between Class 0 and Class 1 prototypes.
        """
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # Class 0 prototypes: indices 0 to n_class0-1
        # Class 1 prototypes: indices n_class0 to n_prototypes-1
        proto_class0 = proto_norm[:self.n_class0]  # (n_class0, D)
        proto_class1 = proto_norm[self.n_class0:]  # (n_class1, D)
        
        # Cross-class similarity (want to minimize)
        cross_sim = torch.mm(proto_class0, proto_class1.t())  # (n_class0, n_class1)
        
        return cross_sim.mean()
    
    def clustering_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Class-aware clustering loss.
        
        Encourages samples to be close to prototypes of their own class.
        """
        distances = self.compute_distances(z)  # (B, P)
        
        # For each sample, only consider distances to same-class prototypes
        class_mask = (labels.unsqueeze(1) == self.prototype_classes.unsqueeze(0).float())  # (B, P)
        
        # Set distances to other-class prototypes to large value
        masked_distances = distances.clone()
        masked_distances[~class_mask] = float('inf')
        
        # Minimum distance to same-class prototype
        min_distances = masked_distances.min(dim=1).values
        
        # Handle inf (shouldn't happen if properly configured)
        min_distances = torch.clamp(min_distances, max=100.0)
        
        return min_distances.mean()
    
    def get_prototype_class_labels(self) -> torch.Tensor:
        """Return class assignment for each prototype."""
        return self.prototype_classes
    
    def get_class_prototypes(self, class_idx: int) -> torch.Tensor:
        """Get prototypes for a specific class."""
        mask = self.prototype_classes == class_idx
        return self.prototypes[mask]
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute similarities to all prototypes."""
        return self.compute_similarity(z)



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
        
        # Bias initialization to -2.0 (approx log(0.14/0.86)) to account for class imbalance
        # This is CRITICAL because weight_raw -> softplus -> positive. 
        # Without negative bias, logits are always positive -> prob > 0.5 -> predicts Class 1 always.
        self.bias = nn.Parameter(torch.ones(n_classes) * -2.0)
        
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


class Projector(nn.Module):
    """
    Projector network for PTaRL.
    
    Maps embedding vectors to K-dimensional coordinates for P-Space construction.
    """
    
    def __init__(self, emb_dim: int, n_prototypes: int, n_layers: int = 3, dropout: float = 0.1):
        """
        Args:
            emb_dim: Dimension of input embedding
            n_prototypes: Number of global prototypes (K)
            n_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        for _ in range(n_layers):
            layers.append(nn.BatchNorm1d(emb_dim))
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(emb_dim, n_prototypes)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate coordinates for P-Space.
        
        Args:
            x: (B, D) embedding vectors from backbone
            
        Returns:
            coordinates: (B, K) coordinate values for each prototype
        """
        hidden = self.hidden(x)
        coordinates = self.output(hidden)
        return coordinates


class GlobalPrototypeLayer(nn.Module):
    """
    Global Prototype Layer for PTaRL.
    
    Uses K global prototypes initialized via KMeans clustering.
    The P-Space is constructed as: p_space = coordinates @ prototypes
    """
    
    def __init__(
        self,
        n_prototypes: int,
        prototype_dim: int,
        random_seed: int = 42
    ):
        """
        Args:
            n_prototypes: Number of global prototypes K (typically ceil(log2(input_dims)))
            prototype_dim: Dimension of each prototype (should match embedding dim)
            random_seed: Random seed for KMeans initialization
        """
        super().__init__()
        self.n_prototypes = n_prototypes
        self.prototype_dim = prototype_dim
        self.random_seed = random_seed
        
        # Global prototypes (will be initialized via KMeans)
        self.prototypes = nn.Parameter(torch.zeros(n_prototypes, prototype_dim))
        nn.init.normal_(self.prototypes, std=1.0)
        
        self._initialized = False
    
    def initialize_from_embeddings(self, embeddings: torch.Tensor):
        """
        Initialize global prototypes using KMeans on embeddings.
        
        Args:
            embeddings: (N, D) embeddings from training data
        """
        from sklearn.cluster import KMeans
        import numpy as np
        
        device = self.prototypes.device
        emb_np = embeddings.detach().cpu().numpy()
        
        kmeans = KMeans(
            n_clusters=self.n_prototypes,
            random_state=self.random_seed,
            n_init='auto'
        ).fit(emb_np)
        
        self.prototypes.data = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.float32,
            device=device
        )
        self._initialized = True
    
    def construct_p_space(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Construct P-Space representation.
        
        Args:
            coordinates: (B, K) coordinates from projector
            
        Returns:
            p_space: (B, D) P-Space representation
        """
        return torch.mm(coordinates, self.prototypes)  # (B, D)
    
    def orthogonalization_loss(self) -> torch.Tensor:
        """
        Compute orthogonalization loss to encourage prototype independence.
        
        Prototypes should be orthogonal to each other for disentangled learning.
        
        Returns:
            loss: Orthogonalization loss
        """
        # Compute normalized prototype similarity matrix
        r = torch.sqrt(torch.sum(self.prototypes ** 2, dim=1, keepdim=True) + 1e-8)
        proto_matrix = torch.mm(self.prototypes, self.prototypes.t()) / (torch.mm(r, r.t()) + 1e-8)
        proto_matrix = torch.clamp(proto_matrix.abs(), 0, 1)
        
        # L1 and L2 norms
        l1 = torch.sum(proto_matrix.abs())
        l2 = torch.sum(proto_matrix ** 2)
        
        # Sparsity loss
        loss_sparse = l1 / (l2 + 1e-8)
        
        # Constraint loss (off-diagonal should be zero)
        loss_constraint = torch.abs(l1 - self.n_prototypes)
        
        return loss_sparse + 0.5 * loss_constraint
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: construct P-Space.
        
        Args:
            coordinates: (B, K) coordinates from projector
            
        Returns:
            p_space: (B, D) P-Space representation
        """
        return self.construct_p_space(coordinates)


def diversifying_loss(
    coordinates: torch.Tensor,
    labels: torch.Tensor,
    is_regression: bool = False,
    sample_ratio: float = 0.5
) -> torch.Tensor:
    """
    Compute diversifying loss (contrastive learning inspired).
    
    Encourages samples of the same class to have similar coordinates.
    
    Args:
        coordinates: (B, K) coordinate vectors
        labels: (B,) class labels
        is_regression: Whether this is a regression task
        sample_ratio: Proportion of samples to use
        
    Returns:
        loss: Diversifying loss
    """
    import numpy as np
    
    if coordinates.shape[0] < 2:
        return torch.tensor(0.0, device=coordinates.device)
    
    # Sample subset for efficiency
    n_samples = max(2, int(coordinates.shape[0] * sample_ratio))
    indices = np.random.choice(coordinates.shape[0], n_samples, replace=False)
    
    coords = coordinates[indices]
    labs = labels[indices]
    
    # Pairwise L1 distance between coordinates
    distance = (coords.unsqueeze(1) - coords.unsqueeze(0)).abs().sum(dim=2)  # (N, N)
    
    if not is_regression:
        # Classification: same label = positive pair
        label_similarity = (labs.unsqueeze(1) == labs.unsqueeze(0)).float()
    else:
        # Regression: bin labels and compare
        device = coordinates.device
        y_min, y_max = labs.min(), labs.max()
        num_bin = 1 + int(np.log2(len(labs)))
        interval_width = (y_max - y_min) / num_bin + 1e-8
        y_assign = torch.clamp(((labs - y_min) / interval_width).long(), 0, num_bin - 1)
        label_similarity = (y_assign.unsqueeze(1) == y_assign.unsqueeze(0)).float()
    
    # Positive loss: minimize distance for same-class pairs
    positive_mask = label_similarity
    positive_loss = torch.sum(distance * positive_mask) / (torch.sum(positive_mask) + 1e-8)
    
    return positive_loss

