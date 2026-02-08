"""
Sinkhorn Distance for optimal transport.

Used in PTaRL for projection loss computation.
"""
import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    """
    Sinkhorn Distance using the entropic regularization.
    
    Computes the optimal transport distance between sample representations
    and global prototypes.
    """
    
    def __init__(
        self,
        eps: float = 0.1,
        max_iter: int = 50,
        metric: str = 'cosine'
    ):
        """
        Args:
            eps: Regularization parameter (entropy weight)
            max_iter: Maximum number of Sinkhorn iterations
            metric: Distance metric ('cosine' or 'euclidean')
        """
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.metric = metric
    
    def _compute_cost_matrix(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cost matrix between two sets of vectors.
        
        Args:
            x: (B, D) batch of vectors
            y: (K, D) prototype vectors
            
        Returns:
            C: (B, K) cost matrix
        """
        if self.metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
            similarity = torch.mm(x_norm, y_norm.t())
            C = 1 - similarity
        elif self.metric == 'euclidean':
            # Squared Euclidean distance
            C = torch.cdist(x, y, p=2).pow(2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return C
    
    def forward(
        self,
        x: torch.Tensor,
        prototypes: torch.Tensor,
        coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sinkhorn distance weighted by coordinates.
        
        Args:
            x: (B, D) sample representations from backbone
            prototypes: (K, D) global prototype vectors
            coordinates: (B, K) projection coordinates
            
        Returns:
            loss: Scalar Sinkhorn distance loss
        """
        B = x.shape[0]
        K = prototypes.shape[0]
        device = x.device
        
        # Compute cost matrix
        C = self._compute_cost_matrix(x, prototypes)  # (B, K)
        
        # Initialize uniform marginals
        mu = torch.ones(B, device=device) / B
        nu = torch.ones(K, device=device) / K
        
        # Sinkhorn iteration
        # Use log-domain for numerical stability
        u = torch.zeros(B, device=device)
        v = torch.zeros(K, device=device)
        
        # Kernel matrix
        K_mat = torch.exp(-C / self.eps)  # (B, K)
        
        for _ in range(self.max_iter):
            u = torch.log(mu + 1e-8) - torch.logsumexp(
                torch.log(K_mat + 1e-8) + v.unsqueeze(0), dim=1
            )
            v = torch.log(nu + 1e-8) - torch.logsumexp(
                torch.log(K_mat.t() + 1e-8) + u.unsqueeze(0), dim=1
            )
        
        # Compute transport plan
        pi = torch.exp(u.unsqueeze(1) + torch.log(K_mat + 1e-8) + v.unsqueeze(0))
        
        # Weighted Sinkhorn distance using coordinates as weights
        # Weight the cost by the absolute coordinates (importance of each prototype)
        coord_weights = torch.softmax(coordinates.abs(), dim=1)  # (B, K)
        weighted_cost = (pi * C * coord_weights).sum()
        
        return weighted_cost


class SimplifiedSinkhornLoss(nn.Module):
    """
    Simplified Sinkhorn-like loss for P-Space projection.
    
    Measures how well the P-Space representation approximates the original.
    """
    
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        x: torch.Tensor,
        prototypes: torch.Tensor,
        coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute projection alignment loss.
        
        Args:
            x: (B, D) original sample representations
            prototypes: (K, D) global prototype vectors
            coordinates: (B, K) learned coordinates
            
        Returns:
            loss: Alignment loss
        """
        # Reconstruct in P-Space
        p_space = torch.mm(coordinates, prototypes)  # (B, D)
        
        # Cosine similarity between original and P-Space representation
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        p_norm = torch.nn.functional.normalize(p_space, p=2, dim=1)
        
        # Want similarity to be high, so loss = 1 - similarity
        similarity = (x_norm * p_norm).sum(dim=1).mean()
        
        return 1 - similarity
