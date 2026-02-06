"""
Loss functions for prototype network training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class PrototypeLoss(nn.Module):
    """
    Combined loss function for prototype network.
    
    Loss = L_classification + λ_recon * L_reconstruction 
         + λ_diversity * L_diversity + λ_clustering * L_clustering
    
    Components:
    1. Classification Loss (BCE): Main task loss
    2. Reconstruction Loss: Ensures latent space preserves information
    3. Diversity Loss: Encourages prototypes to be different
    4. Clustering Loss: Encourages samples to be near prototypes
    """
    
    def __init__(
        self,
        lambda_reconstruction: float = 0.1,
        lambda_diversity: float = 0.01,
        lambda_clustering: float = 0.05,
        num_weight: float = 1.0,
        cat_weight: float = 1.0
    ):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_diversity = lambda_diversity
        self.lambda_clustering = lambda_clustering
        self.num_weight = num_weight
        self.cat_weight = cat_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        prototype_layer
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Model output dictionary
            targets: (batch_size,) target labels
            x_num: (batch_size, n_numerical) input numerical features
            x_cat: (batch_size, n_categorical) input categorical features
            prototype_layer: PrototypeLayer for regularization losses
            
        Returns:
            Dictionary of losses including 'total' loss
        """
        # 1. Classification loss (Binary Cross-Entropy)
        loss_cls = self.bce_loss(outputs['logits'], targets)
        
        # 2. Reconstruction loss
        loss_recon_num = self.mse_loss(outputs['num_recon'], x_num)
        
        loss_recon_cat = torch.tensor(0.0, device=x_num.device)
        for i, cat_logits in enumerate(outputs['cat_recons']):
            loss_recon_cat = loss_recon_cat + self.ce_loss(cat_logits, x_cat[:, i])
        if len(outputs['cat_recons']) > 0:
            loss_recon_cat = loss_recon_cat / len(outputs['cat_recons'])
        
        loss_recon = self.num_weight * loss_recon_num + self.cat_weight * loss_recon_cat
        
        # 3. Diversity loss (encourage prototypes to be different)
        loss_diversity = prototype_layer.diversity_loss()
        
        # 4. Clustering loss (encourage samples to be near prototypes)
        loss_clustering = prototype_layer.clustering_loss(outputs['z'])
        
        # Total loss
        total_loss = (
            loss_cls 
            + self.lambda_reconstruction * loss_recon
            + self.lambda_diversity * loss_diversity
            + self.lambda_clustering * loss_clustering
        )
        
        return {
            'total': total_loss,
            'classification': loss_cls,
            'reconstruction': loss_recon,
            'reconstruction_num': loss_recon_num,
            'reconstruction_cat': loss_recon_cat,
            'diversity': loss_diversity,
            'clustering': loss_clustering
        }


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute alpha weights
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal loss
        focal_loss = alpha_weight * focal_weight * bce
        
        return focal_loss.mean()
