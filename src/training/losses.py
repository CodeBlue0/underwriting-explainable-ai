"""
Loss functions for prototype network training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PrototypeLoss(nn.Module):
    """
    Phase 1 Loss function (aligned with PTaRL paper).
    
    Loss = L_classification + λ_recon * L_reconstruction
    
    Components:
    1. Classification Loss (BCE): Main task loss
    2. Reconstruction Loss: Ensures latent space preserves information (keeps decoder aligned)
    
    Note: Diversity and Clustering losses removed per PTaRL paper.
          Phase 1 is simple supervised learning without prototypes.
    """
    
    def __init__(
        self,
        lambda_reconstruction: float = 0.1,
        lambda_diversity: float = 0.01,  # Kept for backward compat, but unused
        lambda_clustering: float = 0.05,  # Kept for backward compat, but unused
        num_weight: float = 1.0,
        cat_weight: float = 1.0
    ):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
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
        prototype_layer=None  # Not used anymore, kept for backward compat
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Phase 1 losses.
        
        Args:
            outputs: Model output dictionary
            targets: (batch_size,) target labels
            x_num: (batch_size, n_numerical) input numerical features
            x_cat: (batch_size, n_categorical) input categorical features
            prototype_layer: Unused, kept for backward compatibility
            
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
        
        # Total loss (simplified per PTaRL paper)
        total_loss = loss_cls + self.lambda_reconstruction * loss_recon
        
        return {
            'total': total_loss,
            'classification': loss_cls,
            'reconstruction': loss_recon,
            'reconstruction_num': loss_recon_num,
            'reconstruction_cat': loss_recon_cat,
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


class PTaRLLoss(nn.Module):
    """
    PTaRL Loss function for Phase 2 training.
    
    Total Loss = task_weight * L_task 
               + projection_weight * L_projection
               + diversifying_weight * L_diversifying
               + orthogonalization_weight * L_orthogonalization
               + reconstruction_weight * L_reconstruction
    
    Components:
    1. Task Loss: BCE classification loss
    2. Projection Loss: Sinkhorn + L1 reconstruction in P-Space
    3. Diversifying Loss: Contrastive-inspired coordinate alignment
    4. Orthogonalization Loss: Prototype independence
    5. Reconstruction Loss: Feature reconstruction to keep decoder aligned with encoder
    """
    
    def __init__(
        self,
        task_weight: float = 1.0,
        projection_weight: float = 1.0,
        diversifying_weight: float = 0.5,
        orthogonalization_weight: float = 2.5,
        reconstruction_weight: float = 0.1,
        sinkhorn_eps: float = 0.1,
        sinkhorn_max_iter: int = 50
    ):
        super().__init__()
        self.task_weight = task_weight
        self.projection_weight = projection_weight
        self.diversifying_weight = diversifying_weight
        self.orthogonalization_weight = orthogonalization_weight
        self.reconstruction_weight = reconstruction_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Import Sinkhorn
        from ..utils.sinkhorn import SinkhornDistance
        self.sinkhorn = SinkhornDistance(
            eps=sinkhorn_eps,
            max_iter=sinkhorn_max_iter,
            metric='cosine'
        )
    
    def projection_loss(
        self,
        z: torch.Tensor,
        prototypes: torch.Tensor,
        coordinates: torch.Tensor,
        p_space: torch.Tensor
    ) -> torch.Tensor:
        """Compute projection loss: Sinkhorn + L1 reconstruction."""
        sinkhorn_loss = torch.mean(self.sinkhorn(z, prototypes, coordinates))
        l1_loss = F.l1_loss(z, p_space)
        return sinkhorn_loss + l1_loss
    
    def reconstruction_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        x_num: torch.Tensor,
        x_cat: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature reconstruction loss to keep decoder aligned."""
        loss_num = self.mse_loss(outputs['num_recon'], x_num)
        
        loss_cat = torch.tensor(0.0, device=x_num.device)
        for i, cat_logits in enumerate(outputs['cat_recons']):
            loss_cat = loss_cat + self.ce_loss(cat_logits, x_cat[:, i])
        if len(outputs['cat_recons']) > 0:
            loss_cat = loss_cat / len(outputs['cat_recons'])
        
        return loss_num + loss_cat
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        global_prototype_layer,
        x_num: torch.Tensor = None,
        x_cat: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PTaRL losses for Phase 2.
        
        Args:
            outputs: Model output dictionary with z, coordinates, p_space, logits
            targets: (batch_size,) target labels
            global_prototype_layer: GlobalPrototypeLayer for orthogonalization
            x_num: Original numerical features (for reconstruction)
            x_cat: Original categorical features (for reconstruction)
            
        Returns:
            Dictionary of losses including 'total' loss
        """
        from ..models.prototype_layer import diversifying_loss
        
        # 1. Task loss (BCE)
        loss_task = self.bce_loss(outputs['logits'], targets)
        
        # 2. Projection loss
        loss_projection = self.projection_loss(
            outputs['z'],
            global_prototype_layer.prototypes,
            outputs['coordinates'],
            outputs['p_space']
        )
        
        # 3. Diversifying loss (contrastive)
        loss_diversifying = diversifying_loss(
            outputs['coordinates'],
            targets,
            is_regression=False
        )
        
        # 4. Orthogonalization loss
        loss_orthogonalization = global_prototype_layer.orthogonalization_loss()
        
        # 5. Reconstruction loss
        if x_num is not None and x_cat is not None and 'num_recon' in outputs:
            loss_reconstruction = self.reconstruction_loss(outputs, x_num, x_cat)
        else:
            loss_reconstruction = torch.tensor(0.0, device=targets.device)
        
        # Total loss
        total_loss = (
            self.task_weight * loss_task
            + self.projection_weight * loss_projection
            + self.diversifying_weight * loss_diversifying
            + self.orthogonalization_weight * loss_orthogonalization
            + self.reconstruction_weight * loss_reconstruction
        )
        
        return {
            'total': total_loss,
            'task': loss_task,
            'projection': loss_projection,
            'diversifying': loss_diversifying,
            'orthogonalization': loss_orthogonalization,
            'reconstruction': loss_reconstruction
        }
