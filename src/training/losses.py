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


class ClassBalancedPrototypeLoss(nn.Module):
    """
    Loss function for ClassBalancedPrototypeLayer.
    
    Adds class separation loss to ensure prototypes remain distinct per class.
    """
    
    def __init__(
        self,
        lambda_reconstruction: float = 0.1,
        lambda_diversity: float = 0.5,
        lambda_clustering: float = 0.05,
        lambda_separation: float = 0.3,
        num_weight: float = 1.0,
        cat_weight: float = 1.0,
        use_class_aware_clustering: bool = True
    ):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_diversity = lambda_diversity
        self.lambda_clustering = lambda_clustering
        self.lambda_separation = lambda_separation
        self.num_weight = num_weight
        self.cat_weight = cat_weight
        self.use_class_aware_clustering = use_class_aware_clustering
        
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
        Compute all losses including class separation.
        """
        # 1. Classification loss
        loss_cls = self.bce_loss(outputs['logits'], targets)
        
        # 2. Reconstruction loss
        loss_recon_num = self.mse_loss(outputs['num_recon'], x_num)
        
        loss_recon_cat = torch.tensor(0.0, device=x_num.device)
        for i, cat_logits in enumerate(outputs['cat_recons']):
            loss_recon_cat = loss_recon_cat + self.ce_loss(cat_logits, x_cat[:, i])
        if len(outputs['cat_recons']) > 0:
            loss_recon_cat = loss_recon_cat / len(outputs['cat_recons'])
        
        loss_recon = self.num_weight * loss_recon_num + self.cat_weight * loss_recon_cat
        
        # 3. Diversity loss
        loss_diversity = prototype_layer.diversity_loss()
        
        # 4. Clustering loss (class-aware if ClassBalancedPrototypeLayer)
        if self.use_class_aware_clustering and hasattr(prototype_layer, 'clustering_loss'):
            # Check if it's ClassBalancedPrototypeLayer with class-aware clustering
            import inspect
            sig = inspect.signature(prototype_layer.clustering_loss)
            if 'labels' in sig.parameters:
                loss_clustering = prototype_layer.clustering_loss(outputs['z'], targets)
            else:
                loss_clustering = prototype_layer.clustering_loss(outputs['z'])
        else:
            # Fallback to simple min distance
            distances = prototype_layer.compute_distances(outputs['z'])
            loss_clustering = distances.min(dim=1).values.mean()
        
        # 5. Class separation loss (unique to ClassBalancedPrototypeLayer)
        if hasattr(prototype_layer, 'class_separation_loss'):
            loss_separation = prototype_layer.class_separation_loss()
        else:
            loss_separation = torch.tensor(0.0, device=x_num.device)
        
        # Total loss
        total_loss = (
            loss_cls 
            + self.lambda_reconstruction * loss_recon
            + self.lambda_diversity * loss_diversity
            + self.lambda_clustering * loss_clustering
            + self.lambda_separation * loss_separation
        )
        
        return {
            'total': total_loss,
            'classification': loss_cls,
            'reconstruction': loss_recon,
            'diversity': loss_diversity,
            'clustering': loss_clustering,
            'separation': loss_separation
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
    
    Components:
    1. Task Loss: BCE classification loss
    2. Projection Loss: Sinkhorn + L1 reconstruction in P-Space
    3. Diversifying Loss: Contrastive-inspired coordinate alignment
    4. Orthogonalization Loss: Prototype independence
    """
    
    def __init__(
        self,
        task_weight: float = 1.0,
        projection_weight: float = 1.0,
        diversifying_weight: float = 0.5,
        orthogonalization_weight: float = 2.5,
        sinkhorn_eps: float = 0.1,
        sinkhorn_max_iter: int = 50
    ):
        super().__init__()
        self.task_weight = task_weight
        self.projection_weight = projection_weight
        self.diversifying_weight = diversifying_weight
        self.orthogonalization_weight = orthogonalization_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
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
        """
        Compute projection loss: Sinkhorn + L1 reconstruction.
        
        Args:
            z: (B, D) original embeddings
            prototypes: (K, D) global prototypes
            coordinates: (B, K) projection coordinates
            p_space: (B, D) P-Space representation
            
        Returns:
            Projection loss
        """
        # Sinkhorn distance
        sinkhorn_loss = torch.mean(self.sinkhorn(z, prototypes, coordinates))
        
        # L1 reconstruction loss
        l1_loss = F.l1_loss(z, p_space)
        
        return sinkhorn_loss + l1_loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        global_prototype_layer
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PTaRL losses for Phase 2.
        
        Args:
            outputs: Model output dictionary with z, coordinates, p_space, logits
            targets: (batch_size,) target labels
            global_prototype_layer: GlobalPrototypeLayer for orthogonalization
            
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
        
        # Total loss
        total_loss = (
            self.task_weight * loss_task
            + self.projection_weight * loss_projection
            + self.diversifying_weight * loss_diversifying
            + self.orthogonalization_weight * loss_orthogonalization
        )
        
        return {
            'total': total_loss,
            'task': loss_task,
            'projection': loss_projection,
            'diversifying': loss_diversifying,
            'orthogonalization': loss_orthogonalization
        }

