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
            n_classes = cat_logits.shape[1]
            target_i = cat_target[:, i].clamp(0, n_classes - 1)
            cat_losses.append(self.ce_loss(cat_logits, target_i))
        cat_loss = torch.stack(cat_losses).mean() if cat_losses else torch.tensor(0.0)
        
        # Combined loss
        total_loss = self.num_weight * num_loss + self.cat_weight * cat_loss
        
        return total_loss, num_loss, cat_loss


class DecoderEvaluator:
    """
    Evaluator for decoder reconstruction quality.
    
    Metrics:
    - Numerical: MSE, MAE, R² score per feature
    - Categorical: Accuracy per feature
    """
    
    def __init__(self, feature_names: dict = None):
        """
        Args:
            feature_names: Optional dict with 'numerical' and 'categorical' feature name lists
        """
        self.feature_names = feature_names or {}
    
    @torch.no_grad()
    def evaluate(
        self,
        model,
        data_loader,
        device: str = 'cpu'
    ) -> dict:
        """
        Evaluate decoder on a data loader.
        
        Args:
            model: PrototypeNetwork model
            data_loader: DataLoader with (x_num, x_cat, targets)
            device: Device for computation
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        model.to(device)
        
        all_num_targets = []
        all_num_preds = []
        all_cat_targets = []
        all_cat_preds = []
        
        for batch in data_loader:
            x_num, x_cat, _ = batch
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            
            # Forward pass to get latent z
            z = model.encoder(x_num, x_cat)
            
            # Decode
            num_recon, cat_logits = model.decoder(z)
            cat_preds = [logits.argmax(dim=1) for logits in cat_logits]
            
            all_num_targets.append(x_num.cpu())
            all_num_preds.append(num_recon.cpu())
            all_cat_targets.append(x_cat.cpu())
            all_cat_preds.append([p.cpu() for p in cat_preds])
        
        # Concatenate
        num_targets = torch.cat(all_num_targets, dim=0).numpy()
        num_preds = torch.cat(all_num_preds, dim=0).numpy()
        cat_targets = torch.cat(all_cat_targets, dim=0).numpy()
        
        # Stack categorical predictions
        n_cat = cat_targets.shape[1] if len(cat_targets.shape) > 1 else 0
        cat_preds_stacked = []
        for i in range(n_cat):
            cat_preds_stacked.append(
                torch.cat([batch_preds[i] for batch_preds in all_cat_preds], dim=0).numpy()
            )
        
        return self._compute_metrics(num_targets, num_preds, cat_targets, cat_preds_stacked)
    
    def _compute_metrics(self, num_targets, num_preds, cat_targets, cat_preds_list) -> dict:
        """Compute all metrics."""
        import numpy as np
        
        metrics = {
            'numerical': {},
            'categorical': {},
            'summary': {}
        }
        
        # Numerical metrics
        n_numerical = num_targets.shape[1]
        num_names = self.feature_names.get('numerical', [f'num_{i}' for i in range(n_numerical)])
        
        mse_per_feature = []
        mae_per_feature = []
        r2_per_feature = []
        
        for i in range(n_numerical):
            targets_i = num_targets[:, i]
            preds_i = num_preds[:, i]
            
            mse = np.mean((targets_i - preds_i) ** 2)
            mae = np.mean(np.abs(targets_i - preds_i))
            
            # R² score
            ss_res = np.sum((targets_i - preds_i) ** 2)
            ss_tot = np.sum((targets_i - np.mean(targets_i)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            name = num_names[i] if i < len(num_names) else f'num_{i}'
            metrics['numerical'][name] = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            mse_per_feature.append(mse)
            mae_per_feature.append(mae)
            r2_per_feature.append(r2)
        
        # Categorical metrics
        n_categorical = len(cat_preds_list)
        cat_names = self.feature_names.get('categorical', [f'cat_{i}' for i in range(n_categorical)])
        
        acc_per_feature = []
        
        for i, preds_i in enumerate(cat_preds_list):
            targets_i = cat_targets[:, i]
            accuracy = np.mean(preds_i == targets_i)
            
            name = cat_names[i] if i < len(cat_names) else f'cat_{i}'
            metrics['categorical'][name] = {
                'accuracy': float(accuracy)
            }
            acc_per_feature.append(accuracy)
        
        # Summary metrics
        metrics['summary'] = {
            'numerical_mse_mean': float(np.mean(mse_per_feature)) if mse_per_feature else 0.0,
            'numerical_mae_mean': float(np.mean(mae_per_feature)) if mae_per_feature else 0.0,
            'numerical_r2_mean': float(np.mean(r2_per_feature)) if r2_per_feature else 0.0,
            'categorical_accuracy_mean': float(np.mean(acc_per_feature)) if acc_per_feature else 0.0,
        }
        
        return metrics
    
    def print_report(self, metrics: dict):
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("DECODER EVALUATION REPORT")
        print("=" * 60)
        
        print("\n--- Numerical Features ---")
        print(f"{'Feature':<25} {'MSE':<12} {'MAE':<12} {'R²':<12}")
        print("-" * 60)
        for name, m in metrics['numerical'].items():
            print(f"{name:<25} {m['mse']:<12.6f} {m['mae']:<12.6f} {m['r2']:<12.4f}")
        
        print("\n--- Categorical Features ---")
        print(f"{'Feature':<25} {'Accuracy':<12}")
        print("-" * 40)
        for name, m in metrics['categorical'].items():
            print(f"{name:<25} {m['accuracy']:<12.4f}")
        
        print("\n--- Summary ---")
        s = metrics['summary']
        print(f"Numerical MSE (mean):       {s['numerical_mse_mean']:.6f}")
        print(f"Numerical MAE (mean):       {s['numerical_mae_mean']:.6f}")
        print(f"Numerical R² (mean):        {s['numerical_r2_mean']:.4f}")
        print(f"Categorical Accuracy (mean): {s['categorical_accuracy_mean']:.4f}")
        print("=" * 60)

