"""
Complete Prototype Network Model.

Combines all components into a single end-to-end model:
1. FT-Transformer encoder
2. Prototype layer
3. Classification head
4. Feature decoder
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from .ft_transformer import FTTransformer
from .prototype_layer import PrototypeLayer, ClassificationHead
from .decoder import FeatureDecoder


class PrototypeNetwork(nn.Module):
    """
    Complete prototype-based neuro-symbolic model for explainable predictions.
    
    Architecture:
        Input -> FT-Transformer -> Latent z -> Prototype Similarities -> Classification
                                     |
                                     v
                                  Decoder -> Reconstructed Features
    """
    
    def __init__(
        self,
        n_numerical: int,
        categorical_cardinalities: List[int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ffn: int = 128,
        n_prototypes: int = 10,
        similarity_type: str = 'rbf',
        rbf_sigma: float = 1.0,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            n_numerical: Number of numerical features
            categorical_cardinalities: Cardinality for each categorical feature
            d_model: Hidden dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ffn: FFN dimension in transformer
            n_prototypes: Number of learnable prototypes
            similarity_type: 'rbf' or 'cosine' for prototype similarity
            rbf_sigma: Sigma for RBF kernel
            decoder_hidden_dim: Hidden dimension for decoder
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_numerical = n_numerical
        self.n_categorical = len(categorical_cardinalities)
        self.categorical_cardinalities = categorical_cardinalities
        self.n_prototypes = n_prototypes
        self.d_model = d_model
        
        # 1. FT-Transformer Encoder
        self.encoder = FTTransformer(
            n_numerical=n_numerical,
            categorical_cardinalities=categorical_cardinalities,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=d_ffn,
            dropout=dropout
        )
        
        # 2. Prototype Layer
        self.prototype_layer = PrototypeLayer(
            n_prototypes=n_prototypes,
            prototype_dim=d_model,
            similarity_type=similarity_type,
            rbf_sigma=rbf_sigma
        )
        
        # 3. Classification Head (with non-negative weights)
        self.classifier = ClassificationHead(
            n_prototypes=n_prototypes,
            n_classes=1  # Binary classification
        )
        
        # 4. Feature Decoder
        self.decoder = FeatureDecoder(
            latent_dim=d_model,
            n_numerical=n_numerical,
            categorical_cardinalities=categorical_cardinalities,
            hidden_dim=decoder_hidden_dim
        )
    
    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass through the entire model.
        
        Args:
            x_num: (batch_size, n_numerical) scaled numerical features
            x_cat: (batch_size, n_categorical) encoded categorical indices
            return_all: If True, return all intermediate outputs
            
        Returns:
            Dictionary containing:
                - logits: (batch_size,) classification logits
                - probabilities: (batch_size,) sigmoid probabilities
                - z: (batch_size, d_model) latent representations
                - similarities: (batch_size, n_prototypes) prototype similarities
                - contributions: (batch_size, n_prototypes) weighted contributions
                - num_recon: (batch_size, n_numerical) reconstructed numerical
                - cat_recons: List of categorical reconstruction logits
        """
        # 1. Encode features to latent space
        z = self.encoder(x_num, x_cat)
        
        # 2. Compute prototype similarities
        similarities = self.prototype_layer(z)
        
        # 3. Classification
        logits = self.classifier(similarities)
        probabilities = torch.sigmoid(logits)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
        }
        
        if return_all:
            # 4. Decode for reconstruction
            num_recon, cat_recons = self.decoder(z)
            
            # Get weighted contributions of each prototype
            contributions = self.classifier.get_prototype_contributions(similarities)
            
            result.update({
                'z': z,
                'similarities': similarities,
                'contributions': contributions,
                'num_recon': num_recon,
                'cat_recons': cat_recons
            })
        
        return result
    
    def predict(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Simple prediction returning only probabilities."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x_num, x_cat, return_all=False)
        return output['probabilities']
    
    def get_prototype_features(self) -> torch.Tensor:
        """Decode all prototypes to feature space."""
        with torch.no_grad():
            prototypes = self.prototype_layer.prototypes
            num_features, cat_logits = self.decoder(prototypes)
            cat_predictions = [logits.argmax(dim=1) for logits in cat_logits]
        return num_features, cat_predictions
    
    def compute_regularization_losses(
        self,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses.
        
        Args:
            z: Latent representations
            
        Returns:
            Dictionary of regularization losses
        """
        diversity_loss = self.prototype_layer.diversity_loss()
        clustering_loss = self.prototype_layer.clustering_loss(z)
        
        return {
            'diversity_loss': diversity_loss,
            'clustering_loss': clustering_loss
        }
    
    def get_explanation(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for a prediction.
        
        Args:
            x_num: (1, n_numerical) single sample numerical features
            x_cat: (1, n_categorical) single sample categorical features
            top_k: Number of top prototypes to return
            
        Returns:
            Explanation dictionary
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x_num, x_cat, return_all=True)
            
            similarities = output['similarities'][0]  # (n_prototypes,)
            contributions = output['contributions'][0]  # (n_prototypes,)
            
            # Get top-k prototypes
            top_values, top_indices = similarities.topk(top_k)
            
            # Get classifier weights
            weights = self.classifier.weight.squeeze(-1)
            
            explanation = {
                'prediction': output['probabilities'][0].item(),
                'logit': output['logits'][0].item(),
                'top_prototypes': [
                    {
                        'index': idx.item(),
                        'similarity': similarities[idx].item(),
                        'weight': weights[idx].item(),
                        'contribution': contributions[idx].item()
                    }
                    for idx in top_indices
                ],
                'all_similarities': similarities.cpu().numpy(),
                'all_contributions': contributions.cpu().numpy(),
                'latent_z': output['z'][0].cpu().numpy()
            }
            
        return explanation


def create_model_from_config(config) -> PrototypeNetwork:
    """
    Create PrototypeNetwork from configuration object.
    
    Args:
        config: ModelConfig object
        
    Returns:
        Initialized PrototypeNetwork
    """
    return PrototypeNetwork(
        n_numerical=config.n_numerical,
        categorical_cardinalities=config.get_cardinality_list(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        n_prototypes=config.n_prototypes,
        similarity_type=config.similarity_type,
        rbf_sigma=config.rbf_sigma,
        decoder_hidden_dim=config.decoder_hidden_dim,
        dropout=config.dropout
    )
