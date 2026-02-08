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


class ClassBalancedPrototypeNetwork(nn.Module):
    """
    Prototype Network with class-balanced prototypes.
    
    Ensures prototypes are distributed across both Default and Non-Default classes
    by using ClassBalancedPrototypeLayer with:
    - Class-stratified initialization via KMeans
    - Class separation loss to keep prototypes distinct
    - Class-aware clustering loss
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
        n_prototypes_per_class: Optional[int] = None,
        similarity_type: str = 'rbf',
        rbf_sigma: float = 1.0,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        from .prototype_layer import ClassBalancedPrototypeLayer
        
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
        
        # 2. Class-Balanced Prototype Layer
        self.prototype_layer = ClassBalancedPrototypeLayer(
            n_prototypes=n_prototypes,
            prototype_dim=d_model,
            n_prototypes_per_class=n_prototypes_per_class,
            similarity_type=similarity_type,
            rbf_sigma=rbf_sigma
        )
        
        # 3. Classification Head
        self.classifier = ClassificationHead(
            n_prototypes=n_prototypes,
            n_classes=1
        )
        
        # 4. Feature Decoder
        self.decoder = FeatureDecoder(
            latent_dim=d_model,
            n_numerical=n_numerical,
            categorical_cardinalities=categorical_cardinalities,
            hidden_dim=decoder_hidden_dim
        )
        
        self._initialized = False
    
    def initialize_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Initialize prototypes using class-stratified KMeans."""
        self.prototype_layer.initialize_from_data(embeddings, labels)
        self._initialized = True
    
    def generate_embeddings(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Generate embeddings for prototype initialization."""
        return self.encoder(x_num, x_cat)
    
    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, Any]:
        # 1. Encode
        z = self.encoder(x_num, x_cat)
        
        # 2. Compute prototype similarities
        similarities = self.prototype_layer(z)
        
        # 3. Classify
        logits = self.classifier(similarities)
        probabilities = torch.sigmoid(logits)
        
        result = {
            'logits': logits,
            'probabilities': probabilities
        }
        
        if return_all:
            result['z'] = z
            result['similarities'] = similarities
            result['contributions'] = self.classifier.get_prototype_contributions(similarities)
            
            # Decode
            num_recon, cat_recons = self.decoder(z)
            result['num_recon'] = num_recon
            result['cat_recons'] = cat_recons
            
            # Prototype class labels
            result['prototype_classes'] = self.prototype_layer.get_prototype_class_labels()
        
        return result
    
    def get_prototype_class_labels(self) -> torch.Tensor:
        """Get class assignment for each prototype (0=Non-Default, 1=Default)."""
        return self.prototype_layer.get_prototype_class_labels()
    
    def explain_prediction(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        top_k: int = 3
    ) -> dict:
        """Generate detailed explanation for a single prediction."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x_num, x_cat, return_all=True)
            
            similarities = output['similarities'][0]
            contributions = output['contributions'][0]
            prototype_classes = output['prototype_classes']
            weights = self.classifier.weight.squeeze(-1)
            
            top_values, top_indices = similarities.topk(top_k)
            
            explanation = {
                'prediction': output['probabilities'][0].item(),
                'logit': output['logits'][0].item(),
                'top_prototypes': [
                    {
                        'index': idx.item(),
                        'similarity': similarities[idx].item(),
                        'weight': weights[idx].item(),
                        'contribution': contributions[idx].item(),
                        'class': 'Default' if prototype_classes[idx] == 1 else 'Non-Default'
                    }
                    for idx in top_indices
                ],
                'all_similarities': similarities.cpu().numpy(),
                'all_contributions': contributions.cpu().numpy(),
                'prototype_classes': prototype_classes.cpu().numpy(),
                'latent_z': output['z'][0].cpu().numpy()
            }
            
        return explanation


    def get_explanation(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        top_k: int = 3
    ) -> dict:
        """Alias for explain_prediction to match PrototypeNetwork interface."""
        return self.explain_prediction(x_num, x_cat, top_k)


def create_class_balanced_model_from_config(config) -> ClassBalancedPrototypeNetwork:
    """
    Create ClassBalancedPrototypeNetwork from configuration.
    
    This model ensures prototypes are balanced across Default/Non-Default classes.
    """
    n_per_class = getattr(config, 'n_prototypes_per_class', config.n_prototypes // 2)
    
    return ClassBalancedPrototypeNetwork(
        n_numerical=config.n_numerical,
        categorical_cardinalities=config.get_cardinality_list(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        n_prototypes=config.n_prototypes,
        n_prototypes_per_class=n_per_class,
        similarity_type=config.similarity_type,
        rbf_sigma=config.rbf_sigma,
        decoder_hidden_dim=config.decoder_hidden_dim,
        dropout=config.dropout
    )


class PrototypeNetworkPTaRL(nn.Module):
    """
    PTaRL-enhanced Prototype Network.
    
    Implements two-phase learning:
    - Phase 1: Standard supervised learning with the backbone
    - Phase 2: Space Calibration with global prototypes and P-Space
    
    Architecture (Phase 2):
        Input -> FT-Transformer -> Embedding z -> Projector -> Coordinates
                                                      |
                                                      v
                                    Coordinates @ GlobalPrototypes -> P-Space -> Head
    """
    
    def __init__(
        self,
        n_numerical: int,
        categorical_cardinalities: List[int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ffn: int = 128,
        n_global_prototypes: Optional[int] = None,
        n_local_prototypes: int = 10,
        similarity_type: str = 'rbf',
        rbf_sigma: float = 1.0,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.1,
        projector_layers: int = 3,
        random_seed: int = 42
    ):
        """
        Args:
            n_numerical: Number of numerical features
            categorical_cardinalities: Cardinality for each categorical feature
            d_model: Hidden dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ffn: FFN dimension in transformer
            n_global_prototypes: Number of global prototypes (default: ceil(log2(n_features)))
            n_local_prototypes: Number of local prototypes for Phase 1
            similarity_type: 'rbf' or 'cosine' for local prototype similarity
            rbf_sigma: Sigma for RBF kernel
            decoder_hidden_dim: Hidden dimension for decoder
            dropout: Dropout rate
            projector_layers: Number of layers in projector
            random_seed: Random seed for KMeans
        """
        super().__init__()
        
        import math
        
        self.n_numerical = n_numerical
        self.n_categorical = len(categorical_cardinalities)
        self.categorical_cardinalities = categorical_cardinalities
        self.d_model = d_model
        self.random_seed = random_seed
        
        n_features = n_numerical + len(categorical_cardinalities)
        self.n_global_prototypes = n_global_prototypes or int(math.ceil(math.log2(max(n_features, 2))))
        
        # 1. FT-Transformer Encoder (shared across phases)
        self.encoder = FTTransformer(
            n_numerical=n_numerical,
            categorical_cardinalities=categorical_cardinalities,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=d_ffn,
            dropout=dropout
        )
        
        # 2. Local prototype layer (for Phase 1, optional use in Phase 2)
        from .prototype_layer import PrototypeLayer, ClassificationHead, GlobalPrototypeLayer, Projector
        
        self.prototype_layer = PrototypeLayer(
            n_prototypes=n_local_prototypes,
            prototype_dim=d_model,
            similarity_type=similarity_type,
            rbf_sigma=rbf_sigma
        )
        
        # 3. Global Prototype Layer (for Phase 2 - PTaRL)
        self.global_prototype_layer = GlobalPrototypeLayer(
            n_prototypes=self.n_global_prototypes,
            prototype_dim=d_model,
            random_seed=random_seed
        )
        
        # 4. Projector (for Phase 2 - PTaRL)
        self.projector = Projector(
            emb_dim=d_model,
            n_prototypes=self.n_global_prototypes,
            n_layers=projector_layers,
            dropout=dropout
        )
        
        # 5. Classification Heads
        self.classifier = ClassificationHead(
            n_prototypes=n_local_prototypes,
            n_classes=1
        )
        
        self.pspace_classifier = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.pspace_classifier.weight)
        nn.init.constant_(self.pspace_classifier.bias, -2.0)
        
        # 6. Feature Decoder
        self.decoder = FeatureDecoder(
            latent_dim=d_model,
            n_numerical=n_numerical,
            categorical_cardinalities=categorical_cardinalities,
            hidden_dim=decoder_hidden_dim
        )
        
        # Training phase
        self._phase = 1  # 1 or 2
    
    @property
    def phase(self) -> int:
        return self._phase
    
    def set_first_phase(self):
        """Set to first phase: standard supervised learning."""
        self._phase = 1
    
    def set_second_phase(self):
        """Set to second phase: PTaRL space calibration."""
        self._phase = 2
    
    def generate_embeddings(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Generate embeddings using the encoder."""
        return self.encoder(x_num, x_cat)
    
    def initialize_global_prototypes(self, embeddings: torch.Tensor):
        """Initialize global prototypes from embeddings using KMeans."""
        self.global_prototype_layer.initialize_from_embeddings(embeddings)
    
    def first_phase_forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, Any]:
        """
        Phase 1 forward pass: standard prototype-based classification.
        """
        z = self.encoder(x_num, x_cat)
        similarities = self.prototype_layer(z)
        logits = self.classifier(similarities)
        probabilities = torch.sigmoid(logits)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'z': z,
        }
        
        if return_all:
            num_recon, cat_recons = self.decoder(z)
            contributions = self.classifier.get_prototype_contributions(similarities)
            
            result.update({
                'similarities': similarities,
                'contributions': contributions,
                'num_recon': num_recon,
                'cat_recons': cat_recons
            })
        
        return result
    
    def second_phase_forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, Any]:
        """
        Phase 2 forward pass: PTaRL with P-Space.
        """
        # Generate embedding
        z = self.encoder(x_num, x_cat)
        
        # Construct coordinates and P-Space
        coordinates = self.projector(z)
        p_space = self.global_prototype_layer(coordinates)
        
        # Classification from P-Space
        logits = self.pspace_classifier(p_space).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'z': z,
            'coordinates': coordinates,
            'p_space': p_space,
        }
        
        if return_all:
            num_recon, cat_recons = self.decoder(z)
            
            result.update({
                'num_recon': num_recon,
                'cat_recons': cat_recons
            })
        
        return result
    
    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass based on current phase.
        """
        if self._phase == 1:
            return self.first_phase_forward(x_num, x_cat, return_all)
        else:
            return self.second_phase_forward(x_num, x_cat, return_all)
    
    def predict(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Simple prediction returning only probabilities."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x_num, x_cat, return_all=False)
        return output['probabilities']
    
    def get_explanation(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for a prediction.
        
        In Phase 2, explains via P-Space coordinates.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x_num, x_cat, return_all=True)
            
            if self._phase == 1:
                similarities = output['similarities'][0]
                contributions = output['contributions'][0]
                top_values, top_indices = similarities.topk(top_k)
                weights = self.classifier.weight.squeeze(-1)
                
                explanation = {
                    'prediction': output['probabilities'][0].item(),
                    'logit': output['logits'][0].item(),
                    'phase': 1,
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
                    'latent_z': output['z'][0].cpu().numpy()
                }
            else:
                coordinates = output['coordinates'][0]
                top_values, top_indices = coordinates.abs().topk(top_k)
                
                explanation = {
                    'prediction': output['probabilities'][0].item(),
                    'logit': output['logits'][0].item(),
                    'phase': 2,
                    'top_global_prototypes': [
                        {
                            'index': idx.item(),
                            'coordinate': coordinates[idx].item(),
                        }
                        for idx in top_indices
                    ],
                    'all_coordinates': coordinates.cpu().numpy(),
                    'p_space': output['p_space'][0].cpu().numpy(),
                    'latent_z': output['z'][0].cpu().numpy()
                }
        
        return explanation


def create_ptarl_model_from_config(config) -> PrototypeNetworkPTaRL:
    """
    Create PrototypeNetworkPTaRL from configuration object.
    
    Args:
        config: ModelConfig object with PTaRL settings
        
    Returns:
        Initialized PrototypeNetworkPTaRL
    """
    n_global = getattr(config, 'n_global_prototypes', None)
    
    return PrototypeNetworkPTaRL(
        n_numerical=config.n_numerical,
        categorical_cardinalities=config.get_cardinality_list(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        n_global_prototypes=n_global,
        n_local_prototypes=config.n_prototypes,
        similarity_type=config.similarity_type,
        rbf_sigma=config.rbf_sigma,
        decoder_hidden_dim=config.decoder_hidden_dim,
        dropout=config.dropout,
        random_seed=config.seed
    )

