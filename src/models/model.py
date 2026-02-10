"""
Prototype Network Model with PTaRL Two-Phase Learning.

Combines all components into a single end-to-end model:
1. FT-Transformer encoder
2. Class-balanced prototype layer (Phase 1)
3. Global prototype layer with P-Space (Phase 2)
4. Classification heads
5. Feature decoder
"""
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from .ft_transformer import FTTransformer
from .prototype_layer import GlobalPrototypeLayer, Projector
from .decoder import FeatureDecoder


class PrototypeNetwork(nn.Module):
    """
    PTaRL-enhanced Prototype Network with Class-Balanced Local Prototypes.
    
    Implements two-phase learning:
    - Phase 1: Supervised learning with class-balanced local prototypes
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
            decoder_hidden_dim: Hidden dimension for decoder
            dropout: Dropout rate
            projector_layers: Number of layers in projector
            random_seed: Random seed for KMeans initialization
        """
        super().__init__()
        
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
        
        # 2. Global Prototype Layer (for Phase 2 - PTaRL)
        self.global_prototype_layer = GlobalPrototypeLayer(
            n_prototypes=self.n_global_prototypes,
            prototype_dim=d_model,
            random_seed=random_seed
        )
        
        # 3. Projector (for Phase 2 - PTaRL)
        self.projector = Projector(
            emb_dim=d_model,
            n_prototypes=self.n_global_prototypes,
            n_layers=projector_layers,
            dropout=dropout
        )
        
        # 4. Classification Heads
        # Phase 1: Simple linear classifier on embeddings (per PTaRL paper)
        self.phase1_classifier = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.phase1_classifier.weight)
        nn.init.constant_(self.phase1_classifier.bias, -2.0)  # Initialize for imbalanced data
        
        # Phase 2: P-Space classifier
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
    def n_prototypes(self) -> int:
        """Return number of prototypes (only global in Phase 2)."""
        if self._phase == 2:
            return self.n_global_prototypes
        return 0
    
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
        Phase 1 forward pass: Simple encoder → classifier (per PTaRL paper).
        
        No local prototype layer - just learns good representations.
        Reconstruction loss is used to ensure encoder learns useful features.
        """
        z = self.encoder(x_num, x_cat)
        
        # Simple linear classification on embeddings
        logits = self.phase1_classifier(z).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'z': z,
        }
        
        if return_all:
            # Decoder for reconstruction loss
            num_recon, cat_recons = self.decoder(z)
            
            result.update({
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
        """Phase 2 forward pass: PTaRL with P-Space."""
        z = self.encoder(x_num, x_cat)
        coordinates = self.projector(z)
        p_space = self.global_prototype_layer(coordinates)
        
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
        """Forward pass based on current phase."""
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
                # Phase 1 without prototypes (Standard Supervised)
                explanation = {
                    'prediction': output['probabilities'][0].item(),
                    'logit': output['logits'][0].item(),
                    'phase': 1,
                    'top_prototypes': [],
                    'latent_z': output['z'][0].cpu().numpy(),
                    'note': 'Phase 1 does not use prototypes for explanation.'
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


def create_model_from_config(config) -> PrototypeNetwork:
    """
    Create PrototypeNetwork from configuration object.
    
    Args:
        config: ModelConfig object with model settings
        
    Returns:
        Initialized PrototypeNetwork
    """
    n_global = getattr(config, 'n_global_prototypes', None)
    
    return PrototypeNetwork(
        n_numerical=config.n_numerical,
        categorical_cardinalities=config.get_cardinality_list(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        n_global_prototypes=n_global,
        decoder_hidden_dim=config.decoder_hidden_dim,
        dropout=config.dropout,
        random_seed=config.seed
    )
