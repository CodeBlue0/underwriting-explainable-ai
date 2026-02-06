"""
FT-Transformer: Feature Tokenizer + Transformer for tabular data.

This module implements the FT-Transformer architecture that converts
mixed numerical and categorical features into a unified representation
using self-attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class NumericalTokenizer(nn.Module):
    """
    Tokenizes numerical features using linear projection.
    Each numerical feature is transformed into a d_model dimensional vector.
    
    Similar to making each feature a "token" for the transformer.
    """
    
    def __init__(self, n_numerical: int, d_model: int):
        """
        Args:
            n_numerical: Number of numerical features
            d_model: Dimension of each token embedding
        """
        super().__init__()
        self.n_numerical = n_numerical
        self.d_model = d_model
        
        # Each numerical feature gets its own linear projection
        # Weight: (n_numerical, d_model), Bias: (n_numerical, d_model)
        self.weight = nn.Parameter(torch.empty(n_numerical, d_model))
        self.bias = nn.Parameter(torch.empty(n_numerical, d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with small random values (similar to paper)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[0]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num: (batch_size, n_numerical) - scaled numerical features
            
        Returns:
            (batch_size, n_numerical, d_model) - tokenized features
        """
        # x_num: (B, N) -> (B, N, 1)
        # weight: (N, D) -> multiply with x_num
        # Result: (B, N, D)
        x = x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return x


class CategoricalEmbedding(nn.Module):
    """
    Embedding layer for categorical features.
    Each categorical feature gets its own embedding table.
    """
    
    def __init__(self, cardinalities: List[int], d_model: int):
        """
        Args:
            cardinalities: List of cardinality for each categorical feature
            d_model: Dimension of each embedding
        """
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, d_model) for card in cardinalities
        ])
    
    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: (batch_size, n_categorical) - encoded categorical indices
            
        Returns:
            (batch_size, n_categorical, d_model) - embedded features
        """
        embedded = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        return torch.stack(embedded, dim=1)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout1(attended)
        
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular data.
    
    Architecture:
    1. Tokenize numerical features (linear projection)
    2. Embed categorical features (lookup)
    3. Add [CLS] token
    4. Apply transformer layers
    5. Return [CLS] representation as latent vector z
    """
    
    def __init__(
        self,
        n_numerical: int,
        categorical_cardinalities: List[int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ffn: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1
    ):
        """
        Args:
            n_numerical: Number of numerical features
            categorical_cardinalities: List of cardinality for each categorical feature
            d_model: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ffn: FFN hidden dimension
            dropout: General dropout rate
            attention_dropout: Attention dropout rate
            ffn_dropout: FFN dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_numerical = n_numerical
        self.n_categorical = len(categorical_cardinalities)
        self.n_tokens = n_numerical + self.n_categorical + 1  # +1 for CLS
        
        # Tokenizers
        self.numerical_tokenizer = NumericalTokenizer(n_numerical, d_model)
        self.categorical_embedding = CategoricalEmbedding(categorical_cardinalities, d_model)
        
        # Learnable [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional embedding (optional but helps)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                d_ffn,
                dropout,
                attention_dropout,
                ffn_dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_num: (batch_size, n_numerical) - scaled numerical features
            x_cat: (batch_size, n_categorical) - encoded categorical indices
            
        Returns:
            z: (batch_size, d_model) - latent representation from [CLS] token
        """
        batch_size = x_num.shape[0]
        
        # Tokenize numerical features: (B, N_num, D)
        num_tokens = self.numerical_tokenizer(x_num)
        
        # Embed categorical features: (B, N_cat, D)
        cat_tokens = self.categorical_embedding(x_cat)
        
        # Concatenate all feature tokens: (B, N_num + N_cat, D)
        feature_tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        
        # Prepend [CLS] token: (B, 1 + N_num + N_cat, D)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1)
        
        # Add positional embedding
        tokens = tokens + self.position_embedding
        tokens = self.dropout(tokens)
        
        # Apply transformer layers
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        # Final normalization and extract [CLS] token
        tokens = self.final_norm(tokens)
        z = tokens[:, 0]  # [CLS] token is at position 0
        
        return z
    
    def get_attention_weights(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor
    ) -> List[torch.Tensor]:
        """Get attention weights from all layers (for interpretability)."""
        batch_size = x_num.shape[0]
        
        # Same tokenization as forward
        num_tokens = self.numerical_tokenizer(x_num)
        cat_tokens = self.categorical_embedding(x_cat)
        feature_tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1)
        tokens = tokens + self.position_embedding
        
        attention_weights = []
        for block in self.transformer_blocks:
            normed = block.norm1(tokens)
            _, weights = block.attention(normed, normed, normed, need_weights=True)
            attention_weights.append(weights)
            tokens = block(tokens)
        
        return attention_weights
