
# =========================================================================================
# PTaRL: Prototype-based Tabular Representation Learning - Kaggle Inference Script
# For: ICR - Identifying Age-Related Conditions
# =========================================================================================
#
# This is a SELF-CONTAINED inference script for Kaggle submission.
# All model components are embedded directly — no external imports needed.
#
# Usage on Kaggle:
#   1. Upload best_model_phase2.pt as a Kaggle Dataset
#   2. Add the ICR competition dataset as input
#   3. Run this notebook/script
#
# Local testing:
#   python kaggle_submission.py

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------------------

@dataclass
class ICRConfig:
    """Configuration for ICR dataset with PTaRL model."""
    
    numerical_features: List[str] = field(default_factory=lambda: [
        'AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ',
        'BN', 'BP', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL',
        'CR', 'CS', 'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL',
        'DN', 'DU', 'DV', 'DY', 'EB', 'EE', 'EG', 'EH', 'EP', 'EU',
        'FC', 'FD ', 'FE', 'FI', 'FL', 'FR', 'FS', 'GB', 'GE', 'GF',
        'GH', 'GI', 'GL'
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: ['EJ'])
    exclude_columns: List[str] = field(default_factory=lambda: ['Id', 'Class', 'BQ', 'EL'])
    target_column: str = 'Class'
    
    # FT-Transformer architecture (must match training)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ffn: int = 512
    dropout: float = 0.3
    
    # Decoder
    decoder_hidden_dim: int = 256
    
    # PTaRL
    n_global_prototypes: int = 6
    batch_size: int = 32
    seed: int = 42
    
    @property
    def n_numerical(self) -> int:
        return len(self.numerical_features)

# -----------------------------------------------------------------------------------------
# 2. Preprocessor
# -----------------------------------------------------------------------------------------

class ICRPreprocessor:
    """Fits on train data, transforms train/test data."""
    
    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.num_imputers: Dict[str, float] = {}
        self.cat_imputers: Dict[str, Any] = {}
        self._fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'ICRPreprocessor':
        for feat in self.numerical_features:
            if feat in df.columns:
                self.num_imputers[feat] = df[feat].median()
        
        X_num = df[self.numerical_features].copy()
        for feat in self.numerical_features:
            X_num[feat] = X_num[feat].fillna(self.num_imputers.get(feat, 0))
        self.scaler.fit(X_num)
        
        for feat in self.categorical_features:
            if feat in df.columns:
                self.cat_imputers[feat] = df[feat].mode()[0]
                series = df[feat].fillna(self.cat_imputers[feat]).astype(str)
                le = LabelEncoder()
                le.fit(series)
                self.encoders[feat] = le
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_num = df[self.numerical_features].copy()
        for feat in self.numerical_features:
            X_num[feat] = X_num[feat].fillna(self.num_imputers.get(feat, 0))
        X_num_scaled = self.scaler.transform(X_num)
        
        X_cat_list = []
        for feat in self.categorical_features:
            if feat in df.columns:
                series = df[feat].fillna(self.cat_imputers.get(feat, '')).astype(str)
                le = self.encoders[feat]
                mask = ~series.isin(le.classes_)
                if mask.any():
                    series[mask] = str(self.cat_imputers[feat])
                X_cat_list.append(le.transform(series))
            else:
                X_cat_list.append(np.zeros(len(df), dtype=int))
        
        X_cat = np.stack(X_cat_list, axis=1) if X_cat_list else np.zeros((len(df), 0), dtype=np.int64)
        return X_num_scaled.astype(np.float32), X_cat.astype(np.int64)
    
    def get_cardinality_list(self) -> List[int]:
        return [len(self.encoders[f].classes_) for f in self.categorical_features]

# -----------------------------------------------------------------------------------------
# 3. Model Components (must match src/models/ exactly for state_dict compatibility)
# -----------------------------------------------------------------------------------------

class NumericalTokenizer(nn.Module):
    def __init__(self, n_numerical: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_numerical, d_model))
        self.bias = nn.Parameter(torch.empty(n_numerical, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[0]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalEmbedding(nn.Module):
    def __init__(self, cardinalities: List[int], d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, d_model) for c in cardinalities])
    
    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        return torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout, attention_dropout, ffn_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=attention_dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_model), nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout1(attended)
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_numerical, categorical_cardinalities, d_model, n_heads, n_layers, d_ffn,
                 dropout=0.1, attention_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_numerical = n_numerical
        self.n_categorical = len(categorical_cardinalities)
        self.n_tokens = n_numerical + self.n_categorical + 1
        
        self.numerical_tokenizer = NumericalTokenizer(n_numerical, d_model)
        self.categorical_embedding = CategoricalEmbedding(categorical_cardinalities, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ffn, dropout, attention_dropout, ffn_dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_num, x_cat):
        num_tokens = self.numerical_tokenizer(x_num)
        cat_tokens = self.categorical_embedding(x_cat)
        tokens = torch.cat([self.cls_token.expand(x_num.shape[0], -1, -1), num_tokens, cat_tokens], dim=1)
        tokens = tokens + self.position_embedding
        tokens = self.dropout(tokens)
        for block in self.transformer_blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)[:, 0]


class GlobalPrototypeLayer(nn.Module):
    def __init__(self, n_prototypes, prototype_dim, random_seed=42):
        super().__init__()
        self.prototypes = nn.Parameter(torch.zeros(n_prototypes, prototype_dim))
        nn.init.normal_(self.prototypes, std=1.0)
        self.n_prototypes = n_prototypes
        self.random_seed = random_seed
    
    def forward(self, coordinates):
        return torch.mm(coordinates, self.prototypes)


class Projector(nn.Module):
    def __init__(self, emb_dim, n_prototypes, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.BatchNorm1d(emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(emb_dim, n_prototypes)
    
    def forward(self, x):
        return self.output(self.hidden(x))


class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim, n_numerical, categorical_cardinalities, hidden_dim=128):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.numerical_decoder = nn.Linear(hidden_dim, n_numerical)
        self.categorical_decoders = nn.ModuleList([nn.Linear(hidden_dim, c) for c in categorical_cardinalities])
    
    def forward(self, z):
        hidden = self.shared_layers(z)
        return self.numerical_decoder(hidden), [d(hidden) for d in self.categorical_decoders]


class PrototypeNetwork(nn.Module):
    """
    PTaRL PrototypeNetwork — must match src/models/model.py for weight loading.
    Inference-only (no training logic needed).
    """
    def __init__(self, n_numerical, categorical_cardinalities, d_model=64, n_heads=4,
                 n_layers=3, d_ffn=128, n_global_prototypes=None, decoder_hidden_dim=128,
                 dropout=0.1, projector_layers=3, random_seed=42):
        super().__init__()
        self.n_numerical = n_numerical
        self.n_categorical = len(categorical_cardinalities)
        self.d_model = d_model
        self.random_seed = random_seed
        
        n_features = n_numerical + len(categorical_cardinalities)
        self.n_global_prototypes = n_global_prototypes or int(math.ceil(math.log2(max(n_features, 2))))
        
        self.encoder = FTTransformer(
            n_numerical, categorical_cardinalities, d_model, n_heads, n_layers, d_ffn, dropout
        )
        self.global_prototype_layer = GlobalPrototypeLayer(self.n_global_prototypes, d_model, random_seed)
        self.projector = Projector(d_model, self.n_global_prototypes, projector_layers, dropout)
        
        self.phase1_classifier = nn.Linear(d_model, 1)
        self.pspace_classifier = nn.Linear(d_model, 1)
        self.decoder = FeatureDecoder(d_model, n_numerical, categorical_cardinalities, decoder_hidden_dim)
        
        self._phase = 1

    @property
    def phase(self):
        return self._phase

    def set_first_phase(self):
        self._phase = 1

    def set_second_phase(self):
        self._phase = 2

    def forward(self, x_num, x_cat):
        z = self.encoder(x_num, x_cat)
        if self._phase == 1:
            logits = self.phase1_classifier(z).squeeze(-1)
        else:
            coordinates = self.projector(z)
            p_space = self.global_prototype_layer(coordinates)
            logits = self.pspace_classifier(p_space).squeeze(-1)
        return {'logits': logits, 'probabilities': torch.sigmoid(logits)}

# -----------------------------------------------------------------------------------------
# 4. Path Discovery & Inference
# -----------------------------------------------------------------------------------------

def find_paths():
    """Auto-detect data and model paths (works on Kaggle and locally)."""
    train_path, test_path, model_path = None, None, None
    
    # Search Kaggle input
    if os.path.exists('/kaggle/input'):
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in filenames:
                path = os.path.join(dirname, filename)
                if filename == 'train.csv':
                    train_path = path
                elif filename == 'test.csv':
                    test_path = path
                elif filename.endswith('.pt') and 'best_model' in filename:
                    # Prefer phase2 if available
                    if model_path is None or 'phase2' in filename:
                        model_path = path
    
    # Local fallbacks
    if not train_path and os.path.exists('/workspace/data/icr/train.csv'):
        train_path = '/workspace/data/icr/train.csv'
        test_path = '/workspace/data/icr/test.csv'
    
    if not model_path:
        for p in ['/workspace/checkpoints/icr/best_model_phase2.pt',
                  '/workspace/checkpoints/icr/best_model_phase1.pt']:
            if os.path.exists(p):
                model_path = p
                break
    
    return train_path, test_path, model_path


def main():
    print("=" * 60)
    print("PTaRL - ICR Inference")
    print("=" * 60)
    
    train_path, test_path, model_path = find_paths()
    
    if not train_path or not test_path:
        print("ERROR: Could not find train.csv or test.csv")
        return
    if not model_path:
        print("ERROR: Could not find model checkpoint (.pt)")
        return
    
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print(f"  Model: {model_path}")
    
    # Determine phase from model filename
    phase = 2 if 'phase2' in model_path else 1
    print(f"  Phase: {phase}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Config & Preprocessing
    config = ICRConfig()
    preprocessor = ICRPreprocessor(config.numerical_features, config.categorical_features)
    preprocessor.fit(train_df)
    
    test_num, test_cat = preprocessor.transform(test_df)
    cardinalities = preprocessor.get_cardinality_list()
    
    # Load model
    print("\nLoading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Infer n_global_prototypes from weights
    n_global = config.n_global_prototypes
    if 'global_prototype_layer.prototypes' in state_dict:
        n_global = state_dict['global_prototype_layer.prototypes'].shape[0]
    
    model = PrototypeNetwork(
        n_numerical=config.n_numerical,
        categorical_cardinalities=cardinalities,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        n_global_prototypes=n_global,
        decoder_hidden_dim=config.decoder_hidden_dim,
        dropout=config.dropout,
        random_seed=config.seed
    ).to(device)
    
    model.load_state_dict(state_dict, strict=False)
    if phase == 2:
        model.set_second_phase()
    else:
        model.set_first_phase()
    model.eval()
    print(f"  Model loaded (phase={phase}, device={device})")
    
    # Inference (batched to avoid OOM on large test sets)
    print("\nRunning inference...")
    all_probs = []
    batch_size = config.batch_size
    
    with torch.no_grad():
        for i in range(0, len(test_num), batch_size):
            x_num = torch.FloatTensor(test_num[i:i+batch_size]).to(device)
            x_cat = torch.LongTensor(test_cat[i:i+batch_size]).to(device)
            outputs = model(x_num, x_cat)
            all_probs.append(outputs['probabilities'].cpu().numpy())
    
    probs = np.concatenate(all_probs).flatten()
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'class_0': 1 - probs,
        'class_1': probs
    })
    
    output_path = 'submission.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved: {output_path}")
    print(f"  Samples: {len(submission)}")
    print(f"  Mean P(class_1): {probs.mean():.4f}")
    print(f"  Std P(class_1):  {probs.std():.4f}")
    print(submission.head())


if __name__ == '__main__':
    main()
