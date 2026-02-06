"""
Configuration module for Prototype-based Neuro-Symbolic Model.
Contains all hyperparameters and feature definitions.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class ModelConfig:
    """Complete configuration for the prototype network."""
    
    # Feature definitions
    numerical_features: List[str] = field(default_factory=lambda: [
        'person_age',
        'person_income', 
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length'
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ])
    
    # Categorical feature cardinalities (will be updated from data)
    categorical_cardinalities: Dict[str, int] = field(default_factory=lambda: {
        'person_home_ownership': 4,  # RENT, OWN, MORTGAGE, OTHER
        'loan_intent': 6,  # EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT
        'loan_grade': 7,  # A, B, C, D, E, F, G
        'cb_person_default_on_file': 2  # Y, N
    })
    
    # FT-Transformer architecture
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ffn: int = 128
    dropout: float = 0.1
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    
    # Prototype layer
    n_prototypes: int = 10
    prototype_dim: int = 64  # Same as d_model
    similarity_type: str = 'rbf'  # 'rbf' or 'cosine'
    rbf_sigma: float = 1.0
    
    # Decoder architecture
    decoder_hidden_dim: int = 128
    
    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    early_stopping_patience: int = 10
    
    # Loss weights
    lambda_reconstruction: float = 0.1
    lambda_diversity: float = 0.01
    lambda_clustering: float = 0.05
    
    # Paths
    train_path: str = '/workspace/data/train.csv'
    test_path: str = '/workspace/data/test.csv'
    model_save_path: str = '/workspace/checkpoints'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Random seed
    seed: int = 42
    
    @property
    def n_numerical(self) -> int:
        return len(self.numerical_features)
    
    @property
    def n_categorical(self) -> int:
        return len(self.categorical_features)
    
    @property
    def n_features(self) -> int:
        return self.n_numerical + self.n_categorical
    
    def get_cardinality_list(self) -> List[int]:
        """Get list of cardinalities in order of categorical_features."""
        return [self.categorical_cardinalities[f] for f in self.categorical_features]


# Prototype descriptions for explainability
PROTOTYPE_DESCRIPTIONS = {
    0: {
        'ko': '고신용 전문가',
        'en': 'High-Credit Professional',
        'description_ko': '안정적인 고소득과 긴 신용 이력을 가진 우량 고객',
        'description_en': 'Premium customer with stable high income and long credit history'
    },
    1: {
        'ko': '안정적 주택 소유자',
        'en': 'Stable Homeowner',
        'description_ko': '주택담보대출을 보유한 중산층 고객',
        'description_en': 'Middle-class customer with mortgage'
    },
    2: {
        'ko': '젊은 임차인',
        'en': 'Young Renter',
        'description_ko': '짧은 근속기간과 임대 주거 형태의 젊은 고객',
        'description_en': 'Young customer with short employment and renting'
    },
    3: {
        'ko': '교육 투자자',
        'en': 'Education Investor',
        'description_ko': '교육 목적 대출을 받는 고객',
        'description_en': 'Customer taking loan for education purposes'
    },
    4: {
        'ko': '의료비 대출자',
        'en': 'Medical Loan Applicant',
        'description_ko': '의료 비용을 위한 대출 신청 고객',
        'description_en': 'Customer applying for medical expense loan'
    },
    5: {
        'ko': '창업 투자자',
        'en': 'Venture Investor',
        'description_ko': '사업 목적의 대출을 받는 고객',
        'description_en': 'Customer taking loan for business venture'
    },
    6: {
        'ko': '부채 통합자',
        'en': 'Debt Consolidator',
        'description_ko': '기존 부채를 통합하려는 고객',
        'description_en': 'Customer consolidating existing debts'
    },
    7: {
        'ko': '고위험 대출자',
        'en': 'High-Risk Borrower',
        'description_ko': '높은 부채비율과 과거 연체 이력이 있는 고객',
        'description_en': 'Customer with high debt ratio and past default history'
    },
    8: {
        'ko': '주택 개선 투자자',
        'en': 'Home Improver',
        'description_ko': '주택 개선 목적의 대출 고객',
        'description_en': 'Customer taking loan for home improvement'
    },
    9: {
        'ko': '개인 용도 대출자',
        'en': 'Personal Loan Applicant',
        'description_ko': '개인적 용도로 대출을 신청하는 고객',
        'description_en': 'Customer applying for personal use loan'
    }
}


def get_default_config() -> ModelConfig:
    """Returns the default configuration."""
    return ModelConfig()
