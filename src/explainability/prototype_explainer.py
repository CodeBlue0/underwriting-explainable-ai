"""
Prototype-based explainability module.

Extracts human-interpretable explanations from prototype network predictions.
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import torch.nn as nn


class PrototypeExplainer:
    """
    Extracts and formats explanations from prototype network.
    
    Provides:
    1. Top-k most similar prototypes and their contributions
    2. Decoded prototype features in original scale
    3. Feature importance based on attention/contribution
    4. Comparison with similar training cases
    """
    
    def __init__(
        self,
        model: nn.Module,
        preprocessor,
        prototype_descriptions: Optional[Dict[int, Dict[str, str]]] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Trained PrototypeNetwork
            preprocessor: Preprocessor for inverse transforms (e.g., ICRPreprocessor)
            prototype_descriptions: Dict mapping prototype idx -> descriptions
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.device = device
        
        # Default prototype descriptions
        self.prototype_descriptions = prototype_descriptions or {}
        
        # Cache decoded prototypes
        self._decoded_prototypes = None
    
    def get_decoded_prototypes(self) -> Dict[int, Dict[str, Any]]:
        """
        Decode all prototypes to interpretable feature space.
        
        Returns:
            Dict mapping prototype index to decoded features
        """
        if self._decoded_prototypes is not None:
            return self._decoded_prototypes
        
        self._decoded_prototypes = {}
        
        # Phase 1 has no prototypes
        if hasattr(self.model, 'phase') and self.model.phase == 1:
            return self._decoded_prototypes
            
        with torch.no_grad():
            # Phase 2: Global Prototypes
            prototypes = self.model.global_prototype_layer.prototypes
            
            num_recon, cat_logits = self.model.decoder(prototypes)
            
            # Convert to numpy and inverse transform
            num_features = num_recon.cpu().numpy()
            num_original = self.preprocessor.inverse_transform_numerical(num_features)
            
            # Get categorical predictions
            cat_preds = np.stack([
                logits.argmax(dim=1).cpu().numpy() 
                for logits in cat_logits
            ], axis=1)
            cat_labels = self.preprocessor.inverse_transform_categorical(cat_preds)
            
            # Build decoded prototypes dict
            num_names = self.preprocessor.numerical_features
            cat_names = self.preprocessor.categorical_features
            
            n_prototypes = self.model.n_global_prototypes
            
            for i in range(n_prototypes):
                proto_features = {}
                
                # Numerical features
                for j, name in enumerate(num_names):
                    proto_features[name] = float(num_original[i, j])
                
                # Categorical features
                for j, name in enumerate(cat_names):
                    proto_features[name] = cat_labels[name][i]
                
                self._decoded_prototypes[i] = proto_features
        
        return self._decoded_prototypes
    
    def explain_single(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction.
        
        Args:
            x_num: (n_numerical,) preprocessed numerical features
            x_cat: (n_categorical,) encoded categorical features
            top_k: Number of top prototypes to include
            
        Returns:
            Explanation dictionary
        """
        # Convert to tensors
        x_num_t = torch.from_numpy(x_num).float().unsqueeze(0).to(self.device)
        x_cat_t = torch.from_numpy(x_cat).long().unsqueeze(0).to(self.device)
        
        # Get model explanation
        explanation = self.model.get_explanation(x_num_t, x_cat_t, top_k=top_k)
        
        # Compute classifier weights for interpretation
        weights = []
        if hasattr(self.model, 'phase') and self.model.phase == 2:
            weights = self.model.pspace_classifier.weight.squeeze(-1).detach().cpu().numpy().tolist()
        
        explanation['prototype_weights'] = weights

        # Handle PTaRL Phase 2 explanation structure
        if 'top_global_prototypes' in explanation:
            # Map global prototypes to standard format
            # Note: In Phase 2, pspace_classifier operates on d_model-dimensional P-Space,
            # not on per-prototype weights. Use coordinate magnitude as contribution.
            explanation['top_prototypes'] = []
            for p in explanation['top_global_prototypes']:
                coord_val = p['coordinate']
                explanation['top_prototypes'].append({
                    'index': p['index'],
                    'similarity': coord_val,  # Use coordinate as similarity measure
                    'weight': abs(coord_val),  # Use absolute coordinate as weight
                    'contribution': coord_val
                })
        
        # Get decoded prototype features
        decoded_prototypes = self.get_decoded_prototypes()
        
        # Enrich with prototype descriptions and features
        if 'top_prototypes' in explanation:
            for proto_info in explanation['top_prototypes']:
                idx = proto_info['index']
                
                # Add description if available
                if idx in self.prototype_descriptions:
                    proto_info['name_ko'] = self.prototype_descriptions[idx].get('ko', f'프로토타입 {idx}')
                    proto_info['name_en'] = self.prototype_descriptions[idx].get('en', f'Prototype {idx}')
                    proto_info['description_ko'] = self.prototype_descriptions[idx].get('description_ko', '')
                    proto_info['description_en'] = self.prototype_descriptions[idx].get('description_en', '')
                else:
                    proto_info['name_ko'] = f'프로토타입 {idx}'
                    proto_info['name_en'] = f'Prototype {idx}'
                
                # Add decoded features
                if idx in decoded_prototypes:
                    proto_info['features'] = decoded_prototypes[idx]
        
        # Get original scale input features
        num_original = self.preprocessor.inverse_transform_numerical(x_num.reshape(1, -1))[0]
        cat_labels = self.preprocessor.inverse_transform_categorical(x_cat.reshape(1, -1))
        
        input_features = {}
        for j, name in enumerate(self.preprocessor.numerical_features):
            input_features[name] = float(num_original[j])
        for j, name in enumerate(self.preprocessor.categorical_features):
            input_features[name] = cat_labels[name][0]
        
        explanation['input_features'] = input_features
        
        return explanation
    
    def explain_batch(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of samples.
        
        Args:
            x_num: (batch_size, n_numerical) preprocessed numerical features
            x_cat: (batch_size, n_categorical) encoded categorical features
            top_k: Number of top prototypes per sample
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for i in range(len(x_num)):
            exp = self.explain_single(x_num[i], x_cat[i], top_k=top_k)
            explanations.append(exp)
        return explanations
    
    def get_prototype_summary(self) -> Dict[int, Dict[str, Any]]:
        """
        Get summary of all prototypes with descriptions and features.
        
        Returns:
            Dict mapping prototype index to summary info
        """
        decoded = self.get_decoded_prototypes()
        
        if not decoded:
            return {}
            
        weights = []
        if hasattr(self.model, 'phase') and self.model.phase == 2:
            weights = self.model.pspace_classifier.weight.squeeze(-1).detach().cpu().numpy()
        else:
            # Fallback (shouldn't happen in Phase 1 if decoded is empty)
            return {}
        
        summary = {}
        for i in range(len(weights)):
            info = {
                'features': decoded.get(i, {}),
                'weight': float(weights[i])
            }
            
            if i in self.prototype_descriptions:
                info.update(self.prototype_descriptions[i])
            else:
                info['ko'] = f'프로토타입 {i}'
                info['en'] = f'Prototype {i}'
            
            summary[i] = info
        
        return summary
