"""
Training utilities for prototype network.
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .losses import PrototypeLoss, PTaRLLoss


class Trainer:
    """
    Trainer for PrototypeNetwork with two-phase training.
    
    Phase 1: Standard supervised learning with local prototypes
    Phase 2: Space calibration with global prototypes and PTaRL losses
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer (will be re-initialized for Phase 2)
        self.optimizer = None
        self.scheduler = None
        
        # Loss functions
        self.phase1_criterion = PrototypeLoss(
            lambda_reconstruction=config.lambda_reconstruction,
            lambda_diversity=config.lambda_diversity,
            lambda_clustering=config.lambda_clustering
        )
        
        ptarl_weights = getattr(config, 'ptarl_weights', {
            'task_weight': 1.0,
            'projection_weight': 1.0,
            'diversifying_weight': 0.5,
            'orthogonalization_weight': 2.5
        })
        self.phase2_criterion = PTaRLLoss(**ptarl_weights)
        
        # Training state
        self.current_epoch = 0
        self.current_phase = 1
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'phase': [],
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def _init_optimizer(self, phase: int, epochs: int):
        """Initialize optimizer for given phase."""
        lr = self.config.learning_rate
        if phase == 2:
            lr = lr * 0.5  # Lower learning rate for Phase 2
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr / 100
        )
    
    def _collect_embeddings(self) -> torch.Tensor:
        """Collect all embeddings from training data for KMeans initialization."""
        embeddings, _ = self._collect_embeddings_and_labels()
        return embeddings
    
    def _collect_embeddings_and_labels(self) -> tuple:
        """Collect all embeddings and labels from training data."""
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in self.train_loader:
                x_num, x_cat, targets = batch
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                
                z = self.model.generate_embeddings(x_num, x_cat)
                embeddings.append(z.cpu())
                labels.append(targets)
        
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)
    
    def train_phase1_epoch(self) -> Dict[str, float]:
        """Train one epoch in Phase 1."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            x_num, x_cat, targets = batch
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x_num, x_cat, return_all=True)
            
            losses = self.phase1_criterion(
                outputs, targets, x_num, x_cat,
                self.model.prototype_layer
            )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses['total'].item()
            n_batches += 1
        
        return {'train_loss': total_loss / n_batches}
    
    def train_phase2_epoch(self) -> Dict[str, float]:
        """Train one epoch in Phase 2 with PTaRL losses."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        n_batches = 0
        
        for batch in self.train_loader:
            x_num, x_cat, targets = batch
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x_num, x_cat, return_all=True)
            
            losses = self.phase2_criterion(
                outputs, targets,
                self.model.global_prototype_layer,
                x_num=x_num,
                x_cat=x_cat
            )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses['total'].item()
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            n_batches += 1
        
        for key in loss_components:
            loss_components[key] /= n_batches
        
        return {'train_loss': total_loss / n_batches, **loss_components}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.val_loader:
            x_num, x_cat, targets = batch
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(x_num, x_cat, return_all=True)
            
            if self.current_phase == 1:
                losses = self.phase1_criterion(
                    outputs, targets, x_num, x_cat,
                    self.model.prototype_layer
                )
            else:
                losses = self.phase2_criterion(
                    outputs, targets,
                    self.model.global_prototype_layer,
                    x_num=x_num,
                    x_cat=x_cat
                )
            
            total_loss += losses['total'].item()
            n_batches += 1
            
            probs = outputs['probabilities'].cpu().numpy()
            all_predictions.extend(probs)
            all_targets.extend(targets.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        avg_loss = total_loss / n_batches
        auc = roc_auc_score(all_targets, all_predictions)
        acc = accuracy_score(all_targets, (all_predictions > 0.5).astype(int))
        f1 = f1_score(all_targets, (all_predictions > 0.5).astype(int))
        
        return {
            'val_loss': avg_loss,
            'val_auc': auc,
            'val_acc': acc,
            'val_f1': f1
        }
    
    def train_phase(
        self,
        phase: int,
        epochs: int,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ):
        """Train a single phase."""
        self.current_phase = phase
        
        if phase == 1:
            # Initialize class-balanced local prototypes before Phase 1
            if hasattr(self.model, 'initialize_local_prototypes') and not getattr(self.model, '_local_prototypes_initialized', False):
                if verbose:
                    print("Initializing class-balanced local prototypes via KMeans...")
                embeddings, labels = self._collect_embeddings_and_labels()
                self.model.initialize_local_prototypes(embeddings, labels)
            self.model.set_first_phase()
        else:
            # Initialize global prototypes via KMeans
            if verbose:
                print("Initializing global prototypes via KMeans...")
            embeddings = self._collect_embeddings()
            self.model.initialize_global_prototypes(embeddings)
            self.model.set_second_phase()
        
        self._init_optimizer(phase, epochs)
        self.patience_counter = 0
        phase_best_auc = 0.0
        
        if verbose:
            print(f"\n=== Phase {phase} Training ===")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            if phase == 1:
                train_metrics = self.train_phase1_epoch()
            else:
                train_metrics = self.train_phase2_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record history
            self.history['phase'].append(phase)
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_auc'].append(val_metrics['val_auc'])
            self.history['val_acc'].append(val_metrics['val_acc'])
            self.history['val_f1'].append(val_metrics['val_f1'])
            self.history['learning_rate'].append(current_lr)
            
            # Early stopping
            if val_metrics['val_auc'] > phase_best_auc:
                phase_best_auc = val_metrics['val_auc']
                self.patience_counter = 0
                if val_metrics['val_auc'] > self.best_val_auc:
                    self.best_val_auc = val_metrics['val_auc']
            else:
                self.patience_counter += 1
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"Phase {phase} Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) - "
                      f"Train Loss: {train_metrics['train_loss']:.4f} - "
                      f"Val Loss: {val_metrics['val_loss']:.4f} - "
                      f"Val AUC: {val_metrics['val_auc']:.4f}")
            
            if self.patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping Phase {phase} at epoch {epoch+1}")
                break
    
    def train(
        self,
        phase1_epochs: Optional[int] = None,
        phase2_epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: bool = True,
        phase: Optional[int] = None  # NEW: Train specific phase only
    ) -> Dict[str, List[float]]:
        """
        Full two-phase training or single phase training.
        
        Args:
            phase1_epochs: Epochs for Phase 1
            phase2_epochs: Epochs for Phase 2
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
            verbose: Print progress
            phase: If specified, train only this phase (1 or 2)
            
        Returns:
            Training history
        """
        phase1_epochs = phase1_epochs or getattr(self.config, 'phase1_epochs', self.config.epochs // 2)
        phase2_epochs = phase2_epochs or getattr(self.config, 'phase2_epochs', self.config.epochs // 2)
        patience = early_stopping_patience or self.config.early_stopping_patience
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Create phase-specific save paths
            base_path = save_path.rsplit('.', 1)[0]
            ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'pt'
            phase1_save_path = f"{base_path}_phase1.{ext}"
            phase2_save_path = f"{base_path}_phase2.{ext}"
        else:
            phase1_save_path = None
            phase2_save_path = None
        
        if phase == 1:
            # Train Phase 1 only
            self.train_phase(1, phase1_epochs, patience, verbose)
            if save_path:
                self.save_checkpoint(save_path)
        elif phase == 2:
            # Train Phase 2 only (assumes model already trained Phase 1)
            self.train_phase(2, phase2_epochs, patience, verbose)
            if save_path:
                self.save_checkpoint(save_path)
        else:
            # Full two-phase training (default)
            self.train_phase(1, phase1_epochs, patience, verbose)
            # Save Phase 1 checkpoint
            if phase1_save_path:
                self.save_checkpoint(phase1_save_path)
                if verbose:
                    print(f"\nPhase 1 checkpoint saved to: {phase1_save_path}")
            
            self.train_phase(2, phase2_epochs, patience, verbose)
            # Save Phase 2 checkpoint
            if phase2_save_path:
                self.save_checkpoint(phase2_save_path)
                if verbose:
                    print(f"\nPhase 2 checkpoint saved to: {phase2_save_path}")
        
        # Save final model (for backward compatibility)
        if save_path:
            self.save_checkpoint(save_path)
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'phase': self.current_phase,
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_phase = checkpoint.get('phase', 2)
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint['history']
        
        if self.current_phase == 2:
            self.model.set_second_phase()
        else:
            self.model.set_first_phase()


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    save_path: str = '/workspace/checkpoints/best_model.pt',
    verbose: bool = True,
    phase: Optional[int] = None  # NEW: Train specific phase only
) -> Tuple:
    """
    Convenience function to train a PrototypeNetwork.
    
    Args:
        model: PrototypeNetwork model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: ModelConfig
        save_path: Path to save best model
        verbose: Print progress
        phase: If specified, train only this phase (1 or 2)
        
    Returns:
        Trained model and training history
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.device
    )
    
    history = trainer.train(
        save_path=save_path,
        verbose=verbose,
        phase=phase
    )
    
    return trainer.model, history
