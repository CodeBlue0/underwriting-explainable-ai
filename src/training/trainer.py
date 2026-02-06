"""
Training utilities for prototype network.
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .losses import PrototypeLoss


class Trainer:
    """
    Trainer class for PrototypeNetwork.
    
    Handles training loop, validation, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
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
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate / 100
        )
        
        # Loss function
        self.criterion = PrototypeLoss(
            lambda_reconstruction=config.lambda_reconstruction,
            lambda_diversity=config.lambda_diversity,
            lambda_clustering=config.lambda_clustering
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        n_batches = 0
        
        for batch in self.train_loader:
            x_num, x_cat, targets = batch
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x_num, x_cat, return_all=True)
            
            # Compute loss
            losses = self.criterion(
                outputs, targets, x_num, x_cat,
                self.model.prototype_layer
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total'].item()
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            n_batches += 1
        
        # Average losses
        avg_loss = total_loss / n_batches
        for key in loss_components:
            loss_components[key] /= n_batches
        
        return {'train_loss': avg_loss, **loss_components}
    
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
            
            # Forward pass
            outputs = self.model(x_num, x_cat, return_all=True)
            
            # Compute loss
            losses = self.criterion(
                outputs, targets, x_num, x_cat,
                self.model.prototype_layer
            )
            
            total_loss += losses['total'].item()
            n_batches += 1
            
            # Collect predictions
            probs = outputs['probabilities'].cpu().numpy()
            all_predictions.extend(probs)
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
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
    
    def train(
        self,
        epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs (default from config)
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
            verbose: Print progress
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.epochs
        patience = early_stopping_patience or self.config.early_stopping_patience
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_auc'].append(val_metrics['val_auc'])
            self.history['val_acc'].append(val_metrics['val_acc'])
            self.history['val_f1'].append(val_metrics['val_f1'])
            self.history['learning_rate'].append(current_lr)
            
            # Early stopping check
            if val_metrics['val_auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['val_auc']
                self.patience_counter = 0
                
                # Save best model
                if save_path:
                    self.save_checkpoint(save_path)
            else:
                self.patience_counter += 1
            
            # Print progress
            if verbose:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) - "
                      f"Train Loss: {train_metrics['train_loss']:.4f} - "
                      f"Val Loss: {val_metrics['val_loss']:.4f} - "
                      f"Val AUC: {val_metrics['val_auc']:.4f} - "
                      f"Val Acc: {val_metrics['val_acc']:.4f} - "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping
            if self.patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint['history']


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    save_path: str = '/workspace/checkpoints/best_model.pt',
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Convenience function to train a model.
    
    Args:
        model: PrototypeNetwork model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: ModelConfig
        save_path: Path to save best model
        verbose: Print progress
        
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
        epochs=config.epochs,
        early_stopping_patience=config.early_stopping_patience,
        save_path=save_path,
        verbose=verbose
    )
    
    # Load best model
    if os.path.exists(save_path):
        trainer.load_checkpoint(save_path)
    
    return trainer.model, history
