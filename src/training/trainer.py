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


class ClassBalancedTrainer(Trainer):
    """
    Trainer for ClassBalancedPrototypeNetwork.
    
    Extends base Trainer with:
    - Class-stratified prototype initialization at start of training
    - ClassBalancedPrototypeLoss with separation loss
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = 'cuda'
    ):
        # Store train_loader before calling super().__init__
        self._init_train_loader = train_loader
        
        # Call parent init
        super().__init__(model, train_loader, val_loader, config, device)
        
        # Replace loss function with class-balanced version
        from .losses import ClassBalancedPrototypeLoss
        self.criterion = ClassBalancedPrototypeLoss(
            lambda_reconstruction=config.lambda_reconstruction,
            lambda_diversity=getattr(config, 'lambda_diversity', 0.5),
            lambda_clustering=config.lambda_clustering,
            lambda_separation=getattr(config, 'lambda_separation', 0.3)
        )
        
        # Initialize prototypes flag
        self._prototypes_initialized = False
    
    def _collect_embeddings_and_labels(self):
        """Collect all embeddings and labels for prototype initialization."""
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in self._init_train_loader:
                x_num, x_cat, targets = batch
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                
                z = self.model.generate_embeddings(x_num, x_cat)
                embeddings.append(z.cpu())
                labels.append(targets)
        
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)
    
    def initialize_prototypes(self, verbose: bool = True):
        """Initialize prototypes using class-stratified KMeans."""
        if self._prototypes_initialized:
            return
        
        if verbose:
            print("\nInitializing class-balanced prototypes...")
        
        embeddings, labels = self._collect_embeddings_and_labels()
        self.model.initialize_prototypes(embeddings, labels)
        self._prototypes_initialized = True
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train with class-balanced prototypes."""
        # Initialize prototypes before training starts
        self.initialize_prototypes(verbose)
        
        # Continue with normal training
        return super().train(epochs, early_stopping_patience, save_path, verbose)


def train_class_balanced_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    save_path: str = '/workspace/checkpoints/best_class_balanced_model.pt',
    verbose: bool = True
) -> Tuple:
    """
    Train a ClassBalancedPrototypeNetwork.
    
    This function:
    1. Initializes prototypes using class-stratified KMeans
    2. Trains with ClassBalancedPrototypeLoss
    3. Ensures prototypes remain balanced across classes
    
    Args:
        model: ClassBalancedPrototypeNetwork model
        train_loader: Training data loader  
        val_loader: Validation data loader
        config: ModelConfig
        save_path: Path to save best model
        verbose: Print progress
        
    Returns:
        Trained model and training history
    """
    trainer = ClassBalancedTrainer(
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
    
    if save_path and os.path.exists(save_path):
        trainer.load_checkpoint(save_path)
    
    return trainer.model, history


class PTaRLTrainer:
    """
    Trainer for PTaRL model with two-phase training.
    
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
        
        from .losses import PTaRLLoss
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
    
    def _init_optimizer(self, phase: int):
        """Initialize optimizer for given phase."""
        lr = self.config.learning_rate
        if phase == 2:
            lr = lr * 0.5  # Lower learning rate for Phase 2
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay
        )
        
        epochs = getattr(self.config, f'phase{phase}_epochs', self.config.epochs // 2)
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
                x_num=x_num,  # NEW: For reconstruction loss
                x_cat=x_cat   # NEW: For reconstruction loss
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
        
        self._init_optimizer(phase)
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
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full two-phase training.
        
        Args:
            phase1_epochs: Epochs for Phase 1
            phase2_epochs: Epochs for Phase 2
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
            verbose: Print progress
            
        Returns:
            Training history
        """
        phase1_epochs = phase1_epochs or getattr(self.config, 'phase1_epochs', self.config.epochs // 2)
        phase2_epochs = phase2_epochs or getattr(self.config, 'phase2_epochs', self.config.epochs // 2)
        patience = early_stopping_patience or self.config.early_stopping_patience
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Phase 1
        self.train_phase(1, phase1_epochs, patience, verbose)
        
        # Phase 2
        self.train_phase(2, phase2_epochs, patience, verbose)
        
        # Save final model
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


def train_ptarl_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    save_path: str = '/workspace/checkpoints/best_ptarl_model.pt',
    verbose: bool = True
) -> Tuple:
    """
    Convenience function to train a PTaRL model.
    
    Args:
        model: PrototypeNetworkPTaRL model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: ModelConfig
        save_path: Path to save best model
        verbose: Print progress
        
    Returns:
        Trained model and training history
    """
    trainer = PTaRLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.device
    )
    
    history = trainer.train(
        save_path=save_path,
        verbose=verbose
    )
    
    return trainer.model, history

