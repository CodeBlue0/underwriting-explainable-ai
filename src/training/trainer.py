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
from src.utils.visualization import create_tsne_visualization


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
        n_classes = getattr(config, 'n_classes', 1)
        self.n_classes = n_classes
        
        self.phase1_criterion = PrototypeLoss(
            lambda_reconstruction=config.lambda_reconstruction,
            n_classes=n_classes
        )
        
        ptarl_weights = getattr(config, 'ptarl_weights', {
            'task_weight': 1.0,
            'projection_weight': 1.0,
            'diversifying_weight': 0.5,
            'orthogonalization_weight': 2.5
        })
        # Add n_classes to ptarl_weights implies modifying PTaRLLoss init or kwargs
        # Since PTaRLLoss is init with **kwargs, we can add it here if keys match.
        # But PTaRLLoss init definition has n_classes as explicit arg now.
        # Safer to pass explicitly.
        
        # We need to construct PTaRLLoss manually or update dict.
        ptarl_args = ptarl_weights.copy()
        ptarl_args['n_classes'] = n_classes
        self.phase2_criterion = PTaRLLoss(**ptarl_args)
        
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
    
    def _reinitialize_encoder(self):
        """
        Re-initialize encoder parameters for Phase 2 (per PTaRL paper).
        
        The paper re-initializes ALL model parameters when starting Phase 2,
        keeping only the global prototypes learned from Phase 1 embeddings.
        """
        # Re-initialize encoder
        for module in self.model.encoder.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        
        # Re-initialize Phase 1 classifier (will be unused in Phase 2)
        if hasattr(self.model, 'phase1_classifier'):
            self.model.phase1_classifier.reset_parameters()
        
        # Re-initialize projector (Phase 2 component)
        for module in self.model.projector.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        
        # Re-initialize P-Space classifier
        if hasattr(self.model, 'pspace_classifier'):
            self.model.pspace_classifier.reset_parameters()
        
        # Re-initialize decoder
        for module in self.model.decoder.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _collect_embeddings_and_labels(self):
        """Collect all embeddings and labels from training data."""
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for x_num, x_cat, y in self.train_loader:
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                
                z = self.model.encoder(x_num, x_cat)
                all_embeddings.append(z)
                all_labels.append(y)
                
        return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

    def _visualize_latent_space(self, phase, save_dir='/workspace/underwriting-explainable-ai'):
        """Visualize t-SNE for the current phase."""
        try:
            print(f"Generating t-SNE visualization for Phase {phase}...")
            
            # Collect data from train_loader (limit to 5000 samples)
            X_num_list, X_cat_list, y_list = [], [], []
            max_samples = 5000
            current_samples = 0
            
            self.model.eval()
            
            # Use a separate iterator
            with torch.no_grad():
                for batch in self.train_loader:
                    if current_samples >= max_samples:
                        break
                    
                    x_num, x_cat, y = batch
                    
                    X_num_list.append(x_num.cpu().numpy())
                    X_cat_list.append(x_cat.cpu().numpy())
                    y_list.append(y.cpu().numpy())
                    
                    current_samples += x_num.size(0)
            
            if not X_num_list:
                print("Warning: No data collected for visualization.")
                return

            X_num = np.concatenate(X_num_list, axis=0)[:max_samples]
            X_cat = np.concatenate(X_cat_list, axis=0)[:max_samples]
            y = np.concatenate(y_list, axis=0)[:max_samples]
            
            output_path = os.path.join(save_dir, f'tsne_visualization_phase{phase}.png')
            
            # Try to Collect Alpha if available in dataset
            # Note: Dataset might return (x_num, x_cat, y) or (x_num, x_cat, y, alpha) depending on implementation
            # Current implementation of ICRDataset returns (x_num, x_cat, y) based on previous code.
            # So we might not have Alpha here directly from loader.
            # However, we can try to infer or just stick to Class visualization for now in Trainer,
            # and rely on generate_outputs.py for full Alpha viz.
            # BUT, user wants to see it during training.
            
            # Let's check if we can get Alpha. 
            # If not easily possible without changing Dataset, we should at least create Class viz 
            # with correct naming context.
            
            # Actually, let's just create Class visualization for now as "tsne_visualization_phase{phase}_class.png"
            # and if we can't get Alpha, we skip it or leave it to generate_outputs.py.
            # The user's compliant was "only tsne_visualization_phase1.png" exists.
            
            # User requested to disable t-SNE generation during training.
            # create_tsne_visualization(
            #     self.model,
            #     X_num, X_cat, y,
            #     output_path=os.path.join(save_dir, f'tsne_visualization_phase{phase}_class.png'),
            #     n_samples=max_samples,
            #     title_suffix="(Class)",
            #     perplexity=30
            # )
            # print(f"  Saved Class visualization to {os.path.join(save_dir, f'tsne_visualization_phase{phase}_class.png')}")
            print(f"  t-SNE visualization skipped as requested.")
            
            # To get Alpha, we would need to change the DataLoader or Dataset to return it.
            # Given the constraints, I will advise the user to use generate_outputs.py for Alpha,
            # but I will ensure the default training output is at least clearly labeled as Class.
            
        except Exception as e:
            print(f"Error generating t-SNE visualization: {e}")
            import traceback
            traceback.print_exc()
    
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
                outputs, targets, x_num, x_cat
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
                    outputs, targets, x_num, x_cat
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
        
        if self.n_classes > 2:
            # Multiclass
            try:
                auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr', average='macro')
            except ValueError:
                auc = 0.5 # Fallback if single class
                
            y_pred = all_predictions.argmax(axis=1)
            acc = accuracy_score(all_targets, y_pred)
            f1 = f1_score(all_targets, y_pred, average='macro')
        else:
            # Binary
            try:
                auc = roc_auc_score(all_targets, all_predictions)
            except ValueError:
                auc = 0.5
            
            y_pred = (all_predictions > 0.5).astype(int)
            acc = accuracy_score(all_targets, y_pred)
            f1 = f1_score(all_targets, y_pred)
        
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
            # Phase 1: Simple supervised learning (per PTaRL paper)
            # No local prototype initialization needed - just train encoder + classifier
            self.model.set_first_phase()
            if verbose:
                print("Phase 1: Training encoder + classifier (no local prototypes)")
        else:
            # Phase 2: Re-initialize encoder + train with P-Space (per PTaRL paper)
            # Initialize global prototypes via KMeans on Phase 1 embeddings
            if verbose:
                print("Initializing global prototypes via KMeans (using learned Phase 1 embeddings)...")
            embeddings = self._collect_embeddings()
            self.model.initialize_global_prototypes(embeddings)
            
            # Phase 2: Re-initialize encoder + train with P-Space (per PTaRL paper)
            if verbose:
                print("Re-initializing encoder parameters for Phase 2 (start from scratch with P-Space as per PTaRL paper)...")
            self._reinitialize_encoder()
            
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
        
        # Visualize latent space at the end of the phase
        self._visualize_latent_space(phase)
    
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
