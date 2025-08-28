"""
Training modules for DAMP project.

This module contains training classes for GNN scorer and diffusion model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .models import GNNScorer, Denoiser, DiffusionModel
    from .data import DiffusionDataset
except ImportError:
    from models import GNNScorer, Denoiser, DiffusionModel
    from data import DiffusionDataset

class EarlyStopping:
    """Early stopping mechanism to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to be considered as improvement
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement")
                self.early_stop = True
        return self.early_stop

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weight for class balancing
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class GNNTrainer:
    """Trainer for GNN scorer model."""
    
    def __init__(self, model: GNNScorer, device: str = "cpu"):
        """
        Initialize GNN trainer.
        
        Args:
            model: GNN model to train
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        # Enhanced loss function with focal loss for better handling of class imbalance
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        # Enhanced optimizer with better regularization
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4, betas=(0.9, 0.999))
        # Enhanced scheduler with cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        # Early stopping mechanism with more patience
        self.early_stopping = EarlyStopping(patience=15, min_delta=1e-5, verbose=True)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Support batches with or without ESM embeddings
            if len(batch) == 5:
                seq_idx, adj, mask, labels, esm_emb = [b.to(self.device) for b in batch]
                forward_kwargs = {"esm_emb": esm_emb}
            else:
                seq_idx, adj, mask, labels = [b.to(self.device) for b in batch]
                forward_kwargs = {}
            
            self.optimizer.zero_grad()
            predictions = self.model(seq_idx, adj, mask, **forward_kwargs)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (validation loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 5:
                    seq_idx, adj, mask, labels, esm_emb = [b.to(self.device) for b in batch]
                    forward_kwargs = {"esm_emb": esm_emb}
                else:
                    seq_idx, adj, mask, labels = [b.to(self.device) for b in batch]
                    forward_kwargs = {}
                
                predictions = self.model(seq_idx, adj, mask, **forward_kwargs)
                loss = self.criterion(predictions, labels)
                
                # Calculate accuracy
                pred_labels = (predictions > 0.5).float()
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of training epochs
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"Starting training for maximum {epochs} epochs with early stopping...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, save_path)
            
            # Print progress
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch}")
                # Restore best model
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Plot training curves for loss and accuracy.
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if not self.train_losses:
            print("No training data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(len(self.train_losses))
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if self.val_losses:
            ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax1.set_title('GNN Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        if self.val_accuracies:
            ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('GNN Validation Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_comparison(self, other_trainer, labels: Tuple[str, str] = ('Model 1', 'Model 2'), 
                           save_path: Optional[str] = None, show_plot: bool = True):
        """
        Compare training curves with another trainer.
        
        Args:
            other_trainer: Another GNNTrainer instance
            labels: Labels for the two models
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if not self.train_losses or not other_trainer.train_losses:
            print("Insufficient data for comparison")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs1 = range(len(self.train_losses))
        epochs2 = range(len(other_trainer.train_losses))
        
        # Plot training losses
        ax1.plot(epochs1, self.train_losses, 'b-', label=f'{labels[0]} Train', linewidth=2)
        ax1.plot(epochs2, other_trainer.train_losses, 'r-', label=f'{labels[1]} Train', linewidth=2)
        
        if self.val_losses and other_trainer.val_losses:
            ax1.plot(epochs1, self.val_losses, 'b--', label=f'{labels[0]} Val', linewidth=2)
            ax1.plot(epochs2, other_trainer.val_losses, 'r--', label=f'{labels[1]} Val', linewidth=2)
        
        ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot validation accuracies
        if self.val_accuracies and other_trainer.val_accuracies:
            ax2.plot(epochs1, self.val_accuracies, 'g-', label=f'{labels[0]}', linewidth=2)
            ax2.plot(epochs2, other_trainer.val_accuracies, 'orange', label=f'{labels[1]}', linewidth=2)
            ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

class DiffusionTrainer:
    """Trainer for diffusion model."""
    
    def __init__(self, model: DiffusionModel, device: str = "cpu"):
        """
        Initialize diffusion trainer.
        
        Args:
            model: Diffusion model to train
            device: Device to train on
        """
        self.model = model
        self.device = device
        # Move model to device
        self.model.denoiser.to(device)
        # Enhanced loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(ignore_index=20, label_smoothing=0.15)
        # Enhanced optimizer with better regularization and higher learning rate for diffusion
        self.optimizer = optim.AdamW(model.denoiser.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999))
        # Enhanced scheduler with cosine annealing (no loss parameter needed)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, T_mult=2, eta_min=1e-5
        )
        # Early stopping mechanism with more patience for diffusion model
        self.early_stopping = EarlyStopping(patience=25, min_delta=1e-3, verbose=True)
        
        self.train_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.denoiser.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            noised, orig, mask, t = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            predictions = self.model.denoiser(noised, t, mask)
            
            # Reshape for cross entropy
            pred_flat = predictions.view(-1, 20)  # 20 amino acids
            orig_flat = orig.view(-1)
            
            loss = self.criterion(pred_flat, orig_flat)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, epochs: int = 100, 
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the diffusion model with early stopping.
        
        Args:
            train_loader: Training data loader
            epochs: Maximum number of training epochs
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        best_loss = float('inf')
        best_model_state = None
        
        print(f"Starting diffusion training for maximum {epochs} epochs with early stopping...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Learning rate scheduling (cosine annealing doesn't need loss parameter)
            self.scheduler.step()
            
            # Save best model
            if train_loss < best_loss and save_path:
                best_loss = train_loss
                best_model_state = self.model.denoiser.state_dict().copy()
                torch.save({
                    'model_state_dict': self.model.denoiser.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': train_loss
                }, save_path)
            
            # Print progress
            if epoch % 5 == 0:
                print(f"Diffusion Epoch {epoch}: Loss: {train_loss:.4f}")
            
            # Early stopping check
            if self.early_stopping(train_loss):
                print(f"Early stopping at epoch {epoch}")
                # Restore best model
                if best_model_state is not None:
                    self.model.denoiser.load_state_dict(best_model_state)
                break
        
        print(f"Diffusion training completed. Best loss: {best_loss:.4f}")
        
        return {'train_losses': self.train_losses}
    
    def plot_training_curves(self, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Plot training curves for diffusion model loss.
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if not self.train_losses:
            print("No training data available for plotting")
            return
        
        plt.figure(figsize=(10, 6))
        epochs = range(len(self.train_losses))
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.title('Diffusion Model Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_comparison(self, other_trainer, labels: Tuple[str, str] = ('Model 1', 'Model 2'), 
                           save_path: Optional[str] = None, show_plot: bool = True):
        """
        Compare training curves with another diffusion trainer.
        
        Args:
            other_trainer: Another DiffusionTrainer instance
            labels: Labels for the two models
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if not self.train_losses or not other_trainer.train_losses:
            print("Insufficient data for comparison")
            return
        
        plt.figure(figsize=(12, 6))
        
        epochs1 = range(len(self.train_losses))
        epochs2 = range(len(other_trainer.train_losses))
        
        plt.plot(epochs1, self.train_losses, 'b-', label=f'{labels[0]}', linewidth=2)
        plt.plot(epochs2, other_trainer.train_losses, 'r-', label=f'{labels[1]}', linewidth=2)
        
        plt.title('Diffusion Training Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.denoiser.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint 