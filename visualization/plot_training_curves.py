#!/usr/bin/env python3
"""
Script to plot training curves and compare different models
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List, Optional

from src.models import GNNScorer, Denoiser, DiffusionModel
from src.trainer import GNNTrainer, DiffusionTrainer

def setup_plotting_style():
    """Setup matplotlib and seaborn plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def load_training_history(model_path: str) -> Optional[Dict]:
    """Load training history from saved model."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        return checkpoint
    except Exception as e:
        print(f"Could not load {model_path}: {e}")
        return None

def plot_individual_training_curves():
    """Plot individual training curves for each model."""
    print("Plotting individual training curves...")
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Load and plot GNN models
    gnn_models = [
        ("models/gnn_scorer.pth", "Previous GNN"),
        ("models/gnn_scorer_enhanced.pth", "Enhanced GNN")
    ]
    
    for model_path, model_name in gnn_models:
        if os.path.exists(model_path):
            checkpoint = load_training_history(model_path)
            if checkpoint and 'epoch' in checkpoint:
                print(f"Loaded {model_name} training history")
                
                # Create trainer to access plotting methods
                device = "cpu"
                gnn_model = GNNScorer(embed_dim=64, hidden_dim=128, num_layers=3)
                trainer = GNNTrainer(gnn_model, device)
                
                # Load the model state
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Plot training curves (if we had the full history, we would load it here)
                # For now, we'll create a simple plot showing the final metrics
                plt.figure(figsize=(10, 6))
                plt.bar(['Validation Loss', 'Validation Accuracy'], 
                       [checkpoint.get('val_loss', 0), checkpoint.get('val_accuracy', 0)],
                       color=['red', 'green'], alpha=0.7)
                plt.title(f'{model_name} - Final Metrics', fontweight='bold')
                plt.ylabel('Value')
                plt.ylim(0, 1)
                
                # Add value labels on bars
                for i, v in enumerate([checkpoint.get('val_loss', 0), checkpoint.get('val_accuracy', 0)]):
                    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f"plots/{model_name.replace(' ', '_').lower()}_final_metrics.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    # Load and plot Diffusion models
    diff_models = [
        ("models/diffusion_model.pth", "Previous Diffusion"),
        ("models/diffusion_model_enhanced.pth", "Enhanced Diffusion")
    ]
    
    for model_path, model_name in diff_models:
        if os.path.exists(model_path):
            checkpoint = load_training_history(model_path)
            if checkpoint and 'epoch' in checkpoint:
                print(f"Loaded {model_name} training history")
                
                # Create trainer to access plotting methods
                device = "cpu"
                denoiser = Denoiser(embed_dim=64, hidden_dim=256, num_layers=3)
                diffusion_model = DiffusionModel(denoiser, noise_steps=20)
                trainer = DiffusionTrainer(diffusion_model, device)
                
                # Load the model state
                trainer.model.denoiser.load_state_dict(checkpoint['model_state_dict'])
                
                # Plot final loss
                plt.figure(figsize=(8, 6))
                plt.bar(['Training Loss'], [checkpoint.get('loss', 0)], color='blue', alpha=0.7)
                plt.title(f'{model_name} - Final Loss', fontweight='bold')
                plt.ylabel('Loss')
                
                # Add value label on bar
                loss_val = checkpoint.get('loss', 0)
                plt.text(0, loss_val + 0.01, f'{loss_val:.4f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f"plots/{model_name.replace(' ', '_').lower()}_final_loss.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()

def plot_model_comparison():
    """Plot comparison between different models."""
    print("Creating model comparison plots...")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # GNN Validation Loss Comparison
    gnn_models = [
        ("models/gnn_scorer.pth", "Previous"),
        ("models/gnn_scorer_enhanced.pth", "Enhanced")
    ]
    
    gnn_losses = []
    gnn_accuracies = []
    labels = []
    
    for model_path, label in gnn_models:
        if os.path.exists(model_path):
            checkpoint = load_training_history(model_path)
            if checkpoint:
                gnn_losses.append(checkpoint.get('val_loss', 0))
                gnn_accuracies.append(checkpoint.get('val_accuracy', 0))
                labels.append(label)
    
    if gnn_losses:
        ax1.bar(labels, gnn_losses, color=['red', 'blue'], alpha=0.7)
        ax1.set_title('GNN Validation Loss Comparison', fontweight='bold')
        ax1.set_ylabel('Validation Loss')
        ax1.set_ylim(0, max(gnn_losses) * 1.2)
        
        # Add value labels
        for i, v in enumerate(gnn_losses):
            ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    if gnn_accuracies:
        ax2.bar(labels, gnn_accuracies, color=['green', 'orange'], alpha=0.7)
        ax2.set_title('GNN Validation Accuracy Comparison', fontweight='bold')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(gnn_accuracies):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Diffusion Loss Comparison
    diff_models = [
        ("models/diffusion_model.pth", "Previous"),
        ("models/diffusion_model_enhanced.pth", "Enhanced")
    ]
    
    diff_losses = []
    diff_labels = []
    
    for model_path, label in diff_models:
        if os.path.exists(model_path):
            checkpoint = load_training_history(model_path)
            if checkpoint:
                diff_losses.append(checkpoint.get('loss', 0))
                diff_labels.append(label)
    
    if diff_losses:
        ax3.bar(diff_labels, diff_losses, color=['purple', 'cyan'], alpha=0.7)
        ax3.set_title('Diffusion Training Loss Comparison', fontweight='bold')
        ax3.set_ylabel('Training Loss')
        ax3.set_ylim(0, max(diff_losses) * 1.2)
        
        # Add value labels
        for i, v in enumerate(diff_losses):
            ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Model file sizes comparison
    model_sizes = []
    size_labels = []
    
    for model_path, label in gnn_models + diff_models:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            model_sizes.append(size_mb)
            size_labels.append(f"{label}\nGNN" if "gnn" in model_path else f"{label}\nDiffusion")
    
    if model_sizes:
        ax4.bar(size_labels, model_sizes, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'], alpha=0.7)
        ax4.set_title('Model File Sizes Comparison', fontweight='bold')
        ax4.set_ylabel('Size (MB)')
        
        # Add value labels
        for i, v in enumerate(model_sizes):
            ax4.text(i, v + 0.1, f'{v:.1f}MB', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plots/model_comparison_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model comparison plots saved to plots/model_comparison_overview.png")

def plot_training_progress():
    """Plot training progress if we have the training history."""
    print("Creating training progress visualization...")
    
    # This would require saving training history during training
    # For now, we'll create a placeholder for future implementation
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, 'Training Progress Plots\n(Would show loss curves over epochs)', 
             ha='center', va='center', fontsize=16, fontweight='bold',
             transform=plt.gca().transAxes)
    plt.title('Training Progress Visualization', fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("plots/training_progress_placeholder.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training progress placeholder saved to plots/training_progress_placeholder.png")

def create_summary_report():
    """Create a summary report of all model performances."""
    print("Creating summary report...")
    
    report = {
        "model_comparison": {},
        "summary": {}
    }
    
    # Collect data for all models
    models_to_check = [
        ("models/gnn_scorer.pth", "Previous GNN"),
        ("models/gnn_scorer_enhanced.pth", "Enhanced GNN"),
        ("models/diffusion_model.pth", "Previous Diffusion"),
        ("models/diffusion_model_enhanced.pth", "Enhanced Diffusion")
    ]
    
    for model_path, model_name in models_to_check:
        if os.path.exists(model_path):
            checkpoint = load_training_history(model_path)
            if checkpoint:
                report["model_comparison"][model_name] = {
                    "epoch": checkpoint.get('epoch', 0),
                    "val_loss": checkpoint.get('val_loss', 0),
                    "val_accuracy": checkpoint.get('val_accuracy', 0),
                    "loss": checkpoint.get('loss', 0),
                    "file_size_mb": os.path.getsize(model_path) / (1024 * 1024)
                }
    
    # Create summary statistics
    if "Previous GNN" in report["model_comparison"] and "Enhanced GNN" in report["model_comparison"]:
        prev_gnn = report["model_comparison"]["Previous GNN"]
        enh_gnn = report["model_comparison"]["Enhanced GNN"]
        
        report["summary"]["gnn_improvement"] = {
            "val_loss_change": enh_gnn["val_loss"] - prev_gnn["val_loss"],
            "val_accuracy_change": enh_gnn["val_accuracy"] - prev_gnn["val_accuracy"],
            "file_size_change": enh_gnn["file_size_mb"] - prev_gnn["file_size_mb"]
        }
    
    if "Previous Diffusion" in report["model_comparison"] and "Enhanced Diffusion" in report["model_comparison"]:
        prev_diff = report["model_comparison"]["Previous Diffusion"]
        enh_diff = report["model_comparison"]["Enhanced Diffusion"]
        
        report["summary"]["diffusion_improvement"] = {
            "loss_change": enh_diff["loss"] - prev_diff["loss"],
            "file_size_change": enh_diff["file_size_mb"] - prev_diff["file_size_mb"]
        }
    
    # Save report
    with open("plots/training_summary_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Summary report saved to plots/training_summary_report.json")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY REPORT")
    print("="*60)
    
    for model_name, metrics in report["model_comparison"].items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    if "summary" in report:
        print(f"\nImprovements:")
        for improvement_type, changes in report["summary"].items():
            print(f"  {improvement_type}:")
            for change, value in changes.items():
                print(f"    {change}: {value:+.4f}")

def main():
    """Main function."""
    print("Starting training curve visualization...")
    
    # Setup plotting style
    setup_plotting_style()
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Generate all plots
    plot_individual_training_curves()
    plot_model_comparison()
    plot_training_progress()
    create_summary_report()
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("Check the 'plots/' directory for all visualization files.")

if __name__ == "__main__":
    main() 