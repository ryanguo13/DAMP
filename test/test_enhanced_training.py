#!/usr/bin/env python3
"""
Test script to verify enhanced training with improved loss functions and early stopping
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import random

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data import load_sequences, create_dataloaders, DiffusionDataset
from models import GNNScorer, Denoiser, DiffusionModel
from trainer import GNNTrainer, DiffusionTrainer
from generator import SequenceGenerator
from evaluator import QualityEvaluator

def setup_device():
    """Setup device."""
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("Using CPU")
    return device

def load_data_enhanced():
    """Load data with enhanced handling."""
    print("Loading sequences for enhanced training...")
    
    # Load AMP sequences
    amps = load_sequences("dataset/naturalAMPs_APD2024a.fasta", max_length=200, min_length=10)
    print(f"Loaded {len(amps)} AMP sequences")
    
    # Load non-AMP sequences
    non_amps = load_sequences("dataset/non-amp.fasta", max_length=200, min_length=10)
    print(f"Loaded {len(non_amps)} non-AMP sequences")
    
    if len(non_amps) == 0:
        print("ERROR: No non-AMP sequences loaded!")
        return None, None, None
    
    # Balance dataset
    if len(non_amps) > len(amps) * 2:
        random.seed(42)
        non_amps = random.sample(non_amps, len(amps) * 2)
        print(f"Balanced to {len(amps)} AMP and {len(non_amps)} non-AMP sequences")
    
    sequences = amps + non_amps
    labels = [1] * len(amps) + [0] * len(non_amps)
    
    print(f"Final dataset: {len(sequences)} sequences ({len(amps)} AMP, {len(non_amps)} non-AMP)")
    return sequences, labels, amps

def main():
    """Main function."""
    device = setup_device()
    
    # Load data
    sequences, labels, amps = load_data_enhanced()
    if sequences is None:
        return
    
    # Train GNN with enhanced training
    print("\n" + "="*60)
    print("ENHANCED GNN TRAINING WITH FOCAL LOSS & EARLY STOPPING")
    print("="*60)
    
    train_loader, val_loader = create_dataloaders(sequences, labels, batch_size=16)
    
    gnn_model = GNNScorer(embed_dim=64, hidden_dim=128, num_layers=3)
    gnn_trainer = GNNTrainer(gnn_model, device)
    
    import os
    os.makedirs("models", exist_ok=True)
    
    print("Starting GNN training with enhanced loss function and early stopping...")
    gnn_history = gnn_trainer.train(
        train_loader, val_loader, 
        epochs=100,  # Maximum epochs, will stop early if needed
        save_path="models/gnn_scorer_enhanced.pth"
    )
    
    print(f"GNN training completed!")
    print(f"Final validation accuracy: {gnn_history['val_accuracies'][-1]:.4f}")
    print(f"Best validation loss: {min(gnn_history['val_losses']):.4f}")
    print(f"Training epochs: {len(gnn_history['train_losses'])}")
    
    # Train Diffusion with enhanced training
    print("\n" + "="*60)
    print("ENHANCED DIFFUSION TRAINING WITH LABEL SMOOTHING & EARLY STOPPING")
    print("="*60)
    
    max_len = max(len(seq) for seq in sequences)
    diff_dataset = DiffusionDataset(amps, max_len=max_len, noise_steps=20)
    diff_loader = DataLoader(diff_dataset, batch_size=16, shuffle=True)
    
    denoiser = Denoiser(embed_dim=64, hidden_dim=256, num_layers=3)
    diffusion_model = DiffusionModel(denoiser, noise_steps=20)
    diff_trainer = DiffusionTrainer(diffusion_model, device)
    
    print("Starting diffusion training with enhanced loss function and early stopping...")
    diff_history = diff_trainer.train(
        diff_loader, 
        epochs=100,  # Maximum epochs, will stop early if needed
        save_path="models/diffusion_model_enhanced.pth"
    )
    
    print(f"Diffusion training completed!")
    print(f"Final training loss: {diff_history['train_losses'][-1]:.4f}")
    print(f"Best training loss: {min(diff_history['train_losses']):.4f}")
    print(f"Training epochs: {len(diff_history['train_losses'])}")
    
    # Generate and evaluate with enhanced models
    print("\n" + "="*60)
    print("GENERATING SEQUENCES WITH ENHANCED MODELS")
    print("="*60)
    
    generator = SequenceGenerator(diffusion_model, device)
    evaluator = QualityEvaluator(reference_sequences=amps)
    
    # Generate sequences
    print("Generating sequences with enhanced models...")
    generated_sequences = generator.generate_batch(
        num_sequences=15, 
        length=20, 
        temperature=1.0
    )
    
    print("Generated sequences:")
    for i, seq in enumerate(generated_sequences, 1):
        print(f"  {i:2d}. {seq}")
    
    # Evaluate quality
    quality_metrics = evaluator.evaluate_sequence_quality(generated_sequences)
    amp_metrics = evaluator.evaluate_amp_potential(generated_sequences, gnn_model)
    
    print(f"\nEnhanced Model Quality Metrics:")
    print(f"  Diversity Score: {quality_metrics['diversity_score']:.4f}")
    print(f"  Average AMP Score: {amp_metrics['avg_amp_score']:.4f}")
    print(f"  High AMP Potential Ratio: {amp_metrics['high_amp_ratio']:.4f}")
    print(f"  AMP Ratio: {amp_metrics['amp_ratio']:.4f}")
    print(f"  Valid Sequence Ratio: {quality_metrics['valid_sequence_ratio']:.4f}")
    
    # Generate optimized sequences
    print("\nGenerating optimized sequences...")
    optimized_sequences = generator.generate_with_optimization(
        gnn_model, 
        num_sequences=25, 
        length=20, 
        temperature=0.8, 
        top_k=10
    )
    
    print("Top 10 optimized sequences:")
    for i, (seq, score) in enumerate(optimized_sequences, 1):
        print(f"  {i:2d}. {seq} (Score: {score:.4f})")
    
    # Compare with previous results
    print("\n" + "="*60)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("="*60)
    
    # Load previous models for comparison
    try:
        from src.trainer import GNNTrainer as OldGNNTrainer
        old_gnn_model = GNNScorer(embed_dim=64, hidden_dim=128, num_layers=3)
        old_gnn_trainer = OldGNNTrainer(old_gnn_model, device)
        old_gnn_trainer.load_model("models/gnn_scorer.pth")
        
        # Test with old model
        old_amp_metrics = evaluator.evaluate_amp_potential(generated_sequences, old_gnn_model)
        
        print("Comparison Results:")
        print(f"  Enhanced Model - Avg AMP Score: {amp_metrics['avg_amp_score']:.4f}")
        print(f"  Previous Model - Avg AMP Score: {old_amp_metrics['avg_amp_score']:.4f}")
        print(f"  Improvement: {amp_metrics['avg_amp_score'] - old_amp_metrics['avg_amp_score']:.4f}")
        
    except Exception as e:
        print(f"Could not load previous models for comparison: {e}")
    
    print("\n" + "="*60)
    print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main() 