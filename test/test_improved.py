#!/usr/bin/env python3
"""
Improved test script with better data loading and training
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import random

from src.data import load_sequences, create_dataloaders, DiffusionDataset
from src.models import GNNScorer, Denoiser, DiffusionModel
from src.trainer import GNNTrainer, DiffusionTrainer
from src.generator import SequenceGenerator
from src.evaluator import QualityEvaluator

def setup_device():
    """Setup device for training."""
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

def load_data_improved():
    """Load data with better handling."""
    print("Loading sequences with improved handling...")
    
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
    sequences, labels, amps = load_data_improved()
    if sequences is None:
        return
    
    # Train GNN
    print("\n" + "="*50)
    print("TRAINING GNN SCORER")
    print("="*50)
    
    train_loader, val_loader = create_dataloaders(sequences, labels, batch_size=16)
    
    gnn_model = GNNScorer(embed_dim=64, hidden_dim=128, num_layers=3)
    gnn_trainer = GNNTrainer(gnn_model, device)
    
    os.makedirs("models", exist_ok=True)
    gnn_history = gnn_trainer.train(
        train_loader, val_loader, 
        epochs=20, 
        save_path="models/gnn_scorer.pth"
    )
    
    print(f"GNN training completed. Final validation accuracy: {gnn_history['val_accuracies'][-1]:.4f}")
    
    # Train Diffusion
    print("\n" + "="*50)
    print("TRAINING DIFFUSION MODEL")
    print("="*50)
    
    max_len = max(len(seq) for seq in sequences)
    diff_dataset = DiffusionDataset(amps, max_len=max_len, noise_steps=20)
    diff_loader = DataLoader(diff_dataset, batch_size=16, shuffle=True)
    
    denoiser = Denoiser(embed_dim=64, hidden_dim=256, num_layers=3)
    diffusion_model = DiffusionModel(denoiser, noise_steps=20)
    diff_trainer = DiffusionTrainer(diffusion_model, device)
    
    diff_history = diff_trainer.train(
        diff_loader, 
        epochs=20, 
        save_path="models/diffusion_model.pth"
    )
    
    print(f"Diffusion training completed. Final loss: {diff_history['train_losses'][-1]:.4f}")
    
    # Generate and evaluate
    print("\n" + "="*50)
    print("GENERATING AND EVALUATING SEQUENCES")
    print("="*50)
    
    generator = SequenceGenerator(diffusion_model, device)
    evaluator = QualityEvaluator(reference_sequences=amps)
    
    # Generate sequences
    generated_sequences = generator.generate_batch(
        num_sequences=10, 
        length=20, 
        temperature=1.0
    )
    
    print("Generated sequences:")
    for i, seq in enumerate(generated_sequences, 1):
        print(f"  {i:2d}. {seq}")
    
    # Evaluate
    quality_metrics = evaluator.evaluate_sequence_quality(generated_sequences)
    amp_metrics = evaluator.evaluate_amp_potential(generated_sequences, gnn_model)
    
    print(f"\nQuality Metrics:")
    print(f"  Diversity Score: {quality_metrics['diversity_score']:.4f}")
    print(f"  Average AMP Score: {amp_metrics['avg_amp_score']:.4f}")
    print(f"  High AMP Potential Ratio: {amp_metrics['high_amp_ratio']:.4f}")
    
    # Generate optimized sequences
    optimized_sequences = generator.generate_with_optimization(
        gnn_model, 
        num_sequences=20, 
        length=20, 
        temperature=0.8, 
        top_k=5
    )
    
    print(f"\nTop 5 optimized sequences:")
    for i, (seq, score) in enumerate(optimized_sequences, 1):
        print(f"  {i}. {seq} (Score: {score:.4f})")
    
    print("\n" + "="*50)
    print("IMPROVED TRAINING COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main() 