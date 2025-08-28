#!/usr/bin/env python3
"""
Test script to generate sequences using the improved models
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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

def main():
    """Main function."""
    device = setup_device()
    
    # Load trained models
    print("Loading trained models...")
    
    # Load GNN model
    gnn_model = GNNScorer(embed_dim=64, hidden_dim=128, num_layers=3)
    gnn_trainer = GNNTrainer(gnn_model, device)
    gnn_trainer.load_model("models/gnn_scorer.pth")
    print("GNN model loaded successfully")
    
    # Load Diffusion model
    denoiser = Denoiser(embed_dim=64, hidden_dim=256, num_layers=3)
    diffusion_model = DiffusionModel(denoiser, noise_steps=20)
    diff_trainer = DiffusionTrainer(diffusion_model, device)
    diff_trainer.load_model("models/diffusion_model.pth")
    print("Diffusion model loaded successfully")
    
    # Generate sequences
    print("\n" + "="*50)
    print("GENERATING SEQUENCES WITH IMPROVED MODELS")
    print("="*50)
    
    generator = SequenceGenerator(diffusion_model, device)
    
    # Generate basic sequences
    print("Generating basic sequences...")
    basic_sequences = generator.generate_batch(
        num_sequences=10, 
        length=20, 
        temperature=1.0
    )
    
    print("Basic generated sequences:")
    for i, seq in enumerate(basic_sequences, 1):
        print(f"  {i:2d}. {seq}")
    
    # Generate optimized sequences
    print("\nGenerating optimized sequences...")
    optimized_sequences = generator.generate_with_optimization(
        gnn_model, 
        num_sequences=20, 
        length=20, 
        temperature=0.8, 
        top_k=10
    )
    
    print("Top 10 optimized sequences:")
    for i, (seq, score) in enumerate(optimized_sequences, 1):
        print(f"  {i:2d}. {seq} (Score: {score:.4f})")
    
    # Evaluate quality
    print("\n" + "="*50)
    print("QUALITY EVALUATION")
    print("="*50)
    
    evaluator = QualityEvaluator()
    
    # Evaluate basic sequences
    basic_quality = evaluator.evaluate_sequence_quality(basic_sequences)
    basic_amp = evaluator.evaluate_amp_potential(basic_sequences, gnn_model)
    
    print("Basic Generation Quality:")
    print(f"  Diversity Score: {basic_quality['diversity_score']:.4f}")
    print(f"  Average AMP Score: {basic_amp['avg_amp_score']:.4f}")
    print(f"  High AMP Potential Ratio: {basic_amp['high_amp_ratio']:.4f}")
    print(f"  AMP Ratio: {basic_amp['amp_ratio']:.4f}")
    
    # Evaluate optimized sequences
    opt_seqs = [seq for seq, _ in optimized_sequences]
    opt_quality = evaluator.evaluate_sequence_quality(opt_seqs)
    opt_amp = evaluator.evaluate_amp_potential(opt_seqs, gnn_model)
    
    print("\nOptimized Generation Quality:")
    print(f"  Diversity Score: {opt_quality['diversity_score']:.4f}")
    print(f"  Average AMP Score: {opt_amp['avg_amp_score']:.4f}")
    print(f"  High AMP Potential Ratio: {opt_amp['high_amp_ratio']:.4f}")
    print(f"  AMP Ratio: {opt_amp['amp_ratio']:.4f}")
    
    # Generate comprehensive report
    print("\n" + "="*50)
    print("COMPREHENSIVE REPORT")
    print("="*50)
    
    report = evaluator.generate_report(basic_sequences, gnn_model)
    print(report)
    
    print("\n" + "="*50)
    print("GENERATION TEST COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main() 