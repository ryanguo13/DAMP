#!/usr/bin/env python3
"""
DAMP: Diffusion-Driven Antimicrobial Peptide Engineering with GNN

Main script for training models and generating novel AMP sequences.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import json
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_sequences, create_dataloaders, DiffusionDataset
from models import GNNScorer, Denoiser, DiffusionModel
from trainer import GNNTrainer, DiffusionTrainer
from generator import SequenceGenerator
from evaluator import QualityEvaluator

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

def load_data(amp_file: str, non_amp_file: str, max_length: int = 200):
    """Load and prepare training data."""
    print("Loading sequences...")
    
    # Get the project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Make paths absolute and normalize
    amp_file = os.path.abspath(os.path.join(project_root, amp_file))
    non_amp_file = os.path.abspath(os.path.join(project_root, non_amp_file))
    
    # Load sequences with larger max_length to include more non-AMP sequences
    print(f"Loading AMP sequences from {amp_file}...")
    amps = load_sequences(amp_file, max_length, min_length=10)
    print(f"Loaded {len(amps)} AMP sequences")
    
    print(f"Loading non-AMP sequences from {non_amp_file}...")
    non_amps = load_sequences(non_amp_file, max_length, min_length=10)
    print(f"Loaded {len(non_amps)} non-AMP sequences")
    
    if not amps:
        print("Warning: No AMP sequences loaded. Using dummy data.")
        amps = ["GIGKFLHSAKKFGKAFVGEIMNS", "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"]
    
    if not non_amps:
        print("Warning: No non-AMP sequences loaded. Using dummy data.")
        non_amps = ["AAAAAAAAAAAAAAAAAAAAAAAAA", "GGGGGGGGGGGGGGGGGGGGGGGGG"]
    
    print(f"Loaded {len(amps)} AMP sequences and {len(non_amps)} non-AMP sequences")
    
    # Balance the dataset if needed
    if len(non_amps) > len(amps) * 2:
        # Sample non-AMPs to avoid overwhelming the training
        import random
        random.seed(42)  # For reproducibility
        non_amps = random.sample(non_amps, min(len(non_amps), len(amps) * 2))
        print(f"Balanced dataset: {len(amps)} AMP sequences and {len(non_amps)} non-AMP sequences")
    
    # Prepare data
    sequences = amps + non_amps
    labels = [1] * len(amps) + [0] * len(non_amps)
    
    return sequences, labels, amps

def train_gnn_scorer(sequences, labels, device, save_dir="models", epochs=50):
    """Train GNN scorer model."""
    print("\n" + "="*50)
    print("TRAINING GNN SCORER")
    print("="*50)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(sequences, labels, batch_size=32)
    
    # Initialize model with enhanced configuration for better precision
    max_len = max(len(seq) for seq in sequences)
    gnn_model = GNNScorer(embed_dim=128, hidden_dim=256, num_layers=4)
    
    # Initialize trainer
    trainer = GNNTrainer(gnn_model, device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    gnn_save_path = os.path.join(save_dir, "gnn_scorer.pth")
    
    # Train model
    history = trainer.train(
        train_loader, val_loader, 
        epochs=epochs, 
        save_path=gnn_save_path
    )
    
    print(f"GNN training completed. Model saved to {gnn_save_path}")
    return gnn_model, trainer, history

def train_diffusion_model(amp_sequences, device, save_dir="models", epochs=50):
    """Train diffusion model."""
    print("\n" + "="*50)
    print("TRAINING DIFFUSION MODEL")
    print("="*50)
    
    # Create dataset and dataloader
    max_len = max(len(seq) for seq in amp_sequences)
    diff_dataset = DiffusionDataset(amp_sequences, max_len=max_len, noise_steps=50)
    diff_loader = DataLoader(diff_dataset, batch_size=32, shuffle=True)
    
    # Initialize model with enhanced configuration for better precision
    denoiser = Denoiser(embed_dim=128, hidden_dim=512, num_layers=4)
    diffusion_model = DiffusionModel(denoiser, noise_steps=50)
    
    # Initialize trainer
    trainer = DiffusionTrainer(diffusion_model, device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    diff_save_path = os.path.join(save_dir, "diffusion_model.pth")
    
    # Train model
    history = trainer.train(
        diff_loader, 
        epochs=epochs, 
        save_path=diff_save_path
    )
    
    print(f"Diffusion training completed. Model saved to {diff_save_path}")
    return diffusion_model, trainer, history

def generate_and_evaluate(gnn_model, diffusion_model, device, num_sequences=100):
    """Generate sequences and evaluate quality."""
    print("\n" + "="*50)
    print("GENERATING AND EVALUATING SEQUENCES")
    print("="*50)
    
    # Initialize generator
    generator = SequenceGenerator(diffusion_model, device)
    
    # Generate sequences with better temperature for diversity
    print(f"Generating {num_sequences} sequences...")
    generated_sequences = generator.generate_batch(
        num_sequences=num_sequences, 
        length=20, 
        temperature=1.2  # Slightly higher temperature for more diversity
    )
    
    # Initialize evaluator
    evaluator = QualityEvaluator()
    
    # Evaluate quality
    print("Evaluating sequence quality...")
    quality_metrics = evaluator.evaluate_sequence_quality(generated_sequences)
    
    # Evaluate AMP potential
    print("Evaluating AMP potential...")
    amp_metrics = evaluator.evaluate_amp_potential(generated_sequences, gnn_model)
    
    # Generate report
    report = evaluator.generate_report(generated_sequences, gnn_model)
    print("\n" + report)
    
    # Generate optimized sequences with better parameters
    print("\nGenerating optimized sequences...")
    optimized_sequences = generator.generate_with_optimization(
        gnn_model, 
        num_sequences=50, 
        length=20, 
        temperature=0.9,  # Slightly higher for better exploration
        top_k=15  # More candidates for better selection
    )
    
    print("\nTop 10 sequences with highest AMP scores:")
    for i, (seq, score) in enumerate(optimized_sequences, 1):
        print(f"{i:2d}. {seq} (Score: {score:.4f})")
    
    # Save results
    results = {
        'generated_sequences': generated_sequences,
        'optimized_sequences': optimized_sequences,
        'quality_metrics': quality_metrics,
        'amp_metrics': amp_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs("results", exist_ok=True)
    
    # Save JSON results
    with open("results/generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save FASTA files using evaluator methods
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save basic generated sequences as FASTA
    evaluator.save_sequences_to_fasta(
        generated_sequences, 
        f"results/generated_sequences_{timestamp}.fasta",
        sequence_type="generated"
    )
    
    # Save optimized sequences as FASTA
    evaluator.save_optimized_sequences_to_fasta(
        optimized_sequences,
        f"results/optimized_sequences_{timestamp}.fasta"
    )
    
    # Save top sequences as FASTA
    evaluator.save_top_sequences_to_fasta(
        optimized_sequences,
        f"results/top_sequences_{timestamp}.fasta",
        top_k=10
    )
    
    print(f"\nResults saved to:")
    print(f"  - results/generation_results.json")
    print(f"  - results/generated_sequences_{timestamp}.fasta")
    print(f"  - results/optimized_sequences_{timestamp}.fasta")
    print(f"  - results/top_sequences_{timestamp}.fasta")
    
    return generated_sequences, optimized_sequences, quality_metrics, amp_metrics

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="DAMP: Antimicrobial Peptide Generation")
    parser.add_argument("--amp_file", default="dataset/naturalAMPs_APD2024a.fasta", 
                       help="Path to AMP sequences file")
    parser.add_argument("--non_amp_file", default="dataset/non-amp.fasta", 
                       help="Path to non-AMP sequences file")
    parser.add_argument("--max_length", type=int, default=200, 
                       help="Maximum sequence length")
    parser.add_argument("--num_sequences", type=int, default=100, 
                       help="Number of sequences to generate")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Maximum number of training epochs (with early stopping)")
    parser.add_argument("--skip_training", action="store_true", 
                       help="Skip training and load existing models")
    parser.add_argument("--save_dir", default="models", 
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    
    # Load data
    sequences, labels, amp_sequences = load_data(
        args.amp_file, args.non_amp_file, args.max_length
    )
    
    if not args.skip_training:
        # Train GNN scorer
        gnn_model, gnn_trainer, gnn_history = train_gnn_scorer(
            sequences, labels, device, args.save_dir, args.epochs
        )
        
        # Train diffusion model
        diffusion_model, diff_trainer, diff_history = train_diffusion_model(
            amp_sequences, device, args.save_dir, args.epochs
        )
    else:
        # Load existing models with enhanced configuration
        print("Loading existing models...")
        gnn_model = GNNScorer(embed_dim=128, hidden_dim=256, num_layers=4)
        gnn_trainer = GNNTrainer(gnn_model, device)
        try:
            gnn_trainer.load_model(os.path.join(args.save_dir, "gnn_scorer.pth"))
            print("Loaded existing GNN model")
        except:
            print("Warning: Could not load existing GNN model, will train new one")
            gnn_model, gnn_trainer, _ = train_gnn_scorer(sequences, labels, device, args.save_dir, args.epochs)
        
        denoiser = Denoiser(embed_dim=128, hidden_dim=512, num_layers=4)
        diffusion_model = DiffusionModel(denoiser, noise_steps=50)
        diff_trainer = DiffusionTrainer(diffusion_model, device)
        try:
            diff_trainer.load_model(os.path.join(args.save_dir, "diffusion_model.pth"))
            print("Loaded existing Diffusion model")
        except:
            print("Warning: Could not load existing Diffusion model, will train new one")
            diffusion_model, diff_trainer, _ = train_diffusion_model(amp_sequences, device, args.save_dir, args.epochs)
    
    # Generate and evaluate sequences
    generated_sequences, optimized_sequences, quality_metrics, amp_metrics = generate_and_evaluate(
        gnn_model, diffusion_model, device, args.num_sequences
    )
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main() 