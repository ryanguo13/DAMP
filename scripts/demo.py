#!/usr/bin/env python3
"""
DAMP Demo Script

This script demonstrates the core functionality of the DAMP project:
1. Loading and preprocessing data
2. Training GNN scorer and diffusion models
3. Generating novel peptide sequences
4. Evaluating sequence quality
"""

import os
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_sequences, create_dataloaders
from models import GNNScorer, Denoiser, DiffusionModel
from trainer import GNNTrainer, DiffusionTrainer
from generator import SequenceGenerator
from evaluator import QualityEvaluator
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Config

def main():
    """Main demo function."""
    print("=" * 60)
    print("DAMP: Diffusion-Driven Antimicrobial Peptide Engineering")
    print("=" * 60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load configuration
    config = Config()
    config.create_directories()
    
    # Load data
    print("\n1. Loading Data...")
    amps = load_sequences("dataset/naturalAMPs_APD2024a.fasta", max_length=50)
    non_amps = load_sequences("dataset/non_amp.fasta", max_length=50)
    
    if not amps:
        print("Warning: No AMP sequences found. Using dummy data.")
        amps = ["GIGKFLHSAKKFGKAFVGEIMNS", "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"]
    
    if not non_amps:
        print("Warning: No non-AMP sequences found. Using dummy data.")
        non_amps = ["AAAAAAAAAAAAAAAAAAAAAAAAA", "GGGGGGGGGGGGGGGGGGGGGGGGG"]
    
    sequences = amps + non_amps
    labels = [1] * len(amps) + [0] * len(non_amps)
    
    print(f"Loaded {len(amps)} AMP sequences and {len(non_amps)} non-AMP sequences")
    
    # Train GNN Scorer
    print("\n2. Training GNN Scorer...")
    train_loader, val_loader = create_dataloaders(sequences, labels, batch_size=16)
    
    gnn_model = GNNScorer(
        embed_dim=config.model.gnn_embed_dim,
        hidden_dim=config.model.gnn_hidden_dim,
        num_layers=config.model.gnn_num_layers
    )
    
    gnn_trainer = GNNTrainer(gnn_model, device)
    gnn_history = gnn_trainer.train(
        train_loader, val_loader, 
        epochs=5,  # Short training for demo
        save_path=config.paths.gnn_model_path
    )
    
    print(f"GNN training completed. Final validation accuracy: {gnn_history['val_accuracies'][-1]:.4f}")
    
    # Train Diffusion Model
    print("\n3. Training Diffusion Model...")
    from torch.utils.data import DataLoader
    from data import DiffusionDataset
    
    max_len = max(len(seq) for seq in sequences)
    diff_dataset = DiffusionDataset(amps, max_len=max_len, noise_steps=config.model.diffusion_noise_steps)
    diff_loader = DataLoader(diff_dataset, batch_size=16, shuffle=True)
    
    denoiser = Denoiser(
        embed_dim=config.model.diffusion_embed_dim,
        hidden_dim=config.model.diffusion_hidden_dim,
        num_layers=config.model.diffusion_num_layers
    )
    
    diffusion_model = DiffusionModel(denoiser, noise_steps=config.model.diffusion_noise_steps)
    diff_trainer = DiffusionTrainer(diffusion_model, device)
    diff_history = diff_trainer.train(
        diff_loader, 
        epochs=5,  # Short training for demo
        save_path=config.paths.diffusion_model_path
    )
    
    print(f"Diffusion training completed. Final loss: {diff_history['train_losses'][-1]:.4f}")
    
    # Generate Sequences
    print("\n4. Generating Novel Sequences...")
    generator = SequenceGenerator(diffusion_model, device)
    
    # Generate basic sequences
    basic_sequences = generator.generate_batch(
        num_sequences=10, 
        length=20, 
        temperature=1.0
    )
    
    print("Generated sequences:")
    for i, seq in enumerate(basic_sequences, 1):
        print(f"  {i:2d}. {seq}")
    
    # Generate optimized sequences
    print("\n5. Generating Optimized Sequences...")
    optimized_sequences = generator.generate_with_optimization(
        gnn_model, 
        num_sequences=20, 
        length=20, 
        temperature=0.8, 
        top_k=5
    )
    
    print("Top 5 sequences with highest AMP scores:")
    for i, (seq, score) in enumerate(optimized_sequences, 1):
        print(f"  {i}. {seq} (Score: {score:.4f})")
    
    # Evaluate Quality
    print("\n6. Evaluating Sequence Quality...")
    evaluator = QualityEvaluator(reference_sequences=amps)
    
    # Evaluate basic sequences
    basic_metrics = evaluator.evaluate_sequence_quality(basic_sequences)
    basic_amp_metrics = evaluator.evaluate_amp_potential(basic_sequences, gnn_model)
    
    print("Basic Generation Quality Metrics:")
    print(f"  Diversity Score: {basic_metrics['diversity_score']:.4f}")
    print(f"  Average AMP Score: {basic_amp_metrics['avg_amp_score']:.4f}")
    print(f"  High AMP Potential Ratio: {basic_amp_metrics['high_amp_ratio']:.4f}")
    
    # Evaluate optimized sequences
    optimized_seqs = [seq for seq, _ in optimized_sequences]
    opt_metrics = evaluator.evaluate_sequence_quality(optimized_seqs)
    opt_amp_metrics = evaluator.evaluate_amp_potential(optimized_seqs, gnn_model)
    
    print("\nOptimized Generation Quality Metrics:")
    print(f"  Diversity Score: {opt_metrics['diversity_score']:.4f}")
    print(f"  Average AMP Score: {opt_amp_metrics['avg_amp_score']:.4f}")
    print(f"  High AMP Potential Ratio: {opt_amp_metrics['high_amp_ratio']:.4f}")
    
    # Generate comprehensive report
    print("\n7. Generating Comprehensive Report...")
    report = evaluator.generate_report(basic_sequences, gnn_model, amps)
    print(report)
    
    # Save results
    print("\n8. Saving Results...")
    import json
    from datetime import datetime
    import os
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    results = {
        'basic_sequences': basic_sequences,
        'optimized_sequences': optimized_sequences,
        'basic_metrics': basic_metrics,
        'basic_amp_metrics': basic_amp_metrics,
        'optimized_metrics': opt_metrics,
        'optimized_amp_metrics': opt_amp_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save JSON results
    with open("results/demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save FASTA files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save basic sequences as FASTA
    evaluator.save_sequences_to_fasta(
        basic_sequences,
        f"results/demo_basic_sequences_{timestamp}.fasta",
        sequence_type="demo_basic"
    )
    
    # Save optimized sequences as FASTA
    evaluator.save_optimized_sequences_to_fasta(
        optimized_sequences,
        f"results/demo_optimized_sequences_{timestamp}.fasta"
    )
    
    # Save top sequences as FASTA
    evaluator.save_top_sequences_to_fasta(
        optimized_sequences,
        f"results/demo_top_sequences_{timestamp}.fasta",
        top_k=5
    )
    
    print("Results saved to:")
    print("  - results/demo_results.json")
    print(f"  - results/demo_basic_sequences_{timestamp}.fasta")
    print(f"  - results/demo_optimized_sequences_{timestamp}.fasta")
    print(f"  - results/demo_top_sequences_{timestamp}.fasta")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Data loading and preprocessing")
    print("✓ GNN scorer training and validation")
    print("✓ Diffusion model training")
    print("✓ Sequence generation (basic and optimized)")
    print("✓ Quality evaluation with multiple metrics")
    print("✓ Model persistence and loading")
    print("✓ Comprehensive reporting")

if __name__ == "__main__":
    main() 