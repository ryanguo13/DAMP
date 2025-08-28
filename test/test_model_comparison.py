#!/usr/bin/env python3
"""
Script to compare enhanced models with previous models
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

def load_models(device):
    """Load both enhanced and previous models."""
    print("Loading models for comparison...")
    
    # Load enhanced models
    enhanced_gnn = GNNScorer(embed_dim=64, hidden_dim=128, num_layers=3)
    enhanced_gnn_trainer = GNNTrainer(enhanced_gnn, device)
    enhanced_gnn_trainer.load_model("models/gnn_scorer_enhanced.pth")
    
    enhanced_denoiser = Denoiser(embed_dim=64, hidden_dim=256, num_layers=3)
    enhanced_diffusion = DiffusionModel(enhanced_denoiser, noise_steps=20)
    enhanced_diff_trainer = DiffusionTrainer(enhanced_diffusion, device)
    enhanced_diff_trainer.load_model("models/diffusion_model_enhanced.pth")
    
    # Load previous models
    prev_gnn = GNNScorer(embed_dim=64, hidden_dim=128, num_layers=3)
    prev_gnn_trainer = GNNTrainer(prev_gnn, device)
    prev_gnn_trainer.load_model("models/gnn_scorer.pth")
    
    prev_denoiser = Denoiser(embed_dim=64, hidden_dim=256, num_layers=3)
    prev_diffusion = DiffusionModel(prev_denoiser, noise_steps=20)
    prev_diff_trainer = DiffusionTrainer(prev_diffusion, device)
    prev_diff_trainer.load_model("models/diffusion_model.pth")
    
    print("All models loaded successfully!")
    
    return {
        'enhanced': (enhanced_gnn, enhanced_diffusion),
        'previous': (prev_gnn, prev_diffusion)
    }

def generate_and_evaluate(models, device, num_sequences=20):
    """Generate sequences and evaluate with both models."""
    print(f"\nGenerating {num_sequences} sequences for comparison...")
    
    results = {}
    
    for model_type, (gnn_model, diffusion_model) in models.items():
        print(f"\n{'='*50}")
        print(f"TESTING {model_type.upper()} MODELS")
        print(f"{'='*50}")
        
        generator = SequenceGenerator(diffusion_model, device)
        evaluator = QualityEvaluator()
        
        # Generate sequences
        sequences = generator.generate_batch(
            num_sequences=num_sequences,
            length=20,
            temperature=1.0
        )
        
        # Evaluate quality
        quality_metrics = evaluator.evaluate_sequence_quality(sequences)
        amp_metrics = evaluator.evaluate_amp_potential(sequences, gnn_model)
        
        results[model_type] = {
            'sequences': sequences,
            'quality': quality_metrics,
            'amp': amp_metrics
        }
        
        print(f"Generated sequences ({model_type}):")
        for i, seq in enumerate(sequences[:10], 1):  # Show first 10
            print(f"  {i:2d}. {seq}")
        
        print(f"\nQuality Metrics ({model_type}):")
        print(f"  Diversity Score: {quality_metrics['diversity_score']:.4f}")
        print(f"  Average AMP Score: {amp_metrics['avg_amp_score']:.4f}")
        print(f"  High AMP Potential Ratio: {amp_metrics['high_amp_ratio']:.4f}")
        print(f"  AMP Ratio: {amp_metrics['amp_ratio']:.4f}")
        print(f"  Valid Sequence Ratio: {quality_metrics['valid_sequence_ratio']:.4f}")
    
    return results

def compare_results(results):
    """Compare results between enhanced and previous models."""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    enhanced = results['enhanced']
    previous = results['previous']
    
    print("Quality Metrics Comparison:")
    print(f"{'Metric':<25} {'Enhanced':<12} {'Previous':<12} {'Improvement':<12}")
    print("-" * 65)
    
    metrics = [
        ('Diversity Score', 'quality', 'diversity_score'),
        ('Average AMP Score', 'amp', 'avg_amp_score'),
        ('High AMP Ratio', 'amp', 'high_amp_ratio'),
        ('AMP Ratio', 'amp', 'amp_ratio'),
        ('Valid Sequence Ratio', 'quality', 'valid_sequence_ratio')
    ]
    
    for metric_name, metric_type, metric_key in metrics:
        enhanced_val = enhanced[metric_type][metric_key]
        previous_val = previous[metric_type][metric_key]
        improvement = enhanced_val - previous_val
        
        print(f"{metric_name:<25} {enhanced_val:<12.4f} {previous_val:<12.4f} {improvement:<12.4f}")
    
    # Overall assessment
    print(f"\n{'='*60}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*60}")
    
    enhanced_amp_score = enhanced['amp']['avg_amp_score']
    previous_amp_score = previous['amp']['avg_amp_score']
    enhanced_diversity = enhanced['quality']['diversity_score']
    previous_diversity = previous['quality']['diversity_score']
    
    print(f"Enhanced Model Performance:")
    print(f"  - AMP Score: {enhanced_amp_score:.4f} (vs {previous_amp_score:.4f})")
    print(f"  - Diversity: {enhanced_diversity:.4f} (vs {previous_diversity:.4f})")
    
    if enhanced_amp_score > previous_amp_score:
        print(f"  ✅ Enhanced model shows better AMP potential")
    else:
        print(f"  ⚠️  Previous model shows better AMP potential")
    
    if enhanced_diversity > previous_diversity:
        print(f"  ✅ Enhanced model shows better diversity")
    else:
        print(f"  ⚠️  Previous model shows better diversity")
    
    # Calculate overall improvement score
    amp_improvement = enhanced_amp_score - previous_amp_score
    diversity_improvement = enhanced_diversity - previous_diversity
    overall_improvement = amp_improvement + diversity_improvement
    
    print(f"\nOverall Improvement Score: {overall_improvement:.4f}")
    if overall_improvement > 0:
        print("🎉 Enhanced training shows overall improvement!")
    else:
        print("📊 Results are mixed, may need further tuning")

def main():
    """Main function."""
    device = setup_device()
    
    # Load models
    models = load_models(device)
    
    # Generate and evaluate
    results = generate_and_evaluate(models, device, num_sequences=25)
    
    # Compare results
    compare_results(results)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETED!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 