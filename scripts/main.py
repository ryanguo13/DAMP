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

from data import load_sequences, create_dataloaders, DiffusionDataset, get_esm3_embeddings, maybe_load_cached_embeddings, save_embeddings_cache
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


def prepare_esm_embeddings(sequences, use_esm3: bool, esm_model_name: str, device: str,
                           cache_path: str = "dataset/esm3_embeddings.pt", max_len: int = 200,
                           per_residue: bool = True):
    """Compute or load cached ESM embeddings for sequences."""
    if not use_esm3:
        return None, None
    # Try cache first
    cache_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', cache_path))
    cached = maybe_load_cached_embeddings(cache_abs)
    if cached is not None and cached.shape[0] == len(sequences):
        print(f"Loaded cached ESM embeddings from {cache_abs}")
        esm_dim = cached.shape[-1] if cached.dim() >= 2 else cached.shape[0]
        return cached, int(esm_dim)

    # Local ESM .pth path: use EvolutionaryScale esm loader
    if os.path.isfile(esm_model_name) and esm_model_name.endswith('.pth'):
        try:
            import torch
            from esm.pretrained import get_esmc_model_tokenizers, ESMC_600M_202412
        except Exception as e:
            raise RuntimeError("Please install the 'esm' package as per https://github.com/evolutionaryscale/esm") from e
        print(f"Loading local ESM checkpoint: {esm_model_name}")
        # Build model and load local weights
        model = ESMC_600M_202412()
        ckpt = torch.load(esm_model_name, map_location="cpu")
        # Try common keys
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt and isinstance(ckpt['model'], dict):
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys when loading ESM weights: {len(missing)}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading ESM weights: {len(unexpected)}")
        if device in ("cuda", "mps"):
            model = model.to(device)
        model.eval()
        tokenizer_like = get_esmc_model_tokenizers()
        embeddings = get_esm3_embeddings(
            sequences, tokenizer_like, model, max_len=max_len, batch_size=8, per_residue=per_residue
        )
        save_embeddings_cache(embeddings, cache_abs)
        esm_dim = embeddings.shape[-1]
        print(f"Saved ESM embeddings to {cache_abs}")
        return embeddings, int(esm_dim)

    # HF transformers path (expects a repo id or prepared local folder with config)
    try:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
    except Exception as e:
        print(f"Error: transformers not available for ESM embeddings: {e}")
        return None, None
    print(f"Computing ESM embeddings with model {esm_model_name}...")
    local_or_repo = esm_model_name
    # Try AutoProcessor first, fallback to AutoTokenizer; if both fail, raise
    tokenizer_like = None
    try:
        processor = AutoProcessor.from_pretrained(local_or_repo, trust_remote_code=True)
        tokenizer_like = processor
        print("Loaded AutoProcessor for ESM model.")
    except Exception as e_proc:
        print(f"AutoProcessor unavailable ({e_proc}), trying AutoTokenizer...")
        try:
            tokenizer_like = AutoTokenizer.from_pretrained(local_or_repo, trust_remote_code=True)
        except Exception as e_tok:
            raise RuntimeError(
                "Failed to load processor/tokenizer for the specified ESM model. "
                "Per EvolutionaryScale ESM instructions (https://github.com/evolutionaryscale/esm), "
                "prepare a local folder with config (e.g., via save_pretrained) corresponding to your weights, "
                "then pass that folder path with --esm_model."
            )
    model = AutoModel.from_pretrained(local_or_repo, trust_remote_code=True)
    if device in ("cuda", "mps"):
        model.to(device)
    embeddings = get_esm3_embeddings(
        sequences, tokenizer_like, model, max_len=max_len, batch_size=8, per_residue=per_residue
    )
    save_embeddings_cache(embeddings, cache_abs)
    esm_dim = embeddings.shape[-1]
    print(f"Saved ESM embeddings to {cache_abs}")
    return embeddings, int(esm_dim)


def train_gnn_scorer(sequences, labels, device, save_dir="models", epochs=50,
                     use_esm3: bool = False, esm_embeddings: torch.Tensor = None, esm_dim: int = None):
    """Train GNN scorer model."""
    print("\n" + "="*50)
    print("TRAINING GNN SCORER")
    print("="*50)
    
    # Create dataloaders
    if use_esm3 and esm_embeddings is not None:
        train_loader, val_loader = create_dataloaders(sequences, labels, batch_size=32, esm_embeddings=esm_embeddings)
    else:
        train_loader, val_loader = create_dataloaders(sequences, labels, batch_size=32)
    
    # Initialize model with enhanced configuration for better precision
    max_len = max(len(seq) for seq in sequences)
    if use_esm3 and esm_dim is not None:
        gnn_model = GNNScorer(embed_dim=128, hidden_dim=256, num_layers=4, esm_embed_dim=esm_dim, use_esm=True)
    else:
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


def train_diffusion_model(amp_sequences, device, save_dir="models", epochs=50,
                          use_esm3: bool = False, esm_embeddings: torch.Tensor = None, esm_dim: int = None):
    """Train diffusion model."""
    print("\n" + "="*50)
    print("TRAINING DIFFUSION MODEL")
    print("="*50)
    
    # Create dataset and dataloader
    max_len = max(len(seq) for seq in amp_sequences)
    diff_dataset = DiffusionDataset(amp_sequences, max_len=max_len, noise_steps=50)
    diff_loader = DataLoader(diff_dataset, batch_size=32, shuffle=True)
    
    # Initialize model with enhanced configuration for better precision
    if use_esm3 and esm_dim is not None:
        denoiser = Denoiser(embed_dim=128, hidden_dim=512, num_layers=4, esm_embed_dim=esm_dim, use_esm=True)
    else:
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
    # ESM3 flags
    parser.add_argument("--use_esm3", action="store_true", help="Enable ESM3 embeddings integration")
    parser.add_argument("--esm_model", default="models/esmc_600m_2024_12_v0.pth", help="Path to local .pth checkpoint or HF repo/local folder")
    parser.add_argument("--esm_cache", default="dataset/esm3_embeddings.pt", help="Path to cache ESM embeddings")
    parser.add_argument("--esm_per_residue", action="store_true", help="Use per-residue embeddings (recommended)")
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    
    # Load data
    sequences, labels, amp_sequences = load_data(
        args.amp_file, args.non_amp_file, args.max_length
    )
    
    # Prepare ESM embeddings if requested
    esm_embeddings = None
    esm_dim = None
    if args.use_esm3:
        esm_embeddings, esm_dim = prepare_esm_embeddings(
            sequences, use_esm3=True, esm_model_name=args.esm_model, device=device,
            cache_path=args.esm_cache, max_len=args.max_length, per_residue=args.esm_per_residue
        )
    
    if not args.skip_training:
        # Train GNN scorer
        gnn_model, gnn_trainer, gnn_history = train_gnn_scorer(
            sequences, labels, device, args.save_dir, args.epochs,
            use_esm3=args.use_esm3, esm_embeddings=esm_embeddings, esm_dim=esm_dim
        )
        
        # Train diffusion model (currently not feeding ESM during diffusion loop; Phase 4 will add it)
        diffusion_model, diff_trainer, diff_history = train_diffusion_model(
            amp_sequences, device, args.save_dir, args.epochs,
            use_esm3=False
        )
    else:
        # Load existing models with enhanced configuration
        print("Loading existing models...")
        if args.use_esm3 and esm_dim is not None:
            gnn_model = GNNScorer(embed_dim=128, hidden_dim=256, num_layers=4, esm_embed_dim=esm_dim, use_esm=True)
        else:
            gnn_model = GNNScorer(embed_dim=128, hidden_dim=256, num_layers=4)
        gnn_trainer = GNNTrainer(gnn_model, device)
        try:
            gnn_trainer.load_model(os.path.join(args.save_dir, "gnn_scorer.pth"))
            print("Loaded existing GNN model")
        except:
            print("Warning: Could not load existing GNN model, will train new one")
            gnn_model, gnn_trainer, _ = train_gnn_scorer(sequences, labels, device, args.save_dir, args.epochs,
                                                         use_esm3=args.use_esm3, esm_embeddings=esm_embeddings, esm_dim=esm_dim)
        
        denoiser = Denoiser(embed_dim=128, hidden_dim=512, num_layers=4)
        diffusion_model = DiffusionModel(denoiser, noise_steps=50)
        diff_trainer = DiffusionTrainer(diffusion_model, device)
        try:
            diff_trainer.load_model(os.path.join(args.save_dir, "diffusion_model.pth"))
            print("Loaded existing Diffusion model")
        except:
            print("Warning: Could not load existing Diffusion model, will train new one")
            diffusion_model, diff_trainer, _ = train_diffusion_model(amp_sequences, device, args.save_dir, args.epochs,
                                                                     use_esm3=False)
    
    # Generate and evaluate sequences
    generated_sequences, optimized_sequences, quality_metrics, amp_metrics = generate_and_evaluate(
        gnn_model, diffusion_model, device, args.num_sequences
    )
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main() 