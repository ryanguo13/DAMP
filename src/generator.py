"""
Sequence generation module for DAMP project.

This module contains the sequence generator using diffusion models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import random

try:
    from .models import DiffusionModel
    from .data import AA_LIST, IDX_TO_AA, create_adjacency_matrix, sequence_to_indices
except ImportError:
    from models import DiffusionModel
    from data import AA_LIST, IDX_TO_AA, create_adjacency_matrix, sequence_to_indices

class SequenceGenerator:
    """Generator for novel peptide sequences using diffusion model."""
    
    def __init__(self, diffusion_model: DiffusionModel, device: str = "cpu"):
        """
        Initialize sequence generator.
        
        Args:
            diffusion_model: Trained diffusion model
            device: Device to run generation on
        """
        self.diffusion_model = diffusion_model
        self.device = device
        self.diffusion_model.denoiser.to(device)
        self.diffusion_model.denoiser.eval()
    
    def generate_sequence(self, length: int = 20, temperature: float = 1.0) -> str:
        """
        Generate a single sequence.
        
        Args:
            length: Length of sequence to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated amino acid sequence
        """
        # Start with full noise
        current = torch.tensor([random.randint(0, 19) for _ in range(length)], 
                             device=self.device)
        
        # Denoising process
        for step in reversed(range(1, self.diffusion_model.noise_steps + 1)):
            t = torch.tensor([step / self.diffusion_model.noise_steps], device=self.device)
            mask = torch.ones(length, device=self.device)
            
            with torch.no_grad():
                logits = self.diffusion_model.denoiser(
                    current.unsqueeze(0), t, mask.unsqueeze(0)
                )
                
                # Apply temperature
                logits = logits / temperature
                probs = F.softmax(logits[0], dim=-1)
                
                # Sample next token
                current = torch.multinomial(probs, 1).squeeze(1)
        
        # Convert indices to amino acids
        sequence = ''.join(IDX_TO_AA[i.item()] for i in current)
        return sequence
    
    def generate_batch(self, num_sequences: int = 10, length: int = 20, 
                      temperature: float = 1.0) -> List[str]:
        """
        Generate multiple sequences.
        
        Args:
            num_sequences: Number of sequences to generate
            length: Length of each sequence
            temperature: Sampling temperature
            
        Returns:
            List of generated sequences
        """
        sequences = []
        for _ in range(num_sequences):
            seq = self.generate_sequence(length, temperature)
            sequences.append(seq)
        return sequences
    
    def generate_with_constraints(self, length: int = 20, 
                                 start_motif: Optional[str] = None,
                                 end_motif: Optional[str] = None,
                                 temperature: float = 1.0) -> str:
        """
        Generate sequence with constraints.
        
        Args:
            length: Length of sequence
            start_motif: Required motif at start
            end_motif: Required motif at end
            temperature: Sampling temperature
            
        Returns:
            Generated sequence with constraints
        """
        # Start with full noise
        current = torch.tensor([random.randint(0, 19) for _ in range(length)], 
                             device=self.device)
        
        # Set constraints if provided
        if start_motif:
            for i, aa in enumerate(start_motif):
                if i < length:
                    current[i] = AA_LIST.index(aa)
        
        if end_motif:
            for i, aa in enumerate(end_motif):
                if i < len(end_motif) and length - len(end_motif) + i >= 0:
                    current[length - len(end_motif) + i] = AA_LIST.index(aa)
        
        # Denoising process
        for step in reversed(range(1, self.diffusion_model.noise_steps + 1)):
            t = torch.tensor([step / self.diffusion_model.noise_steps], device=self.device)
            mask = torch.ones(length, device=self.device)
            
            with torch.no_grad():
                logits = self.diffusion_model.denoiser(
                    current.unsqueeze(0), t, mask.unsqueeze(0)
                )
                
                # Apply temperature
                logits = logits / temperature
                probs = F.softmax(logits[0], dim=-1)
                
                # Sample next token, but preserve constraints
                for i in range(length):
                    if start_motif and i < len(start_motif):
                        # Force start motif
                        continue
                    if end_motif and i >= length - len(end_motif):
                        # Force end motif
                        continue
                    
                    # Sample for non-constrained positions
                    current[i] = torch.multinomial(probs[i], 1)
        
        # Convert indices to amino acids
        sequence = ''.join(IDX_TO_AA[i.item()] for i in current)
        return sequence
    
    def generate_diverse_batch(self, num_sequences: int = 10, length: int = 20,
                             temperature: float = 1.0, diversity_threshold: float = 0.7) -> List[str]:
        """
        Generate diverse sequences.
        
        Args:
            num_sequences: Number of sequences to generate
            length: Length of each sequence
            temperature: Sampling temperature
            diversity_threshold: Minimum similarity threshold for diversity
            
        Returns:
            List of diverse sequences
        """
        sequences = []
        attempts = 0
        max_attempts = num_sequences * 10  # Prevent infinite loop
        
        while len(sequences) < num_sequences and attempts < max_attempts:
            new_seq = self.generate_sequence(length, temperature)
            
            # Check diversity
            is_diverse = True
            for existing_seq in sequences:
                similarity = self._calculate_similarity(new_seq, existing_seq)
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                sequences.append(new_seq)
            
            attempts += 1
        
        return sequences
    
    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate similarity between two sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Similarity score (0-1)
        """
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def generate_with_optimization(self, gnn_scorer, num_sequences: int = 10,
                                 length: int = 20, temperature: float = 1.0,
                                 top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Generate sequences and score them with GNN.
        
        Args:
            gnn_scorer: Trained GNN model for scoring
            num_sequences: Number of sequences to generate
            length: Length of each sequence
            temperature: Sampling temperature
            top_k: Number of top sequences to return
            
        Returns:
            List of (sequence, score) tuples
        """
        sequences = self.generate_batch(num_sequences, length, temperature)
        scored_sequences = []
        
        for seq in sequences:
            score = self._score_sequence(gnn_scorer, seq)
            scored_sequences.append((seq, score))
        
        # Sort by score and return top_k
        scored_sequences.sort(key=lambda x: x[1], reverse=True)
        return scored_sequences[:top_k]
    
    def _score_sequence(self, gnn_scorer, sequence: str) -> float:
        """
        Score a sequence using GNN with calibration.
        
        Args:
            gnn_scorer: Trained GNN model
            sequence: Sequence to score
            
        Returns:
            AMP score (0-1) with calibration
        """
        gnn_scorer.eval()
        
        # Prepare sequence for GNN
        seq_idx, mask = sequence_to_indices(sequence, len(sequence))
        adj = create_adjacency_matrix(sequence, len(sequence))
        
        # Convert to tensors
        seq_tensor = torch.tensor(seq_idx, device=self.device).unsqueeze(0)
        adj_tensor = torch.tensor(adj, dtype=torch.float, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(mask, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            raw_score = gnn_scorer(seq_tensor, adj_tensor, mask_tensor).item()
        
        # Apply calibration to prevent scores of exactly 1.0
        # Use sigmoid with temperature scaling for better calibration
        calibrated_score = 1.0 / (1.0 + np.exp(-raw_score * 2.0))  # Temperature scaling
        
        # Apply additional calibration to prevent scores > 0.99
        if calibrated_score > 0.99:
            calibrated_score = 0.99 + (calibrated_score - 0.99) * 0.1
        
        return calibrated_score 