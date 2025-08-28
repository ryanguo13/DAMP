"""
Quality evaluation module for DAMP project.

This module contains metrics and evaluation functions for assessing
the quality of generated peptide sequences.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.manifold import TSNE
import torch

try:
    from .data import AA_LIST, AA_TO_IDX
except ImportError:
    from data import AA_LIST, AA_TO_IDX

class QualityEvaluator:
    """Evaluator for peptide sequence quality metrics."""
    
    def __init__(self, reference_sequences: Optional[List[str]] = None):
        """
        Initialize quality evaluator.
        
        Args:
            reference_sequences: Reference sequences for comparison
        """
        self.reference_sequences = reference_sequences or []
        self.aa_frequencies = self._calculate_aa_frequencies()
    
    def _calculate_aa_frequencies(self) -> Dict[str, float]:
        """Calculate amino acid frequencies from reference sequences."""
        if not self.reference_sequences:
            return {aa: 1.0/len(AA_LIST) for aa in AA_LIST}
        
        all_aas = ''.join(self.reference_sequences)
        aa_counts = Counter(all_aas)
        total = sum(aa_counts.values())
        
        frequencies = {}
        for aa in AA_LIST:
            frequencies[aa] = aa_counts.get(aa, 0) / total
        
        return frequencies
    
    def evaluate_sequence_quality(self, sequences: List[str]) -> Dict[str, float]:
        """
        Evaluate quality of generated sequences.
        
        Args:
            sequences: List of sequences to evaluate
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['num_sequences'] = len(sequences)
        metrics['avg_length'] = np.mean([len(seq) for seq in sequences])
        metrics['length_std'] = np.std([len(seq) for seq in sequences])
        
        # Amino acid composition
        metrics.update(self._evaluate_aa_composition(sequences))
        
        # Sequence diversity
        metrics.update(self._evaluate_diversity(sequences))
        
        # Validity checks
        metrics.update(self._evaluate_validity(sequences))
        
        return metrics
    
    def _evaluate_aa_composition(self, sequences: List[str]) -> Dict[str, float]:
        """Evaluate amino acid composition."""
        all_aas = ''.join(sequences)
        aa_counts = Counter(all_aas)
        total = sum(aa_counts.values())
        
        # Calculate frequencies
        gen_frequencies = {aa: aa_counts.get(aa, 0) / total for aa in AA_LIST}
        
        # KL divergence from reference
        kl_div = 0.0
        for aa in AA_LIST:
            if self.aa_frequencies[aa] > 0 and gen_frequencies[aa] > 0:
                kl_div += gen_frequencies[aa] * np.log(gen_frequencies[aa] / self.aa_frequencies[aa])
        
        # Amino acid coverage
        coverage = len([aa for aa in AA_LIST if gen_frequencies[aa] > 0]) / len(AA_LIST)
        
        return {
            'kl_divergence': kl_div,
            'aa_coverage': coverage,
            'avg_aa_frequency': np.mean(list(gen_frequencies.values()))
        }
    
    def _evaluate_diversity(self, sequences: List[str]) -> Dict[str, float]:
        """Evaluate sequence diversity."""
        if len(sequences) < 2:
            return {'diversity_score': 0.0, 'unique_ratio': 1.0}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(sequences)):
            for j in range(i+1, len(sequences)):
                sim = self._calculate_similarity(sequences[i], sequences[j])
                similarities.append(sim)
        
        # Diversity metrics
        avg_similarity = np.mean(similarities)
        diversity_score = 1.0 - avg_similarity
        
        # Unique sequences ratio
        unique_sequences = len(set(sequences))
        unique_ratio = unique_sequences / len(sequences)
        
        return {
            'diversity_score': diversity_score,
            'unique_ratio': unique_ratio,
            'avg_similarity': avg_similarity
        }
    
    def _evaluate_validity(self, sequences: List[str]) -> Dict[str, float]:
        """Evaluate sequence validity."""
        valid_sequences = 0
        valid_aas = 0
        total_aas = 0
        
        for seq in sequences:
            is_valid = True
            for aa in seq:
                total_aas += 1
                if aa in AA_LIST:
                    valid_aas += 1
                else:
                    is_valid = False
            
            if is_valid:
                valid_sequences += 1
        
        return {
            'valid_sequence_ratio': valid_sequences / len(sequences) if sequences else 0.0,
            'valid_aa_ratio': valid_aas / total_aas if total_aas > 0 else 0.0
        }
    
    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two sequences."""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def evaluate_amp_potential(self, sequences: List[str], gnn_scorer, 
                             threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate AMP potential of sequences.
        
        Args:
            sequences: List of sequences to evaluate
            gnn_scorer: Trained GNN model for scoring
            threshold: Classification threshold
            
        Returns:
            Dictionary of AMP potential metrics
        """
        scores = []
        gnn_scorer.eval()
        
        for seq in sequences:
            score = self._score_sequence(gnn_scorer, seq)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Calculate metrics
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        amp_ratio = np.mean(scores > threshold)
        
        # Score distribution
        high_amp_ratio = np.mean(scores > 0.8)
        low_amp_ratio = np.mean(scores < 0.2)
        
        return {
            'avg_amp_score': avg_score,
            'amp_score_std': std_score,
            'amp_ratio': amp_ratio,
            'high_amp_ratio': high_amp_ratio,
            'low_amp_ratio': low_amp_ratio,
            'score_range': (np.min(scores), np.max(scores))
        }
    
    def _score_sequence(self, gnn_scorer, sequence: str) -> float:
        """Score a sequence using GNN with calibration."""
        try:
            from .data import sequence_to_indices, create_adjacency_matrix
        except ImportError:
            from data import sequence_to_indices, create_adjacency_matrix
        
        # Prepare sequence for GNN
        seq_idx, mask = sequence_to_indices(sequence, len(sequence))
        adj = create_adjacency_matrix(sequence, len(sequence))
        
        # Convert to tensors
        seq_tensor = torch.tensor(seq_idx).unsqueeze(0)
        adj_tensor = torch.tensor(adj, dtype=torch.float).unsqueeze(0)
        mask_tensor = torch.tensor(mask).unsqueeze(0)
        
        with torch.no_grad():
            raw_score = gnn_scorer(seq_tensor, adj_tensor, mask_tensor).item()
        
        # Apply calibration to prevent scores of exactly 1.0
        # Use sigmoid with temperature scaling for better calibration
        calibrated_score = 1.0 / (1.0 + np.exp(-raw_score * 2.0))  # Temperature scaling
        
        # Apply additional calibration to prevent scores > 0.99
        if calibrated_score > 0.99:
            calibrated_score = 0.99 + (calibrated_score - 0.99) * 0.1
        
        return calibrated_score
    
    def compare_with_reference(self, generated_sequences: List[str], 
                             reference_sequences: List[str]) -> Dict[str, float]:
        """
        Compare generated sequences with reference sequences.
        
        Args:
            generated_sequences: Generated sequences
            reference_sequences: Reference sequences
            
        Returns:
            Dictionary of comparison metrics
        """
        # Calculate metrics for both sets
        gen_metrics = self.evaluate_sequence_quality(generated_sequences)
        ref_metrics = self.evaluate_sequence_quality(reference_sequences)
        
        # Calculate differences
        differences = {}
        for key in gen_metrics:
            if key in ref_metrics:
                differences[f'{key}_diff'] = gen_metrics[key] - ref_metrics[key]
        
        # Novelty (how different from reference)
        novelty_scores = []
        for gen_seq in generated_sequences:
            max_similarity = max([
                self._calculate_similarity(gen_seq, ref_seq) 
                for ref_seq in reference_sequences
            ]) if reference_sequences else 0.0
            novelty_scores.append(1.0 - max_similarity)
        
        differences['avg_novelty'] = np.mean(novelty_scores)
        differences['high_novelty_ratio'] = np.mean(np.array(novelty_scores) > 0.8)
        
        return differences
    
    def generate_report(self, sequences: List[str], gnn_scorer=None, 
                       reference_sequences: List[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            sequences: Sequences to evaluate
            gnn_scorer: Optional GNN scorer for AMP potential
            reference_sequences: Optional reference sequences for comparison
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 50)
        report.append("PEPTIDE SEQUENCE QUALITY EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Basic quality metrics
        quality_metrics = self.evaluate_sequence_quality(sequences)
        report.append("BASIC QUALITY METRICS:")
        report.append("-" * 30)
        for key, value in quality_metrics.items():
            report.append(f"{key}: {value:.4f}")
        report.append("")
        
        # AMP potential if scorer provided
        if gnn_scorer:
            amp_metrics = self.evaluate_amp_potential(sequences, gnn_scorer)
            report.append("AMP POTENTIAL METRICS:")
            report.append("-" * 30)
            for key, value in amp_metrics.items():
                if isinstance(value, tuple):
                    report.append(f"{key}: {value[0]:.4f} - {value[1]:.4f}")
                else:
                    report.append(f"{key}: {value:.4f}")
            report.append("")
        
        # Comparison with reference if provided
        if reference_sequences:
            comp_metrics = self.compare_with_reference(sequences, reference_sequences)
            report.append("COMPARISON WITH REFERENCE:")
            report.append("-" * 30)
            for key, value in comp_metrics.items():
                report.append(f"{key}: {value:.4f}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 30)
        report.append(f"Total sequences evaluated: {len(sequences)}")
        report.append(f"Average length: {quality_metrics.get('avg_length', 0):.1f}")
        report.append(f"Diversity score: {quality_metrics.get('diversity_score', 0):.4f}")
        report.append(f"Valid sequence ratio: {quality_metrics.get('valid_sequence_ratio', 0):.4f}")
        
        if gnn_scorer:
            amp_metrics = self.evaluate_amp_potential(sequences, gnn_scorer)
            report.append(f"Average AMP score: {amp_metrics.get('avg_amp_score', 0):.4f}")
            report.append(f"High AMP potential ratio: {amp_metrics.get('high_amp_ratio', 0):.4f}")
        
        return "\n".join(report)
    
    def save_sequences_to_fasta(self, sequences: List[str], file_path: str, 
                               sequence_type: str = "generated", 
                               scores: Optional[List[float]] = None):
        """
        Save sequences to FASTA format file.
        
        Args:
            sequences: List of sequences to save
            file_path: Path to save the FASTA file
            sequence_type: Type of sequences (e.g., "generated", "optimized")
            scores: Optional list of scores for each sequence
        """
        with open(file_path, 'w') as f:
            for i, seq in enumerate(sequences, 1):
                if scores and i <= len(scores):
                    score = scores[i-1]
                    f.write(f">{sequence_type}_sequence_{i:03d}_score_{score:.4f}\n{seq}\n")
                else:
                    f.write(f">{sequence_type}_sequence_{i:03d}\n{seq}\n")
    
    def save_optimized_sequences_to_fasta(self, optimized_sequences: List[tuple], 
                                        file_path: str):
        """
        Save optimized sequences (with scores) to FASTA format file.
        
        Args:
            optimized_sequences: List of (sequence, score) tuples
            file_path: Path to save the FASTA file
        """
        with open(file_path, 'w') as f:
            for i, (seq, score) in enumerate(optimized_sequences, 1):
                f.write(f">optimized_sequence_{i:03d}_score_{score:.4f}\n{seq}\n")
    
    def save_top_sequences_to_fasta(self, optimized_sequences: List[tuple], 
                                  file_path: str, top_k: int = 10):
        """
        Save top-k optimized sequences to FASTA format file.
        
        Args:
            optimized_sequences: List of (sequence, score) tuples
            file_path: Path to save the FASTA file
            top_k: Number of top sequences to save
        """
        top_sequences = optimized_sequences[:top_k]
        with open(file_path, 'w') as f:
            for i, (seq, score) in enumerate(top_sequences, 1):
                f.write(f">top_sequence_{i:02d}_score_{score:.4f}\n{seq}\n")

    def plot_quality_metrics(self, sequences: List[str], gnn_scorer=None, 
                           save_path: Optional[str] = None):
        """
        Create visualization of quality metrics.
        
        Args:
            sequences: Sequences to evaluate
            gnn_scorer: Optional GNN scorer for AMP scores
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Length distribution
        lengths = [len(seq) for seq in sequences]
        axes[0, 0].hist(lengths, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Sequence Length Distribution')
        axes[0, 0].set_xlabel('Length')
        axes[0, 0].set_ylabel('Count')
        
        # Amino acid composition
        all_aas = ''.join(sequences)
        aa_counts = Counter(all_aas)
        aa_list = [aa for aa in AA_LIST if aa in aa_counts]
        aa_freqs = [aa_counts[aa] for aa in aa_list]
        
        axes[0, 1].bar(aa_list, aa_freqs)
        axes[0, 1].set_title('Amino Acid Composition')
        axes[0, 1].set_xlabel('Amino Acid')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # AMP scores if available
        if gnn_scorer:
            scores = []
            for seq in sequences:
                score = self._score_sequence(gnn_scorer, seq)
                scores.append(score)
            
            axes[1, 0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('AMP Score Distribution')
            axes[1, 0].set_xlabel('AMP Score')
            axes[1, 0].set_ylabel('Count')
        
        # Diversity analysis
        if len(sequences) > 1:
            similarities = []
            for i in range(min(100, len(sequences))):  # Sample for efficiency
                for j in range(i+1, min(100, len(sequences))):
                    sim = self._calculate_similarity(sequences[i], sequences[j])
                    similarities.append(sim)
            
            axes[1, 1].hist(similarities, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Pairwise Similarity Distribution')
            axes[1, 1].set_xlabel('Similarity')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 