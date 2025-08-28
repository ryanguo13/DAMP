"""
Data handling module for DAMP project.

This module contains classes and functions for loading, preprocessing,
and managing peptide sequence data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from typing import List, Tuple, Optional
import random

# Amino acid vocabulary
AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard AAs
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LIST)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}
NUM_AA = len(AA_LIST)

def load_sequences(file_path: str, max_length: int = 50, min_length: int = 5) -> List[str]:
    """
    Load sequences from FASTA file.
    
    Args:
        file_path: Path to FASTA file
        max_length: Maximum sequence length to include
        min_length: Minimum sequence length to include
        
    Returns:
        List of sequence strings
    """
    try:
        sequences = [str(record.seq) for record in SeqIO.parse(file_path, "fasta") 
                    if min_length <= len(str(record.seq)) <= max_length]
        return sequences
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using dummy data.")
        return []
    except Exception as e:
        print(f"Warning: Error loading {file_path}: {e}")
        return []

def sequence_to_indices(sequence: str, max_len: int) -> Tuple[List[int], List[int]]:
    """
    Convert sequence to indices and create mask.
    
    Args:
        sequence: Amino acid sequence string
        max_len: Maximum sequence length
        
    Returns:
        Tuple of (indices, mask)
    """
    seq_idx = [AA_TO_IDX.get(aa, 0) for aa in sequence]  # 0 for unknown
    seq_idx += [NUM_AA] * (max_len - len(sequence))  # Pad with special token
    mask = [1 if i < len(sequence) else 0 for i in range(max_len)]
    return seq_idx, mask

def create_adjacency_matrix(sequence: str, max_len: int) -> np.ndarray:
    """
    Create adjacency matrix for sequence graph.
    
    Args:
        sequence: Amino acid sequence string
        max_len: Maximum sequence length
        
    Returns:
        Normalized adjacency matrix
    """
    adj = np.zeros((max_len, max_len))
    
    # Add sequential connections
    for i in range(len(sequence) - 1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    
    # Add self-loops
    for i in range(len(sequence)):
        adj[i, i] = 1
    
    # Normalize
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-5))  # Avoid div0
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    
    return adj_norm

class PeptideDataset(Dataset):
    """Dataset for peptide sequences with labels."""
    
    def __init__(self, sequences: List[str], labels: List[int], max_len: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            sequences: List of peptide sequences
            labels: List of labels (1 for AMP, 0 for non-AMP)
            max_len: Maximum sequence length (auto-determined if None)
        """
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len if max_len else max(len(seq) for seq in sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        seq_idx, mask = sequence_to_indices(seq, self.max_len)
        adj = create_adjacency_matrix(seq, self.max_len)
        
        return (
            torch.tensor(seq_idx),
            torch.tensor(adj, dtype=torch.float),
            torch.tensor(mask),
            torch.tensor(label, dtype=torch.float)
        )

class DiffusionDataset(Dataset):
    """Dataset for diffusion model training."""
    
    def __init__(self, sequences: List[str], max_len: Optional[int] = None, noise_steps: int = 10):
        """
        Initialize diffusion dataset.
        
        Args:
            sequences: List of AMP sequences for training
            max_len: Maximum sequence length
            noise_steps: Number of noise levels
        """
        self.sequences = [[AA_TO_IDX[aa] for aa in seq] for seq in sequences]
        self.max_len = max_len if max_len else max(len(seq) for seq in sequences)
        self.noise_steps = noise_steps
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        t = random.randint(1, self.noise_steps)  # noise level
        
        # Add noise to sequence with better noise scheduling
        noise_prob = t / self.noise_steps
        noised = []
        for s in seq:
            if random.random() < noise_prob:
                # Add random amino acid noise
                noised.append(random.randint(0, NUM_AA-1))
            else:
                # Keep original amino acid
                noised.append(s)
        
        # Pad sequences
        pad_noised = noised + [NUM_AA] * (self.max_len - len(seq))
        pad_orig = seq + [NUM_AA] * (self.max_len - len(seq))
        mask = [1 if i < len(seq) else 0 for i in range(self.max_len)]
        
        return (
            torch.tensor(pad_noised),
            torch.tensor(pad_orig),
            torch.tensor(mask),
            torch.tensor(t / self.noise_steps)
        )

def create_dataloaders(sequences: List[str], labels: List[int], 
                      batch_size: int = 32, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        sequences: List of sequences
        labels: List of labels
        batch_size: Batch size for training
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Split data
    n_train = int(len(sequences) * train_split)
    train_sequences = sequences[:n_train]
    train_labels = labels[:n_train]
    val_sequences = sequences[n_train:]
    val_labels = labels[n_train:]
    
    # Create datasets
    train_dataset = PeptideDataset(train_sequences, train_labels)
    val_dataset = PeptideDataset(val_sequences, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 