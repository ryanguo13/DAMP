"""
Data handling module for DAMP project.

This module contains classes and functions for loading, preprocessing,
and managing peptide sequence data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from typing import List, Tuple, Optional, Union
import random
import os

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

# -----------------------------
# ESM3 integration utilities
# -----------------------------

def _call_model_for_embeddings(model, tokenizer_outputs, per_residue: bool):
    """
    Call either a HF-style model or an ESM C local model to obtain embeddings.
    Returns a tensor of shape (B, L, D) if per_residue else (B, D).
    """
    # Prefer hidden states via HF-style API
    try:
        out = model(**tokenizer_outputs, output_hidden_states=True)
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            last_hidden = out.hidden_states[-1]
        elif isinstance(out, dict) and 'hidden_states' in out:
            last_hidden = out['hidden_states'][-1]
        elif hasattr(out, 'last_hidden_state'):
            last_hidden = out.last_hidden_state
        else:
            last_hidden = None
        if last_hidden is not None:
            return last_hidden
    except TypeError:
        pass
    except Exception:
        pass
    # ESM C local path: positional input_ids and flags
    input_ids = None
    if isinstance(tokenizer_outputs, dict) and 'input_ids' in tokenizer_outputs:
        input_ids = tokenizer_outputs['input_ids']
    elif hasattr(tokenizer_outputs, 'input_ids'):
        input_ids = tokenizer_outputs.input_ids
    if input_ids is None:
        raise RuntimeError("Tokenizer outputs do not contain 'input_ids' for ESM model call")
    try:
        # Try to request embeddings directly
        out = model(input_ids, return_embeddings=True)
        # Common return patterns
        if hasattr(out, 'embeddings'):
            return out.embeddings
        if isinstance(out, dict) and 'embeddings' in out:
            return out['embeddings']
        if isinstance(out, tuple) and len(out) > 1:
            return out[1]
    except TypeError:
        # Some variants may require different flags
        out = model(input_ids)
        if hasattr(out, 'embeddings'):
            return out.embeddings
        if isinstance(out, dict) and 'embeddings' in out:
            return out['embeddings']
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            return out.hidden_states[-1]
    # If we reach here, we could not extract embeddings
    raise RuntimeError("Unable to extract embeddings from ESM model output")

def get_esm3_embeddings(
    sequences: List[str],
    tokenizer: "PreTrainedTokenizerBase",
    model: "torch.nn.Module",
    max_len: int = 50,
    batch_size: int = 16,
    per_residue: bool = True,
) -> torch.Tensor:
    """
    Compute ESM embeddings for a list of sequences using a provided tokenizer/model.

    Args:
        sequences: List of amino-acid sequences
        tokenizer: Tokenizer/processor compatible with the ESM model
        model: Model returning hidden states or embeddings
        max_len: Tokenization/truncation length
        batch_size: Batch size for embedding computation
        per_residue: If True, return per-residue embeddings (B, L, D). If False, return pooled (B, D)

    Returns:
        Tensor of embeddings. Shape is (N, L, D) if per_residue else (N, D)
    """
    device = next(model.parameters()).device if hasattr(model, "parameters") else getattr(model, "device", torch.device("cpu"))
    outputs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            # Move tensors to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            else:
                # Tokenizer might return an object with tensors
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(device)
            # Get embeddings
            emb = _call_model_for_embeddings(model, inputs, per_residue=True)
            # Ensure tensor on CPU
            if emb.is_cuda or str(emb.device) != 'cpu':
                emb = emb.detach().cpu()
            # For per_residue, emb is (B, L, D). For pooled, we mean-pool later
            if not per_residue:
                pooled = emb.mean(dim=1)
                outputs.append(pooled)
            else:
                outputs.append(emb)
    all_emb = torch.cat(outputs, dim=0)
    return all_emb

def maybe_load_cached_embeddings(cache_path: Optional[str]) -> Optional[torch.Tensor]:
    """
    Load cached embeddings if available.
    """
    if cache_path and os.path.isfile(cache_path):
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception as e:
            print(f"Warning: Failed to load cached embeddings from {cache_path}: {e}")
    return None

def save_embeddings_cache(embeddings: torch.Tensor, cache_path: Optional[str]):
    """
    Save embeddings tensor to disk if path provided.
    """
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(embeddings, cache_path)

class PeptideDataset(Dataset):
    """Dataset for peptide sequences with labels."""
    
    def __init__(self, sequences: List[str], labels: List[int], max_len: Optional[int] = None,
                 esm_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None):
        """
        Initialize dataset.
        
        Args:
            sequences: List of peptide sequences
            labels: List of labels (1 for AMP, 0 for non-AMP)
            max_len: Maximum sequence length (auto-determined if None)
            esm_embeddings: Optional per-residue embeddings aligned to sequences.
                Expected shape: (N, L, D) where L may be <= max_len. Will be padded.
        """
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len if max_len else max(len(seq) for seq in sequences)
        if esm_embeddings is not None and isinstance(esm_embeddings, np.ndarray):
            esm_embeddings = torch.from_numpy(esm_embeddings)
        self.esm_embeddings = esm_embeddings  # (N, L, D) or None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        seq_idx, mask = sequence_to_indices(seq, self.max_len)
        adj = create_adjacency_matrix(seq, self.max_len)
        
        if self.esm_embeddings is not None:
            emb = self.esm_embeddings[idx]
            # If pooled embeddings provided (D,), expand to per-residue tiled
            if emb.dim() == 1:
                emb = emb.unsqueeze(0).repeat(len(seq), 1)
            # Pad or truncate to max_len along sequence length
            L = emb.size(0)
            if L < self.max_len:
                pad = torch.zeros(self.max_len - L, emb.size(1))
                emb = torch.cat([emb, pad], dim=0)
            elif L > self.max_len:
                emb = emb[:self.max_len]
            return (
                torch.tensor(seq_idx),
                torch.tensor(adj, dtype=torch.float),
                torch.tensor(mask),
                torch.tensor(label, dtype=torch.float),
                emb.float(),
            )
        
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
                      batch_size: int = 32, train_split: float = 0.8,
                      esm_embeddings: Optional[torch.Tensor] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        sequences: List of sequences
        labels: List of labels
        batch_size: Batch size for training
        train_split: Fraction of data for training
        esm_embeddings: Optional per-residue or pooled embeddings aligned with `sequences`.
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Split data
    n_train = int(len(sequences) * train_split)
    train_sequences = sequences[:n_train]
    train_labels = labels[:n_train]
    val_sequences = sequences[n_train:]
    val_labels = labels[n_train:]
    
    train_emb = None
    val_emb = None
    if esm_embeddings is not None:
        train_emb = esm_embeddings[:n_train]
        val_emb = esm_embeddings[n_train:]
    
    # Create datasets
    train_dataset = PeptideDataset(train_sequences, train_labels, esm_embeddings=train_emb)
    val_dataset = PeptideDataset(val_sequences, val_labels, esm_embeddings=val_emb)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 