"""
Neural network models for DAMP project.

This module contains the GNN scorer and diffusion denoiser models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
try:
    from .data import NUM_AA
except ImportError:
    from data import NUM_AA

class GCNLayer(nn.Module):
    """Graph Convolutional Network layer."""
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize GCN layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (batch_size, seq_len, in_features)
            adj: Adjacency matrix (batch_size, seq_len, seq_len)
            
        Returns:
            Updated node features
        """
        x = self.linear(x)
        x = torch.bmm(adj, x)
        return x

class GNNScorer(nn.Module):
    """Graph Neural Network for scoring peptide sequences."""
    
    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64, num_layers: int = 2):
        """
        Initialize GNN scorer.
        
        Args:
            embed_dim: Embedding dimension for amino acids
            hidden_dim: Hidden dimension for GCN layers
            num_layers: Number of GCN layers
        """
        super().__init__()
        self.embedding = nn.Embedding(NUM_AA + 1, embed_dim)  # +1 for pad token
        
        # GCN layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNLayer(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, seq_idx: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            seq_idx: Sequence indices (batch_size, seq_len)
            adj: Adjacency matrix (batch_size, seq_len, seq_len)
            mask: Sequence mask (batch_size, seq_len)
            
        Returns:
            AMP scores (batch_size,)
        """
        # Ensure tensors are on the same device as the model
        device = next(self.parameters()).device
        seq_idx = seq_idx.to(device)
        adj = adj.to(device)
        mask = mask.to(device)
        
        x = self.embedding(seq_idx)
        
        # Apply GCN layers
        for conv in self.conv_layers:
            x = F.relu(conv(x, adj))
            x = self.dropout(x)
        
        # Global pooling with mask
        x = x * mask.unsqueeze(-1)
        x = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        
        # Final prediction
        x = self.fc(x).squeeze(1)
        return torch.sigmoid(x)

class Denoiser(nn.Module):
    """Denoiser network for diffusion model."""
    
    def __init__(self, embed_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2):
        """
        Initialize denoiser.
        
        Args:
            embed_dim: Embedding dimension for amino acids
            hidden_dim: Hidden dimension for MLP layers
            num_layers: Number of MLP layers
        """
        super().__init__()
        self.embedding = nn.Embedding(NUM_AA + 1, embed_dim)  # +1 for pad token
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # MLP layers
        layers = []
        input_dim = embed_dim * 2  # sequence + time embedding
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, NUM_AA)
    
    def forward(self, noised: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            noised: Noised sequence indices (batch_size, seq_len)
            t: Time step (batch_size,)
            mask: Sequence mask (batch_size, seq_len)
            
        Returns:
            Logits for amino acid prediction (batch_size, seq_len, num_aa)
        """
        # Ensure tensors are on the same device as the model
        device = next(self.parameters()).device
        noised = noised.to(device)
        t = t.to(device)
        mask = mask.to(device)
        
        # Embed sequences and time
        x = self.embedding(noised)  # (batch_size, seq_len, embed_dim)
        t_emb = self.time_embed(t.unsqueeze(1))  # (batch_size, embed_dim)
        
        # Expand time embedding to match sequence length
        t_expanded = t_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Concatenate and process
        x = torch.cat([x, t_expanded], dim=-1)
        x = self.mlp(x)
        x = self.out(x)
        
        # Apply mask
        x = x * mask.unsqueeze(-1)
        
        return x

class DiffusionModel:
    """Wrapper for diffusion model with training and generation."""
    
    def __init__(self, denoiser: Denoiser, noise_steps: int = 10):
        """
        Initialize diffusion model.
        
        Args:
            denoiser: Denoiser network
            noise_steps: Number of noise levels
        """
        self.denoiser = denoiser
        self.noise_steps = noise_steps
    
    def add_noise(self, sequences: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add noise to sequences.
        
        Args:
            sequences: Original sequences (batch_size, seq_len)
            t: Noise level (batch_size,)
            
        Returns:
            Noised sequences
        """
        batch_size, seq_len = sequences.shape
        noised = sequences.clone()
        
        # Randomly replace amino acids based on noise level
        for i in range(batch_size):
            noise_prob = t[i].item()
            mask = torch.rand(seq_len) < noise_prob
            random_aas = torch.randint(0, NUM_AA, (seq_len,))
            noised[i, mask] = random_aas[mask]
        
        return noised
    
    def denoise_step(self, noised: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Single denoising step.
        
        Args:
            noised: Noised sequences
            t: Current time step
            mask: Sequence mask
            
        Returns:
            Denoised sequences
        """
        with torch.no_grad():
            logits = self.denoiser(noised, t, mask)
            probs = F.softmax(logits, dim=-1)
            denoised = torch.multinomial(probs.view(-1, NUM_AA), 1).view(noised.shape)
        return denoised 