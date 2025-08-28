"""
Configuration file for DAMP project.

This module contains all configuration parameters for the project.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for model architectures."""
    
    # GNN Scorer
    gnn_embed_dim: int = 32
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.1
    
    # Diffusion Model
    diffusion_embed_dim: int = 32
    diffusion_hidden_dim: int = 128
    diffusion_num_layers: int = 2
    diffusion_noise_steps: int = 10

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # General
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    train_split: float = 0.8
    
    # GNN Training
    gnn_patience: int = 3
    gnn_lr_factor: float = 0.5
    
    # Diffusion Training
    diffusion_patience: int = 3
    diffusion_lr_factor: float = 0.5

@dataclass
class GenerationConfig:
    """Configuration for sequence generation."""
    
    # Generation parameters
    default_length: int = 20
    default_temperature: float = 1.0
    num_sequences: int = 100
    
    # Optimization
    optimization_batch_size: int = 50
    optimization_top_k: int = 10
    optimization_temperature: float = 0.8
    
    # Diversity
    diversity_threshold: float = 0.7
    max_diversity_attempts: int = 10

@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    # File paths
    amp_file: str = "dataset/naturalAMPs_APD2024a.fasta"
    non_amp_file: str = "dataset/non_amp.fasta"
    
    # Processing
    max_length: int = 50
    min_length: int = 5
    
    # Amino acids
    aa_list: str = 'ACDEFGHIKLMNPQRSTVWY'

@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    # Directories
    models_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    
    # Files
    gnn_model_path: str = "models/gnn_scorer.pth"
    diffusion_model_path: str = "models/diffusion_model.pth"
    results_file: str = "results/generation_results.json"
    
    def create_directories(self):
        """Create necessary directories."""
        for directory in [self.models_dir, self.results_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # AMP scoring
    amp_threshold: float = 0.5
    high_amp_threshold: float = 0.8
    low_amp_threshold: float = 0.2
    
    # Quality metrics
    novelty_threshold: float = 0.8
    similarity_threshold: float = 0.7

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.generation = GenerationConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        self.evaluation = EvaluationConfig()
    
    def create_directories(self):
        """Create all necessary directories."""
        self.paths.create_directories()
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'generation': self.generation.__dict__,
            'data': self.data.__dict__,
            'paths': self.paths.__dict__,
            'evaluation': self.evaluation.__dict__
        }
    
    def save(self, filepath: str):
        """Save config to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load config from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls()
        config.model = ModelConfig(**data['model'])
        config.training = TrainingConfig(**data['training'])
        config.generation = GenerationConfig(**data['generation'])
        config.data = DataConfig(**data['data'])
        config.paths = PathConfig(**data['paths'])
        config.evaluation = EvaluationConfig(**data['evaluation'])
        
        return config

# Default configuration
default_config = Config() 