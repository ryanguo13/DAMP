"""
DAMP: Diffusion-Driven Antimicrobial Peptide Engineering with GNN

A comprehensive pipeline for generating and scoring antimicrobial peptides using
diffusion models and graph neural networks.
"""

__version__ = "0.1.0"
__author__ = "Ryan KWOK"

from .data import PeptideDataset, DiffusionDataset, load_sequences
from .models import GNNScorer, Denoiser, GCNLayer, DiffusionModel
from .trainer import GNNTrainer, DiffusionTrainer
from .generator import SequenceGenerator
from .evaluator import QualityEvaluator

__all__ = [
    "PeptideDataset",
    "DiffusionDataset", 
    "load_sequences",
    "GNNScorer",
    "Denoiser",
    "GCNLayer",
    "DiffusionModel",
    "GNNTrainer",
    "DiffusionTrainer",
    "SequenceGenerator",
    "QualityEvaluator"
] 