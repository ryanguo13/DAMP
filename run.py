#!/usr/bin/env python3
"""
DAMP (Diffusion-based Antimicrobial Peptide) Generator
Main entry point for the DAMP project

Usage:
    python run.py [options]

Examples:
    # Basic training and generation
    python run.py --epochs 100 --num_sequences 100
    
    # Skip training and use existing models
    python run.py --skip_training --num_sequences 50
    
    # Demo mode with quick training
    python run.py --demo
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scripts.main import main

if __name__ == "__main__":
    main() 