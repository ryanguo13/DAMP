#!/usr/bin/env python3
"""
Debug script to check data loading issues
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data import load_sequences
import traceback

def test_load():
    print("Testing data loading...")
    
    try:
        print("Loading AMP sequences...")
        amps = load_sequences('dataset/naturalAMPs_APD2024a.fasta', max_length=200, min_length=10)
        print(f"Loaded {len(amps)} AMP sequences")
        
        print("Loading non-AMP sequences...")
        non_amps = load_sequences('dataset/non-amp.fasta', max_length=200, min_length=10)
        print(f"Loaded {len(non_amps)} non-AMP sequences")
        
        if len(non_amps) == 0:
            print("ERROR: No non-AMP sequences loaded!")
        else:
            print(f"First non-AMP sequence: {non_amps[0][:50]}...")
            print(f"Length: {len(non_amps[0])}")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_load() 