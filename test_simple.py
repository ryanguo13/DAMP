#!/usr/bin/env python3
"""
Simple test to verify project structure and basic imports
"""

import os
import sys

def test_imports():
    """Test that core modules can be imported."""
    print("🧪 Testing Imports")
    print("="*50)
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from data import load_sequences, AA_LIST
        print("  ✅ src.data imported successfully")
    except Exception as e:
        print(f"  ❌ src.data import failed: {e}")
    
    try:
        from models import GNNScorer, Denoiser, DiffusionModel
        print("  ✅ src.models imported successfully")
    except Exception as e:
        print(f"  ❌ src.models import failed: {e}")
    
    try:
        from trainer import GNNTrainer, DiffusionTrainer
        print("  ✅ src.trainer imported successfully")
    except Exception as e:
        print(f"  ❌ src.trainer import failed: {e}")
    
    try:
        from generator import SequenceGenerator
        print("  ✅ src.generator imported successfully")
    except Exception as e:
        print(f"  ❌ src.generator import failed: {e}")
    
    try:
        from evaluator import QualityEvaluator
        print("  ✅ src.evaluator imported successfully")
    except Exception as e:
        print(f"  ❌ src.evaluator import failed: {e}")
    
    print("\n" + "="*50)

def test_scripts():
    """Test that scripts can be imported."""
    print("🧪 Testing Scripts")
    print("="*50)
    
    try:
        # Test main script import
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
        import main
        print("  ✅ scripts/main.py imported successfully")
    except Exception as e:
        print(f"  ❌ scripts/main.py import failed: {e}")
    
    try:
        # Test demo script import
        import demo
        print("  ✅ scripts/demo.py imported successfully")
    except Exception as e:
        print(f"  ❌ scripts/demo.py import failed: {e}")
    
    print("\n" + "="*50)

def main():
    """Run all tests."""
    print("🚀 DAMP Project Simple Test")
    print("="*60)
    
    test_imports()
    test_scripts()
    
    print("✅ Simple test completed!")

if __name__ == "__main__":
    main() 