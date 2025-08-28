#!/usr/bin/env python3
"""
Test runner script for DAMP project
Runs all tests in the test directory
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_test(test_file: str, description: str = None):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Add src to path for imports
        env = os.environ.copy()
        src_path = str(Path(__file__).parent.parent / 'src')
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{src_path}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = src_path
        
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=Path(__file__).parent.parent,
            env=env,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED - {test_file} (Duration: {duration:.2f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ FAILED - {test_file} (Duration: {duration:.2f}s)")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ ERROR - {test_file}: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 DAMP Test Suite")
    print("="*60)
    
    # Define tests to run
    tests = [
        ("test_data_loading.py", "Data loading and validation tests"),
        ("test_generation.py", "Sequence generation tests"),
        ("test_model_comparison.py", "Model comparison tests"),
        ("test_enhanced_training.py", "Enhanced training tests"),
    ]
    
    # Get test directory
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Change to project root
    os.chdir(project_root)
    
    # Run tests
    passed = 0
    total = len(tests)
    
    for test_file, description in tests:
        test_path = test_dir / test_file
        if test_path.exists():
            if run_test(str(test_path), description):
                passed += 1
        else:
            print(f"⚠️  SKIPPED - {test_file} (file not found)")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 