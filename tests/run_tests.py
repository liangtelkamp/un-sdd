#!/usr/bin/env python3
"""
Test runner for LLM Model Strategy Pattern tests.

This script runs all tests in the tests directory with proper configuration.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests in the tests directory."""
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    print("Running LLM Model Strategy Pattern Tests")
    print("=" * 50)
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest", 
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("\n" + "=" * 50)
        print("All tests passed! ✅")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nTests failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
