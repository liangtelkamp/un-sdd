#!/usr/bin/env python3
"""
Test runner script for the classifier test suite.
This script demonstrates how to run all tests with proper configuration.
"""
import subprocess
import sys
import os


def run_tests():
    """Run all tests in the test suite."""
    print("🧪 Running comprehensive classifier test suite...")
    print("=" * 60)
    
    # Change to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Test commands to run
    test_commands = [
        # Run all tests with verbose output
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        
        # Run tests with coverage (if pytest-cov is installed)
        ["python", "-m", "pytest", "tests/", "--cov=classifiers", "--cov-report=term-missing"],
        
        # Run specific test files
        ["python", "-m", "pytest", "tests/test_pii_classifier.py", "-v"],
        ["python", "-m", "pytest", "tests/test_pii_sensitivity_classifier.py", "-v"],
        ["python", "-m", "pytest", "tests/test_non_pii_classifier.py", "-v"],
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n🔍 Running test command {i}: {' '.join(cmd)}")
        print("-" * 40)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Tests passed successfully!")
                if result.stdout:
                    print("Output:", result.stdout)
            else:
                print("❌ Tests failed!")
                if result.stderr:
                    print("Error:", result.stderr)
                if result.stdout:
                    print("Output:", result.stdout)
                    
        except subprocess.TimeoutExpired:
            print("⏰ Tests timed out after 5 minutes")
        except FileNotFoundError:
            print("❌ pytest not found. Please install pytest: pip install pytest")
        except Exception as e:
            print(f"❌ Error running tests: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test suite execution completed!")


if __name__ == "__main__":
    run_tests()