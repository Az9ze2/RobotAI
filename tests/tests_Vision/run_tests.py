"""
Test runner for all vision tests.

Run this script to execute all vision-related unit tests.
"""

import pytest
import sys
from pathlib import Path


def main():
    """Run all vision tests."""
    # Get tests directory
    tests_dir = Path(__file__).parent
    
    # Run pytest with verbose output
    args = [
        str(tests_dir),
        "-v",
        "--tb=short",
        "-s",
        "--color=yes"
    ]
    
    print("=" * 80)
    print("Running Vision System Unit Tests")
    print("=" * 80)
    
    exit_code = pytest.main(args)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
