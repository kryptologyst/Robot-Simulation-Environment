#!/usr/bin/env python3
"""Setup script for Robot Simulation Environment."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True


def install_dependencies():
    """Install project dependencies."""
    commands = [
        ("python -m pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install -e .", "Installing package in development mode")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def setup_pre_commit():
    """Setup pre-commit hooks."""
    commands = [
        ("pip install pre-commit", "Installing pre-commit"),
        ("pre-commit install", "Installing pre-commit hooks")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  {description} failed, but continuing...")
    
    return True


def run_tests():
    """Run unit tests."""
    return run_command("python -m pytest tests/ -v", "Running unit tests")


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "logs", 
        "assets",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Robot Simulation Environment...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup pre-commit
    setup_pre_commit()
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but setup completed")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run basic demo: python scripts/run_demo.py basic")
    print("2. Run interactive demo: python scripts/run_demo.py interactive")
    print("3. Run algorithm comparison: python scripts/run_demo.py comparison")
    print("\nFor interactive visualization:")
    print("streamlit run demo/app.py")
    
    print("\n⚠️  SAFETY WARNING:")
    print("This simulation is for RESEARCH AND EDUCATION ONLY.")
    print("Do not use on real robots without expert review!")


if __name__ == "__main__":
    main()
