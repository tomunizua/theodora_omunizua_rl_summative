#!/usr/bin/env python3
"""
Installation Script for Recycling Sorting Agent Project
Helps users install all required dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✓ Python {sys.version.split()[0]} is compatible")
    return True

def install_requirements():
    """Install requirements from requirements.txt"""
    print("\nInstalling requirements...")
    
    if not Path("requirements.txt").exists():
        print("✗ requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("Installing project dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✓ Requirements installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def verify_installation():
    """Verify that all packages are installed correctly"""
    print("\nVerifying installation...")
    
    required_packages = [
        'gymnasium', 'stable_baselines3', 'pygame', 
        'numpy', 'matplotlib', 'torch', 'imageio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Try installing them manually:")
        for package in missing_packages:
            print(f"pip install {package}")
        return False
    
    print("\n✓ All packages installed successfully!")
    return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = ["models", "results", "models/dqn", "models/pg"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}/")

def main():
    """Main installation function"""
    print("="*60)
    print(" Recycling Sorting Agent - Installation Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("\nInstallation failed. Please check the error messages above.")
        return
    
    # Verify installation
    if not verify_installation():
        print("\nSome packages are missing. Please install them manually.")
        return
    
    # Create directories
    create_directories()
    
    print("\n" + "="*60)
    print(" INSTALLATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Run the project: python run_project.py")
    print("2. Or test the environment: python test_environment.py")
    print("3. Or run the demo: python demo_random_agent.py")
    
    print("\nFor help, see the README.md file.")

if __name__ == "__main__":
    main() 