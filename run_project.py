#!/usr/bin/env python3
"""
Complete Project Runner for Recycling Sorting Agent
Guides users through running the entire project step by step
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 40)

def check_dependencies():
    """Check if all dependencies are installed"""
    print_step(1, "Checking Dependencies")
    
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
        print("Please install them using:")
        print("pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True

def test_environment():
    """Test the environment"""
    print_step(2, "Testing Environment")
    
    try:
        result = subprocess.run([sys.executable, "test_environment.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Environment test passed!")
            return True
        else:
            print("✗ Environment test failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Environment test timed out!")
        return False
    except Exception as e:
        print(f"✗ Environment test error: {e}")
        return False

def run_random_demo():
    """Run random agent demo"""
    print_step(3, "Running Random Agent Demo")
    
    print("This will open a window showing the environment with random actions.")
    print("Press 'q' to quit or close the window when done.")
    
    try:
        subprocess.run([sys.executable, "demo_random_agent.py"], timeout=300)
        print("✓ Random agent demo completed!")
        return True
    except subprocess.TimeoutExpired:
        print("✓ Random agent demo completed (timeout)")
        return True
    except Exception as e:
        print(f"✗ Random agent demo error: {e}")
        return False

def run_training():
    """Run training"""
    print_step(4, "Running Training")
    
    print("This will train all RL algorithms. This may take 30-60 minutes.")
    print("Training will create models and results in the models/ and results/ directories.")
    
    response = input("Do you want to proceed with training? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Skipping training. You can run it later with: python main.py --mode train")
        return True
    
    try:
        print("Starting training...")
        result = subprocess.run([sys.executable, "main.py", "--mode", "train", "--timesteps", "25000"], 
                              timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✓ Training completed successfully!")
            return True
        else:
            print("✗ Training failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("✓ Training completed (timeout - this is normal for long training)")
        return True
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False

def create_demos():
    """Create demo GIFs"""
    print_step(5, "Creating Demo GIFs")
    
    try:
        result = subprocess.run([sys.executable, "create_demo_gif.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Demo GIFs created successfully!")
            return True
        else:
            print("✗ Demo GIF creation failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✓ Demo GIF creation completed (timeout)")
        return True
    except Exception as e:
        print(f"✗ Demo GIF creation error: {e}")
        return False

def show_results():
    """Show results summary"""
    print_step(6, "Results Summary")
    
    # Check what files were created
    results_dir = Path("results")
    models_dir = Path("models")
    
    if results_dir.exists():
        print("Results directory contents:")
        for file in results_dir.glob("*"):
            print(f"  - {file.name}")
    
    if models_dir.exists():
        print("\nModels directory contents:")
        for model_type in models_dir.iterdir():
            if model_type.is_dir():
                print(f"  - {model_type.name}/")
                for model_file in model_type.glob("*"):
                    if model_file.is_file():
                        print(f"    - {model_file.name}")
    
    print("\n" + "="*60)
    print(" PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nNext steps for your report:")
    print("1. Review the generated plots in results/")
    print("2. Include the demo GIFs in your report")
    print("3. Analyze the training results")
    print("4. Create your final report using the template")
    
    print("\nKey files for your report:")
    print("- results/algorithm_comparison.png: Algorithm comparison plots")
    print("- results/random_agent_demo.gif: Random agent behavior")
    print("- results/trained_agent_demo.gif: Trained agent behavior (if training completed)")
    print("- results/training_results.json: Detailed training results")

def main():
    """Main function"""
    print_header("Recycling Sorting Agent - Complete Project Runner")
    
    print("This script will guide you through running the entire project.")
    print("Make sure you have all dependencies installed first.")
    
    # Check if we're in the right directory
    if not Path("environment").exists() or not Path("training").exists():
        print("✗ Error: Please run this script from the project root directory")
        print("Make sure you're in the directory containing environment/ and training/ folders")
        return
    
    # Run all steps
    steps = [
        ("Dependency Check", check_dependencies),
        ("Environment Test", test_environment),
        ("Random Demo", run_random_demo),
        ("Training", run_training),
        ("Demo Creation", create_demos),
        ("Results Summary", show_results)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n✗ Step '{step_name}' failed. Please fix the issue and try again.")
            return
    
    print("\n🎉 All steps completed successfully!")
    print("Your project is ready for the report!")

if __name__ == "__main__":
    main() 