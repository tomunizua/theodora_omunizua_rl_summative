#!/usr/bin/env python3
"""
Create Training Stability Analysis Plots
Specifically for DQN loss curves and Policy Gradient entropy analysis
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def create_stability_plots():
    """Create training stability analysis plots"""
    print("🔍 Creating Training Stability Analysis...")
    
    # Create results directory
    results_dir = Path("results/training_plots")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Method configurations
    methods = {
        'DQN_Default': 'models/dqn/tensorboard_default',
        'DQN_Optimized': 'models/dqn/tensorboard_optimized', 
        'DQN_Less_Optimized': 'models/dqn/tensorboard_less_optimized',
        'PPO_Default': 'models/pg/ppo/tensorboard',
        'PPO_Optimized': 'models/pg/ppo_optimized/tensorboard',
        'A2C_Default': 'models/pg/a2c/tensorboard',
        'A2C_Optimized': 'models/pg/a2c_optimized/tensorboard',
        'REINFORCE': 'models/pg/reinforce/tensorboard'
    }
    
    # Colors for consistent plotting
    colors = {
        'DQN_Default': '#1f77b4',
        'DQN_Optimized': '#ff7f0e', 
        'DQN_Less_Optimized': '#2ca02c',
        'PPO_Default': '#d62728',
        'PPO_Optimized': '#9467bd',
        'A2C_Default': '#8c564b',
        'A2C_Optimized': '#e377c2',
        'REINFORCE': '#7f7f7f'
    }
    
    # Extract data
    all_data = {}
    for method_name, log_path in methods.items():
        full_path = Path(log_path)
        if not full_path.exists():
            continue
        
        print(f"📊 Processing {method_name}...")
        
        # Find latest run
        run_dirs = [d for d in full_path.iterdir() if d.is_dir()]
        if not run_dirs:
            continue
        
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        
        try:
            ea = EventAccumulator(str(latest_run))
            ea.Reload()
            scalar_tags = ea.Tags()['scalars']
            
            method_data = {}
            for tag in scalar_tags:
                try:
                    scalar_events = ea.Scalars(tag)
                    steps = [event.step for event in scalar_events]
                    values = [event.value for event in scalar_events]
                    method_data[tag] = {
                        'steps': np.array(steps),
                        'values': np.array(values)
                    }
                except:
                    continue
            
            all_data[method_name] = method_data
            
        except Exception as e:
            print(f"❌ Failed to process {method_name}: {e}")
    
    # Create stability plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Stability Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: DQN Loss Functions
    ax1 = axes[0, 0]
    ax1.set_title('DQN Loss Functions', fontweight='bold')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    dqn_methods = [m for m in all_data.keys() if 'DQN' in m]
    for method_name in dqn_methods:
        data = all_data[method_name]
        loss_keys = [k for k in data.keys() if 'loss' in k.lower() and 'train' in k.lower()]
        
        for loss_key in loss_keys:
            steps = data[loss_key]['steps']
            values = data[loss_key]['values']
            
            # Smooth loss curves
            if len(values) > 10:
                window = min(100, len(values) // 20)
                smoothed = pd.Series(values).rolling(window=window).mean()
                ax1.plot(steps, smoothed, 
                        label=f"{method_name}", 
                        color=colors.get(method_name, None), linewidth=2)
    
    ax1.legend()
    ax1.set_yscale('log')  # Log scale for loss
    
    # Plot 2: Policy Entropy (for PG methods)
    ax2 = axes[0, 1]
    ax2.set_title('Policy Entropy (Policy Gradient Methods)', fontweight='bold')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Entropy')
    ax2.grid(True, alpha=0.3)
    
    pg_methods = [m for m in all_data.keys() if any(pg in m for pg in ['PPO', 'A2C', 'REINFORCE'])]
    for method_name in pg_methods:
        data = all_data[method_name]
        entropy_keys = [k for k in data.keys() if 'entropy' in k.lower()]
        
        for entropy_key in entropy_keys:
            steps = data[entropy_key]['steps']
            values = data[entropy_key]['values']
            
            # Smooth entropy curves
            if len(values) > 10:
                window = min(50, len(values) // 10)
                smoothed = pd.Series(values).rolling(window=window).mean()
                ax2.plot(steps, smoothed, 
                        label=f"{method_name}", 
                        color=colors.get(method_name, None), linewidth=2)
    
    ax2.legend()
    
    # Plot 3: Value Function Loss (for actor-critic methods)
    ax3 = axes[1, 0]
    ax3.set_title('Value Function Loss', fontweight='bold')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Value Loss')
    ax3.grid(True, alpha=0.3)
    
    for method_name, data in all_data.items():
        value_loss_keys = [k for k in data.keys() if 'value' in k.lower() and 'loss' in k.lower()]
        
        for loss_key in value_loss_keys:
            steps = data[loss_key]['steps']
            values = data[loss_key]['values']
            
            if len(values) > 10:
                window = min(50, len(values) // 10)
                smoothed = pd.Series(values).rolling(window=window).mean()
                ax3.plot(steps, smoothed, 
                        label=f"{method_name}", 
                        color=colors.get(method_name, None), linewidth=2)
    
    ax3.legend()
    ax3.set_yscale('log')
    
    # Plot 4: Learning Rate Schedules
    ax4 = axes[1, 1]
    ax4.set_title('Learning Rate Schedules', fontweight='bold')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Learning Rate')
    ax4.grid(True, alpha=0.3)
    
    for method_name, data in all_data.items():
        lr_keys = [k for k in data.keys() if 'learning_rate' in k.lower()]
        
        for lr_key in lr_keys:
            steps = data[lr_key]['steps']
            values = data[lr_key]['values']
            
            ax4.plot(steps, values, 
                    label=f"{method_name}", 
                    color=colors.get(method_name, None), linewidth=2)
    
    ax4.legend()
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_stability_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Training stability plot saved to {results_dir / 'training_stability_analysis.png'}")

if __name__ == "__main__":
    create_stability_plots()