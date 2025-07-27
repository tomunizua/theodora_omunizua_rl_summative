#!/usr/bin/env python3
"""
Main Entry Point for Recycling Sorting Agent RL Project
Runs training, evaluation, and visualization for all RL algorithms
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pygame
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environment.custom_env import RecyclingSortingEnv
from environment.rendering import RecyclingSortingRenderer
from training.dqn_training import DQNTrainer
from training.pg_training import PolicyGradientTrainer

def create_random_agent_demo():
    """Create a demo showing random agent behavior"""
    print("Creating random agent demo...")
    
    env = RecyclingSortingEnv(batch_size=20, conveyor_length=10)
    renderer = RecyclingSortingRenderer()
    
    # Record frames for GIF
    frames = []
    
    obs, info = env.reset()
    done = False
    step = 0
    
    while not done and step < 200:  # Limit steps for demo
        # Random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        # Render
        renderer.render(info, action, reward)
        
        # Capture frame (simplified - in real implementation would save pygame surface)
        time.sleep(0.1)  # Slow down for visualization
        
        step += 1
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    renderer.close()
    env.close()
    print("Random agent demo completed!")

def run_training_comparison():
    """Run training for all algorithms and compare results"""
    print("=== Starting Comprehensive RL Training Comparison ===")
    
    # Training parameters
    total_timesteps = 50000
    
    # Initialize trainers
    dqn_trainer = DQNTrainer(total_timesteps=total_timesteps)
    pg_trainer = PolicyGradientTrainer(total_timesteps=total_timesteps)
    
    results = {}
    
    try:
        # Train DQN variants
        print("\n--- Training DQN Variants ---")
        dqn_default = dqn_trainer.train_default_dqn()
        results['DQN_Default'] = dqn_trainer.evaluate_model("models/dqn/dqn_final")
        
        dqn_optimized = dqn_trainer.train_optimized_dqn()
        results['DQN_Optimized'] = dqn_trainer.evaluate_model("models/dqn/dqn_optimized")
        
        dqn_exploration = dqn_trainer.train_exploration_dqn()
        results['DQN_Exploration'] = dqn_trainer.evaluate_model("models/dqn/dqn_exploration")
        
        # Train Policy Gradient variants
        print("\n--- Training Policy Gradient Variants ---")
        ppo_default = pg_trainer.train_ppo_default()
        results['PPO_Default'] = pg_trainer.evaluate_model("models/pg/ppo/ppo_final", "PPO")
        
        ppo_optimized = pg_trainer.train_ppo_optimized()
        results['PPO_Optimized'] = pg_trainer.evaluate_model("models/pg/ppo_optimized/ppo_optimized", "PPO")
        
        a2c_default = pg_trainer.train_a2c_default()
        results['A2C_Default'] = pg_trainer.evaluate_model("models/pg/a2c/a2c_final", "A2C")
        
        a2c_optimized = pg_trainer.train_a2c_optimized()
        results['A2C_Optimized'] = pg_trainer.evaluate_model("models/pg/a2c_optimized/a2c_optimized", "A2C")
        
        reinforce = pg_trainer.train_reinforce()
        results['REINFORCE'] = pg_trainer.evaluate_model("models/pg/reinforce/reinforce_final", "REINFORCE")
        
        # Generate plots
        dqn_trainer.plot_training_results()
        pg_trainer.plot_training_results()
        
        # Create comparison plots
        create_comparison_plots(results)
        
        # Print final comparison
        print_final_comparison(results)
        
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        dqn_trainer.close()
        pg_trainer.close()
    
    return results

def create_comparison_plots(results):
    """Create comparison plots for all algorithms"""
    print("Creating comparison plots...")
    
    # Extract data
    models = list(results.keys())
    avg_rewards = [results[model]['avg_reward'] for model in models]
    avg_efficiencies = [results[model]['avg_efficiency'] for model in models]
    avg_correct = [results[model]['avg_correct'] for model in models]
    avg_wrong = [results[model]['avg_wrong'] for model in models]
    avg_missed = [results[model]['avg_missed'] for model in models]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RL Algorithm Comparison - Recycling Sorting Agent', fontsize=16)
    
    # Plot 1: Average Rewards
    bars1 = ax1.bar(models, avg_rewards, color=['blue', 'lightblue', 'cyan', 'green', 'lightgreen', 'orange', 'red', 'purple', 'brown'])
    ax1.set_title('Average Rewards')
    ax1.set_ylabel('Reward')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, avg_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{value:.1f}', 
                ha='center', va='bottom')
    
    # Plot 2: Efficiency
    bars2 = ax2.bar(models, [eff * 100 for eff in avg_efficiencies], color=['blue', 'lightblue', 'cyan', 'green', 'lightgreen', 'orange', 'red', 'purple', 'brown'])
    ax2.set_title('Sorting Efficiency')
    ax2.set_ylabel('Efficiency (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, avg_efficiencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{value:.1%}', 
                ha='center', va='bottom')
    
    # Plot 3: Correct vs Wrong Sorts
    x = np.arange(len(models))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, avg_correct, width, label='Correct Sorts', color='green', alpha=0.7)
    bars3b = ax3.bar(x + width/2, avg_wrong, width, label='Wrong Sorts', color='red', alpha=0.7)
    
    ax3.set_title('Correct vs Wrong Sorts')
    ax3.set_ylabel('Number of Sorts')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Missed Items
    bars4 = ax4.bar(models, avg_missed, color='orange', alpha=0.7)
    ax4.set_title('Missed Items')
    ax4.set_ylabel('Number of Missed Items')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars4, avg_missed):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{value:.1f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison plots saved to results/algorithm_comparison.png")

def print_final_comparison(results):
    """Print final comparison table"""
    print("\n" + "="*80)
    print("FINAL ALGORITHM COMPARISON")
    print("="*80)
    
    # Sort by average reward
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_reward'], reverse=True)
    
    print(f"{'Algorithm':<20} {'Avg Reward':<12} {'Efficiency':<12} {'Correct':<10} {'Wrong':<10} {'Missed':<10}")
    print("-" * 80)
    
    for model_name, result in sorted_results:
        print(f"{model_name:<20} {result['avg_reward']:<12.2f} {result['avg_efficiency']:<12.2%} "
              f"{result['avg_correct']:<10.2f} {result['avg_wrong']:<10.2f} {result['avg_missed']:<10.2f}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Find best performers
    best_reward = max(results.items(), key=lambda x: x[1]['avg_reward'])
    best_efficiency = max(results.items(), key=lambda x: x[1]['avg_efficiency'])
    best_correct = max(results.items(), key=lambda x: x[1]['avg_correct'])
    
    print(f"• Best Average Reward: {best_reward[0]} ({best_reward[1]['avg_reward']:.2f})")
    print(f"• Best Efficiency: {best_efficiency[0]} ({best_efficiency[1]['avg_efficiency']:.2%})")
    print(f"• Most Correct Sorts: {best_correct[0]} ({best_correct[1]['avg_correct']:.2f})")
    
    # Algorithm family comparison
    dqn_models = [k for k in results.keys() if 'DQN' in k]
    pg_models = [k for k in results.keys() if any(x in k for x in ['PPO', 'A2C', 'REINFORCE'])]
    
    if dqn_models and pg_models:
        dqn_avg_reward = np.mean([results[m]['avg_reward'] for m in dqn_models])
        pg_avg_reward = np.mean([results[m]['avg_reward'] for m in pg_models])
        
        print(f"\n• Value-Based (DQN) Average: {dqn_avg_reward:.2f}")
        print(f"• Policy-Based (PG) Average: {pg_avg_reward:.2f}")
        
        if dqn_avg_reward > pg_avg_reward:
            print("• Value-Based methods performed better on average")
        else:
            print("• Policy-Based methods performed better on average")

def run_visualization_demo():
    """Run a visualization demo with trained models"""
    print("Running visualization demo...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load best model (assuming PPO optimized performed well)
    try:
        from stable_baselines3 import PPO
        model = PPO.load("models/pg/ppo_optimized/ppo_optimized")
        
        env = RecyclingSortingEnv(batch_size=20, conveyor_length=10)
        renderer = RecyclingSortingRenderer()
        
        print("Running 3 episodes with trained agent...")
        
        for episode in range(3):
            print(f"Episode {episode + 1}/3")
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                renderer.render(info, action, reward)
                time.sleep(0.05)  # Faster for demo
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
            
            print(f"Episode {episode + 1} completed with total reward: {total_reward:.2f}")
        
        renderer.close()
        env.close()
        
    except Exception as e:
        print(f"Visualization demo error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Recycling Sorting Agent RL Project')
    parser.add_argument('--mode', choices=['train', 'demo', 'visualize', 'all'], 
                       default='all', help='Mode to run')
    parser.add_argument('--timesteps', type=int, default=50000, 
                       help='Number of training timesteps')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    if args.mode in ['train', 'all']:
        print("Starting training phase...")
        results = run_training_comparison()
        
        # Save results to file
        import json
        with open('results/training_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                json_results[key] = {
                    'avg_reward': float(value['avg_reward']),
                    'avg_efficiency': float(value['avg_efficiency']),
                    'avg_correct': float(value['avg_correct']),
                    'avg_wrong': float(value['avg_wrong']),
                    'avg_missed': float(value['avg_missed'])
                }
            json.dump(json_results, f, indent=2)
        
        print("Training results saved to results/training_results.json")
    
    if args.mode in ['demo', 'all']:
        print("Starting random agent demo...")
        create_random_agent_demo()
    
    if args.mode in ['visualize', 'all']:
        print("Starting visualization demo...")
        run_visualization_demo()
    
    print("\n" + "="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Generated files:")
    print("- Trained models in models/ directory")
    print("- Results and plots in results/ directory")
    print("- Training logs and checkpoints")
    print("\nNext steps:")
    print("1. Review the comparison plots")
    print("2. Analyze the training results")
    print("3. Create your report using the generated data")
    print("4. Record video demonstrations if needed")

if __name__ == "__main__":
    main() 