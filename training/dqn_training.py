import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
import sys
sys.path.append('..')

from environment.custom_env import RecyclingSortingEnv

class DQNTrainer:
    """
    DQN (Deep Q-Network) Trainer for Fresh Recycling Sorting Environment
    
    Environment features:
    - Observation: Confidence vector [paper, plastic, organic, metal]
    - Actions: 0=Paper, 1=Plastic, 2=Organic, 3=Metal, 4=Skip
    - Rewards: +10 correct, -10 incorrect, -3 skip, -0.1 step penalty
    """
    
    def __init__(self, log_dir="models/dqn", total_timesteps=100000):
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir}/eval", exist_ok=True)
        os.makedirs(f"{log_dir}/plots", exist_ok=True)
        
        # Environment setup with new parameters
        self.env = Monitor(RecyclingSortingEnv(batch_size=20, max_skips_per_episode=10))
        self.eval_env = Monitor(RecyclingSortingEnv(batch_size=20, max_skips_per_episode=10))
        
        # Callbacks
        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{log_dir}/best_model",
            log_path=f"{log_dir}/eval",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        self.checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=f"{log_dir}/checkpoints",
            name_prefix="dqn_model"
        )
        
    def train_default_dqn(self):
        """Train DQN with default hyperparameters"""
        print("Training DQN with default hyperparameters...")
        
        model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard_default",
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            max_grad_norm=10.0,
            device="auto"
        )
        
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[self.eval_callback, self.checkpoint_callback]
        )
        
        # Save final model
        model.save(f"{self.log_dir}/dqn_default")
        print(f"Default DQN training completed. Model saved to {self.log_dir}/dqn_default")
        
        return model
    
    def train_less_optimized_dqn(self):
        """Train DQN with less optimized hyperparameters"""
        print("Training DQN with less optimized hyperparameters...")
        
        model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard_less_optimized",
            learning_rate=5e-5,  # Lower learning rate
            buffer_size=50000,   # Smaller buffer
            learning_starts=2000, # Start learning later
            batch_size=16,       # Smaller batch size
            tau=1.0,             # No soft update
            gamma=0.99,
            train_freq=8,        # Train less frequently
            gradient_steps=1,
            target_update_interval=2000,  # Update target less frequently
            exploration_fraction=0.15,    # Longer exploration
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,    # Higher final epsilon
            max_grad_norm=10.0,
            device="auto"
        )
        
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[self.eval_callback, self.checkpoint_callback]
        )
        
        # Save less optimized model
        model.save(f"{self.log_dir}/dqn_less_optimized")
        print(f"Less optimized DQN training completed. Model saved to {self.log_dir}/dqn_less_optimized")
        
        return model
    
    def train_optimized_dqn(self):
        """Train DQN with optimized hyperparameters"""
        print("Training DQN with optimized hyperparameters...")
        
        model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard_optimized",
            learning_rate=5e-4,  # Higher learning rate for faster convergence
            buffer_size=50000,   # Smaller buffer for more recent experiences
            learning_starts=500,  # Start learning earlier
            batch_size=64,       # Larger batch size for stability
            tau=0.005,           # Soft update for target network
            gamma=0.95,          # Slightly lower discount factor
            train_freq=1,        # Train every step
            gradient_steps=1,
            target_update_interval=500,  # Update target more frequently
            exploration_fraction=0.2,    # Longer exploration
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,  # Lower final epsilon
            max_grad_norm=1.0,   # Gradient clipping
            device="auto"
        )
        
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[self.eval_callback, self.checkpoint_callback]
        )
        
        # Save optimized model
        model.save(f"{self.log_dir}/dqn_optimized")
        print(f"Optimized DQN training completed. Model saved to {self.log_dir}/dqn_optimized")
        
        return model
    
    def evaluate_model(self, model_path, model_name="DQN", num_episodes=20):
        """Evaluate a trained model with comprehensive metrics"""
        print(f"Evaluating {model_name} model from {model_path}...")
        
        model = DQN.load(model_path)
        
        total_rewards = []
        efficiencies = []
        correct_sorts = []
        wrong_sorts = []
        skips_used = []
        steps_per_episode = []
        convergence_data = []
        
        for episode in range(num_episodes):
            obs, info = self.eval_env.reset()
            episode_reward = 0
            episode_correct = 0
            episode_wrong = 0
            episode_skips = 0
            episode_steps = 0
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                if 'correct_sorts' in info:
                    episode_correct = info['correct_sorts']
                    episode_wrong = info['wrong_sorts']
                    episode_skips = info['skips_used']
            
            total_rewards.append(episode_reward)
            efficiencies.append(info.get('efficiency', 0))
            correct_sorts.append(episode_correct)
            wrong_sorts.append(episode_wrong)
            skips_used.append(episode_skips)
            steps_per_episode.append(episode_steps)
            
            # Track convergence (reward improvement over episodes)
            if episode > 0:
                convergence_data.append(episode_reward - total_rewards[episode-1])
        
        # Calculate comprehensive statistics
        avg_reward = np.mean(total_rewards)
        avg_efficiency = np.mean(efficiencies)
        avg_correct = np.mean(correct_sorts)
        avg_wrong = np.mean(wrong_sorts)
        avg_skips = np.mean(skips_used)
        avg_steps = np.mean(steps_per_episode)
        std_reward = np.std(total_rewards)
        
        # Convergence analysis
        convergence_rate = np.mean(convergence_data) if convergence_data else 0
        stability = 1.0 / (1.0 + std_reward)  # Higher stability = lower variance
        
        print(f"{model_name} Evaluation Results ({num_episodes} episodes):")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Efficiency: {avg_efficiency:.2%}")
        print(f"Average Correct Sorts: {avg_correct:.2f}")
        print(f"Average Wrong Sorts: {avg_wrong:.2f}")
        print(f"Average Skips Used: {avg_skips:.2f}")
        print(f"Average Steps per Episode: {avg_steps:.2f}")
        print(f"Convergence Rate: {convergence_rate:.3f}")
        print(f"Stability Score: {stability:.3f}")
        
        return {
            'model_name': model_name,
            'avg_reward': avg_reward,
            'avg_efficiency': avg_efficiency,
            'avg_correct': avg_correct,
            'avg_wrong': avg_wrong,
            'avg_skips': avg_skips,
            'avg_steps': avg_steps,
            'std_reward': std_reward,
            'convergence_rate': convergence_rate,
            'stability': stability,
            'rewards': total_rewards,
            'efficiencies': efficiencies,
            'steps_per_episode': steps_per_episode,
            'convergence_data': convergence_data
        }
    
    def create_performance_visualizations(self, results_dict):
        """Create comprehensive performance visualizations"""
        print("Creating performance visualizations...")
        
        models = list(results_dict.keys())
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DQN Performance Analysis - Fresh Recycling Sorting Environment', fontsize=16)
        
        # Plot 1: Average Rewards with Error Bars
        rewards = [results_dict[model]['avg_reward'] for model in models]
        std_rewards = [results_dict[model]['std_reward'] for model in models]
        
        bars1 = ax1.bar(models, rewards, yerr=std_rewards, capsize=5, 
                       color=['blue', 'orange', 'green'], alpha=0.7)
        ax1.set_title('Average Rewards with Standard Deviation')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 2: Efficiency and Steps per Episode
        efficiencies = [results_dict[model]['avg_efficiency'] * 100 for model in models]
        steps = [results_dict[model]['avg_steps'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars2a = ax2.bar(x - width/2, efficiencies, width, label='Efficiency (%)', 
                        color='green', alpha=0.7)
        ax2_twin = ax2.twinx()
        bars2b = ax2_twin.bar(x + width/2, steps, width, label='Steps per Episode', 
                             color='red', alpha=0.7)
        
        ax2.set_title('Efficiency vs Steps per Episode')
        ax2.set_ylabel('Efficiency (%)', color='green')
        ax2_twin.set_ylabel('Steps per Episode', color='red')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence Analysis
        convergence_rates = [results_dict[model]['convergence_rate'] for model in models]
        stability_scores = [results_dict[model]['stability'] for model in models]
        
        bars3a = ax3.bar(x - width/2, convergence_rates, width, label='Convergence Rate', 
                        color='purple', alpha=0.7)
        bars3b = ax3.bar(x + width/2, stability_scores, width, label='Stability Score', 
                        color='cyan', alpha=0.7)
        
        ax3.set_title('Convergence and Stability Analysis')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Detailed Performance Breakdown
        correct_sorts = [results_dict[model]['avg_correct'] for model in models]
        wrong_sorts = [results_dict[model]['avg_wrong'] for model in models]
        skips_used = [results_dict[model]['avg_skips'] for model in models]
        
        x_pos = np.arange(len(models))
        bars4a = ax4.bar(x_pos - width, correct_sorts, width, label='Correct Sorts', 
                        color='green', alpha=0.7)
        bars4b = ax4.bar(x_pos, wrong_sorts, width, label='Wrong Sorts', 
                        color='red', alpha=0.7)
        bars4c = ax4.bar(x_pos + width, skips_used, width, label='Skips Used', 
                        color='orange', alpha=0.7)
        
        ax4.set_title('Sorting Performance Breakdown')
        ax4.set_ylabel('Number of Items')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(models, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/plots/dqn_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance visualizations saved to {self.log_dir}/plots/dqn_performance_analysis.png")
    
    def analyze_exploration_exploitation(self, results_dict):
        """Analyze exploration vs exploitation balance"""
        print("\n=== Exploration vs Exploitation Analysis ===")
        
        for model_name, results in results_dict.items():
            print(f"\n{model_name}:")
            
            # Analyze reward consistency (exploitation)
            reward_std = results['std_reward']
            avg_reward = results['avg_reward']
            
            if reward_std < 5.0:
                exploitation_score = "High"
            elif reward_std < 10.0:
                exploitation_score = "Medium"
            else:
                exploitation_score = "Low"
            
            # Analyze convergence (exploration effectiveness)
            convergence_rate = results['convergence_rate']
            if convergence_rate > 0.5:
                exploration_score = "Effective"
            elif convergence_rate > 0:
                exploration_score = "Moderate"
            else:
                exploration_score = "Poor"
            
            print(f"  Exploitation Score: {exploitation_score} (Reward Std: {reward_std:.2f})")
            print(f"  Exploration Score: {exploration_score} (Convergence Rate: {convergence_rate:.3f})")
            
            # Identify weaknesses
            weaknesses = []
            if results['avg_efficiency'] < 0.7:
                weaknesses.append("Low sorting efficiency")
            if results['avg_skips'] > 5.0:
                weaknesses.append("Excessive skipping")
            if reward_std > 10.0:
                weaknesses.append("Inconsistent performance")
            if convergence_rate < 0:
                weaknesses.append("Poor convergence")
            
            if weaknesses:
                print(f"  Weaknesses: {', '.join(weaknesses)}")
            else:
                print("  No major weaknesses identified")
            
            # Suggest improvements
            improvements = []
            if results['avg_efficiency'] < 0.7:
                improvements.append("Increase learning rate or reduce exploration")
            if results['avg_skips'] > 5.0:
                improvements.append("Reduce skip penalty or improve confidence threshold")
            if reward_std > 10.0:
                improvements.append("Increase batch size or adjust learning rate")
            if convergence_rate < 0:
                improvements.append("Extend training time or adjust exploration schedule")
            
            if improvements:
                print(f"  Suggested Improvements: {', '.join(improvements)}")
    
    def plot_training_results(self):
        """Plot training results from tensorboard logs"""
        try:
            plot_results([self.log_dir], self.total_timesteps, "timesteps", "DQN Training Results")
            plt.savefig(f"{self.log_dir}/plots/training_results.png")
            plt.close()
            print(f"Training results plot saved to {self.log_dir}/plots/training_results.png")
        except Exception as e:
            print(f"Could not plot training results: {e}")
    
    def close(self):
        """Close environments"""
        self.env.close()
        self.eval_env.close()

if __name__ == "__main__":
    # Initialize trainer
    trainer = DQNTrainer(total_timesteps=50000)
    
    try:
        # Train different DQN variants
        print("=== DQN Training Session ===")
        
        # Default DQN
        default_model = trainer.train_default_dqn()
        default_results = trainer.evaluate_model(f"{trainer.log_dir}/dqn_default", "DQN Default")
        
        # Less Optimized DQN
        less_optimized_model = trainer.train_less_optimized_dqn()
        less_optimized_results = trainer.evaluate_model(f"{trainer.log_dir}/dqn_less_optimized", "DQN Less Optimized")
        
        # Optimized DQN
        optimized_model = trainer.train_optimized_dqn()
        optimized_results = trainer.evaluate_model(f"{trainer.log_dir}/dqn_optimized", "DQN Optimized")
        
        # Combine results
        results_dict = {
            'DQN Default': default_results,
            'DQN Less Optimized': less_optimized_results,
            'DQN Optimized': optimized_results
        }
        
        # Create visualizations
        trainer.create_performance_visualizations(results_dict)
        trainer.plot_training_results()
        
        # Analyze exploration vs exploitation
        trainer.analyze_exploration_exploitation(results_dict)
        
        # Compare results
        print("\n=== DQN Model Comparison ===")
        print(f"{'Model':<20} {'Avg Reward':<12} {'Efficiency':<12} {'Steps':<8} {'Stability':<10} {'Convergence':<12}")
        print("-" * 80)
        for model_name, result in results_dict.items():
            print(f"{model_name:<20} {result['avg_reward']:<12.2f} {result['avg_efficiency']:<12.2%} "
                  f"{result['avg_steps']:<8.1f} {result['stability']:<10.3f} {result['convergence_rate']:<12.3f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        trainer.close() 