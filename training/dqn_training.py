import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import RecyclingSortingEnv

class DQNTrainer:
    """
    DQN (Deep Q-Network) Trainer for Recycling Sorting Environment (OPTIMIZED)

    Environment features (FIXED FOR OPTIMAL LEARNING):
    - Observation: Confidence vector [paper, plastic, organic, metal]
    - Actions: 0=Paper, 1=Plastic, 2=Organic, 3=Metal, 4=Discard, 5=Scan (FIXED MAPPING - NO SHUFFLING)
    - Rewards (STRATEGIC STRUCTURE):
      * Correct & confident: +50 (+1 participation)
      * Correct & not confident: +30 (+1 participation)
      * Correct discard: +20 (+1 participation)
      * False discard: -10 penalty (prevents discard abuse)
      * Wrong sorts: +2-8 (still positive for exploration)
      * Scan action: -0.2 time penalty (strategic cost)
      * Item timeout: -15 penalty (missed opportunity)
    """

    def __init__(self, log_dir="models/dqn", total_timesteps=100000):
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps

        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir}/eval", exist_ok=True)
        os.makedirs(f"{log_dir}/plots", exist_ok=True)

        # Environment setup - optimized for learning with fixed bins and proper terminal conditions
        env_params = {
            'batch_size': 10,                     # For internal logic (not termination)
            'confidence_discard_threshold': 0.6,  # Higher threshold for clearer decisions
            'confidence_noise_std': 0.02,         # Minimal noise for clearer learning signals
            'item_timeout': 150,                  # Longer timeout to eliminate time pressure
            'max_wrong_sorts': 8,                 # Episode ends after 8 wrong sorts
            'bin_capacity': 15                    # Episode ends when any bin reaches 15 items
        }
        
        self.env = Monitor(RecyclingSortingEnv(**env_params))
        self.eval_env = Monitor(RecyclingSortingEnv(**env_params))

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
        """Train DQN with default hyperparameters (adjusted for less harsh rewards)"""
        print("Training DQN with default hyperparameters...")

        model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard_default",
            learning_rate=2e-4,  # Slightly higher for faster learning with gentler rewards
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
            exploration_fraction=0.2,    # Longer exploration
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,    # Higher final epsilon
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
        """Train DQN with high-performance optimized hyperparameters"""
        print("Training DQN with high-performance optimized hyperparameters...")

        # Import torch for policy kwargs
        try:
            import torch
        except ImportError:
            print("PyTorch not available, using default policy")
            torch = None

        model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard_optimized",
            # Learning parameters optimized for recycling task
            learning_rate=3e-4,  # Balanced learning rate for stable convergence
            buffer_size=200000,  # Larger buffer for more diverse experiences
            learning_starts=2000,  # More initial exploration for better foundation
            batch_size=64,       # Optimal batch size for stability vs speed
            
            # Advanced DQN features for better performance
            tau=0.01,            # Soft target update for stability
            gamma=0.995,         # Higher discount factor for better long-term planning
            train_freq=4,        # Train every 4 steps
            gradient_steps=2,    # Multiple gradient steps per update for better convergence
            target_update_interval=1000,  # Balanced target network updates
            
            # Optimized exploration strategy
            exploration_fraction=0.2,     # Longer exploration phase for complex environment
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,   # Very low final epsilon for maximum exploitation
            
            # Enhanced network architecture (if torch available)
            policy_kwargs=dict(
                net_arch=[256, 256, 128] if torch else [64, 64],  # Deeper network for complex patterns
                activation_fn=torch.nn.ReLU if torch else "relu",
                normalize_images=False
            ),
            
            # Training stability improvements
            max_grad_norm=1.0,   # Tighter gradient clipping for stability
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

    def train_high_performance_dqn(self):
        """Train DQN with high-performance configuration for recycling environment"""
        print("Training High-Performance DQN with advanced techniques...")

        # Import torch for policy kwargs
        try:
            import torch
        except ImportError:
            print("PyTorch not available, using default policy")
            torch = None

        model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard_high_performance",
            # Learning parameters optimized for recycling task
            learning_rate=3e-4,  # Balanced learning rate
            buffer_size=200000,  # Larger buffer for more diverse experiences
            learning_starts=2000,  # More initial exploration
            batch_size=64,       # Optimal batch size for stability vs speed
            
            # Advanced DQN features
            tau=0.01,            # Soft target update for stability
            gamma=0.995,         # High discount for long-term planning
            train_freq=4,        # Train every 4 steps
            gradient_steps=2,    # Multiple gradient steps per update
            target_update_interval=1000,  # Balanced target network updates
            
            # Exploration strategy
            exploration_fraction=0.2,     # Longer exploration phase
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,   # Very low final epsilon
            
            # Network architecture improvements (if torch available)
            policy_kwargs=dict(
                net_arch=[256, 256, 128] if torch else [64, 64],  # Deeper network for complex patterns
                activation_fn=torch.nn.ReLU if torch else "relu",
                normalize_images=False
            ),
            
            # Training stability
            max_grad_norm=1.0,   # Tighter gradient clipping
            device="auto"
        )

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[self.eval_callback, self.checkpoint_callback]
        )

        # Save high-performance model
        model.save(f"{self.log_dir}/dqn_high_performance")
        print(f"High-Performance DQN training completed. Model saved to {self.log_dir}/dqn_high_performance")

        return model

    def train_curriculum_dqn(self):
        """Train DQN with curriculum learning - start easy, increase difficulty"""
        print("Training DQN with Curriculum Learning...")
        
        # Stage 1: Easy environment (less noise, longer timeout)
        print("Stage 1: Easy training (low noise, long timeout)")
        easy_env_params = {
            'bin_capacity': 20,            # Larger bins for easier learning
            'max_wrong_sorts': 12,         # More forgiving error limit
            'confidence_discard_threshold': 0.4,
            'confidence_noise_std': 0.05,  # Less noise
            'item_timeout': 120            # Longer timeout
        }
        
        easy_env = Monitor(RecyclingSortingEnv(**easy_env_params))
        
        model = DQN(
            "MlpPolicy",
            easy_env,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard_curriculum",
            learning_rate=5e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            tau=0.01,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.3,  # More exploration in easy stage
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            max_grad_norm=10.0,
            device="auto"
        )
        
        # Train on easy environment
        model.learn(total_timesteps=self.total_timesteps // 3)
        
        # Stage 2: Medium difficulty
        print("Stage 2: Medium training (normal noise, normal timeout)")
        medium_env = Monitor(RecyclingSortingEnv(**{
            'bin_capacity': 15,            # Standard bin capacity
            'max_wrong_sorts': 8,          # Standard error limit
            'confidence_discard_threshold': 0.4,
            'confidence_noise_std': 0.1,
            'item_timeout': 80
        }))
        
        model.set_env(medium_env)
        model.exploration_final_eps = 0.05  # Reduce exploration
        model.learn(total_timesteps=self.total_timesteps // 3)
        
        # Stage 3: Hard environment (more noise, shorter timeout)
        print("Stage 3: Hard training (high noise, short timeout)")
        hard_env = Monitor(RecyclingSortingEnv(**{
            'bin_capacity': 12,            # Smaller bins for challenge
            'max_wrong_sorts': 6,          # Stricter error limit
            'confidence_discard_threshold': 0.4,
            'confidence_noise_std': 0.15,  # More noise
            'item_timeout': 60             # Shorter timeout
        }))
        
        model.set_env(hard_env)
        model.exploration_final_eps = 0.02  # Minimal exploration
        model.learn(total_timesteps=self.total_timesteps // 3)
        
        # Clean up
        easy_env.close()
        medium_env.close()
        hard_env.close()
        
        # Save curriculum model
        model.save(f"{self.log_dir}/dqn_curriculum")
        print(f"Curriculum DQN training completed. Model saved to {self.log_dir}/dqn_curriculum")
        
        return model

    def evaluate_model(self, model_path, model_name="DQN", num_episodes=20):
        """Evaluate a trained model with comprehensive metrics"""
        print(f"Evaluating {model_name} model from {model_path}...")

        model = DQN.load(model_path)

        total_rewards = []
        efficiencies = []
        correct_sorts = []
        wrong_sorts = []
        discards = []
        steps_per_episode = []
        convergence_data = []

        for episode in range(num_episodes):
            obs, info = self.eval_env.reset()
            episode_reward = 0
            episode_correct = 0
            episode_wrong = 0
            episode_discards = 0
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
                    episode_discards = info['discarded']

            total_rewards.append(episode_reward)
            efficiencies.append(info.get('efficiency', 0))
            correct_sorts.append(episode_correct)
            wrong_sorts.append(episode_wrong)
            discards.append(episode_discards)
            steps_per_episode.append(episode_steps)

            # Track convergence (reward improvement over episodes)
            if episode > 0:
                convergence_data.append(episode_reward - total_rewards[episode-1])

        # Calculate comprehensive statistics
        avg_reward = np.mean(total_rewards)
        avg_efficiency = np.mean(efficiencies)
        avg_correct = np.mean(correct_sorts)
        avg_wrong = np.mean(wrong_sorts)
        avg_discards = np.mean(discards)
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
        print(f"Average Discards: {avg_discards:.2f}")
        print(f"Average Steps per Episode: {avg_steps:.2f}")
        print(f"Convergence Rate: {convergence_rate:.3f}")
        print(f"Stability Score: {stability:.3f}")

        return {
            'model_name': model_name,
            'avg_reward': avg_reward,
            'avg_efficiency': avg_efficiency,
            'avg_correct': avg_correct,
            'avg_wrong': avg_wrong,
            'avg_discards': avg_discards,
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
        fig.suptitle('DQN Performance Analysis - Recycling Sorting Environment', fontsize=16)

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
        discards = [results_dict[model]['avg_discards'] for model in models]

        x_pos = np.arange(len(models))
        bars4a = ax4.bar(x_pos - width, correct_sorts, width, label='Correct Sorts',
                        color='green', alpha=0.7)
        bars4b = ax4.bar(x_pos, wrong_sorts, width, label='Wrong Sorts',
                        color='red', alpha=0.7)
        bars4c = ax4.bar(x_pos + width, discards, width, label='Discards',
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
            if results['avg_discards'] > 5.0:
                weaknesses.append("Excessive discarding")
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
            if results['avg_discards'] > 5.0:
                improvements.append("Reduce discard penalty or improve confidence threshold")
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

    def train_and_compare_all_models(self):
        """Train all DQN variants and compare their performance"""
        print(" Training and Comparing All DQN Variants")
        print("=" * 60)
        
        models_to_train = [
            ("Default DQN", self.train_default_dqn),
            ("Less Optimized DQN", self.train_less_optimized_dqn),
            ("Optimized DQN", self.train_optimized_dqn),
            ("High-Performance DQN", self.train_high_performance_dqn),
            ("Curriculum DQN", self.train_curriculum_dqn)
        ]
        
        results = {}
        
        for model_name, train_method in models_to_train:
            print(f"\n Training {model_name}...")
            try:
                model = train_method()
                
                # Evaluate the model
                model_path = f"{self.log_dir}/dqn_{model_name.lower().replace(' ', '_').replace('-', '_')}"
                results[model_name] = self.evaluate_model(model_path, model_name, num_episodes=30)
                
                print(f"✅ {model_name} completed successfully")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                continue
        
        if len(results) > 1:
            # Create comprehensive comparison
            print(f"\n Creating Performance Visualizations...")
            self.create_performance_visualizations(results)
            
            # Find best model
            best_model = max(results.keys(), key=lambda x: results[x]['avg_reward'])
            best_efficiency = max(results.keys(), key=lambda x: results[x]['avg_efficiency'])
            most_stable = max(results.keys(), key=lambda x: results[x]['stability'])
            
            print(f"\n PERFORMANCE SUMMARY")
            print("=" * 50)
            print(f" Best Average Reward: {best_model} ({results[best_model]['avg_reward']:.2f})")
            print(f" Best Efficiency: {best_efficiency} ({results[best_efficiency]['avg_efficiency']:.2%})")
            print(f" Most Stable: {most_stable} (stability: {results[most_stable]['stability']:.3f})")
            
            # Detailed recommendations
            print(f"\n RECOMMENDATIONS")
            print("=" * 50)
            
            if results[best_model]['avg_reward'] > 150:
                print("✅ Excellent performance achieved!")
            elif results[best_model]['avg_reward'] > 100:
                print("✅ Good performance, consider fine-tuning hyperparameters")
            else:
                print("⚠️  Performance needs improvement:")
                print("   - Try longer training (increase total_timesteps)")
                print("   - Adjust reward structure in environment")
                print("   - Consider different network architectures")
            
            return results
        else:
            print(" Not enough models trained for comparison")
            return results

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