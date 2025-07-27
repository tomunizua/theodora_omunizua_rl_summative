import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
import sys
sys.path.append('..')

from environment.custom_env import RecyclingSortingEnv

class PolicyGradientTrainer:
    """
    Policy Gradient Trainer for Recycling Sorting Environment
    Implements REINFORCE, PPO, and Actor-Critic algorithms
    """
    
    def __init__(self, log_dir="models/pg", total_timesteps=100000):
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f"{log_dir}/ppo", exist_ok=True)
        os.makedirs(f"{log_dir}/a2c", exist_ok=True)
        os.makedirs(f"{log_dir}/reinforce", exist_ok=True)
        
        # Environment setup
        self.env = Monitor(RecyclingSortingEnv(batch_size=20, conveyor_length=10))
        self.eval_env = Monitor(RecyclingSortingEnv(batch_size=20, conveyor_length=10))
        
    def train_ppo_default(self):
        """Train PPO with default hyperparameters"""
        print("Training PPO with default hyperparameters...")
        
        log_dir_ppo = f"{self.log_dir}/ppo"
        os.makedirs(f"{log_dir_ppo}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir_ppo}/eval", exist_ok=True)
        
        # Callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{log_dir_ppo}/best_model",
            log_path=f"{log_dir_ppo}/eval",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=f"{log_dir_ppo}/checkpoints",
            name_prefix="ppo_model"
        )
        
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_ppo}/tensorboard",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            device="auto"
        )
        
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # Save final model
        model.save(f"{log_dir_ppo}/ppo_final")
        print(f"PPO training completed. Model saved to {log_dir_ppo}/ppo_final")
        
        return model
    
    def train_ppo_optimized(self):
        """Train PPO with optimized hyperparameters"""
        print("Training PPO with optimized hyperparameters...")
        
        log_dir_ppo = f"{self.log_dir}/ppo_optimized"
        os.makedirs(f"{log_dir_ppo}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir_ppo}/eval", exist_ok=True)
        
        # Callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{log_dir_ppo}/best_model",
            log_path=f"{log_dir_ppo}/eval",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=f"{log_dir_ppo}/checkpoints",
            name_prefix="ppo_optimized"
        )
        
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_ppo}/tensorboard",
            learning_rate=1e-4,  # Lower learning rate for stability
            n_steps=1024,        # Smaller batch size
            batch_size=32,       # Smaller batch size
            n_epochs=4,          # Fewer epochs
            gamma=0.95,          # Lower discount factor
            gae_lambda=0.9,      # Lower GAE lambda
            clip_range=0.1,      # Smaller clip range
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,       # Add entropy for exploration
            vf_coef=0.25,        # Lower value function coefficient
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=0.01,      # Target KL divergence
            device="auto"
        )
        
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # Save optimized model
        model.save(f"{log_dir_ppo}/ppo_optimized")
        print(f"PPO optimized training completed. Model saved to {log_dir_ppo}/ppo_optimized")
        
        return model
    
    def train_a2c_default(self):
        """Train Actor-Critic (A2C) with default hyperparameters"""
        print("Training Actor-Critic (A2C) with default hyperparameters...")
        
        log_dir_a2c = f"{self.log_dir}/a2c"
        os.makedirs(f"{log_dir_a2c}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir_a2c}/eval", exist_ok=True)
        
        # Callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{log_dir_a2c}/best_model",
            log_path=f"{log_dir_a2c}/eval",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=f"{log_dir_a2c}/checkpoints",
            name_prefix="a2c_model"
        )
        
        model = A2C(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_a2c}/tensorboard",
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.0,
            vf_coef=0.25,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            use_sde=False,
            sde_sample_freq=-1,
            device="auto"
        )
        
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # Save final model
        model.save(f"{log_dir_a2c}/a2c_final")
        print(f"A2C training completed. Model saved to {log_dir_a2c}/a2c_final")
        
        return model
    
    def train_a2c_optimized(self):
        """Train Actor-Critic (A2C) with optimized hyperparameters"""
        print("Training Actor-Critic (A2C) with optimized hyperparameters...")
        
        log_dir_a2c = f"{self.log_dir}/a2c_optimized"
        os.makedirs(f"{log_dir_a2c}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir_a2c}/eval", exist_ok=True)
        
        # Callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{log_dir_a2c}/best_model",
            log_path=f"{log_dir_a2c}/eval",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=f"{log_dir_a2c}/checkpoints",
            name_prefix="a2c_optimized"
        )
        
        model = A2C(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_a2c}/tensorboard",
            learning_rate=5e-4,  # Lower learning rate
            n_steps=10,          # More steps per update
            gamma=0.95,          # Lower discount factor
            gae_lambda=0.9,      # Lower GAE lambda
            ent_coef=0.01,       # Add entropy for exploration
            vf_coef=0.5,         # Higher value function coefficient
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            use_sde=False,
            sde_sample_freq=-1,
            device="auto"
        )
        
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # Save optimized model
        model.save(f"{log_dir_a2c}/a2c_optimized")
        print(f"A2C optimized training completed. Model saved to {log_dir_a2c}/a2c_optimized")
        
        return model
    
    def train_reinforce(self):
        """Train REINFORCE algorithm (using PPO with specific settings)"""
        print("Training REINFORCE algorithm...")
        
        log_dir_reinforce = f"{self.log_dir}/reinforce"
        os.makedirs(f"{log_dir_reinforce}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir_reinforce}/eval", exist_ok=True)
        
        # Callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{log_dir_reinforce}/best_model",
            log_path=f"{log_dir_reinforce}/eval",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=f"{log_dir_reinforce}/checkpoints",
            name_prefix="reinforce_model"
        )
        
        # REINFORCE-like settings (no clipping, high entropy)
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_reinforce}/tensorboard",
            learning_rate=1e-3,  # Higher learning rate
            n_steps=1,           # Single step updates (like REINFORCE)
            batch_size=1,        # Single sample updates
            n_epochs=1,          # Single epoch
            gamma=0.99,
            gae_lambda=1.0,      # No GAE (like REINFORCE)
            clip_range=1.0,      # No clipping
            clip_range_vf=None,
            normalize_advantage=False,  # No advantage normalization
            ent_coef=0.1,        # High entropy for exploration
            vf_coef=0.0,         # No value function (pure policy gradient)
            max_grad_norm=1.0,   # No gradient clipping
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            device="auto"
        )
        
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # Save REINFORCE model
        model.save(f"{log_dir_reinforce}/reinforce_final")
        print(f"REINFORCE training completed. Model saved to {log_dir_reinforce}/reinforce_final")
        
        return model
    
    def evaluate_model(self, model_path, model_type="PPO", num_episodes=10):
        """Evaluate a trained model"""
        print(f"Evaluating {model_type} model from {model_path}...")
        
        if model_type == "PPO":
            model = PPO.load(model_path)
        elif model_type == "A2C":
            model = A2C.load(model_path)
        else:
            model = PPO.load(model_path)  # REINFORCE uses PPO
        
        total_rewards = []
        efficiencies = []
        correct_sorts = []
        wrong_sorts = []
        missed_items = []
        
        for episode in range(num_episodes):
            obs, info = self.eval_env.reset()
            episode_reward = 0
            episode_correct = 0
            episode_wrong = 0
            episode_missed = 0
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                
                if 'correct_sorts' in info:
                    episode_correct = info['correct_sorts']
                    episode_wrong = info['wrong_sorts']
                    episode_missed = info['missed_items']
            
            total_rewards.append(episode_reward)
            efficiencies.append(info.get('efficiency', 0))
            correct_sorts.append(episode_correct)
            wrong_sorts.append(episode_wrong)
            missed_items.append(episode_missed)
        
        # Calculate statistics
        avg_reward = np.mean(total_rewards)
        avg_efficiency = np.mean(efficiencies)
        avg_correct = np.mean(correct_sorts)
        avg_wrong = np.mean(wrong_sorts)
        avg_missed = np.mean(missed_items)
        
        print(f"{model_type} Evaluation Results ({num_episodes} episodes):")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Efficiency: {avg_efficiency:.2%}")
        print(f"Average Correct Sorts: {avg_correct:.2f}")
        print(f"Average Wrong Sorts: {avg_wrong:.2f}")
        print(f"Average Missed Items: {avg_missed:.2f}")
        
        return {
            'avg_reward': avg_reward,
            'avg_efficiency': avg_efficiency,
            'avg_correct': avg_correct,
            'avg_wrong': avg_wrong,
            'avg_missed': avg_missed,
            'rewards': total_rewards,
            'efficiencies': efficiencies
        }
    
    def plot_training_results(self):
        """Plot training results from tensorboard logs"""
        try:
            # Plot PPO results
            ppo_dir = f"{self.log_dir}/ppo"
            if os.path.exists(ppo_dir):
                plot_results([ppo_dir], self.total_timesteps, "timesteps", "PPO Training Results")
                plt.savefig(f"{ppo_dir}/training_results.png")
                plt.close()
                print(f"PPO training results plot saved to {ppo_dir}/training_results.png")
            
            # Plot A2C results
            a2c_dir = f"{self.log_dir}/a2c"
            if os.path.exists(a2c_dir):
                plot_results([a2c_dir], self.total_timesteps, "timesteps", "A2C Training Results")
                plt.savefig(f"{a2c_dir}/training_results.png")
                plt.close()
                print(f"A2C training results plot saved to {a2c_dir}/training_results.png")
                
        except Exception as e:
            print(f"Could not plot training results: {e}")
    
    def close(self):
        """Close environments"""
        self.env.close()
        self.eval_env.close()

if __name__ == "__main__":
    # Initialize trainer
    trainer = PolicyGradientTrainer(total_timesteps=50000)
    
    try:
        # Train different Policy Gradient variants
        print("=== Policy Gradient Training Session ===")
        
        # PPO Default
        ppo_default = trainer.train_ppo_default()
        ppo_default_results = trainer.evaluate_model(f"{trainer.log_dir}/ppo/ppo_final", "PPO")
        
        # PPO Optimized
        ppo_optimized = trainer.train_ppo_optimized()
        ppo_optimized_results = trainer.evaluate_model(f"{trainer.log_dir}/ppo_optimized/ppo_optimized", "PPO")
        
        # A2C Default
        a2c_default = trainer.train_a2c_default()
        a2c_default_results = trainer.evaluate_model(f"{trainer.log_dir}/a2c/a2c_final", "A2C")
        
        # A2C Optimized
        a2c_optimized = trainer.train_a2c_optimized()
        a2c_optimized_results = trainer.evaluate_model(f"{trainer.log_dir}/a2c_optimized/a2c_optimized", "A2C")
        
        # REINFORCE
        reinforce = trainer.train_reinforce()
        reinforce_results = trainer.evaluate_model(f"{trainer.log_dir}/reinforce/reinforce_final", "REINFORCE")
        
        # Plot results
        trainer.plot_training_results()
        
        # Compare results
        print("\n=== Policy Gradient Model Comparison ===")
        print(f"{'Model':<20} {'Avg Reward':<12} {'Efficiency':<12} {'Correct':<10} {'Wrong':<10} {'Missed':<10}")
        print("-" * 75)
        print(f"{'PPO Default':<20} {ppo_default_results['avg_reward']:<12.2f} {ppo_default_results['avg_efficiency']:<12.2%} {ppo_default_results['avg_correct']:<10.2f} {ppo_default_results['avg_wrong']:<10.2f} {ppo_default_results['avg_missed']:<10.2f}")
        print(f"{'PPO Optimized':<20} {ppo_optimized_results['avg_reward']:<12.2f} {ppo_optimized_results['avg_efficiency']:<12.2%} {ppo_optimized_results['avg_correct']:<10.2f} {ppo_optimized_results['avg_wrong']:<10.2f} {ppo_optimized_results['avg_missed']:<10.2f}")
        print(f"{'A2C Default':<20} {a2c_default_results['avg_reward']:<12.2f} {a2c_default_results['avg_efficiency']:<12.2%} {a2c_default_results['avg_correct']:<10.2f} {a2c_default_results['avg_wrong']:<10.2f} {a2c_default_results['avg_missed']:<10.2f}")
        print(f"{'A2C Optimized':<20} {a2c_optimized_results['avg_reward']:<12.2f} {a2c_optimized_results['avg_efficiency']:<12.2%} {a2c_optimized_results['avg_correct']:<10.2f} {a2c_optimized_results['avg_wrong']:<10.2f} {a2c_optimized_results['avg_missed']:<10.2f}")
        print(f"{'REINFORCE':<20} {reinforce_results['avg_reward']:<12.2f} {reinforce_results['avg_efficiency']:<12.2%} {reinforce_results['avg_correct']:<10.2f} {reinforce_results['avg_wrong']:<10.2f} {reinforce_results['avg_missed']:<10.2f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        trainer.close() 