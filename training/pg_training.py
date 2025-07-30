import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import RecyclingSortingEnv

class PolicyGradientTrainer:
    """
    Policy Gradient Trainer for Recycling Sorting Environment (OPTIMIZED)
    Implements PPO and A2C algorithms with ultra-aggressive hyperparameters

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

    def __init__(self, log_dir="models/pg", total_timesteps=100000):
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps

        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f"{log_dir}/ppo", exist_ok=True)
        os.makedirs(f"{log_dir}/ppo_optimized", exist_ok=True)
        os.makedirs(f"{log_dir}/a2c", exist_ok=True)
        os.makedirs(f"{log_dir}/a2c_optimized", exist_ok=True)
        os.makedirs(f"{log_dir}/reinforce", exist_ok=True)

        # Environment setup - optimized for PG methods with proper terminal conditions
        env_params = {
            'batch_size': 10,  # For internal logic (not termination)
            'confidence_discard_threshold': 0.6,  # Higher threshold for clearer decisions
            'confidence_noise_std': 0.02,  # Minimal noise for clearer learning signals
            'item_timeout': 150,  # Longer timeout to eliminate time pressure
            'max_wrong_sorts': 8,  # Episode ends after 8 wrong sorts
            'bin_capacity': 15  # Episode ends when any bin reaches 15 items
        }
        
        self.env = Monitor(RecyclingSortingEnv(**env_params))
        self.eval_env = Monitor(RecyclingSortingEnv(**env_params))

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

        # Import torch for policy kwargs
        try:
            import torch
        except ImportError:
            print("PyTorch not available, using default policy")
            torch = None

        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_ppo}/tensorboard",
            # Properly optimized learning parameters (better than defaults but stable)
            learning_rate=1e-3,  # Higher than default (3e-4) but not too aggressive
            n_steps=1024,        # Smaller than default (2048) for more frequent updates
            batch_size=32,       # Smaller than default (64) for more updates
            n_epochs=15,         # More than default (10) for better policy updates
            gamma=0.99,          # Keep default gamma for stability
            gae_lambda=0.95,     # Keep default GAE lambda
            clip_range=0.2,      # Keep default clip range for stability
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,       # Small entropy for some exploration (default is 0.0)
            vf_coef=0.5,         # Keep default value function coefficient
            max_grad_norm=0.5,   # Keep default gradient norm for stability
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,      # Keep default (no KL constraint)
            
            # Simpler but effective network architecture
            policy_kwargs=dict(
                net_arch=[128, 128, 64] if torch else [64, 64],  # Simpler network for faster training
                activation_fn=torch.nn.Tanh if torch else "tanh",  # Tanh activation for better gradients
                normalize_images=False
            ),
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

        # Import torch for policy kwargs
        try:
            import torch
        except ImportError:
            print("PyTorch not available, using default policy")
            torch = None

        model = A2C(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_a2c}/tensorboard",
            # Properly optimized learning parameters (better than defaults but stable)
            learning_rate=1e-3,  # Higher than default (7e-4) but not too aggressive
            n_steps=8,           # Slightly larger than default (5) for better stability
            gamma=0.99,          # Keep default gamma for stability
            gae_lambda=1.0,      # Keep default GAE lambda
            ent_coef=0.01,       # Small entropy for some exploration (default is 0.0)
            vf_coef=0.5,         # Higher than default (0.25) for better value learning
            max_grad_norm=0.5,   # Keep default gradient norm for stability
            rms_prop_eps=1e-5,   # Keep default epsilon
            use_sde=False,
            sde_sample_freq=-1,
            
            # Simpler but effective network architecture
            policy_kwargs=dict(
                net_arch=[128, 128, 64] if torch else [64, 64],  # Simpler network for faster training
                activation_fn=torch.nn.Tanh if torch else "tanh",  # Tanh activation for better gradients
                normalize_images=False
            ),
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

    def train_ppo_high_performance(self):
        """Train PPO with high-performance configuration for recycling environment"""
        print("Training High-Performance PPO with advanced techniques...")

        log_dir_ppo = f"{self.log_dir}/ppo_high_performance"
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
            name_prefix="ppo_high_performance"
        )

        # Import torch for policy kwargs
        try:
            import torch
        except ImportError:
            print("PyTorch not available, using default policy")
            torch = None

        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_ppo}/tensorboard",
            # High-performance parameters (more aggressive than optimized but stable)
            learning_rate=2e-3,  # Higher than optimized (1e-3) for faster learning
            n_steps=512,         # Smaller than optimized (1024) for more frequent updates
            batch_size=16,       # Smaller than optimized (32) for more updates
            n_epochs=20,         # More than optimized (15) for better policy updates
            gamma=0.99,          # Keep stable gamma
            gae_lambda=0.95,     # Keep stable GAE lambda
            clip_range=0.3,      # Slightly larger than default (0.2) but not too aggressive
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.02,       # Higher than optimized (0.01) for more exploration
            vf_coef=0.75,        # Higher than optimized (0.5) for better value learning
            max_grad_norm=1.0,   # Higher than default (0.5) but stable
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,      # Keep stable (no KL constraint)
            
            # Optimized network architecture for recycling task
            policy_kwargs=dict(
                net_arch=[64, 64, 32] if torch else [32, 32],  # Compact network for fast training
                activation_fn=torch.nn.Tanh if torch else "tanh",  # Tanh for better gradients
                normalize_images=False
            ),
            device="auto"
        )

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )

        # Save high-performance model
        model.save(f"{log_dir_ppo}/ppo_high_performance")
        print(f"High-Performance PPO training completed. Model saved to {log_dir_ppo}/ppo_high_performance")

        return model

    def train_a2c_high_performance(self):
        """Train A2C with high-performance configuration for recycling environment"""
        print("Training High-Performance A2C with advanced techniques...")

        log_dir_a2c = f"{self.log_dir}/a2c_high_performance"
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
            name_prefix="a2c_high_performance"
        )

        # Import torch for policy kwargs
        try:
            import torch
        except ImportError:
            print("PyTorch not available, using default policy")
            torch = None

        model = A2C(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=f"{log_dir_a2c}/tensorboard",
            # High-performance parameters (more aggressive than optimized but stable)
            learning_rate=2e-3,  # Higher than optimized (1e-3) for faster learning
            n_steps=5,           # Smaller than optimized (8) for more frequent updates
            gamma=0.99,          # Keep stable gamma
            gae_lambda=1.0,      # Keep stable GAE lambda
            ent_coef=0.02,       # Higher than optimized (0.01) for more exploration
            vf_coef=0.75,        # Higher than optimized (0.5) for better value learning
            max_grad_norm=1.0,   # Higher than default (0.5) but stable
            rms_prop_eps=1e-5,   # Keep stable epsilon
            use_sde=False,
            sde_sample_freq=-1,
            
            # Optimized network architecture for recycling task
            policy_kwargs=dict(
                net_arch=[64, 64, 32] if torch else [32, 32],  # Compact network for fast training
                activation_fn=torch.nn.Tanh if torch else "tanh",  # Tanh for better gradients
                normalize_images=False
            ),
            device="auto"
        )

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )

        # Save high-performance model
        model.save(f"{log_dir_a2c}/a2c_high_performance")
        print(f"High-Performance A2C training completed. Model saved to {log_dir_a2c}/a2c_high_performance")

        return model

    def train_ppo_ultra_performance(self):
        """Train PPO with ultra-aggressive parameters and curriculum learning"""
        print("Training Ultra-Performance PPO with curriculum learning...")

        log_dir_ppo = f"{self.log_dir}/ppo_ultra_performance"
        os.makedirs(f"{log_dir_ppo}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir_ppo}/eval", exist_ok=True)

        # Import torch for policy kwargs
        try:
            import torch
        except ImportError:
            print("PyTorch not available, using default policy")
            torch = None

        # Stage 1: Very easy environment for initial learning
        print("Stage 1: Very easy environment training...")
        easy_env = Monitor(RecyclingSortingEnv(
            bin_capacity=20,     # Larger bins for easier learning
            max_wrong_sorts=12,  # More forgiving error limit
            confidence_discard_threshold=0.7,  # Very high threshold
            confidence_noise_std=0.01,  # Almost no noise
            item_timeout=200     # Very long timeout
        ))

        model = PPO(
            "MlpPolicy",
            easy_env,
            verbose=1,
            tensorboard_log=f"{log_dir_ppo}/tensorboard",
            # Ultra-aggressive parameters for maximum learning
            learning_rate=2e-2,  # Extremely high learning rate
            n_steps=64,          # Very small rollout
            batch_size=4,        # Very small batch
            n_epochs=100,        # Many epochs for thorough learning
            gamma=0.9,           # Low gamma for immediate rewards
            gae_lambda=0.99,
            clip_range=0.8,      # Very large clip range
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.2,        # Very high entropy for exploration
            vf_coef=5.0,         # Very high value coefficient
            max_grad_norm=20.0,  # High gradient norm
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=0.5,       # Very high KL tolerance
            
            policy_kwargs=dict(
                net_arch=[32, 32] if torch else [16, 16],
                activation_fn=torch.nn.Tanh if torch else "tanh",
                normalize_images=False
            ),
            device="auto"
        )

        # Train on easy environment
        model.learn(total_timesteps=self.total_timesteps // 3)

        # Stage 2: Normal environment
        print("Stage 2: Normal environment training...")
        model.set_env(self.env)
        model.learning_rate = 1e-2  # Still high learning rate
        model.ent_coef = 0.1        # Still high entropy
        model.learn(total_timesteps=self.total_timesteps // 3)

        # Stage 3: Slightly harder environment (but not too hard)
        print("Stage 3: Slightly harder environment training...")
        hard_env = Monitor(RecyclingSortingEnv(
            bin_capacity=15,         # Standard bin capacity
            max_wrong_sorts=8,       # Standard error limit
            confidence_discard_threshold=0.5,  # Still reasonable
            confidence_noise_std=0.05,  # Moderate noise
            item_timeout=100         # Still reasonable timeout
        ))
        
        model.set_env(hard_env)
        model.learning_rate = 5e-3  # Reduce learning rate but keep it high
        model.ent_coef = 0.05       # Reduce entropy but keep exploration
        model.learn(total_timesteps=self.total_timesteps // 3)

        # Clean up
        easy_env.close()
        hard_env.close()

        # Save ultra-performance model
        model.save(f"{log_dir_ppo}/ppo_ultra_performance")
        print(f"Ultra-Performance PPO training completed. Model saved to {log_dir_ppo}/ppo_ultra_performance")

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
        discards = []
        steps_per_episode = []

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

        # Calculate statistics
        avg_reward = np.mean(total_rewards)
        avg_efficiency = np.mean(efficiencies)
        avg_correct = np.mean(correct_sorts)
        avg_wrong = np.mean(wrong_sorts)
        avg_discards = np.mean(discards)
        avg_steps = np.mean(steps_per_episode)

        print(f"{model_type} Evaluation Results ({num_episodes} episodes):")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Efficiency: {avg_efficiency:.2%}")
        print(f"Average Correct Sorts: {avg_correct:.2f}")
        print(f"Average Wrong Sorts: {avg_wrong:.2f}")
        print(f"Average Discards: {avg_discards:.2f}")
        print(f"Average Steps per Episode: {avg_steps:.2f}")

        return {
            'avg_reward': avg_reward,
            'avg_efficiency': avg_efficiency,
            'avg_correct': avg_correct,
            'avg_wrong': avg_wrong,
            'avg_discards': avg_discards,
            'avg_steps': avg_steps,
            'rewards': total_rewards,
            'efficiencies': efficiencies,
            'steps_per_episode': steps_per_episode
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

    def train_and_compare_all_models(self):
        """Train all PG variants and compare their performance"""
        print(" Training and Comparing All Policy Gradient Variants")
        print("=" * 70)
        
        models_to_train = [
            ("PPO Optimized", self.train_ppo_optimized),
            ("PPO High-Performance", self.train_ppo_high_performance),
            ("PPO Ultra-Performance", self.train_ppo_ultra_performance),
            ("A2C Optimized", self.train_a2c_optimized),
            ("A2C High-Performance", self.train_a2c_high_performance)
        ]
        
        results = {}
        
        for model_name, train_method in models_to_train:
            print(f"\n Training {model_name}...")
            try:
                model = train_method()
                
                # Determine model path based on algorithm type
                if "PPO" in model_name:
                    if "Ultra-Performance" in model_name:
                        model_path = f"{self.log_dir}/ppo_ultra_performance/ppo_ultra_performance"
                        model_type = "PPO"
                    elif "High-Performance" in model_name:
                        model_path = f"{self.log_dir}/ppo_high_performance/ppo_high_performance"
                        model_type = "PPO"
                    elif "Optimized" in model_name:
                        model_path = f"{self.log_dir}/ppo_optimized/ppo_optimized"
                        model_type = "PPO"
                    else:
                        model_path = f"{self.log_dir}/ppo/ppo_default"
                        model_type = "PPO"
                elif "A2C" in model_name:
                    if "High-Performance" in model_name:
                        model_path = f"{self.log_dir}/a2c_high_performance/a2c_high_performance"
                        model_type = "A2C"
                    elif "Optimized" in model_name:
                        model_path = f"{self.log_dir}/a2c_optimized/a2c_optimized"
                        model_type = "A2C"
                    else:
                        model_path = f"{self.log_dir}/a2c/a2c_default"
                        model_type = "A2C"
                else:  # REINFORCE
                    model_path = f"{self.log_dir}/reinforce/reinforce"
                    model_type = "PPO"
                
                # Evaluate the model
                results[model_name] = self.evaluate_model(model_path, model_type, num_episodes=30)
                
                print(f"✅ {model_name} completed successfully")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                continue
        
        if len(results) > 1:
            # Find best models
            best_model = max(results.keys(), key=lambda x: results[x]['avg_reward'])
            best_efficiency = max(results.keys(), key=lambda x: results[x]['avg_efficiency'])
            
            print(f"\n POLICY GRADIENT PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f" Best Average Reward: {best_model} ({results[best_model]['avg_reward']:.2f})")
            print(f" Best Efficiency: {best_efficiency} ({results[best_efficiency]['avg_efficiency']:.2%})")
            
            # Algorithm comparison
            ppo_models = {k: v for k, v in results.items() if "PPO" in k}
            a2c_models = {k: v for k, v in results.items() if "A2C" in k}
            
            if ppo_models:
                best_ppo = max(ppo_models.keys(), key=lambda x: ppo_models[x]['avg_reward'])
                print(f"🔵 Best PPO: {best_ppo} ({ppo_models[best_ppo]['avg_reward']:.2f})")
            
            if a2c_models:
                best_a2c = max(a2c_models.keys(), key=lambda x: a2c_models[x]['avg_reward'])
                print(f"🟢 Best A2C: {best_a2c} ({a2c_models[best_a2c]['avg_reward']:.2f})")
            
            # Detailed recommendations
            print(f"\n POLICY GRADIENT RECOMMENDATIONS")
            print("=" * 60)
            
            if results[best_model]['avg_reward'] > 150:
                print("✅ Excellent performance achieved!")
                print("   - Policy gradient methods are working well for this task")
                print("   - Consider fine-tuning hyperparameters for even better results")
            elif results[best_model]['avg_reward'] > 100:
                print("✅ Good performance, room for improvement:")
                print("   - Try longer training (increase total_timesteps)")
                print("   - Experiment with different network architectures")
                print("   - Consider curriculum learning approaches")
            else:
                print("⚠️  Performance needs improvement:")
                print("   - Policy gradient methods may need significant tuning")
                print("   - Consider using DQN for comparison")
                print("   - Adjust reward structure or environment parameters")
            
            # Algorithm-specific advice
            if ppo_models and a2c_models:
                best_ppo_reward = max(ppo_models.values(), key=lambda x: x['avg_reward'])['avg_reward']
                best_a2c_reward = max(a2c_models.values(), key=lambda x: x['avg_reward'])['avg_reward']
                
                if best_ppo_reward > best_a2c_reward + 10:
                    print("   - PPO performs better than A2C for this task")
                    print("   - Focus on PPO variants for future training")
                elif best_a2c_reward > best_ppo_reward + 10:
                    print("   - A2C performs better than PPO for this task")
                    print("   - Focus on A2C variants for future training")
                else:
                    print("   - PPO and A2C perform similarly")
                    print("   - Both algorithms are viable for this task")
            
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
        print(f"{'Model':<20} {'Avg Reward':<12} {'Efficiency':<12} {'Correct':<10} {'Wrong':<10} {'Discards':<10} {'Steps':<10}")
        print("-" * 90)
        print(f"{'PPO Default':<20} {ppo_default_results['avg_reward']:<12.2f} {ppo_default_results['avg_efficiency']:<12.2%} {ppo_default_results['avg_correct']:<10.2f} {ppo_default_results['avg_wrong']:<10.2f} {ppo_default_results['avg_discards']:<10.2f} {ppo_default_results['avg_steps']:<10.2f}")
        print(f"{'PPO Optimized':<20} {ppo_optimized_results['avg_reward']:<12.2f} {ppo_optimized_results['avg_efficiency']:<12.2%} {ppo_optimized_results['avg_correct']:<10.2f} {ppo_optimized_results['avg_wrong']:<10.2f} {ppo_optimized_results['avg_discards']:<10.2f} {ppo_optimized_results['avg_steps']:<10.2f}")
        print(f"{'A2C Default':<20} {a2c_default_results['avg_reward']:<12.2f} {a2c_default_results['avg_efficiency']:<12.2%} {a2c_default_results['avg_correct']:<10.2f} {a2c_default_results['avg_wrong']:<10.2f} {a2c_default_results['avg_discards']:<10.2f} {a2c_default_results['avg_steps']:<10.2f}")
        print(f"{'A2C Optimized':<20} {a2c_optimized_results['avg_reward']:<12.2f} {a2c_optimized_results['avg_efficiency']:<12.2%} {a2c_optimized_results['avg_correct']:<10.2f} {a2c_optimized_results['avg_wrong']:<10.2f} {a2c_optimized_results['avg_discards']:<10.2f} {a2c_optimized_results['avg_steps']:<10.2f}")
        print(f"{'REINFORCE':<20} {reinforce_results['avg_reward']:<12.2f} {reinforce_results['avg_efficiency']:<12.2%} {reinforce_results['avg_correct']:<10.2f} {reinforce_results['avg_wrong']:<10.2f} {reinforce_results['avg_discards']:<10.2f} {reinforce_results['avg_steps']:<10.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        trainer.close()