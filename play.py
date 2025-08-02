#!/usr/bin/env python3
"""
Model Testing Script
Load and run a single trained model with the recycling sorting environment
"""

import os
import sys
import time
import pygame
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import RecyclingSortingEnv
from environment.rendering import RecyclingSortingRenderer

# ==============================================================================
# ModelPlayer Class
# This class handles model loading and the simulation loop.
# ==============================================================================
class ModelPlayer:
    """Simple model testing interface"""
    
    def __init__(self):
        # 🎯 MODEL CONFIGURATION - CHANGE THESE TO TEST DIFFERENT MODELS
        self.model_path = 'models/pg/ppo/ppo_final.zip'  # ← CHANGE THIS PATH
        self.model_type = 'PPO'  # ← CHANGE THIS TYPE: 'DQN', 'PPO', 'A2C'
        
        # Action and bin names for display
        self.action_names = ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard', 'Missed']
        self.bin_names = ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard']
        
    def load_model(self):
        """Load the configured model"""
        if not os.path.exists(self.model_path):
            print(f" Model file not found: {self.model_path}")
            print("   Please run training first or check the path.")
            return None
        
        try:
            if self.model_type == 'DQN':
                from stable_baselines3 import DQN
                model = DQN.load(self.model_path)
            elif self.model_type == 'PPO':
                from stable_baselines3 import PPO
                model = PPO.load(self.model_path)
            elif self.model_type == 'A2C':
                from stable_baselines3 import A2C
                model = A2C.load(self.model_path)
            else:
                print(f" Unknown model type: {self.model_type}")
                return None
            
            print(f" Loaded {self.model_type} model from: {self.model_path}")
            return model
            
        except Exception as e:
            print(f" Failed to load model: {e}")
            return None
    
    def print_model_info(self):
        """Print current model information"""
        print(f"🤖 Testing Model: {self.model_type}")
        print(f"📁 Path: {self.model_path}")
        print()
    
    def play_episode(self, model, episodes=3):
        """
        Plays a fixed number of episodes using the loaded model.
        The game loop is simplified to step through the environment
        and render the state at each step.
        """
        print(f"🎮 Running {episodes} episodes...")
        print()
        
        # Temporarily suppress environment print statements
        import builtins
        original_print = builtins.print
        def quiet_print(*args, **kwargs):
            # Only suppress termination messages
            if args and isinstance(args[0], str) and ("Episode terminated" in args[0] or "🗂️" in args[0] or "❌" in args[0]):
                return
            original_print(*args, **kwargs)
        builtins.print = quiet_print
        
        # Initialize environment and renderer
        env = RecyclingSortingEnv(
            bin_capacity=15,
            max_wrong_sorts=8,
            item_timeout=80
        )
        renderer = RecyclingSortingRenderer()
        
        episode_rewards = []
        
        try:
            # Main loop for all episodes
            for episode in range(episodes):
                obs, info = env.reset()
                episode_reward = 0.0
                done = False
                
                # State variables for rendering
                last_action: Optional[int] = None
                last_reward: Optional[float] = None
                
                # This loop runs for the duration of a single episode
                while not done:
                    # Handle Pygame events to allow closing the window
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            # End the main loop as well
                            episode = episodes
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                done = True
                                episode = episodes
                                break

                    if done:
                        break
                    
                    # Determine item progress based on the environment's internal timer
                    timeout_remaining = info.get('item_timeout_remaining', env.item_timeout)
                    progress = 1.0 - (timeout_remaining / env.item_timeout)
                    # Let the renderer handle its own progress, don't override it
                    # renderer.item_progress = progress

                    # Determine what action to take
                    if progress >= 0.5 and timeout_remaining > 0:
                        # Model makes a decision when the item is halfway across
                        action, _ = model.predict(obs, deterministic=True)
                    elif timeout_remaining <= 0:
                        # Item timed out - missed sort
                        action = 5  # Missed sort action
                    else:
                        # Item is still moving - use a "wait/scan" action to advance timer
                        # This advances the environment timer without making a sorting decision
                        action = 5  # Use missed sort as the "wait" action
                    
                    # Always take a step to advance the environment timer
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    # Only update display variables for actual sorting decisions
                    if progress >= 0.5 or timeout_remaining <= 0:
                        last_action = action
                        last_reward = reward
                        
                        # Show the sorting result
                        renderer.render(info, action=last_action, reward=last_reward)
                        pygame.display.flip()
                        time.sleep(0.5)  # Pause to show the result
                        
                        # Reset for next item (new item will appear after this action)
                        # Let the renderer handle its own state
                        # renderer.item_progress = 0.0
                        # renderer.item_moving = True
                    else:
                        # Just render the moving item without action feedback
                        renderer.render(info, action=None, reward=None)

                    # Update done flag
                    done = terminated or truncated
                    
                    # Small delay to keep the animation at a reasonable pace
                    time.sleep(0.015)
                
                # Store episode reward and show why episode ended
                episode_rewards.append(episode_reward)
                original_print(f"Episode {episode + 1}: {episode_reward:.1f} (Ended: {info.get('termination_reason', 'Unknown')})")
                
                # Small delay between episodes to make output readable
                if episode < episodes - 1:
                    time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n⏹   Stopped by user")
        
        finally:
            # Restore original print function
            builtins.print = original_print
            renderer.close()
            env.close()
        
        # Final summary
        if episode_rewards:
            cumulative_reward = sum(episode_rewards)
            print()
            print("📊 Results:")
            print(f"  Cumulative Reward: {cumulative_reward:.1f}")
            print(f"  Average Reward: {cumulative_reward/len(episode_rewards):.1f}")
    
    def run_model_demo(self):
        """Run model demonstration"""
        self.print_model_info()
        model = self.load_model()
        if model is None:
            print(" Failed to load model. Please check the model path and run training if needed.")
            return
        
        self.play_episode(model, episodes=3)
        
def main():
    """Main function"""
    print("🚀 Recycling Sorting Agent - Model Testing")
    print("=" * 50)
    player = ModelPlayer()
    player.run_model_demo()

if __name__ == "__main__":
    main()
