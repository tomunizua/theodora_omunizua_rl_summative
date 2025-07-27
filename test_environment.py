#!/usr/bin/env python3
"""
Test script for Recycling Sorting Environment
Verifies basic functionality and environment setup
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environment.custom_env import RecyclingSortingEnv

def test_environment_basics():
    """Test basic environment functionality"""
    print("Testing basic environment functionality...")
    
    # Create environment
    env = RecyclingSortingEnv(batch_size=10, conveyor_length=5)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test action space
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few steps
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: Action={action}, Reward={reward:.1f}, Done={done}")
        
        if done:
            print(f"Episode completed after {step} steps")
            print(f"Final info: {info}")
            break
    
    print(f"Total reward: {total_reward:.1f}")
    env.close()
    print("Basic environment test completed!\n")

def test_environment_consistency():
    """Test environment consistency and reward structure"""
    print("Testing environment consistency...")
    
    env = RecyclingSortingEnv(batch_size=5, conveyor_length=3)
    
    # Test with deterministic actions
    obs, info = env.reset()
    print(f"Starting item type: {obs[0]}")
    
    # Wait until item is at sorting position
    while obs[2] < 2:  # item_position < conveyor_length - 1
        obs, reward, done, truncated, info = env.step(5)  # Wait action
        print(f"Waiting... Position: {obs[2]}, Reward: {reward}")
    
    # Now test sorting actions
    item_type = obs[0]
    correct_action = env.correct_bins[item_type]
    
    print(f"Item type: {item_type}, Correct action: {correct_action}")
    
    # Test correct action
    obs, reward, done, truncated, info = env.step(correct_action)
    print(f"Correct action reward: {reward}")
    
    # Test wrong action (if possible)
    wrong_action = (correct_action + 1) % 5
    obs, reward, done, truncated, info = env.step(wrong_action)
    print(f"Wrong action reward: {reward}")
    
    env.close()
    print("Environment consistency test completed!\n")

def test_observation_space():
    """Test observation space bounds"""
    print("Testing observation space bounds...")
    
    env = RecyclingSortingEnv(batch_size=10, conveyor_length=5)
    
    # Run multiple episodes to check observation bounds
    for episode in range(5):
        obs, info = env.reset()
        
        # Check initial observation bounds
        assert 0 <= obs[0] <= 4, f"Item type out of bounds: {obs[0]}"
        assert 0 <= obs[1] <= 2, f"Speed out of bounds: {obs[1]}"
        assert 0 <= obs[2] <= 4, f"Position out of bounds: {obs[2]}"
        assert 0 <= obs[3] <= 10, f"Time remaining out of bounds: {obs[3]}"
        
        step = 0
        while step < 50:  # Limit steps
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            # Check observation bounds
            assert 0 <= obs[0] <= 4, f"Item type out of bounds: {obs[0]}"
            assert 0 <= obs[1] <= 2, f"Speed out of bounds: {obs[1]}"
            assert 0 <= obs[2] <= 4, f"Position out of bounds: {obs[2]}"
            assert 0 <= obs[3] <= 10, f"Time remaining out of bounds: {obs[3]}"
            
            step += 1
            if done:
                break
    
    env.close()
    print("Observation space bounds test completed!\n")

def test_reward_structure():
    """Test reward structure"""
    print("Testing reward structure...")
    
    env = RecyclingSortingEnv(batch_size=5, conveyor_length=3)
    
    rewards = []
    
    for episode in range(10):
        obs, info = env.reset()
        episode_reward = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
    
    env.close()
    
    print(f"Reward statistics over {len(rewards)} episodes:")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Std reward: {np.std(rewards):.2f}")
    print(f"  Min reward: {np.min(rewards):.2f}")
    print(f"  Max reward: {np.max(rewards):.2f}")
    print("Reward structure test completed!\n")

def main():
    """Run all tests"""
    print("=== Recycling Sorting Environment Tests ===\n")
    
    try:
        test_environment_basics()
        test_environment_consistency()
        test_observation_space()
        test_reward_structure()
        
        print("=== All Tests Passed! ===")
        print("The environment is working correctly.")
        print("You can now run the training scripts.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 