#!/usr/bin/env python3
"""
Random Agent Demo for Recycling Sorting Environment
Demonstrates the environment with random actions (no training)
"""

import pygame
import numpy as np
import random
import time
import sys
import os
sys.path.append('.')

from environment.custom_env import RecyclingSortingEnv
from environment.rendering import RecyclingSortingRenderer

def run_random_agent_demo():
    """Run a demo of the environment with a random agent and smooth conveyor animation"""
    print("Starting Random Agent Demo...")
    print("Environment: Fresh Recycling Sorting with Confidence Vectors")
    print("Actions: 0=Paper, 1=Plastic, 2=Organic, 3=Metal, 4=Discard")
    print("Rewards: +10 correct, -10 incorrect, -3 wrong, -0.1 step penalty, discard only rewarded if confidence is low")
    print("Press 'q' to quit, 'r' to reset, 'space' to pause/resume")
    
    # Initialize environment and renderer
    env = RecyclingSortingEnv(batch_size=20)
    renderer = RecyclingSortingRenderer()
    
    # Demo parameters
    max_episodes = 3
    current_episode = 0
    paused = False
    
    try:
        while current_episode < max_episodes:
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            
            print(f"\n=== Episode {current_episode + 1}/{max_episodes} ===")
            print(f"Initial confidence vector: {obs}")
            print(f"True class: {info['true_class']} ({info['bin_names'][info['true_class']]})")
            
            done = False
            while not done:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            return
                        elif event.key == pygame.K_r:
                            print("Resetting episode...")
                            break
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            print("Paused" if paused else "Resumed")
                if paused:
                    time.sleep(0.1)
                    continue
                # Take random action
                action = random.randint(0, 4)  # 0-4: Paper, Plastic, Organic, Metal, Discard
                action_names = ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard']
                # Animate item moving across the conveyor before stepping
                renderer.item_progress = 0.0
                renderer.item_moving = True
                while renderer.item_moving:
                    renderer.render(info, action, None)
                    time.sleep(0.02)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                return
                            elif event.key == pygame.K_r:
                                print("Resetting episode...")
                                break
                            elif event.key == pygame.K_SPACE:
                                paused = not paused
                                print("Paused" if paused else "Resumed")
                    if paused:
                        time.sleep(0.1)
                # Now take the step in the environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                print(f"Step {step_count}: Action={action_names[action]}, "
                      f"Reward={reward:.1f}, True={info['bin_names'][info['true_class']]}, "
                      f"Confidence={obs}")
                # Show the result for a short moment
                renderer.render(info, action, reward)
                time.sleep(0.15)
            print(f"\nEpisode {current_episode + 1} completed!")
            print(f"Total Reward: {episode_reward:.1f}")
            print(f"Correct Sorts: {info['correct_sorts']}")
            print(f"Wrong Sorts: {info['wrong_sorts']}")
            print(f"Discarded: {info['discarded']}")
            print(f"Efficiency: {info['efficiency']:.2%}")
            current_episode += 1
            time.sleep(1)
        print("\nDemo completed! Press any key to exit...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        env.close()
        renderer.close()

if __name__ == "__main__":
    run_random_agent_demo() 