#!/usr/bin/env python3
"""
Create Demo GIF for Random Agent
Generates a GIF showing random agent behavior for the report
"""

import sys
import time
import pygame
import numpy as np
import os
from pathlib import Path
import imageio

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environment.custom_env import RecyclingSortingEnv
from environment.rendering import RecyclingSortingRenderer

def capture_screen(surface):
    """Capture pygame surface as numpy array"""
    # Get the surface data
    view = pygame.surfarray.array3d(surface)
    # Convert from (width, height, channel) to (height, width, channel)
    view = view.transpose([1, 0, 2])
    return view

def create_demo_gif():
    """Create a GIF demo of random agent behavior"""
    print("Creating demo GIF...")
    
    # Create directories
    os.makedirs("results", exist_ok=True)
    
    # Initialize environment and renderer
    env = RecyclingSortingEnv(batch_size=15, conveyor_length=8)
    renderer = RecyclingSortingRenderer()
    
    # GIF parameters
    frames = []
    max_frames = 300  # 10 seconds at 30 fps
    frame_count = 0
    
    obs, info = env.reset()
    done = False
    
    print("Capturing frames...")
    
    while not done and frame_count < max_frames:
        # Take random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Render
        renderer.render(info, action, reward)
        
        # Capture frame
        frame = capture_screen(renderer.screen)
        frames.append(frame)
        
        frame_count += 1
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Small delay to slow down the demo
        time.sleep(0.05)
    
    renderer.close()
    env.close()
    
    # Save GIF
    print(f"Saving GIF with {len(frames)} frames...")
    imageio.mimsave('results/random_agent_demo.gif', frames, fps=10)
    print("GIF saved to results/random_agent_demo.gif")
    
    return len(frames)

def create_trained_agent_gif():
    """Create a GIF demo of trained agent behavior"""
    print("Creating trained agent GIF...")
    
    try:
        from stable_baselines3 import PPO
        
        # Load best model (assuming PPO optimized performed well)
        model_path = "models/pg/ppo_optimized/ppo_optimized"
        if not os.path.exists(model_path + ".zip"):
            print(f"Model not found at {model_path}. Please run training first.")
            return
        
        model = PPO.load(model_path)
        
        # Initialize environment and renderer
        env = RecyclingSortingEnv(batch_size=15, conveyor_length=8)
        renderer = RecyclingSortingRenderer()
        
        # GIF parameters
        frames = []
        max_frames = 300
        frame_count = 0
        
        obs, info = env.reset()
        done = False
        
        print("Capturing trained agent frames...")
        
        while not done and frame_count < max_frames:
            # Take trained action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Render
            renderer.render(info, action, reward)
            
            # Capture frame
            frame = capture_screen(renderer.screen)
            frames.append(frame)
            
            frame_count += 1
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            # Small delay
            time.sleep(0.05)
        
        renderer.close()
        env.close()
        
        # Save GIF
        print(f"Saving trained agent GIF with {len(frames)} frames...")
        imageio.mimsave('results/trained_agent_demo.gif', frames, fps=10)
        print("Trained agent GIF saved to results/trained_agent_demo.gif")
        
        return len(frames)
        
    except Exception as e:
        print(f"Error creating trained agent GIF: {e}")
        print("Make sure to run training first.")

def main():
    """Main function"""
    import os
    
    print("=== Creating Demo GIFs ===")
    
    # Create random agent GIF
    random_frames = create_demo_gif()
    
    # Try to create trained agent GIF
    trained_frames = create_trained_agent_gif()
    
    print("\n=== GIF Creation Complete ===")
    print(f"Random agent GIF: {random_frames} frames")
    if trained_frames:
        print(f"Trained agent GIF: {trained_frames} frames")
    else:
        print("Trained agent GIF: Not created (run training first)")
    
    print("\nGIFs saved to results/ directory")
    print("You can now include these in your report!")

if __name__ == "__main__":
    main() 