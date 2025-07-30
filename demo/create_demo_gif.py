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
    env = RecyclingSortingEnv(
        bin_capacity=15,
        max_wrong_sorts=8,
        item_timeout=100
    )
    renderer = RecyclingSortingRenderer()
    
    # GIF parameters
    frames = []
    max_frames = 300  # 10 seconds at 30 fps
    frame_count = 0
    
    obs, info = env.reset()
    done = False
    
    print("Capturing frames...")
    
    step_count = 0
    action_names = ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard', 'Scan']
    
    while not done and frame_count < max_frames:
        # Take strategic random action (similar to demo)
        if np.random.random() < 0.15:  # 15% chance to scan
            action = 5  # Scan
        else:
            action = np.random.randint(0, 5)  # 0-4: sorting actions
        
        print(f"Step {step_count + 1}: Action={action_names[action]}, "
              f"True={info.get('bin_names', ['Paper', 'Plastic', 'Organic', 'Metal'])[info.get('true_class', 0)]}")
        
        # Animate item movement along conveyor belt
        movement_frames = 15
        for i in range(movement_frames):
            # Update item progress from 0.0 to 1.0
            renderer.item_progress = i / (movement_frames - 1)
            renderer.item_moving = True
            
            # Render with current progress
            renderer.render(info, action, None)
            
            # Capture frame during movement
            frame = capture_screen(renderer.screen)
            frames.append(frame)
            frame_count += 1
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
            
            if done or frame_count >= max_frames:
                break
                
            time.sleep(0.05)  # Smooth animation timing
        
        if done or frame_count >= max_frames:
            break
        
        # Take the actual step in environment
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        # Reset item position for next item (if episode continues)
        if not done:
            renderer.item_progress = 0.0
            renderer.item_moving = False
        
        # Render result with action feedback
        renderer.render(info, action, reward)
        
        # Capture result frame (hold for a moment to show action result)
        for _ in range(3):  # Hold result for 3 frames
            frame = capture_screen(renderer.screen)
            frames.append(frame)
            frame_count += 1
            
            if frame_count >= max_frames:
                break
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Brief pause between items
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
        env = RecyclingSortingEnv(
            bin_capacity=15,
            max_wrong_sorts=8,
            item_timeout=100
        )
        renderer = RecyclingSortingRenderer()
        
        # GIF parameters
        frames = []
        max_frames = 300
        frame_count = 0
        
        obs, info = env.reset()
        done = False
        
        print("Capturing trained agent frames...")
        
        step_count = 0
        action_names = ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard', 'Scan']
        
        while not done and frame_count < max_frames:
            # Take trained action
            action, _ = model.predict(obs, deterministic=True)
            
            print(f"Trained Step {step_count + 1}: Action={action_names[action]}, "
                  f"True={info.get('bin_names', ['Paper', 'Plastic', 'Organic', 'Metal'])[info.get('true_class', 0)]}")
            
            # Animate item movement along conveyor belt
            movement_frames = 12  # Slightly fewer frames for trained agent
            for i in range(movement_frames):
                # Update item progress from 0.0 to 1.0
                renderer.item_progress = i / (movement_frames - 1)
                renderer.item_moving = True
                
                # Render with current progress
                renderer.render(info, action, None)
                
                # Capture frame during movement
                frame = capture_screen(renderer.screen)
                frames.append(frame)
                frame_count += 1
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                
                if done or frame_count >= max_frames:
                    break
                    
                time.sleep(0.04)  # Smooth animation timing
            
            if done or frame_count >= max_frames:
                break
            
            # Take the actual step in environment
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Reset item position for next item (if episode continues)
            if not done:
                renderer.item_progress = 0.0
                renderer.item_moving = False
            
            # Render result with action feedback
            renderer.render(info, action, reward)
            
            # Capture result frame (hold for a moment to show action result)
            for _ in range(2):  # Hold result for 2 frames (faster for trained agent)
                frame = capture_screen(renderer.screen)
                frames.append(frame)
                frame_count += 1
                
                if frame_count >= max_frames:
                    break
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            # Brief pause between items
            time.sleep(0.04)  # Slightly faster for trained agent
        
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
    print(f"📹 Random agent GIF: {random_frames} frames")
    if trained_frames:
        print(f"🎯 Trained agent GIF: {trained_frames} frames")
    else:
        print("❌ Trained agent GIF: Not created (run training first)")
    
    print("\n📁 Files created:")
    print("   • results/random_agent_demo.gif - Shows random agent behavior")
    if trained_frames:
        print("   • results/trained_agent_demo.gif - Shows trained agent behavior")
    
    print("\n🎨 GIF Features:")
    print("   • 6-action support (including Scan)")
    print("   • Smooth item movement animation")
    print("   • Visual urgency indicators")
    print("   • Enhanced statistics display")
    print("   • Strategic action logging")
    
    print("\n✅ Ready for your report and presentations!")

if __name__ == "__main__":
    main() 