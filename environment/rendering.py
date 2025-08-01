import pygame
import numpy as np
from typing import Dict, Any, Optional
import math
import random

class RecyclingSortingRenderer:
    """
    Pygame-based renderer for the Recycling Sorting Environment
    - Items move smoothly along the conveyor belt
    - Gradient and shadow visuals
    - Cooler color tones
    """
    
    def __init__(self, screen_width=900, screen_height=600):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Recycling Sorting Agent - RL Environment")
        
        # Cooler, softer colors
        self.WHITE = (248, 250, 252)
        self.BLACK = (30, 41, 59)
        self.GRAY = (100, 116, 139)
        self.LIGHT_GRAY = (203, 213, 225)
        self.DARK_GRAY = (51, 65, 85)
        
        # Cooler tones - blues, teals, purples, muted colors
        self.SOFT_BLUE = (59, 130, 246)       # Paper - soft blue
        self.TEAL = (20, 184, 166)            # Plastic - teal
        self.SAGE_GREEN = (34, 197, 94)       # Organic - sage green
        self.SLATE = (71, 85, 105)            # Metal - slate
        self.MUTED_RED = (239, 68, 68)        # Discard - muted red
        
        # Additional cooler tones
        self.LAVENDER = (139, 92, 246)
        self.MINT = (16, 185, 129)
        self.STEEL_BLUE = (30, 64, 175)
        
        # Item colors for each class (cooler tones)
        self.item_colors = {
            0: self.SOFT_BLUE,   # Paper
            1: self.TEAL,        # Plastic
            2: self.SAGE_GREEN,  # Organic
            3: self.SLATE,       # Metal
            4: self.MUTED_RED    # Discard
        }
        
        # Bin colors (cooler tones)
        self.bin_colors = {
            0: self.SOFT_BLUE,   # Paper
            1: self.TEAL,        # Plastic
            2: self.SAGE_GREEN,  # Organic
            3: self.SLATE,       # Metal
            4: self.MUTED_RED    # Discard
        }
        
        # Fonts (smaller)
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)
        
        # Conveyor belt properties
        self.conveyor_start_x = 60
        self.conveyor_end_x = 600
        self.conveyor_y = 250
        self.conveyor_width = 500
        self.conveyor_height = 36
        
        # Bin properties
        self.bin_width = 48
        self.bin_height = 70
        self.bin_spacing = 60
        self.n_bins = 5  # 4 material bins + 1 discard
        
        # For smooth item movement (slowed down)
        self.item_progress = 0.0  # 0.0 (left) to 1.0 (right)
        self.item_moving = True
        self.item_speed = 0.015  # Slower movement for better observation
        
        # Time-limited conveyor belt (extended for slower pace)
        self.item_timeout = 80  # More frames for slower environment
        
    def render(self, env_info: Dict[str, Any], action: Optional[int] = None, reward: Optional[float] = None):
        
        self.screen.fill(self.WHITE)
        
        # Draw background
        self._draw_background()
        
        # Draw conveyor belt
        self._draw_conveyor_belt(env_info)
        
        # Draw bins and bin counters
        self._draw_bins(env_info)
        
        # Draw moving item
        self._draw_moving_item(env_info)
        
        # Update item progress (after drawing, so it's ready for next frame)
        self._update_item_progress(env_info, action)
        
        # Draw confidence vector
        self._draw_confidence_vector(env_info)
        
        # Draw statistics
        self._draw_statistics(env_info)
        
        # Draw action feedback
        if action is not None:
            self._draw_action_feedback(action, reward)
        
        # Draw progress bar
        self._draw_progress_bar(env_info)
        
        pygame.display.flip()
        
    def _draw_background(self):
        for x in range(0, self.screen_width, 30):
            pygame.draw.line(self.screen, self.LIGHT_GRAY, (x, 0), (x, self.screen_height), 1)
        for y in range(0, self.screen_height, 30):
            pygame.draw.line(self.screen, self.LIGHT_GRAY, (0, y), (self.screen_width, y), 1)
        title = self.font_large.render("Recycling Sorting Agent", True, self.BLACK)
        self.screen.blit(title, (self.screen_width // 2 - title.get_width() // 2, 10))
    
    def _draw_conveyor_belt(self, env_info: Dict[str, Any]):
        pygame.draw.rect(self.screen, self.GRAY, 
                         (self.conveyor_start_x, self.conveyor_y, self.conveyor_width, self.conveyor_height))
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, 
                         (self.conveyor_start_x, self.conveyor_y + 3, self.conveyor_width, self.conveyor_height - 6))
        # Moving belt pattern
        offset = (pygame.time.get_ticks() // 80) % 30
        for i in range(0, self.conveyor_width, 30):
            x = self.conveyor_start_x + i + offset
            if x < self.conveyor_start_x + self.conveyor_width:
                pygame.draw.line(self.screen, self.GRAY, 
                                 (x, self.conveyor_y + 6), (x, self.conveyor_y + self.conveyor_height - 6), 2)
        pygame.draw.rect(self.screen, self.BLACK, 
                         (self.conveyor_start_x, self.conveyor_y, self.conveyor_width, self.conveyor_height), 2)
        # Conveyor belt label
        label = self.font_medium.render("Conveyor Belt", True, self.BLACK)
        self.screen.blit(label, (self.conveyor_start_x + self.conveyor_width // 2 - label.get_width() // 2, self.conveyor_y + self.conveyor_height + 10))
    
    def _draw_bins(self, env_info: Dict[str, Any]):
        bin_labels = ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard']
        bin_x_start = 600
        bin_y = 200
        bin_counters = env_info.get('bin_counters', [0, 0, 0, 0, 0])
        
        for i in range(self.n_bins):
            x = bin_x_start + i * self.bin_spacing
            
            bin_type = i  # Fixed order: 0=Paper, 1=Plastic, 2=Organic, 3=Metal, 4=Discard

            color = self.bin_colors[bin_type]
            label_text = bin_labels[bin_type]
            counter_value = bin_counters[i] if i < len(bin_counters) else 0
            
            # Draw shadow
            shadow_rect = pygame.Rect(x + 3, bin_y + 3, self.bin_width, self.bin_height)
            self._draw_bin_shadow(shadow_rect)
            
            # Draw bin with gradient
            bin_rect = pygame.Rect(x, bin_y, self.bin_width, self.bin_height)
            self._draw_bin_gradient(bin_rect, color)
            
            # Bin outline
            pygame.draw.rect(self.screen, self.DARK_GRAY, (x, bin_y, self.bin_width, self.bin_height), 2)
            
            # Bin label
            label = self.font_medium.render(label_text, True, self.BLACK)
            self.screen.blit(label, (x + self.bin_width // 2 - label.get_width() // 2, bin_y - 18))
            
            # Bin counter above
            counter = self.font_large.render(str(counter_value), True, self.BLACK)
            self.screen.blit(counter, (x + self.bin_width // 2 - counter.get_width() // 2, bin_y - 40))
    
    def _draw_moving_item(self, env_info: Dict[str, Any]):
        # Animate item from left to right
        true_class = env_info.get('true_class', 0)
        true_class = max(0, min(true_class, 4))  # Avoid out-of-bounds
        
        # Check if item should be visible based on environment timeout
        timeout_remaining = env_info.get('item_timeout_remaining', self.item_timeout)
        
        # Only draw if there's time remaining for the item
        if timeout_remaining <= 0:
            # If item has timed out, ensure it's not moving and reset progress for next item
            self.item_moving = False
            self.item_progress = 0.0
            return  # Don't draw if item has timed out

        # Calculate item position
        progress = self.item_progress 
        item_x = int(self.conveyor_start_x + progress * (self.conveyor_width - 50))
        item_y = self.conveyor_y + self.conveyor_height // 2
        color = self.item_colors[true_class] 
        size = 25  # Larger size for better visibility
        
        # Check for urgency indicators
        item_blinking = env_info.get('item_blinking', False)
        time_urgency = env_info.get('time_urgency', 0.0)
        
        # Apply blinking effect when urgent
        if item_blinking and int(pygame.time.get_ticks() / 200) % 2:  # Blink every 200ms
            # Make item flash red when urgent
            color = self.MUTED_RED
            size = 28  # Slightly larger when blinking
        
        # Draw item with bright outline for visibility
        pygame.draw.circle(self.screen, color, (item_x, item_y), size)
        
        # Draw urgency outline - gets redder as urgency increases
        if time_urgency > 0.5:
            urgency_color = (int(255 * time_urgency), 0, 0)  # Red intensity based on urgency
            outline_thickness = int(3 + time_urgency * 3)  # Thicker outline when urgent
            pygame.draw.circle(self.screen, urgency_color, (item_x, item_y), size, outline_thickness)
        else:
            pygame.draw.circle(self.screen, self.BLACK, (item_x, item_y), size, 3)  # Normal outline
        
        # Item type indicator - use text symbols for better visibility
        item_types = ['P', 'PL', 'O', 'M', 'D']
        item_text = self.font_large.render(item_types[true_class], True, self.WHITE)
        
        self.screen.blit(item_text, (item_x - item_text.get_width() // 2, item_y - item_text.get_height() // 2))
    
    def _update_item_progress(self, env_info: Dict[str, Any], action: Optional[int]):
        # Synchronize with environment timer instead of using internal speed
        timeout_remaining = env_info.get('item_timeout_remaining', self.item_timeout)
        item_timeout = env_info.get('item_timeout', self.item_timeout)
        
        # Calculate progress based on environment timer
        if timeout_remaining > 0:
            self.item_progress = 1.0 - (timeout_remaining / item_timeout)
            self.item_moving = True
        else:
            # Item has timed out
            self.item_progress = 1.0
            self.item_moving = False
            
        # Only reset for new item when the current item has been fully processed
        # Don't reset immediately when action is taken - wait for item to finish moving
        if action is not None and (self.item_progress >= 1.0 or timeout_remaining <= 0):
            # Item has reached the end or timed out, reset for new item
            self.item_progress = 0.0
            self.item_moving = True
    
    def _draw_bin_shadow(self, rect):
        """Draw a soft shadow for bins"""
        shadow_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        for i in range(5):
            alpha = 30 - i * 5
            shadow_color = (*self.DARK_GRAY, alpha)
            pygame.draw.rect(shadow_surf, shadow_color, (0, 0, rect.width - i, rect.height - i))
        self.screen.blit(shadow_surf, rect)
    
    def _draw_bin_gradient(self, rect, base_color):
        """Draw a vertical gradient for bins"""
        # Create lighter and darker versions of the base color
        lighter = tuple(min(255, c + 40) for c in base_color)
        darker = tuple(max(0, c - 40) for c in base_color)
        
        # Draw gradient
        for y in range(rect.height):
            ratio = y / rect.height
            color = tuple(int(lighter[i] * (1 - ratio) + darker[i] * ratio) for i in range(3))
            pygame.draw.line(self.screen, color, 
                             (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))
    
    def _draw_confidence_vector(self, env_info: Dict[str, Any]):
        confidence_vector = env_info.get('confidence_vector', [0.25, 0.25, 0.25, 0.25])
        bin_names = env_info.get('bin_names', ['Paper', 'Plastic', 'Organic', 'Metal'])
        conf_x = 20
        conf_y = 60
        bar_width = 120
        bar_height = 15
        
        title = self.font_medium.render("Confidence Vector:", True, self.BLACK)
        self.screen.blit(title, (conf_x, conf_y - 20))
        
        for i, (conf, name) in enumerate(zip(confidence_vector, bin_names)):
            y_pos = conf_y + i * (bar_height + 5)
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, (conf_x, y_pos, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.BLACK, (conf_x, y_pos, bar_width, bar_height), 1)
            fill_width = int(bar_width * conf)
            color = self.bin_colors[i]
            pygame.draw.rect(self.screen, color, (conf_x, y_pos, fill_width, bar_height))
            label = self.font_small.render(f"{name}: {conf:.2f}", True, self.BLACK)
            self.screen.blit(label, (conf_x + bar_width + 5, y_pos))
    
    def _draw_statistics(self, env_info: Dict[str, Any]):
        stats_x = 20
        stats_y = 200
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (stats_x - 6, stats_y - 6, 180, 120))
        pygame.draw.rect(self.screen, self.BLACK, (stats_x - 6, stats_y - 6, 180, 120), 1)
        
        # Show timeout remaining - use environment data if available
        timeout_remaining = env_info.get('item_timeout_remaining', self.item_timeout)
        
        # Enhanced stats with new metrics
        missed_sorts = env_info.get('missed_sorts', 0)
        time_urgency = env_info.get('time_urgency', 0.0)
        
        stats = [
            f"Total Reward: {env_info.get('total_reward', 0):.1f}",
            f"Correct Sorts: {env_info.get('correct_sorts', 0)}",
            f"Wrong Sorts: {env_info.get('wrong_sorts', 0)}",
            f"Missed Sorts: {missed_sorts}",  # New: timeout tracking
            f"Discarded: {env_info.get('discarded', 0)}",
            f"Efficiency: {env_info.get('efficiency', 0):.2%}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font_medium.render(stat, True, self.BLACK)
            self.screen.blit(text, (stats_x, stats_y + i * 18))
        
        # Visual timeout bar
        timeout_y = stats_y + len(stats) * 18 + 10
        timeout_label = self.font_medium.render("Time Left:", True, self.BLACK)
        self.screen.blit(timeout_label, (stats_x, timeout_y))
        
        # Timeout progress bar
        bar_width = 160
        bar_height = 12
        bar_x = stats_x
        bar_y = timeout_y + 18
        
        # Background bar
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.BLACK, (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Progress fill - green to red based on urgency
        if timeout_remaining > 0:
            progress = timeout_remaining / env_info.get('item_timeout', self.item_timeout)
            fill_width = int(bar_width * progress)
            
            # Color based on urgency: green -> yellow -> red
            if time_urgency < 0.5:
                color = self.SAGE_GREEN
            elif time_urgency < 0.7:
                color = (255, 255, 0)  # Yellow
            else:
                color = self.MUTED_RED
            
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
        
        # Timeout number
        timeout_text = self.font_small.render(f"{timeout_remaining}", True, self.BLACK)
        self.screen.blit(timeout_text, (bar_x + bar_width + 5, bar_y))
    
    def _draw_action_feedback(self, action: int, reward: Optional[float]):
        action_names = ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard', 'Scan']
        action_x = self.screen_width // 2
        action_y = 80
        
        # Determine color based on reward value and action type
        # High rewards (30+) = Green (correct sort)
        # Medium rewards (10-29) = Yellow (uncertain/exploration)
        # Low positive rewards (1-9) = Red (wrong sort with exploration bonus)
        # Negative rewards = Red (penalties)
        if reward is None:
            color = self.LIGHT_GRAY
        elif reward >= 30:
            color = self.SAGE_GREEN  # Correct sort
        elif reward >= 10:
            color = (255, 255, 0)  # Yellow for medium rewards
        elif reward > 0:
            color = self.MUTED_RED  # Wrong sort (still gets exploration bonus)
        else:
            color = self.MUTED_RED  # Negative rewards (penalties)
            
        pygame.draw.rect(self.screen, color, (action_x - 60, action_y - 14, 120, 28))
        pygame.draw.rect(self.screen, self.DARK_GRAY, (action_x - 60, action_y - 14, 120, 28), 1)
        
        action_text = self.font_medium.render(f"Action: {action_names[action]}", True, self.BLACK)
        self.screen.blit(action_text, (action_x - action_text.get_width() // 2, action_y - 10))
        
        if reward is not None:
            reward_color = self.SAGE_GREEN if reward > 0 else self.MUTED_RED
            reward_text = self.font_medium.render(f"Reward: {reward:.1f}", True, reward_color)
            self.screen.blit(reward_text, (action_x - reward_text.get_width() // 2, action_y + 22))
    
    def _draw_progress_bar(self, env_info: Dict[str, Any]):
        step = env_info.get('step', 0)
        batch_size = env_info.get('batch_size', 20)
        progress = step / batch_size if batch_size > 0 else 0
        
        bar_x = 20
        bar_y = 450
        bar_width = 160  # Match the time left bar width
        bar_height = 12
        
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.DARK_GRAY, (bar_x, bar_y, bar_width, bar_height), 1)
        fill_width = int(bar_width * progress)
        pygame.draw.rect(self.screen, self.SOFT_BLUE, (bar_x, bar_y, fill_width, bar_height))
        
        progress_text = self.font_small.render(f"Batch Progress: {step}/{batch_size}", True, self.BLACK)
        self.screen.blit(progress_text, (bar_x, bar_y - 18))
    
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    import time
    renderer = RecyclingSortingRenderer()
    
    # Simulation parameters
    num_items_to_simulate = 5
    
    # Global stats for the simulation
    total_reward = 0
    correct_sorts = 0
    wrong_sorts = 0
    discarded_items = 0
    missed_sorts = 0
    
    # Simulate items appearing and moving
    for item_idx in range(num_items_to_simulate):
        # Determine the class of the new item
        current_item_class = random.randint(0, 3) # Random material type (0-3)
        if item_idx == num_items_to_simulate - 1: # Make the last one a discard item for variety
            current_item_class = 4 

        # Reset renderer state for the new item
        renderer.item_progress = 0.0
        renderer.item_moving = True
        
        print(f"\n--- New item (class: {current_item_class}) appeared ---")

        # Simulate the item moving for its full timeout duration or until an action
        # We add a few extra frames to allow for timeout to be clearly visible if no action
        for frame_count in range(renderer.item_timeout + 20): 
            # Calculate remaining time for the current item
            current_item_timeout_remaining = max(0, renderer.item_timeout - frame_count)
            
            # Calculate urgency for visual feedback
            time_urgency = 1.0 - (current_item_timeout_remaining / renderer.item_timeout)
            item_blinking = time_urgency > 0.7 and int(pygame.time.get_ticks() / 200) % 2 == 0 # Blink if urgent

            # Prepare environment info for rendering
            mock_env_info = {
                'true_class': current_item_class, # This stays constant for the current item
                'confidence_vector': [0.7 if i == current_item_class else 0.1 for i in range(4)],
                'correct_sorts': correct_sorts,
                'wrong_sorts': wrong_sorts,
                'discarded': discarded_items,
                'total_reward': total_reward,
                'efficiency': (correct_sorts + discarded_items) / (item_idx * renderer.item_timeout + frame_count + 1) if (item_idx * renderer.item_timeout + frame_count + 1) > 0 else 0,
                'step': item_idx * renderer.item_timeout + frame_count, # Global step count
                'batch_size': num_items_to_simulate * renderer.item_timeout, # Total frames for all items
                'bin_names': ['Paper', 'Plastic', 'Organic', 'Metal'],
                'bin_counters': [correct_sorts, wrong_sorts, discarded_items, 0, 0], # Simplified for demo
                'item_timeout_remaining': current_item_timeout_remaining,
                'time_urgency': time_urgency,
                'item_blinking': item_blinking
            }
            
            action_taken = None
            reward_received = None

            # Simulate taking an action when the item is about halfway across the belt
            if frame_count == renderer.item_timeout // 2:
                action_taken = current_item_class # Simulate a correct sort
                reward_received = 10
                total_reward += reward_received
                if action_taken == current_item_class:
                    correct_sorts += 1
                else:
                    wrong_sorts += 1
                print(f"Item {item_idx} sorted into bin {action_taken}. Reward: {reward_received}")
                renderer.item_moving = False # Stop item movement after action
                renderer.item_progress = 1.0 # Immediately move it to end visually
                # Render one last frame with the action feedback before breaking
                renderer.render(mock_env_info, action=action_taken, reward=reward_received)
                time.sleep(0.5) # Pause to see action feedback
                break # Move to the next item

            # Simulate item timing out if it reaches the end of its time
            if current_item_timeout_remaining <= 0 and renderer.item_moving:
                action_taken = 5 # Representing no action/missed
                reward_received = -20 # Penalty for missed item
                total_reward += reward_received
                missed_sorts += 1
                print(f"Item {item_idx} timed out. Reward: {reward_received}")
                renderer.item_moving = False # Stop item movement
                # Render one last frame with the action feedback before breaking
                renderer.render(mock_env_info, action=action_taken, reward=reward_received)
                time.sleep(0.5) # Pause to see timeout feedback
                break # Move to the next item

            # Render the current frame
            renderer.render(mock_env_info, action=action_taken, reward=reward_received)
            time.sleep(0.05) # Control animation speed

    # Keep the window open after the simulation ends until manually closed
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Allow for re-rendering the final state if needed
            renderer.render(mock_env_info) 
        time.sleep(0.1) # Small delay to prevent busy-waiting

    renderer.close()