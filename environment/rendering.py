import pygame
import numpy as np
from typing import Dict, Any, Optional
import math

class RecyclingSortingRenderer:
    """
    Pygame-based renderer for the Recycling Sorting Environment
    - Items move smoothly along the conveyor belt
    - Red discard bin at the end
    - Bin counters above each bin
    - 'Conveyor Belt' label below the belt
    """
    
    def __init__(self, screen_width=900, screen_height=600):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Recycling Sorting Agent - RL Environment")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        self.BROWN = (139, 69, 19)
        self.CYAN = (0, 255, 255)
        
        # Item colors for each class
        self.item_colors = {
            0: self.YELLOW,  # Paper
            1: self.CYAN,    # Plastic
            2: self.BROWN,   # Organic
            3: self.GRAY,    # Metal
        }
        
        # Bin colors
        self.bin_colors = {
            0: self.YELLOW,  # Paper
            1: self.CYAN,    # Plastic
            2: self.BROWN,   # Organic
            3: self.GRAY,    # Metal
            4: self.RED      # Discard
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
        
        # For smooth item movement
        self.item_progress = 0.0  # 0.0 (left) to 1.0 (right)
        self.last_step = -1
        self.last_true_class = 0
        self.last_action = None
        self.last_bin_counters = [0, 0, 0, 0, 0]
        self.item_moving = True
        self.item_speed = 0.04  # Controls how fast the item moves
        
    def render(self, env_info: Dict[str, Any], action: Optional[int] = None, reward: Optional[float] = None):
        self.screen.fill(self.WHITE)
        
        # Draw background
        self._draw_background()
        
        # Draw conveyor belt
        self._draw_conveyor_belt(env_info)
        
        # Draw bins and bin counters
        self._draw_bins(env_info)
        
        # Animate item moving along the conveyor
        self._draw_moving_item(env_info)
        
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
        
        # Update item movement state
        self._update_item_progress(env_info, action)
        
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
            color = self.bin_colors[i]
            # Bin container
            pygame.draw.rect(self.screen, color, (x, bin_y, self.bin_width, self.bin_height))
            pygame.draw.rect(self.screen, self.BLACK, (x, bin_y, self.bin_width, self.bin_height), 2)
            # Bin label
            label = self.font_medium.render(bin_labels[i], True, self.BLACK)
            self.screen.blit(label, (x + self.bin_width // 2 - label.get_width() // 2, bin_y - 18))
            # Bin counter above
            counter = self.font_large.render(str(bin_counters[i]), True, self.BLACK)
            self.screen.blit(counter, (x + self.bin_width // 2 - counter.get_width() // 2, bin_y - 40))
    
    def _draw_moving_item(self, env_info: Dict[str, Any]):
        # Animate item from left to right
        true_class = env_info.get('true_class', 0)
        # If new step, reset progress
        if env_info.get('step', 0) != self.last_step:
            self.item_progress = 0.0
            self.last_true_class = true_class
            self.item_moving = True
        # Calculate item position
        progress = self.item_progress
        item_x = int(self.conveyor_start_x + progress * (self.conveyor_width - 24))
        item_y = self.conveyor_y + self.conveyor_height // 2
        color = self.item_colors[self.last_true_class]
        size = 18
        pygame.draw.circle(self.screen, color, (item_x, item_y), size)
        pygame.draw.circle(self.screen, self.BLACK, (item_x, item_y), size, 1)
        # Item type indicator
        item_types = ['📄', '🥤', '🍎', '🔧']
        item_text = self.font_medium.render(item_types[self.last_true_class], True, self.BLACK)
        self.screen.blit(item_text, (item_x - item_text.get_width() // 2, item_y - item_text.get_height() // 2))
    
    def _update_item_progress(self, env_info: Dict[str, Any], action: Optional[int]):
        # If new step, reset progress
        if env_info.get('step', 0) != self.last_step:
            self.item_progress = 0.0
            self.last_step = env_info.get('step', 0)
            self.last_action = action
            self.last_bin_counters = env_info.get('bin_counters', [0, 0, 0, 0, 0])
            self.item_moving = True
        # Move item smoothly
        if self.item_moving:
            self.item_progress += self.item_speed
            if self.item_progress >= 1.0:
                self.item_progress = 1.0
                self.item_moving = False
    
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
        stats = [
            f"Total Reward: {env_info.get('total_reward', 0):.1f}",
            f"Correct Sorts: {env_info.get('correct_sorts', 0)}",
            f"Wrong Sorts: {env_info.get('wrong_sorts', 0)}",
            f"Discarded: {env_info.get('discarded', 0)}",
            f"Efficiency: {env_info.get('efficiency', 0):.2%}",
            f"Step: {env_info.get('step', 0)}/{env_info.get('batch_size', 20)}"
        ]
        for i, stat in enumerate(stats):
            text = self.font_medium.render(stat, True, self.BLACK)
            self.screen.blit(text, (stats_x, stats_y + i * 18))
    
    def _draw_action_feedback(self, action: int, reward: Optional[float]):
        action_names = ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard']
        action_x = self.screen_width // 2
        action_y = 80
        color = self.GREEN if reward and reward > 0 else self.RED if reward and reward < 0 else self.GRAY
        pygame.draw.rect(self.screen, color, (action_x - 60, action_y - 14, 120, 28))
        pygame.draw.rect(self.screen, self.BLACK, (action_x - 60, action_y - 14, 120, 28), 1)
        action_text = self.font_medium.render(f"Action: {action_names[action]}", True, self.BLACK)
        self.screen.blit(action_text, (action_x - action_text.get_width() // 2, action_y - 10))
        if reward is not None:
            reward_color = self.GREEN if reward > 0 else self.RED
            reward_text = self.font_medium.render(f"Reward: {reward:.1f}", True, reward_color)
            self.screen.blit(reward_text, (action_x - reward_text.get_width() // 2, action_y + 22))
    
    def _draw_progress_bar(self, env_info: Dict[str, Any]):
        step = env_info.get('step', 0)
        batch_size = env_info.get('batch_size', 20)
        progress = step / batch_size
        bar_x = 20
        bar_y = 450
        bar_width = 180
        bar_height = 12
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.BLACK, (bar_x, bar_y, bar_width, bar_height), 1)
        fill_width = int(bar_width * progress)
        pygame.draw.rect(self.screen, self.GREEN, (bar_x, bar_y, fill_width, bar_height))
        progress_text = self.font_small.render(f"Batch Progress: {step}/{batch_size}", True, self.BLACK)
        self.screen.blit(progress_text, (bar_x, bar_y - 18))
    
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    import time
    renderer = RecyclingSortingRenderer()
    for step in range(30):
        mock_env_info = {
            'true_class': step % 4,
            'confidence_vector': [0.7, 0.1, 0.1, 0.1] if step % 4 == 0 else [0.1, 0.7, 0.1, 0.1] if step % 4 == 1 else [0.1, 0.1, 0.7, 0.1] if step % 4 == 2 else [0.1, 0.1, 0.1, 0.7],
            'correct_sorts': step // 3,
            'wrong_sorts': step % 4 == 0,
            'discarded': step % 5 == 0,
            'total_reward': 10 * step - 5,
            'efficiency': 0.8,
            'step': step,
            'batch_size': 30,
            'bin_names': ['Paper', 'Plastic', 'Organic', 'Metal', 'Discard'],
            'bin_counters': [step // 5, step // 6, step // 7, step // 8, step // 9]
        }
        renderer.render(mock_env_info, action=step % 5, reward=10 if step % 2 == 0 else -5)
        time.sleep(0.15)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    renderer.close()