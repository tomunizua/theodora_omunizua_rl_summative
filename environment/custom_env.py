import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import random

class RecyclingSortingEnv(gym.Env):
    """
    Recycling Sorting Environment (Conveyor, Discard Bin, Confidence-based)
    - 5 bins: Paper, Plastic, Organic, Metal, Discard (red)
    - Observation: Confidence vector [paper, plastic, organic, metal]
    - Actions: 0=Paper, 1=Plastic, 2=Organic, 3=Metal, 4=Discard
    - Rewards: Confidence-based, see below
    """
    
    def __init__(self, batch_size=20, confidence_discard_threshold=0.4):
        super(RecyclingSortingEnv, self).__init__()
        
        self.batch_size = batch_size
        self.confidence_discard_threshold = confidence_discard_threshold
        self.current_step = 0
        self.total_reward = 0
        self.correct_sorts = 0
        self.wrong_sorts = 0
        self.discarded = 0
        self.n_classes = 4
        self.bin_names = ["Paper", "Plastic", "Organic", "Metal", "Discard"]
        self.bin_counters = [0, 0, 0, 0, 0]  # Paper, Plastic, Organic, Metal, Discard
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.reset()
        self._renderer = None
        self._last_action = None
        self._last_reward = None
        self._last_confidence = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0
        self.correct_sorts = 0
        self.wrong_sorts = 0
        self.discarded = 0
        self.bin_counters = [0, 0, 0, 0, 0]
        self.true_class = self._generate_item()
        self.confidence_vector = self._generate_confidence_vector(self.true_class)
        observation = self.confidence_vector
        info = self._get_info()
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0
        done = False
        truncated = False
        confidence = float(self.confidence_vector[action]) if action < 4 else float(np.max(self.confidence_vector))
        max_conf = float(np.max(self.confidence_vector))
        correct_bin = self.true_class
        # Action: 0-3 = sort to bin, 4 = discard
        if action == 4:  # Discard
            self.bin_counters[4] += 1
            self.discarded += 1
            if max_conf < self.confidence_discard_threshold:
                reward = 2  # Small reward for correct discard (uncertain)
            else:
                reward = -10  # Heavy penalty for discarding when confident
        else:
            self.bin_counters[action] += 1
            if action == correct_bin:
                if self.confidence_vector[action] >= self.confidence_discard_threshold:
                    reward = 10  # Correct and confident
                else:
                    reward = 2  # Correct but not confident
                self.correct_sorts += 1
            else:
                if self.confidence_vector[action] >= self.confidence_discard_threshold:
                    reward = -10  # Wrong but confident
                else:
                    reward = -3  # Wrong and not confident
                self.wrong_sorts += 1
        reward -= 0.1  # Step penalty
        self.total_reward += reward
        self.current_step += 1
        self._last_action = action
        self._last_reward = reward
        self._last_confidence = confidence
        # Next item
        self.true_class = self._generate_item()
        self.confidence_vector = self._generate_confidence_vector(self.true_class)
        if self.current_step >= self.batch_size:
            done = True
        observation = self.confidence_vector
        info = self._get_info()
        return observation, reward, done, truncated, info
    
    def _generate_item(self) -> int:
        weights = [0.35, 0.35, 0.2, 0.1]
        return random.choices(range(self.n_classes), weights=weights)[0]
    def _generate_confidence_vector(self, true_class: int) -> np.ndarray:
        confidence = np.random.uniform(0.05, 0.15, self.n_classes)
        confidence[true_class] = np.random.uniform(0.6, 0.9)
        confidence = confidence / np.sum(confidence)
        return confidence.astype(np.float32)
    def _get_info(self) -> Dict[str, Any]:
        return {
            'total_reward': self.total_reward,
            'correct_sorts': self.correct_sorts,
            'wrong_sorts': self.wrong_sorts,
            'discarded': self.discarded,
            'true_class': self.true_class,
            'step': self.current_step,
            'efficiency': self.correct_sorts / max(1, self.correct_sorts + self.wrong_sorts),
            'confidence_vector': self.confidence_vector,
            'bin_names': self.bin_names,
            'bin_counters': self.bin_counters,
            'batch_size': self.batch_size
        }
    def render(self, mode='human'):
        if self._renderer is None:
            from environment.rendering import RecyclingSortingRenderer
            self._renderer = RecyclingSortingRenderer()
        info = self._get_info()
        self._renderer.render(info, action=self._last_action, reward=self._last_reward)
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None 