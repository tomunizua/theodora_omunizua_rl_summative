import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import random

class RecyclingSortingEnv(gym.Env):
    """
    Enhanced Recycling Sorting Environment (Optimized for Learning)
    - Features: Ambiguous Items, Time-Limited Conveyor, Fixed Bin Mapping
    - Observation: Confidence vector of item type [paper, plastic, organic, metal]
    - Actions: 0=Paper, 1=Plastic, 2=Organic, 3=Metal, 4=Discard (Fixed mapping)
    """

    def __init__(self, batch_size=20, confidence_discard_threshold=0.4, confidence_noise_std=0.0, item_timeout=80, 
                 max_wrong_sorts=8, bin_capacity=15):
        super(RecyclingSortingEnv, self).__init__()

        self.batch_size = batch_size  # Still used for some internal logic, but not for termination
        self.confidence_discard_threshold = confidence_discard_threshold
        self.confidence_noise_std = confidence_noise_std
        self.item_timeout = item_timeout  # Aligned with renderer timeout (80 frames)
        
        # Proper terminal conditions
        self.max_wrong_sorts = max_wrong_sorts  # Episode ends after too many mistakes
        self.bin_capacity = bin_capacity  # Episode ends when any bin gets full

        self.n_classes = 4  # paper, plastic, organic, metal
        self.bin_names = ["Paper", "Plastic", "Organic", "Metal", "Discard", "Scan"]
        self.current_bin_order = [0, 1, 2, 3, 4]  # Fixed mapping: Action=Material

        self.action_space = spaces.Discrete(6)  # 4 bins + discard + scan
        self.observation_space = spaces.Box(
            low=np.array([0.0] * self.n_classes),
            high=np.array([1.0] * self.n_classes),
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
        self.missed_sorts = 0  # Items that timed out
        self.bin_counters = [0, 0, 0, 0, 0]
        self.termination_reason = ""  # Track why episode ended

        # Fixed bin layout - no shuffling for easier learning
        # Action 0=Paper, 1=Plastic, 2=Organic, 3=Metal, 4=Discard
        self.current_bin_order = [0, 1, 2, 3]  # Fixed order for consistent learning

        self._current_item = {
            "true_class": self._generate_item(),
            "timer": self.item_timeout
        }
        self.confidence_vector = self._generate_confidence_vector(self._current_item["true_class"])
        return self.confidence_vector, self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0
        done = False
        truncated = False

        true_class = self._current_item["true_class"]
        
        # Find which physical bin position corresponds to the correct material type
        # The action refers to the physical bin position (0-4), we need to check
        # what material type is actually in that position
        correct_bin_position = None
        for pos, material_type in enumerate(self.current_bin_order):
            if material_type == true_class:
                correct_bin_position = pos
                break

        # Calculate confidence based on the action taken
        if action < 4:
            # Get the material type that's actually in this bin position (0-3)
            material_in_bin = self.current_bin_order[action]
            confidence = float(self.confidence_vector[material_in_bin])
        elif action == 4:
            # For discard action, use max confidence
            confidence = float(np.max(self.confidence_vector))
        else:  # action == 5 (scan)
            # For scan action, no confidence needed
            confidence = 0.0
        
        max_conf = float(np.max(self.confidence_vector))

        # Handle different actions
        if action == 5:  # Scan action
            # Time penalty for scanning - encourages strategic rather than indefinite observation
            reward -= 0.2  # Small time penalty to discourage excessive scanning
            # No processing of item, just observation
        elif action == 4:  # Discard action
            reward += 1  # Participation reward
            self.bin_counters[4] += 1
            self.discarded += 1
            if max_conf < self.confidence_discard_threshold:
                # Correct discard (low confidence item)
                reward += 20  # High reward for correct discard
                self.correct_sorts += 1
            else:
                # False discard (high confidence item that should have been sorted)
                reward -= 10  # Penalty for discarding sortable items
                self.wrong_sorts += 1
        else:
            # Material bin action (0-3)
            reward += 1  # Participation reward
            if action < 4:
                self.bin_counters[action] += 1
                if action == correct_bin_position:
                    self.correct_sorts += 1
                    # Very high rewards for correct sorting
                    if confidence >= self.confidence_discard_threshold:
                        reward += 50  # Extremely high reward for confident correct sorting
                    else:
                        reward += 30  # High reward for correct but not confident
                else:
                    self.wrong_sorts += 1
                    # Positive reinforcement even for wrong actions to encourage exploration
                    if confidence >= self.confidence_discard_threshold:
                        reward += 2  # Small positive reward even for confident wrong sorting
                    else:
                        reward += 8  # Higher positive reward for uncertain wrong sorting (exploration bonus)

        self.total_reward += reward
        self.current_step += 1

        self._last_action = action
        self._last_reward = reward
        self._last_confidence = confidence

        # Handle item timeout and progression
        # Decrease timer for current item each step
        self._current_item["timer"] -= 1
        
        # Check terminal conditions BEFORE processing timeout
        done = self._check_terminal_conditions()
        
        # If not done, handle item progression
        if not done:
            # Check if item timed out
            if self._current_item["timer"] <= 0:
                # Item timeout - penalty event, not terminal condition
                self.missed_sorts += 1  # Track missed items
                self.wrong_sorts += 1  # Count as error for quality tracking
                reward -= 15  # Penalty for letting item timeout
                print(f"⏰ Item timed out! Missed sort penalty: -15. Total missed: {self.missed_sorts}")
                
                # Generate new item and continue episode
                self._current_item = {
                    "true_class": self._generate_item(),
                    "timer": self.item_timeout
                }
            elif action is not None and action != 5:  # if a processing action was taken (not scan)
                # Only generate new item after processing actions (sort/discard), not scan
                self._current_item = {
                    "true_class": self._generate_item(),
                    "timer": self.item_timeout
                }
            # If action == 5 (scan), keep the same item and let timer continue counting down

        self.confidence_vector = self._generate_confidence_vector(self._current_item["true_class"])

        return self.confidence_vector, reward, done, truncated, self._get_info()

    def _generate_item(self) -> int:
        weights = [0.35, 0.35, 0.2, 0.1]
        return random.choices(range(self.n_classes), weights=weights)[0]

    def _generate_confidence_vector(self, true_class: int) -> np.ndarray:
        confidence = np.random.uniform(0.02, 0.08, self.n_classes)  # Lower base confidence
        ambiguous = np.random.rand() < 0.1  # Only 10% chance it's ambiguous (was 20%)

        if ambiguous:
            # Make ambiguous cases less confusing
            other = random.choice([i for i in range(self.n_classes) if i != true_class])
            confidence[true_class] = np.random.uniform(0.4, 0.5)  # Still slightly higher
            confidence[other] = np.random.uniform(0.35, 0.45)    # But close
        else:
            # Make clear cases very clear
            confidence[true_class] = np.random.uniform(0.7, 0.95)  # Very high confidence

        confidence = confidence / np.sum(confidence)

        if self.confidence_noise_std > 0:
            confidence = self._apply_confidence_noise(confidence)

        return confidence.astype(np.float32)

    def _check_terminal_conditions(self) -> bool:
        """Check if episode should terminate based on realistic conditions"""
        
        # Terminal condition 1: Any material bin reached capacity
        for i in range(4):  # Only check material bins (0-3), not discard bin (4)
            if self.bin_counters[i] >= self.bin_capacity:
                self.termination_reason = f"Bin {self.bin_names[i]} reached capacity ({self.bin_capacity} items)"
                print(f"🗂️ Episode terminated: {self.termination_reason}")
                return True
        
        # Terminal condition 2: Too many wrong sorts (quality control failure)
        if self.wrong_sorts >= self.max_wrong_sorts:
            self.termination_reason = f"Quality control failure: {self.wrong_sorts} wrong sorts (max: {self.max_wrong_sorts})"
            print(f"❌ Episode terminated: {self.termination_reason}")
            return True
            
        return False

    def _apply_confidence_noise(self, conf: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.confidence_noise_std, size=conf.shape)
        noisy_conf = conf + noise
        noisy_conf = np.clip(noisy_conf, 0, 1)
        noisy_conf = noisy_conf / np.sum(noisy_conf)
        return noisy_conf.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        # Calculate time urgency indicator (for visual feedback)
        time_urgency = 1.0 - (self._current_item["timer"] / self.item_timeout)  # 0.0 = just started, 1.0 = about to timeout
        
        return {
            'total_reward': self.total_reward,
            'correct_sorts': self.correct_sorts,
            'wrong_sorts': self.wrong_sorts,
            'discarded': self.discarded,
            'missed_sorts': self.missed_sorts,  # New: track timeouts
            'true_class': self._current_item["true_class"],
            'step': self.current_step,
            'efficiency': self.correct_sorts / max(1, self.correct_sorts + self.wrong_sorts),
            'confidence_vector': self.confidence_vector,
            'bin_names': self.bin_names,
            'bin_counters': self.bin_counters,
            'current_bin_order': self.current_bin_order,  # Aligned with renderer
            'batch_size': self.batch_size,
            'item_timeout_remaining': self._current_item["timer"],  # For renderer timeout display
            'time_urgency': time_urgency,  # 0.0-1.0 urgency indicator for visual feedback
            'item_blinking': time_urgency > 0.7,  # Item should blink when urgency > 70%
            # Terminal condition info
            'bin_capacity': self.bin_capacity,
            'max_wrong_sorts': self.max_wrong_sorts,
            'bin_full': any(self.bin_counters[i] >= self.bin_capacity for i in range(4)),
            'too_many_errors': self.wrong_sorts >= self.max_wrong_sorts,
            'termination_reason': self.termination_reason,  # Why episode ended
            # Quality metrics
            'total_processed': self.correct_sorts + self.wrong_sorts + self.missed_sorts,
            'accuracy': self.correct_sorts / max(1, self.correct_sorts + self.wrong_sorts + self.missed_sorts),
            'miss_rate': self.missed_sorts / max(1, self.correct_sorts + self.wrong_sorts + self.missed_sorts)
        }

    def render(self, mode='human'):
        if self._renderer is None:
            from environment.rendering import RecyclingSortingRenderer
            self._renderer = RecyclingSortingRenderer()
        self._renderer.render(self._get_info(), action=self._last_action, reward=self._last_reward)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
