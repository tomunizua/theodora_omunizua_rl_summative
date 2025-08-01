# Recycling Sorting Agent - Reinforcement Learning Project

## Overview

This project implements a comprehensive Reinforcement Learning (RL) system for training an intelligent agent to sort recyclable items on a conveyor belt. The agent learns to classify and sort different types of waste (Paper, Plastic, Organic, Metal, Non-recyclable) into appropriate bins while maximizing efficiency and minimizing errors.

## 🎮 Interactive Model Testing

The project includes an interactive testing interface (`play.py`) that allows you to:
- **Watch trained models in action** with real-time visualization
- **Test different algorithms** (DQN, PPO, A2C) by changing the model path
- **See decision-making process** with color-coded action feedback
- **Monitor performance** with episode rewards and termination reasons

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Advanced Pygame visualization
├── training/
│   ├── dqn_training.py          # DQN training with hyperparameter variants
│   ├── pg_training.py           # Policy Gradient methods (PPO, A2C, REINFORCE)
├── models/                      # Saved trained models
│   ├── dqn/                     # DQN model variants
│   └── pg/                      # Policy gradient models
├── results/                     # Training results and plots
├── main.py                      # Main entry point for experiments
├── play.py                      # Interactive model testing interface
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Environment Design

### State Space
- **Confidence Vector**: 4-dimensional vector representing classification confidence for each material type
  - Paper confidence (0.0 - 1.0)
  - Plastic confidence (0.0 - 1.0)
  - Organic confidence (0.0 - 1.0)
  - Metal confidence (0.0 - 1.0)
- **Dynamic Bin Assignment**: Bins are randomly shuffled each episode to prevent position memorization
- **Item Timeout**: Each item has 80 steps to be processed before automatic timeout

### Action Space (Discrete)
- **0**: Sort to Bin A (Paper)
- **1**: Sort to Bin B (Plastic)
- **2**: Sort to Bin C (Organic)
- **3**: Sort to Bin D (Metal)
- **4**: Discard (Non-recyclable)
- **5**: Scan (Continue observing item)

### Reward System
- **+50**: Confident correct sorting (high confidence + correct bin)
- **+30**: Uncertain correct sorting (low confidence + correct bin)
- **+20**: Correct discard (low confidence item)
- **+8**: Wrong sort with low confidence (exploration bonus)
- **+2**: Wrong sort with high confidence (small exploration bonus)
- **+1**: Participation reward for any sorting action
- **-0.2**: Time penalty for scanning (encourages decisive action)
- **-10**: False discard (discarding sortable item)
- **-15**: Item timeout penalty (missed sort)

## Implemented Algorithms

### Value-Based Methods
1. **DQN (Deep Q-Network)**
   - Default hyperparameters
   - Optimized hyperparameters
   - Exploration-focused variant

### Policy Gradient Methods
1. **PPO (Proximal Policy Optimization)**
   - Default hyperparameters
   - Optimized hyperparameters
2. **A2C (Advantage Actor-Critic)**
   - Default hyperparameters
   - Optimized hyperparameters
3. **REINFORCE**
   - Pure policy gradient implementation

## Features

### Advanced Visualization
- **Real-time Pygame rendering** with smooth animations
- **Conveyor belt movement** with realistic item progression
- **Item identification** with material type indicators (P, PL, O, M, D)
- **Real-time statistics** display with sorting metrics
- **Smart action feedback** with color-coded performance:
  - 🟢 **Green**: Correct sorts (high rewards 30+)
  - 🟡 **Yellow**: Uncertain actions (medium rewards 10-29)
  - 🔴 **Red**: Wrong sorts or penalties (low/negative rewards)
- **Progress tracking** with synchronized progress bars
- **Confidence vector** visualization for decision transparency

### Comprehensive Training
- **Multiple algorithm variants** for comparison
- **Hyperparameter optimization** for each method
- **Automatic evaluation** and comparison
- **Training visualization** with TensorBoard support
- **Model checkpointing** and best model saving

### Analysis Tools
- **Performance comparison** across all algorithms
- **Statistical analysis** of sorting efficiency
- **Training curve visualization**
- **Results export** to JSON format

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd theodora_omunizua_rl_summative
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import gymnasium, stable_baselines3, pygame; print('Installation successful!')"
   ```

## Usage

### Quick Start
Run the complete project with all features:
```bash
python main.py --mode all
```

### Individual Components

#### Training Only
```bash
python main.py --mode train --timesteps 50000
```

#### Random Agent Demo
```bash
python main.py --mode demo
```

#### Visualization Demo
```bash
python main.py --mode visualize
```

#### Interactive Model Testing
```bash
python play.py
```

#### Custom Training Duration
```bash
python main.py --mode train --timesteps 100000
```

## 🎯 Interactive Model Testing (play.py)

The `play.py` script provides an interactive interface to test and visualize trained models:

### Features
- **Real-time model testing** with visual feedback
- **Configurable model selection** - easily switch between different trained models
- **Episode-by-episode results** with rewards and termination reasons
- **Color-coded action feedback** showing model performance
- **Synchronized item movement** with proper material type display

### Configuration
Edit the model configuration in `play.py`:
```python
# 🎯 MODEL CONFIGURATION - CHANGE THESE TO TEST DIFFERENT MODELS
self.model_path = 'models/dqn/dqn_less_optimized.zip'  # ← CHANGE THIS PATH
self.model_type = 'DQN'  # ← CHANGE THIS TYPE: 'DQN', 'PPO', 'A2C'
```

### Available Models
- **DQN Models**: `models/dqn/dqn_final.zip`, `models/dqn/dqn_optimized.zip`, `models/dqn/dqn_exploration.zip`
- **PPO Models**: `models/pg/ppo/ppo_final.zip`, `models/pg/ppo_optimized/ppo_optimized.zip`
- **A2C Models**: `models/pg/a2c/a2c_final.zip`, `models/pg/a2c_optimized/a2c_optimized.zip`
- **REINFORCE**: `models/pg/reinforce/reinforce_final.zip`

### Controls
- **ESC key** or **close window** to exit
- **Automatic progression** through 3 episodes
- **Real-time statistics** and performance metrics

### Sample Output
```
🚀 Recycling Sorting Agent - Model Testing
==================================================
🤖 Testing Model: DQN
📁 Path: models/dqn/dqn_less_optimized.zip

 Loaded DQN model from: models/dqn/dqn_less_optimized.zip
🎮 Running 3 episodes...

Episode 1: 51.0 (Ended: Quality control failure: 8 wrong sorts (max: 8))
Episode 2: 94.3 (Ended: Quality control failure: 8 wrong sorts (max: 8))
Episode 3: 127.8 (Ended: Bin Paper reached capacity (15 items))

📊 Results:
  Cumulative Reward: 273.1
  Average Reward: 91.0
```

### Individual Training Scripts

#### DQN Training
```bash
cd training
python dqn_training.py
```

#### Policy Gradient Training
```bash
cd training
python pg_training.py
```

## Expected Results

### Performance Metrics
The system tracks and compares:
- **Average Reward**: Total reward per episode
- **Sorting Efficiency**: Percentage of correct sorts
- **Correct Sorts**: Number of items correctly classified
- **Wrong Sorts**: Number of misclassified items
- **Missed Items**: Items that reached the end unsorted

### Algorithm Comparison
The project provides comprehensive comparison of:
- **Value-based vs Policy-based** methods
- **Exploration strategies** and their impact
- **Hyperparameter sensitivity** for each algorithm
- **Convergence speed** and stability

## Output Files

### Generated Results
- `results/training_results.json`: Complete training results
- `results/algorithm_comparison.png`: Visual comparison plots
- `models/*/training_results.png`: Individual algorithm training curves
- `models/*/best_model/`: Best performing models for each variant

### Model Files
- `models/dqn/`: DQN model variants
- `models/pg/ppo/`: PPO models
- `models/pg/a2c/`: A2C models
- `models/pg/reinforce/`: REINFORCE models

## Technical Details

### Environment Specifications
- **Observation Space**: Box(4,) - Confidence vector for material classification
- **Action Space**: Discrete(6) - 6 possible actions (4 bins + discard + scan)
- **Episode Length**: Variable (ends when bin reaches capacity or too many errors)
- **Reward Range**: [-15, +51] with exploration bonuses
- **Item Timeout**: 80 steps per item (realistic conveyor timing)
- **Quality Control**: Maximum 8 wrong sorts before episode termination

### Training Parameters
- **Default Timesteps**: 50,000 per algorithm
- **Evaluation Frequency**: Every 1,000 steps
- **Checkpoint Frequency**: Every 5,000 steps
- **Evaluation Episodes**: 10 per model

### Hardware Requirements
- **Minimum**: CPU-only training (slower)
- **Recommended**: GPU with CUDA support
- **Memory**: 4GB RAM minimum, 8GB recommended

## Customization

### Environment Modifications
- Adjust `batch_size` and `conveyor_length` in environment initialization
- Modify reward structure in `_calculate_reward()` method
- Change item distribution in `_generate_item()` method

### Algorithm Tuning
- Modify hyperparameters in training scripts
- Add new algorithm variants
- Implement custom exploration strategies

### Visualization Customization
- Adjust colors and layout in `rendering.py`
- Modify animation speeds and effects
- Add new visual elements

## Troubleshooting

### Common Issues

1. **Pygame Display Error**
   - Ensure X11 forwarding if running on remote server
   - Use `export DISPLAY=:0` for local display

2. **CUDA Out of Memory**
   - Reduce batch sizes in training scripts
   - Use CPU-only training with `device="cpu"`

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path includes project root

4. **Training Convergence Issues**
   - Increase training timesteps
   - Adjust learning rates
   - Modify exploration parameters

## Contributing

This project is designed for educational purposes. Feel free to:
- Experiment with different reward structures
- Implement additional RL algorithms
- Enhance the visualization system
- Optimize hyperparameters for better performance

## License

This project is created for educational purposes as part of the ML Techniques II course at African Leadership University.

## Contact

For questions or issues related to this project, please refer to the course instructor or create an issue in the repository.

---

**Note**: This project demonstrates advanced RL concepts including environment design, algorithm comparison, hyperparameter optimization, and real-time visualization. It serves as a comprehensive example of practical reinforcement learning implementation.