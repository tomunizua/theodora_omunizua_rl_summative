# Recycling Sorting Agent - Reinforcement Learning Project

## Overview

This project implements a comprehensive Reinforcement Learning (RL) system for training an intelligent agent to sort recyclable items on a conveyor belt. The agent learns to classify and sort different types of waste (Paper, Plastic, Organic, Metal, Non-recyclable) into appropriate bins while maximizing efficiency and minimizing errors.

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
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Environment Design

### State Space
- **Item Type**: 5 types (Paper, Plastic, Organic, Metal, Non-recyclable)
- **Conveyor Speed**: 3 levels (Slow, Medium, Fast)
- **Item Position**: Position on conveyor belt (0-9)
- **Time Remaining**: Steps left in current batch

### Action Space (Discrete)
- **0**: Sort to Bin A (Paper)
- **1**: Sort to Bin B (Plastic)
- **2**: Sort to Bin C (Organic)
- **3**: Sort to Bin D (Metal)
- **4**: Discard (Non-recyclable)
- **5**: Wait (No action)

### Reward System
- **+5**: Correct sorting
- **-5**: Wrong sorting
- **-10**: Missed item (item reaches end without sorting)
- **-1**: Waiting when item is ready to sort
- **-2**: Invalid action (trying to sort too early)
- **+20 × efficiency**: Batch completion bonus

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
- **Conveyor belt movement** with speed-dependent animation
- **Item identification** with emoji indicators
- **Real-time statistics** display
- **Action feedback** with color-coded rewards
- **Progress tracking** with visual progress bars

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

#### Custom Training Duration
```bash
python main.py --mode train --timesteps 100000
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
- **Observation Space**: Box(4,) - [item_type, speed, position, time_remaining]
- **Action Space**: Discrete(6) - 6 possible actions
- **Episode Length**: Variable (batch-based)
- **Reward Range**: [-10, +25] approximately

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