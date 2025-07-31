#!/usr/bin/env python3
"""
Comprehensive Training Results Analysis
Generates all required plots and metrics for the report
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingAnalyzer:
    """Comprehensive training results analyzer for RL methods"""
    
    def __init__(self, models_dir="models", results_dir="results"):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create plots subdirectory
        self.plots_dir = self.results_dir / "training_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Method configurations
        self.methods = {
            'DQN_Default': 'models/dqn/tensorboard_default',
            'DQN_Optimized': 'models/dqn/tensorboard_optimized', 
            'DQN_Less_Optimized': 'models/dqn/tensorboard_less_optimized',
            'PPO_Default': 'models/pg/ppo/tensorboard',
            'PPO_Optimized': 'models/pg/ppo_optimized/tensorboard',
            'A2C_Default': 'models/pg/a2c/tensorboard',
            'A2C_Optimized': 'models/pg/a2c_optimized/tensorboard',
            'REINFORCE': 'models/pg/reinforce/tensorboard'
        }
        
        # Colors for consistent plotting
        self.colors = {
            'DQN_Default': '#1f77b4',
            'DQN_Optimized': '#ff7f0e', 
            'DQN_Less_Optimized': '#2ca02c',
            'PPO_Default': '#d62728',
            'PPO_Optimized': '#9467bd',
            'A2C_Default': '#8c564b',
            'A2C_Optimized': '#e377c2',
            'REINFORCE': '#7f7f7f'
        }
        
        print("🔍 Training Results Analyzer Initialized")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📊 Results directory: {self.results_dir}")
        print(f"🎨 Plots directory: {self.plots_dir}")
    
    def extract_tensorboard_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract training data from tensorboard logs"""
        print("\\n📈 Extracting Tensorboard Data...")
        
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            print("❌ TensorBoard not available. Installing...")
            os.system("pip install tensorboard")
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        all_data = {}
        
        for method_name, log_path in self.methods.items():
            full_path = Path(log_path)
            if not full_path.exists():
                print(f"⚠️  {method_name}: No tensorboard logs found at {full_path}")
                continue
            
            print(f"📊 Processing {method_name}...")
            method_data = {}
            
            # Find all subdirectories (different runs)
            run_dirs = [d for d in full_path.iterdir() if d.is_dir()]
            
            if not run_dirs:
                print(f"⚠️  {method_name}: No run directories found")
                continue
            
            # Process the most recent run
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            print(f"   📂 Using run: {latest_run.name}")
            
            try:
                # Load tensorboard data
                ea = EventAccumulator(str(latest_run))
                ea.Reload()
                
                # Extract available scalar tags
                scalar_tags = ea.Tags()['scalars']
                print(f"   📋 Available metrics: {scalar_tags}")
                
                # Extract key metrics
                for tag in scalar_tags:
                    try:
                        scalar_events = ea.Scalars(tag)
                        steps = [event.step for event in scalar_events]
                        values = [event.value for event in scalar_events]
                        
                        method_data[tag] = {
                            'steps': np.array(steps),
                            'values': np.array(values)
                        }
                        print(f"   ✅ Extracted {tag}: {len(values)} data points")
                    except Exception as e:
                        print(f"   ⚠️  Failed to extract {tag}: {e}")
                
                all_data[method_name] = method_data
                
            except Exception as e:
                print(f"❌ Failed to process {method_name}: {e}")
        
        print(f"✅ Extracted data for {len(all_data)} methods")
        return all_data
    
    def extract_evaluation_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract evaluation data from saved evaluation files"""
        print("\\n📊 Extracting Evaluation Data...")
        
        eval_data = {}
        
        # Check for evaluation files
        eval_paths = {
            'DQN': 'models/dqn/eval/evaluations.npz',
            'PPO_Default': 'models/pg/ppo/eval/evaluations.npz',
            'PPO_Optimized': 'models/pg/ppo_optimized/eval/evaluations.npz',
            'A2C_Default': 'models/pg/a2c/eval/evaluations.npz',
            'A2C_Optimized': 'models/pg/a2c_optimized/eval/evaluations.npz',
            'REINFORCE': 'models/pg/reinforce/eval/evaluations.npz'
        }
        
        for method_name, eval_path in eval_paths.items():
            full_path = Path(eval_path)
            if full_path.exists():
                try:
                    data = np.load(full_path)
                    eval_data[method_name] = {
                        'timesteps': data['timesteps'],
                        'results': data['results'],
                        'ep_lengths': data['ep_lengths'] if 'ep_lengths' in data else None
                    }
                    print(f"✅ {method_name}: {len(data['results'])} evaluation points")
                except Exception as e:
                    print(f"❌ Failed to load {method_name} evaluation data: {e}")
            else:
                print(f"⚠️  {method_name}: No evaluation data found")
        
        return eval_data
    
    def plot_cumulative_rewards(self, tensorboard_data: Dict, eval_data: Dict):
        """Plot cumulative rewards over episodes for all methods"""
        print("\\n📈 Creating Cumulative Rewards Plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Training Rewards (from tensorboard)
        ax1.set_title('Training Rewards Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Episode Reward')
        ax1.grid(True, alpha=0.3)
        
        for method_name, data in tensorboard_data.items():
            # Look for reward-related metrics
            reward_keys = [k for k in data.keys() if 'reward' in k.lower() or 'return' in k.lower()]
            
            if reward_keys:
                # Use the first reward metric found
                reward_key = reward_keys[0]
                steps = data[reward_key]['steps']
                values = data[reward_key]['values']
                
                # Smooth the data for better visualization
                if len(values) > 10:
                    window = min(50, len(values) // 10)
                    smoothed = pd.Series(values).rolling(window=window, center=True).mean()
                    ax1.plot(steps, smoothed, label=method_name, 
                            color=self.colors.get(method_name, None), linewidth=2)
                    ax1.fill_between(steps, values, alpha=0.1, 
                                   color=self.colors.get(method_name, None))
                else:
                    ax1.plot(steps, values, label=method_name, 
                            color=self.colors.get(method_name, None), linewidth=2)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Evaluation Rewards
        ax2.set_title('Evaluation Performance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Mean Evaluation Reward')
        ax2.grid(True, alpha=0.3)
        
        for method_name, data in eval_data.items():
            timesteps = data['timesteps']
            mean_rewards = np.mean(data['results'], axis=1)
            std_rewards = np.std(data['results'], axis=1)
            
            color = self.colors.get(method_name, None)
            ax2.plot(timesteps, mean_rewards, label=method_name, 
                    color=color, linewidth=2)
            ax2.fill_between(timesteps, mean_rewards - std_rewards, 
                           mean_rewards + std_rewards, alpha=0.2, color=color)
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'cumulative_rewards_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Cumulative rewards plot saved to {self.plots_dir / 'cumulative_rewards_comparison.png'}")
    
    def analyze_convergence(self, tensorboard_data: Dict, eval_data: Dict) -> Dict[str, Dict]:
        """Analyze episodes/steps to convergence for each method"""
        print("\\n🎯 Analyzing Convergence Metrics...")
        
        convergence_analysis = {}
        
        # Define convergence criteria
        convergence_threshold = 0.8  # 80% of maximum performance
        
        for method_name in set(list(tensorboard_data.keys()) + list(eval_data.keys())):
            analysis = {
                'method': method_name,
                'convergence_step': None,
                'final_performance': None,
                'max_performance': None,
                'stability_score': None,
                'training_efficiency': None
            }
            
            # Use evaluation data if available, otherwise tensorboard
            if method_name in eval_data:
                timesteps = eval_data[method_name]['timesteps']
                rewards = np.mean(eval_data[method_name]['results'], axis=1)
            elif method_name in tensorboard_data:
                data = tensorboard_data[method_name]
                reward_keys = [k for k in data.keys() if 'reward' in k.lower() or 'return' in k.lower()]
                if reward_keys:
                    timesteps = data[reward_keys[0]]['steps']
                    rewards = data[reward_keys[0]]['values']
                else:
                    continue
            else:
                continue
            
            if len(rewards) == 0:
                continue
            
            # Calculate metrics
            analysis['max_performance'] = float(np.max(rewards))
            analysis['final_performance'] = float(np.mean(rewards[-10:]))  # Last 10 evaluations
            
            # Find convergence point
            target_performance = analysis['max_performance'] * convergence_threshold
            convergence_mask = rewards >= target_performance
            
            if np.any(convergence_mask):
                convergence_idx = np.where(convergence_mask)[0][0]
                analysis['convergence_step'] = int(timesteps[convergence_idx])
            
            # Calculate stability (coefficient of variation in final 20% of training)
            final_20_percent = int(len(rewards) * 0.8)
            if final_20_percent < len(rewards):
                final_rewards = rewards[final_20_percent:]
                analysis['stability_score'] = float(np.std(final_rewards) / (np.mean(final_rewards) + 1e-8))
            
            # Training efficiency (area under curve / total steps)
            if len(timesteps) > 1:
                auc = np.trapz(rewards, timesteps)
                analysis['training_efficiency'] = float(auc / timesteps[-1])
            
            convergence_analysis[method_name] = analysis
            
            print(f"📊 {method_name}:")
            print(f"   🎯 Convergence: {analysis['convergence_step']} steps")
            print(f"   📈 Max Performance: {analysis['max_performance']:.2f}")
            print(f"   🏁 Final Performance: {analysis['final_performance']:.2f}")
            print(f"   📊 Stability Score: {analysis['stability_score']:.4f}")
        
        return convergence_analysis
    
    def create_convergence_plot(self, convergence_analysis: Dict):
        """Create convergence comparison plots"""
        print("\\n📊 Creating Convergence Analysis Plot...")
        
        # Prepare data for plotting
        methods = list(convergence_analysis.keys())
        convergence_steps = [convergence_analysis[m]['convergence_step'] or 0 for m in methods]
        final_performance = [convergence_analysis[m]['final_performance'] or 0 for m in methods]
        stability_scores = [convergence_analysis[m]['stability_score'] or 0 for m in methods]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Convergence Analysis Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Steps to Convergence
        ax1.bar(range(len(methods)), convergence_steps, 
               color=[self.colors.get(m, 'gray') for m in methods])
        ax1.set_title('Steps to Convergence (80% of Max Performance)')
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Training Steps')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(convergence_steps):
            if v > 0:
                ax1.text(i, v + max(convergence_steps) * 0.01, f'{v:,}', 
                        ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Final Performance
        ax2.bar(range(len(methods)), final_performance, 
               color=[self.colors.get(m, 'gray') for m in methods])
        ax2.set_title('Final Performance (Mean Reward)')
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Mean Reward')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(final_performance):
            ax2.text(i, v + max(final_performance) * 0.01, f'{v:.1f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Training Stability (lower is better)
        ax3.bar(range(len(methods)), stability_scores, 
               color=[self.colors.get(m, 'gray') for m in methods])
        ax3.set_title('Training Stability (Lower = More Stable)')
        ax3.set_xlabel('Methods')
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(stability_scores):
            ax3.text(i, v + max(stability_scores) * 0.01, f'{v:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'convergence_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Convergence analysis plot saved to {self.plots_dir / 'convergence_analysis.png'}")
    
    def generate_summary_report(self, convergence_analysis: Dict):
        """Generate a comprehensive summary report"""
        print("\\n📋 Generating Summary Report...")
        
        report = []
        report.append("# Training Results Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Method comparison table
        report.append("## Method Performance Comparison")
        report.append("")
        report.append("| Method | Convergence Steps | Final Performance | Max Performance | Stability Score |")
        report.append("|--------|------------------|-------------------|-----------------|-----------------|")
        
        # Sort by final performance
        sorted_methods = sorted(convergence_analysis.items(), 
                              key=lambda x: x[1]['final_performance'] or 0, reverse=True)
        
        for method_name, analysis in sorted_methods:
            conv_steps = analysis['convergence_step'] or "N/A"
            final_perf = f"{analysis['final_performance']:.2f}" if analysis['final_performance'] else "N/A"
            max_perf = f"{analysis['max_performance']:.2f}" if analysis['max_performance'] else "N/A"
            stability = f"{analysis['stability_score']:.4f}" if analysis['stability_score'] else "N/A"
            
            report.append(f"| {method_name} | {conv_steps} | {final_perf} | {max_perf} | {stability} |")
        
        report.append("")
        
        # Key findings
        report.append("## Key Findings")
        report.append("")
        
        if sorted_methods:
            best_method = sorted_methods[0]
            report.append(f"**Best Overall Performance:** {best_method[0]}")
            report.append(f"- Final Performance: {best_method[1]['final_performance']:.2f}")
            report.append(f"- Convergence: {best_method[1]['convergence_step']} steps")
            report.append("")
        
        # Save report
        report_text = "\\n".join(report)
        report_path = self.results_dir / "training_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Summary report saved to {report_path}")
        print("\\n" + "="*50)
        print("SUMMARY REPORT")
        print("="*50)
        print(report_text)
        
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete training analysis pipeline"""
        print("🚀 Starting Complete Training Analysis")
        print("=" * 60)
        
        # Step 1: Extract data
        tensorboard_data = self.extract_tensorboard_data()
        eval_data = self.extract_evaluation_data()
        
        if not tensorboard_data and not eval_data:
            print("❌ No training data found! Please run training first.")
            return
        
        # Step 2: Create plots
        if tensorboard_data or eval_data:
            self.plot_cumulative_rewards(tensorboard_data, eval_data)
        
        # Step 3: Analyze convergence
        convergence_analysis = self.analyze_convergence(tensorboard_data, eval_data)
        
        if convergence_analysis:
            self.create_convergence_plot(convergence_analysis)
            self.generate_summary_report(convergence_analysis)
        
        print("\\n🎉 Complete Training Analysis Finished!")
        print(f"📁 All results saved to: {self.results_dir}")
        print(f"🎨 All plots saved to: {self.plots_dir}")
        
        return {
            'tensorboard_data': tensorboard_data,
            'eval_data': eval_data,
            'convergence_analysis': convergence_analysis
        }

def main():
    """Main function to run the training analysis"""
    analyzer = TrainingAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\\n📊 Analysis Complete! Check the results directory for:")
    print("   • training_plots/cumulative_rewards_comparison.png")
    print("   • training_plots/convergence_analysis.png")
    print("   • training_analysis_report.md")
    
    return results

if __name__ == "__main__":
    main()