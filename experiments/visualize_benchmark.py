"""
Script pentru vizualizarea rezultatelor benchmark-ului pe EasyFrozenLake.

Genereaza grafice comparative pentru toti cei 5 agenti.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_latest_benchmark():
    """Incarca cel mai recent fisier de benchmark."""
    results_dir = Path("../results")

    # Gaseste toate fisierele benchmark
    benchmark_files = list(results_dir.glob("benchmark_easy_*.json"))

    if not benchmark_files:
        print("No benchmark files found!")
        return None

    # Cel mai recent
    latest_file = max(benchmark_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)


def smooth_curve(values, weight=0.9):
    """Exponential moving average pentru netezirea curbelor."""
    smoothed = []
    last = values[0] if values else 0
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_comparison_bars(results):
    """Grafice bar pentru comparatie."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Benchmark Comparison - EasyFrozenLake (4x4)',
                 fontsize=16, fontweight='bold')

    algorithms = [r['algorithm'] for r in results]
    success_rates = [r['eval']['success_rate'] * 100 for r in results]
    mean_rewards = [r['eval']['mean_reward'] for r in results]
    mean_steps = [r['eval']['mean_steps'] for r in results]

    # Colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # Success Rate
    ax1 = axes[0]
    bars1 = ax1.bar(algorithms, success_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Success Rate', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 110])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')

    # Add values on bars
    for bar, val in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')

    # Mean Reward
    ax2 = axes[1]
    bars2 = ax2.bar(algorithms, mean_rewards, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Reward', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add values on bars
    for bar, val in zip(bars2, mean_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Mean Steps
    ax3 = axes[2]
    bars3 = ax3.bar(algorithms, mean_steps, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Mean Steps', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Steps to Goal', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add values on bars
    for bar, val in zip(bars3, mean_steps):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # Rotate x labels
    for ax in axes:
        ax.set_xticklabels(algorithms, rotation=15, ha='right')

    plt.tight_layout()

    # Save
    output_file = "../results/benchmark_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.show()


def plot_learning_curves(results):
    """Curbe de invatare pentru algoritmii episodici."""
    # Filter algoritmi cu training_rewards
    episodic_agents = [r for r in results if 'training_rewards' in r]

    if not episodic_agents:
        print("No episodic training data found.")
        return

    fig, axes = plt.subplots(1, len(episodic_agents), figsize=(6*len(episodic_agents), 5))

    if len(episodic_agents) == 1:
        axes = [axes]

    fig.suptitle('Learning Curves - Episode Rewards', fontsize=16, fontweight='bold')

    colors = {'Q-Learning': '#3498db', 'DQN': '#e74c3c', 'DQN+PER': '#2ecc71'}

    for idx, result in enumerate(episodic_agents):
        ax = axes[idx]
        algo = result['algorithm']
        rewards = result['training_rewards']

        # Plot raw
        episodes = list(range(len(rewards)))
        ax.plot(episodes, rewards, alpha=0.3, color=colors.get(algo, 'gray'), linewidth=0.5)

        # Plot smoothed
        smoothed = smooth_curve(rewards, weight=0.95)
        ax.plot(episodes, smoothed, color=colors.get(algo, 'gray'),
                linewidth=2, label=f'{algo} (smoothed)')

        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title(f'{algo}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend()

    plt.tight_layout()

    # Save
    output_file = "../results/learning_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.show()


def plot_efficiency_scatter(results):
    """Scatter plot: Success Rate vs Mean Steps."""
    fig, ax = plt.subplots(figsize=(10, 8))

    algorithms = [r['algorithm'] for r in results]
    success_rates = [r['eval']['success_rate'] * 100 for r in results]
    mean_steps = [r['eval']['mean_steps'] for r in results]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # Scatter
    for i, (algo, sr, ms) in enumerate(zip(algorithms, success_rates, mean_steps)):
        ax.scatter(ms, sr, s=500, color=colors[i], alpha=0.7,
                  edgecolors='black', linewidth=2, label=algo)

        # Annotate
        ax.annotate(algo, (ms, sr), fontsize=11, fontweight='bold',
                   ha='center', va='center')

    ax.set_xlabel('Mean Steps to Goal', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Efficiency Analysis: Success Rate vs Steps',
                 fontsize=16, fontweight='bold')

    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([-5, 110])

    # Add quadrants
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.3)
    ax.axvline(x=20, color='red', linestyle='--', linewidth=1, alpha=0.3)

    # Annotations for quadrants
    ax.text(5, 95, 'BEST\n(High Success, Low Steps)',
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    # Save
    output_file = "../results/efficiency_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.show()


def plot_winner_highlight(results):
    """Highlight pentru castigator."""
    fig, ax = plt.subplots(figsize=(12, 6))

    algorithms = [r['algorithm'] for r in results]
    success_rates = [r['eval']['success_rate'] * 100 for r in results]
    mean_steps = [r['eval']['mean_steps'] for r in results]

    # Calculate score: success_rate / mean_steps (higher is better)
    scores = [sr / ms if ms > 0 else 0 for sr, ms in zip(success_rates, mean_steps)]

    # Find winner
    winner_idx = scores.index(max(scores))

    # Colors (winner in gold)
    colors = ['#3498db' if i != winner_idx else '#FFD700'
              for i in range(len(algorithms))]

    # Bar chart with combined metric
    bars = ax.bar(algorithms, scores, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)

    ax.set_ylabel('Efficiency Score\n(Success Rate / Mean Steps)',
                  fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Ranking', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add values and crown for winner
    for i, (bar, val, algo) in enumerate(zip(bars, scores, algorithms)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        if i == winner_idx:
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   '👑', ha='center', va='bottom', fontsize=30)

    ax.set_xticklabels(algorithms, rotation=15, ha='right')

    # Winner box
    winner = algorithms[winner_idx]
    winner_sr = success_rates[winner_idx]
    winner_steps = mean_steps[winner_idx]

    textstr = f'WINNER: {winner}\n'
    textstr += f'Success Rate: {winner_sr:.0f}%\n'
    textstr += f'Mean Steps: {winner_steps:.2f}\n'
    textstr += f'Efficiency: {scores[winner_idx]:.2f}'

    props = dict(boxstyle='round', facecolor='gold', alpha=0.8, edgecolor='black', linewidth=2)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = "../results/winner_ranking.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.show()


def main():
    print("="*60)
    print("BENCHMARK VISUALIZATION")
    print("="*60)

    # Load results
    results = load_latest_benchmark()

    if not results:
        return

    print(f"\nLoaded {len(results)} algorithms")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10

    # Generate plots
    print("\nGenerating plots...")

    plot_comparison_bars(results)
    plot_learning_curves(results)
    plot_efficiency_scatter(results)
    plot_winner_highlight(results)

    print("\n" + "="*60)
    print("ALL PLOTS GENERATED!")
    print("="*60)
    print("\nCheck ../results/ folder for:")
    print("  - benchmark_comparison.png")
    print("  - learning_curves.png")
    print("  - efficiency_scatter.png")
    print("  - winner_ranking.png")


if __name__ == "__main__":
    main()
