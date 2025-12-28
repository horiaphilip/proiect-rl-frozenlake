"""
Script pentru vizualizarea rezultatelor experimentelor.

Generează grafice și tabele comparative pentru Q-Learning, DQN, PPO și PPO+RND.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(results_dir):
    """Încarcă rezultatele din fișierul JSON."""
    results_file = os.path.join(results_dir, 'results.json')
    with open(results_file, 'r') as f:
        return json.load(f)


def smooth_curve(values, weight=0.9):
    """Netezește o curbă folosind exponential moving average."""
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_learning_curves(results, save_dir):
    """
    Plotează curbele de învățare pentru Q-Learning și DQN (pe episoade).
    PPO/PPO+RND sunt pe timesteps -> nu sunt incluse aici ca să nu amestecăm axe diferite.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curbe de Învățare - Comparație Algoritmi', fontsize=16, fontweight='bold')

    algorithms = ['q_learning', 'dqn']
    colors = {'q_learning': 'blue', 'dqn': 'green', 'ppo': 'red', 'ppo_rnd': 'purple'}
    labels = {'q_learning': 'Q-Learning', 'dqn': 'DQN', 'ppo': 'PPO', 'ppo_rnd': 'PPO+RND'}

    # 1. Reward per Episode
    ax = axes[0, 0]
    for algo_name in algorithms:
        if algo_name not in results:
            continue
        algo_results = results[algo_name]
        all_rewards = [run['episode_rewards'] for run in algo_results['runs']]

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        smoothed_rewards = smooth_curve(mean_rewards)
        episodes = range(len(mean_rewards))

        ax.plot(episodes, smoothed_rewards, label=labels[algo_name],
                color=colors[algo_name], linewidth=2)
        ax.fill_between(episodes,
                        np.array(smoothed_rewards) - std_rewards,
                        np.array(smoothed_rewards) + std_rewards,
                        alpha=0.2, color=colors[algo_name])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Steps per Episode
    ax = axes[0, 1]
    for algo_name in algorithms:
        if algo_name not in results:
            continue
        algo_results = results[algo_name]
        all_steps = [run['episode_steps'] for run in algo_results['runs']]

        mean_steps = np.mean(all_steps, axis=0)
        std_steps = np.std(all_steps, axis=0)
        smoothed_steps = smooth_curve(mean_steps)

        episodes = range(len(mean_steps))
        ax.plot(episodes, smoothed_steps, label=labels[algo_name],
                color=colors[algo_name], linewidth=2)
        ax.fill_between(episodes,
                        np.array(smoothed_steps) - std_steps,
                        np.array(smoothed_steps) + std_steps,
                        alpha=0.2, color=colors[algo_name])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Epsilon Decay
    ax = axes[1, 0]
    for algo_name in algorithms:
        if algo_name not in results:
            continue
        algo_results = results[algo_name]
        all_epsilons = [run['epsilons'] for run in algo_results['runs']]
        mean_epsilons = np.mean(all_epsilons, axis=0)

        episodes = range(len(mean_epsilons))
        ax.plot(episodes, mean_epsilons, label=labels[algo_name],
                color=colors[algo_name], linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Epsilon Decay (Exploration Rate)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Loss (doar pentru DQN)
    ax = axes[1, 1]
    if 'dqn' in results:
        dqn_results = results['dqn']
        all_losses = [run['losses'] for run in dqn_results['runs']]

        mean_losses = np.mean(all_losses, axis=0)
        std_losses = np.std(all_losses, axis=0)
        smoothed_losses = smooth_curve(mean_losses, weight=0.95)

        episodes = range(len(mean_losses))
        ax.plot(episodes, smoothed_losses, label='DQN',
                color=colors['dqn'], linewidth=2)
        ax.fill_between(episodes,
                        np.array(smoothed_losses) - std_losses,
                        np.array(smoothed_losses) + std_losses,
                        alpha=0.2, color=colors['dqn'])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('DQN Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Salvat: {save_path}")
    plt.close()


def plot_final_comparison(results, save_dir):
    """
    Plotează comparație finală între algoritmi (include PPO+RND).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Comparație Finală între Algoritmi', fontsize=16, fontweight='bold')

    algorithms = ['q_learning', 'dqn', 'ppo', 'ppo_rnd']
    labels = {'q_learning': 'Q-Learning', 'dqn': 'DQN', 'ppo': 'PPO', 'ppo_rnd': 'PPO+RND'}
    colors = {'q_learning': 'blue', 'dqn': 'green', 'ppo': 'red', 'ppo_rnd': 'purple'}

    algo_names = []
    mean_rewards = []
    std_rewards = []
    success_rates = []
    std_success_rates = []
    bar_colors = []

    for algo_name in algorithms:
        if algo_name not in results:
            continue

        algo_results = results[algo_name]
        final_evals = [run['final_eval'] for run in algo_results['runs']]

        rewards = [e['mean_reward'] for e in final_evals]
        successes = [e['success_rate'] for e in final_evals]

        algo_names.append(labels[algo_name])
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))
        success_rates.append(np.mean(successes))
        std_success_rates.append(np.std(successes))
        bar_colors.append(colors[algo_name])

    # 1. Mean Reward Comparison
    ax = axes[0]
    x_pos = np.arange(len(algo_names))
    bars = ax.bar(x_pos, mean_rewards, yerr=std_rewards,
                  capsize=5, alpha=0.7, color=bar_colors)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Reward (Final Evaluation)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_names)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val, std in zip(bars, mean_rewards, std_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9)

    # 2. Success Rate Comparison
    ax = axes[1]
    bars = ax.bar(x_pos, success_rates, yerr=std_success_rates,
                  capsize=5, alpha=0.7, color=bar_colors)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate (Final Evaluation)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_names)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val, std in zip(bars, success_rates, std_success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}±{std:.1%}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'final_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Salvat: {save_path}")
    plt.close()


def plot_convergence_analysis(results, save_dir):
    """
    Analizează convergența (doar Q-Learning și DQN, că sunt pe episoade).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Analiză Convergență', fontsize=16, fontweight='bold')

    algorithms = ['q_learning', 'dqn']
    labels = {'q_learning': 'Q-Learning', 'dqn': 'DQN'}
    colors = {'q_learning': 'blue', 'dqn': 'green'}

    # 1. Rolling Average Reward
    ax = axes[0]
    window_size = 50

    for algo_name in algorithms:
        if algo_name not in results:
            continue
        algo_results = results[algo_name]
        all_rewards = [run['episode_rewards'] for run in algo_results['runs']]

        rolling_avgs = []
        for rewards in all_rewards:
            rolling_avg = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
            rolling_avgs.append(rolling_avg.values)

        mean_rolling = np.mean(rolling_avgs, axis=0)
        std_rolling = np.std(rolling_avgs, axis=0)

        episodes = range(len(mean_rolling))
        ax.plot(episodes, mean_rolling, label=labels[algo_name],
                color=colors[algo_name], linewidth=2)
        ax.fill_between(episodes,
                        mean_rolling - std_rolling,
                        mean_rolling + std_rolling,
                        alpha=0.2, color=colors[algo_name])

    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Rolling Average Reward (window={window_size})')
    ax.set_title('Convergență (Rolling Average)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Rolling Variance
    ax = axes[1]
    window_size = 50

    for algo_name in algorithms:
        if algo_name not in results:
            continue
        algo_results = results[algo_name]
        all_rewards = [run['episode_rewards'] for run in algo_results['runs']]

        rolling_vars = []
        for rewards in all_rewards:
            rolling_var = pd.Series(rewards).rolling(window=window_size, min_periods=1).var()
            rolling_vars.append(rolling_var.values)

        mean_var = np.mean(rolling_vars, axis=0)

        episodes = range(len(mean_var))
        ax.plot(episodes, mean_var, label=labels[algo_name],
                color=colors[algo_name], linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Rolling Variance (window={window_size})')
    ax.set_title('Stabilitate (Variance în Reward)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'convergence_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Salvat: {save_path}")
    plt.close()


def create_comparison_table(results, save_dir):
    """
    Creează tabel comparativ cu metrici (include PPO+RND).
    """
    algorithms = ['q_learning', 'dqn', 'ppo', 'ppo_rnd']
    labels = {'q_learning': 'Q-Learning', 'dqn': 'DQN', 'ppo': 'PPO', 'ppo_rnd': 'PPO+RND'}

    table_data = []

    for algo_name in algorithms:
        if algo_name not in results:
            continue

        algo_results = results[algo_name]
        final_evals = [run['final_eval'] for run in algo_results['runs']]

        mean_rewards = [e['mean_reward'] for e in final_evals]
        std_rewards_list = [e['std_reward'] for e in final_evals]
        mean_steps = [e['mean_steps'] for e in final_evals]
        success_rates = [e['success_rate'] for e in final_evals]

        table_data.append({
            'Algoritm': labels[algo_name],
            'Mean Reward': f"{np.mean(mean_rewards):.4f} ± {np.std(mean_rewards):.4f}",
            'Std Reward': f"{np.mean(std_rewards_list):.4f}",
            'Mean Steps': f"{np.mean(mean_steps):.2f} ± {np.std(mean_steps):.2f}",
            'Success Rate': f"{np.mean(success_rates):.2%} ± {np.std(success_rates):.2%}"
        })

    df = pd.DataFrame(table_data)

    csv_path = os.path.join(save_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSalvat tabel: {csv_path}")

    print("\n" + "="*80)
    print("TABEL COMPARATIV")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    return df


def main():
    """Funcția principală pentru vizualizare."""
    print("="*60)
    print("VIZUALIZARE REZULTATE EXPERIMENTE")
    print("="*60)

    results_base = "../results"
    experiment_dirs = sorted([d for d in Path(results_base).iterdir() if d.is_dir()],
                             key=lambda x: x.stat().st_mtime, reverse=True)

    if not experiment_dirs:
        print("Nu s-au găsit rezultate. Rulați mai întâi experiments/run_experiments.py")
        return

    results_dir = str(experiment_dirs[0])
    print(f"\nÎncărcare rezultate din: {results_dir}")

    results = load_results(results_dir)

    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("\nGenerare grafice...")

    print("\n1. Curbe de învățare...")
    plot_learning_curves(results, plots_dir)

    print("2. Comparație finală...")
    plot_final_comparison(results, plots_dir)

    print("3. Analiză convergență...")
    plot_convergence_analysis(results, plots_dir)

    print("4. Tabel comparativ...")
    create_comparison_table(results, plots_dir)

    print(f"\n{'='*60}")
    print("VIZUALIZARE COMPLETĂ!")
    print(f"Grafice salvate în: {plots_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
