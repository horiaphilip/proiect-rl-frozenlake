"""
Vizualizare rezultate benchmark multi-seed.
Genereaza grafice cu error bars (mean +/- std).
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def load_latest_results():
    """Incarca cele mai recente rezultate multi-seed."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    pattern = os.path.join(results_dir, "benchmark_multi_seed_*.json")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("Nu s-au gasit rezultate multi-seed!")

    latest_file = max(files, key=os.path.getctime)
    print(f"Se incarca: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)


def create_comparison_plot(data):
    """Creeaza grafic comparativ cu error bars."""
    results = data['results']
    algorithms = list(results.keys())

    # Extrage datele
    success_rates = [results[algo]['success_rate']['mean'] * 100 for algo in algorithms]
    success_stds = [results[algo]['success_rate']['std'] * 100 for algo in algorithms]

    mean_rewards = [results[algo]['mean_reward']['mean'] for algo in algorithms]
    reward_stds = [results[algo]['mean_reward']['std'] for algo in algorithms]

    mean_steps = [results[algo]['mean_steps']['mean'] for algo in algorithms]
    steps_stds = [results[algo]['mean_steps']['std'] for algo in algorithms]

    # Configurare plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    x = np.arange(len(algorithms))

    # Plot 1: Success Rate
    ax1 = axes[0]
    bars1 = ax1.bar(x, success_rates, yerr=success_stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate (mean +/- std)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.set_ylim(0, 120)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Target 100%')
    ax1.legend()

    # Adauga valorile pe bare
    for i, (bar, val, std) in enumerate(zip(bars1, success_rates, success_stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 2: Mean Reward
    ax2 = axes[1]
    bars2 = ax2.bar(x, mean_rewards, yerr=reward_stds, capsize=5, color=colors, alpha=0.8)
    ax2.set_ylabel('Mean Reward', fontsize=12)
    ax2.set_title('Mean Reward (mean +/- std)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target 1.0')
    ax2.legend()

    for i, (bar, val, std) in enumerate(zip(bars2, mean_rewards, reward_stds)):
        y_pos = max(bar.get_height() + std + 0.05, 0.1)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Mean Steps
    ax3 = axes[2]
    bars3 = ax3.bar(x, mean_steps, yerr=steps_stds, capsize=5, color=colors, alpha=0.8)
    ax3.set_ylabel('Mean Steps', fontsize=12)
    ax3.set_title('Mean Steps (mean +/- std)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')

    for i, (bar, val, std) in enumerate(zip(bars3, mean_steps, steps_stds)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def create_stability_plot(data):
    """Creeaza grafic de stabilitate (std deviation comparison)."""
    results = data['results']
    algorithms = list(results.keys())

    # Extrage std-urile (normalizate)
    success_stds = [results[algo]['success_rate']['std'] * 100 for algo in algorithms]
    reward_stds_norm = [results[algo]['mean_reward']['std'] / max(0.01, results[algo]['mean_reward']['mean']) * 100
                        for algo in algorithms]  # Coeficient de variatie
    steps_stds_norm = [results[algo]['mean_steps']['std'] / max(0.01, results[algo]['mean_steps']['mean']) * 100
                       for algo in algorithms]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(algorithms))
    width = 0.25

    bars1 = ax.bar(x - width, success_stds, width, label='Success Rate Std (%)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, reward_stds_norm, width, label='Reward CV (%)', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, steps_stds_norm, width, label='Steps CV (%)', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Variabilitate (%)', fontsize=12)
    ax.set_title('Stabilitatea Algoritmilor (mai mic = mai stabil)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(max(success_stds), max(reward_stds_norm), max(steps_stds_norm)) * 1.2)

    plt.tight_layout()
    return fig


def create_seed_distribution_plot(data):
    """Creeaza grafic cu distributia rezultatelor per seed."""
    results = data['results']
    algorithms = list(results.keys())
    seeds = data['experiment_info']['seeds']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    # Success Rate per seed
    ax1 = axes[0]
    for i, algo in enumerate(algorithms):
        values = [v * 100 for v in results[algo]['success_rate']['values']]
        ax1.scatter([i] * len(values), values, c=colors[i], s=100, alpha=0.7, label=algo)
        ax1.scatter([i], [np.mean(values)], c='black', s=200, marker='_', linewidths=3)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate per Seed', fontweight='bold')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.set_ylim(-5, 110)

    # Mean Reward per seed
    ax2 = axes[1]
    for i, algo in enumerate(algorithms):
        values = results[algo]['mean_reward']['values']
        ax2.scatter([i] * len(values), values, c=colors[i], s=100, alpha=0.7, label=algo)
        ax2.scatter([i], [np.mean(values)], c='black', s=200, marker='_', linewidths=3)
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('Mean Reward per Seed', fontweight='bold')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')

    # Mean Steps per seed
    ax3 = axes[2]
    for i, algo in enumerate(algorithms):
        values = results[algo]['mean_steps']['values']
        ax3.scatter([i] * len(values), values, c=colors[i], s=100, alpha=0.7, label=algo)
        ax3.scatter([i], [np.mean(values)], c='black', s=200, marker='_', linewidths=3)
    ax3.set_ylabel('Mean Steps')
    ax3.set_title('Mean Steps per Seed', fontweight='bold')
    ax3.set_xticks(range(len(algorithms)))
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')

    plt.tight_layout()
    return fig


def print_latex_table(data):
    """Genereaza tabel LaTeX pentru documentatie."""
    results = data['results']
    algorithms = list(results.keys())

    print("\n" + "="*70)
    print("TABEL LATEX")
    print("="*70)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Rezultate Multi-Seed Benchmark (5 seeds)}")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Algorithm} & \\textbf{Success Rate} & \\textbf{Mean Reward} & \\textbf{Mean Steps} \\\\")
    print("\\hline")

    for algo in algorithms:
        sr = results[algo]['success_rate']
        mr = results[algo]['mean_reward']
        ms = results[algo]['mean_steps']
        print(f"{algo} & {sr['mean']*100:.1f}\\% $\\pm$ {sr['std']*100:.1f}\\% & "
              f"{mr['mean']:.2f} $\\pm$ {mr['std']:.2f} & "
              f"{ms['mean']:.1f} $\\pm$ {ms['std']:.1f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    # Incarca rezultatele
    data = load_latest_results()

    print(f"\nExperiment: {data['experiment_info']['type']}")
    print(f"Seeds: {data['experiment_info']['seeds']}")
    print(f"Environment: {data['experiment_info']['environment']}")

    # Creeaza graficele
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    # 1. Grafic comparativ principal
    fig1 = create_comparison_plot(data)
    fig1.savefig(os.path.join(results_dir, "multi_seed_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"\nSalvat: multi_seed_comparison.png")

    # 2. Grafic stabilitate
    fig2 = create_stability_plot(data)
    fig2.savefig(os.path.join(results_dir, "multi_seed_stability.png"), dpi=150, bbox_inches='tight')
    print(f"Salvat: multi_seed_stability.png")

    # 3. Distributia per seed
    fig3 = create_seed_distribution_plot(data)
    fig3.savefig(os.path.join(results_dir, "multi_seed_distribution.png"), dpi=150, bbox_inches='tight')
    print(f"Salvat: multi_seed_distribution.png")

    # 4. Tabel LaTeX
    print_latex_table(data)

    plt.show()
    print("\nVizualizare completa!")


if __name__ == "__main__":
    main()
