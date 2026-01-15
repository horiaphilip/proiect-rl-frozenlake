import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional




AGENT_COLORS = {
    'Q-Learning': '#3498db',
    'DQN': '#e74c3c',
    'DQN-PER': '#2ecc71',
    'PPO': '#f39c12',
    'PPO-RND': '#9b59b6',
}

ENV_COLORS = {
    'easy': '#27ae60',
    'medium': '#f1c40f',
    'hard': '#c0392b',
}

ENV_DISPLAY_NAMES = {
    'easy': 'Easy (4x4)',
    'medium': 'Medium (8x8)',
    'hard': 'Hard (8x8 Dynamic)',
}


def load_results(filepath: Optional[str] = None) -> Dict:
    results_dir = Path("../results")

    if filepath:
        with open(filepath, 'r') as f:
            return json.load(f)

    files = list(results_dir.glob("comparative_study_*.json"))

    if not files:
        raise FileNotFoundError("Nu am gasit fisiere comparative_study_*.json in results/")

    latest = max(files, key=lambda p: p.stat().st_mtime)
    print(f"Incarc: {latest}")

    with open(latest, 'r') as f:
        return json.load(f)


def smooth_curve(values: List[float], weight: float = 0.9) -> List[float]:
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed



def plot_heatmap_success_rate(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    matrix = np.zeros((len(agents), len(envs)))

    for i, agent in enumerate(agents):
        for j, env in enumerate(envs):
            agg = results['results'][env][agent]['aggregated']
            if 'error' not in agg:
                matrix[i, j] = agg['success_rate']['mean'] * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=[ENV_DISPLAY_NAMES.get(e, e) for e in envs],
        yticklabels=agents,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Success Rate (%)'},
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'}
    )

    ax.set_title('Success Rate: Agenti vs Environment-uri', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Environment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Agent', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_success_rate.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: heatmap_success_rate.png")
    plt.close()


def plot_bars_per_environment(results: Dict, output_dir: str) -> None:

    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(len(envs), 3, figsize=(18, 5 * len(envs)))

    for env_idx, env in enumerate(envs):
        env_results = results['results'][env]

        success_rates = []
        mean_rewards = []
        mean_steps = []
        errors_sr = []
        errors_mr = []
        errors_ms = []

        for agent in agents:
            agg = env_results[agent]['aggregated']
            if 'error' in agg:
                success_rates.append(0)
                mean_rewards.append(0)
                mean_steps.append(0)
                errors_sr.append(0)
                errors_mr.append(0)
                errors_ms.append(0)
            else:
                success_rates.append(agg['success_rate']['mean'] * 100)
                mean_rewards.append(agg['mean_reward']['mean'])
                mean_steps.append(agg['mean_steps']['mean'])
                errors_sr.append(agg['success_rate']['std'] * 100)
                errors_mr.append(agg['mean_reward']['std'])
                errors_ms.append(agg['mean_steps']['std'])

        colors = [AGENT_COLORS.get(a, 'gray') for a in agents]
        x = np.arange(len(agents))

        ax1 = axes[env_idx, 0] if len(envs) > 1 else axes[0]
        bars1 = ax1.bar(x, success_rates, color=colors, alpha=0.8, edgecolor='black', yerr=errors_sr, capsize=5)
        ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Success Rate', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, rotation=30, ha='right')
        ax1.set_ylim([0, 110])
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5)
        ax1.grid(axis='y', alpha=0.3)

        ax2 = axes[env_idx, 1] if len(envs) > 1 else axes[1]
        bars2 = ax2.bar(x, mean_rewards, color=colors, alpha=0.8, edgecolor='black', yerr=errors_mr, capsize=5)
        ax2.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
        ax2.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Mean Reward', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(agents, rotation=30, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        ax3 = axes[env_idx, 2] if len(envs) > 1 else axes[2]
        bars3 = ax3.bar(x, mean_steps, color=colors, alpha=0.8, edgecolor='black', yerr=errors_ms, capsize=5)
        ax3.set_ylabel('Mean Steps', fontsize=11, fontweight='bold')
        ax3.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Mean Steps', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(agents, rotation=30, ha='right')
        ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('Comparatie Metrici per Environment', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bars_per_environment.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: bars_per_environment.png")
    plt.close()


def plot_boxplots_variance(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 6))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        data = []
        labels = []

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r]

            if valid_runs:
                sr_values = [r['eval']['success_rate'] * 100 for r in valid_runs]
                data.append(sr_values)
                labels.append(agent)

        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

            for patch, agent in zip(bp['boxes'], labels):
                patch.set_facecolor(AGENT_COLORS.get(agent, 'gray'))
                patch.set_alpha(0.7)

        ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)}', fontsize=13, fontweight='bold')
        ax.set_ylim([-5, 110])
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

    plt.suptitle('Varianta Success Rate intre Seed-uri', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplots_variance.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: boxplots_variance.png")
    plt.close()


def plot_learning_curves(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    episodic_agents = ['Q-Learning', 'DQN', 'DQN-PER']
    episodic_agents = [a for a in episodic_agents if a in agents]

    if not episodic_agents:
        print("  Skip: Nu exista agenti episodici pentru learning curves")
        return

    fig, axes = plt.subplots(len(envs), len(episodic_agents), figsize=(5 * len(episodic_agents), 4 * len(envs)))

    if len(envs) == 1 and len(episodic_agents) == 1:
        axes = [[axes]]
    elif len(envs) == 1:
        axes = [axes]
    elif len(episodic_agents) == 1:
        axes = [[ax] for ax in axes]

    for env_idx, env in enumerate(envs):
        env_results = results['results'][env]

        for agent_idx, agent in enumerate(episodic_agents):
            ax = axes[env_idx][agent_idx]

            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and 'training_rewards' in r]

            if not valid_runs:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{agent}')
                continue

            for run in valid_runs:
                rewards = run['training_rewards']
                episodes = range(len(rewards))
                ax.plot(episodes, rewards, alpha=0.2, color=AGENT_COLORS.get(agent, 'gray'), linewidth=0.5)

            all_rewards = [r['training_rewards'] for r in valid_runs]
            min_len = min(len(r) for r in all_rewards)
            all_rewards = [r[:min_len] for r in all_rewards]
            mean_rewards = np.mean(all_rewards, axis=0)
            smoothed = smooth_curve(list(mean_rewards), weight=0.95)

            ax.plot(range(len(smoothed)), smoothed,
                   color=AGENT_COLORS.get(agent, 'gray'),
                   linewidth=2.5, label=f'{agent} (mean)')

            ax.set_xlabel('Episode', fontsize=10)
            ax.set_ylabel('Reward', fontsize=10)
            ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - {agent}', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)

    plt.suptitle('Curbe de Invatare (Agenti Episodici)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: learning_curves.png")
    plt.close()


def plot_radar_chart(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']

    env = 'hard'
    if env not in results['results']:
        env = results['metadata']['environments'][-1]

    env_results = results['results'][env]

    metrics = ['Success Rate', 'Mean Reward', 'Efficiency', 'Consistency']
    n_metrics = len(metrics)

    values_per_agent = {}

    for agent in agents:
        agg = env_results[agent]['aggregated']
        if 'error' in agg:
            continue

        sr = agg['success_rate']['mean']
        mr = agg['mean_reward']['mean']
        ms = agg['mean_steps']['mean']
        sr_std = agg['success_rate']['std']

        efficiency = sr / max(ms, 1) * 10
        consistency = 1 - sr_std

        values_per_agent[agent] = [
            sr,
            max(0, mr + 1) / 2,
            min(efficiency, 1),
            consistency
        ]

    if not values_per_agent:
        print("  Skip: Nu exista date pentru radar chart")
        return


    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for agent, values in values_per_agent.items():
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=2,
                label=agent, color=AGENT_COLORS.get(agent, 'gray'))
        ax.fill(angles, values_plot, alpha=0.15, color=AGENT_COLORS.get(agent, 'gray'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title(f'Comparatie Multi-dimensionala ({ENV_DISPLAY_NAMES.get(env, env)})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_chart.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: radar_chart.png")
    plt.close()


def plot_ranking_table(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    agent_scores = {}

    for agent in agents:
        scores = []
        for env in envs:
            agg = results['results'][env][agent]['aggregated']
            if 'error' not in agg:
                scores.append(agg['success_rate']['mean'])
        if scores:
            agent_scores[agent] = np.mean(scores)

    sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    agents_sorted = [a[0] for a in sorted_agents]
    scores_sorted = [a[1] * 100 for a in sorted_agents]
    colors = [AGENT_COLORS.get(a, 'gray') for a in agents_sorted]

    bars = ax.barh(agents_sorted, scores_sorted, color=colors, alpha=0.8, edgecolor='black')

    for bar, score in zip(bars, scores_sorted):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}%', va='center', fontweight='bold', fontsize=12)

    medals = ['1st', '2nd', '3rd', '4th', '5th']
    medal_colors = ['gold', 'silver', '#CD7F32', 'gray', 'gray']

    for idx, (bar, medal, mcolor) in enumerate(zip(bars, medals, medal_colors)):
        if idx < 3:
            ax.text(2, bar.get_y() + bar.get_height()/2,
                   medal, va='center', ha='left',
                   fontsize=14, fontweight='bold',
                   color=mcolor,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Average Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ranking Final: Media Success Rate pe Toate Environment-urile',
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 110])
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ranking_final.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: ranking_final.png")
    plt.close()


def plot_scalability_analysis(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    env_order = ['easy', 'medium', 'hard']
    envs_ordered = [e for e in env_order if e in envs]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(envs_ordered))
    width = 0.15
    n_agents = len(agents)

    for idx, agent in enumerate(agents):
        success_rates = []
        errors = []

        for env in envs_ordered:
            agg = results['results'][env][agent]['aggregated']
            if 'error' in agg:
                success_rates.append(0)
                errors.append(0)
            else:
                success_rates.append(agg['success_rate']['mean'] * 100)
                errors.append(agg['success_rate']['std'] * 100)

        offset = (idx - n_agents/2 + 0.5) * width
        bars = ax.bar(x + offset, success_rates, width,
                     label=agent, color=AGENT_COLORS.get(agent, 'gray'),
                     alpha=0.8, edgecolor='black', yerr=errors, capsize=3)

    ax.set_xlabel('Environment Difficulty', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Analiza Scalabilitatii: Performanta vs Dificultate',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_DISPLAY_NAMES.get(e, e) for e in envs_ordered])
    ax.set_ylim([0, 110])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    ax.annotate('', xy=(len(envs_ordered)-0.5, -12), xytext=(-0.5, -12),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               annotation_clip=False)
    ax.text(len(envs_ordered)/2 - 0.5, -18, 'Dificultate Crescuta',
           ha='center', fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/scalability_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: scalability_analysis.png")
    plt.close()


def generate_summary_figure(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    matrix = np.zeros((len(agents), len(envs)))
    for i, agent in enumerate(agents):
        for j, env in enumerate(envs):
            agg = results['results'][env][agent]['aggregated']
            if 'error' not in agg:
                matrix[i, j] = agg['success_rate']['mean'] * 100

    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='RdYlGn',
               xticklabels=[ENV_DISPLAY_NAMES.get(e, e) for e in envs],
               yticklabels=agents, vmin=0, vmax=100, ax=ax1,
               cbar_kws={'label': 'Success %'}, annot_kws={'size': 12, 'weight': 'bold'})
    ax1.set_title('Success Rate Matrix', fontsize=14, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 2])
    agent_avg = {}
    for agent in agents:
        scores = []
        for env in envs:
            agg = results['results'][env][agent]['aggregated']
            if 'error' not in agg:
                scores.append(agg['success_rate']['mean'])
        if scores:
            agent_avg[agent] = np.mean(scores) * 100

    sorted_agents = sorted(agent_avg.items(), key=lambda x: x[1], reverse=True)
    y_pos = range(len(sorted_agents))
    bars = ax2.barh(y_pos, [s[1] for s in sorted_agents],
                   color=[AGENT_COLORS.get(s[0], 'gray') for s in sorted_agents],
                   alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([s[0] for s in sorted_agents])
    ax2.set_xlabel('Avg Success Rate (%)')
    ax2.set_title('Overall Ranking', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 110])
    ax2.invert_yaxis()

    ax3 = fig.add_subplot(gs[1, 0])
    env_order = ['easy', 'medium', 'hard']
    envs_ordered = [e for e in env_order if e in envs]
    for agent in agents:
        sr_per_env = []
        for env in envs_ordered:
            agg = results['results'][env][agent]['aggregated']
            sr = agg['success_rate']['mean'] * 100 if 'error' not in agg else 0
            sr_per_env.append(sr)
        ax3.plot(envs_ordered, sr_per_env, 'o-', label=agent,
                color=AGENT_COLORS.get(agent, 'gray'), linewidth=2, markersize=8)
    ax3.set_xlabel('Environment')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Scalability', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 110])

    ax4 = fig.add_subplot(gs[1, 1])
    hard_env = 'hard' if 'hard' in envs else envs[-1]
    data_box = []
    labels_box = []
    for agent in agents:
        runs = results['results'][hard_env][agent]['runs']
        valid = [r['eval']['success_rate'] * 100 for r in runs if 'error' not in r]
        if valid:
            data_box.append(valid)
            labels_box.append(agent)
    if data_box:
        bp = ax4.boxplot(data_box, tick_labels=labels_box, patch_artist=True)
        for patch, agent in zip(bp['boxes'], labels_box):
            patch.set_facecolor(AGENT_COLORS.get(agent, 'gray'))
            patch.set_alpha(0.7)
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title(f'Variance ({ENV_DISPLAY_NAMES.get(hard_env, hard_env)})',
                  fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=30)
    ax4.set_ylim([-5, 110])
    ax4.grid(axis='y', alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    table_data = []
    headers = ['Agent', 'Easy', 'Medium', 'Hard', 'Avg']
    table_data.append(headers)

    for agent in agents:
        row = [agent]
        scores = []
        for env in ['easy', 'medium', 'hard']:
            if env in envs:
                agg = results['results'][env][agent]['aggregated']
                if 'error' not in agg:
                    sr = agg['success_rate']['mean'] * 100
                    row.append(f'{sr:.0f}%')
                    scores.append(sr)
                else:
                    row.append('-')
            else:
                row.append('-')
        avg = np.mean(scores) if scores else 0
        row.append(f'{avg:.0f}%')
        table_data.append(row)

    table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4a90d9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax5.set_title('Summary Table', fontsize=14, fontweight='bold', pad=20)

    fig.suptitle('STUDIU COMPARATIV: 5 Agenti x 3 Environment-uri',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(f"{output_dir}/summary_figure.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: summary_figure.png")
    plt.close()


def plot_loss_curves(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    loss_agents = ['DQN', 'DQN-PER']
    loss_agents = [a for a in loss_agents if a in agents]

    if not loss_agents:
        print("  Skip: Nu exista agenti cu loss")
        return

    fig, axes = plt.subplots(len(envs), len(loss_agents), figsize=(6 * len(loss_agents), 4 * len(envs)))

    if len(envs) == 1 and len(loss_agents) == 1:
        axes = [[axes]]
    elif len(envs) == 1:
        axes = [axes]
    elif len(loss_agents) == 1:
        axes = [[ax] for ax in axes]

    for env_idx, env in enumerate(envs):
        env_results = results['results'][env]

        for agent_idx, agent in enumerate(loss_agents):
            ax = axes[env_idx][agent_idx]

            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('losses')]

            if not valid_runs:
                ax.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{agent}')
                continue

            for run in valid_runs:
                losses = run['losses']
                ax.plot(range(len(losses)), losses, alpha=0.3,
                       color=AGENT_COLORS.get(agent, 'gray'), linewidth=0.5)

            all_losses = [r['losses'] for r in valid_runs]
            min_len = min(len(l) for l in all_losses)
            all_losses = [l[:min_len] for l in all_losses]
            mean_losses = np.mean(all_losses, axis=0)
            smoothed = smooth_curve(list(mean_losses), weight=0.95)

            ax.plot(range(len(smoothed)), smoothed,
                   color=AGENT_COLORS.get(agent, 'gray'),
                   linewidth=2.5, label=f'{agent} (mean)')

            ax.set_xlabel('Episode', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - {agent} Loss', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Curbe de Loss (DQN Agents)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curves.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: loss_curves.png")
    plt.close()


def plot_q_values_evolution(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    q_agents = ['Q-Learning', 'DQN', 'DQN-PER']
    q_agents = [a for a in q_agents if a in agents]

    if not q_agents:
        print("  Skip: Nu exista agenti cu Q-values")
        return

    fig, axes = plt.subplots(len(envs), 2, figsize=(14, 4 * len(envs)))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        env_results = results['results'][env]

        ax_mean = axes[env_idx][0]
        ax_max = axes[env_idx][1]

        for agent in q_agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('q_values_mean')]

            if not valid_runs:
                continue

            all_q_mean = [r['q_values_mean'] for r in valid_runs]
            all_q_max = [r['q_values_max'] for r in valid_runs]

            min_len = min(len(q) for q in all_q_mean)
            all_q_mean = [q[:min_len] for q in all_q_mean]
            all_q_max = [q[:min_len] for q in all_q_max]

            mean_q = np.mean(all_q_mean, axis=0)
            max_q = np.mean(all_q_max, axis=0)

            x_vals = np.linspace(0, 100, len(mean_q))

            ax_mean.plot(x_vals, mean_q, 'o-', label=agent,
                        color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)
            ax_max.plot(x_vals, max_q, 'o-', label=agent,
                       color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

        ax_mean.set_xlabel('Training Progress (%)', fontsize=10)
        ax_mean.set_ylabel('Mean Q-value', fontsize=10)
        ax_mean.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Mean Q-values', fontsize=11, fontweight='bold')
        ax_mean.legend(loc='best', fontsize=8)
        ax_mean.grid(alpha=0.3)

        ax_max.set_xlabel('Training Progress (%)', fontsize=10)
        ax_max.set_ylabel('Max Q-value', fontsize=10)
        ax_max.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Max Q-values', fontsize=11, fontweight='bold')
        ax_max.legend(loc='best', fontsize=8)
        ax_max.grid(alpha=0.3)

    plt.suptitle('Evolutia Q-values in Timpul Antrenarii', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/q_values_evolution.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: q_values_evolution.png")
    plt.close()


def plot_epsilon_decay(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    epsilon_agents = ['Q-Learning', 'DQN', 'DQN-PER']
    epsilon_agents = [a for a in epsilon_agents if a in agents]

    if not epsilon_agents:
        print("  Skip: Nu exista agenti cu epsilon")
        return

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        for agent in epsilon_agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('epsilons')]

            if not valid_runs:
                continue

            epsilons = valid_runs[0]['epsilons']
            ax.plot(range(len(epsilons)), epsilons, '-', label=agent,
                   color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Epsilon', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Epsilon Decay', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.suptitle('Decaderea Epsilon (Exploration Rate)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epsilon_decay.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: epsilon_decay.png")
    plt.close()


def plot_td_errors(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    td_agents = ['Q-Learning', 'DQN', 'DQN-PER']
    td_agents = [a for a in td_agents if a in agents]

    if not td_agents:
        print("  Skip: Nu exista agenti cu TD errors")
        return

    fig, axes = plt.subplots(len(envs), 1, figsize=(14, 4 * len(envs)))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        has_data = False

        for agent in td_agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('td_errors')]

            if not valid_runs:
                continue

            has_data = True

            all_td = [r['td_errors'] for r in valid_runs]
            min_len = min(len(t) for t in all_td)
            all_td = [t[:min_len] for t in all_td]
            mean_td = np.mean(all_td, axis=0)
            smoothed = smooth_curve(list(mean_td), weight=0.95)

            ax.plot(range(len(smoothed)), smoothed,
                   color=AGENT_COLORS.get(agent, 'gray'),
                   linewidth=2, label=agent)

        if not has_data:
            ax.text(0.5, 0.5, 'No TD error data', ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('TD Error', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - TD Errors', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('TD Errors (Q-Learning, DQN, DQN-PER)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/td_errors.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: td_errors.png")
    plt.close()


def plot_training_stability(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(len(envs), 1, figsize=(14, 4 * len(envs)))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('training_successes')]

            if not valid_runs:
                continue

            all_successes = [r['training_successes'] for r in valid_runs]
            min_len = min(len(s) for s in all_successes)
            all_successes = [s[:min_len] for s in all_successes]
            mean_successes = np.mean(all_successes, axis=0)


            window = min(50, len(mean_successes) // 5)
            if window > 1:
                running_sr = np.convolve(mean_successes, np.ones(window)/window, mode='valid') * 100
                x_vals = range(window-1, len(mean_successes))
                ax.plot(x_vals, running_sr, '-', label=agent,
                       color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Running Success Rate (%)', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Training Stability', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 105])

    plt.suptitle('Stabilitatea Training-ului (Running Success Rate)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_stability.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: training_stability.png")
    plt.close()


def generate_metrics_table(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    lines = []
    lines.append("=" * 100)
    lines.append("TABEL COMPLET METRICI - STUDIU COMPARATIV")
    lines.append("=" * 100)
    lines.append("")

    for env in envs:
        lines.append(f"\n{'='*80}")
        lines.append(f"Environment: {ENV_DISPLAY_NAMES.get(env, env)}")
        lines.append(f"{'='*80}")

        header = f"{'Agent':<12} | {'Success Rate':<18} | {'Mean Reward':<18} | {'Mean Steps':<15} | {'Std Steps':<10}"
        lines.append(header)
        lines.append("-" * len(header))

        for agent in agents:
            agg = results['results'][env][agent]['aggregated']
            if 'error' in agg:
                lines.append(f"{agent:<12} | {'ERROR':<18} | {'-':<18} | {'-':<15} | {'-':<10}")
            else:
                sr = f"{agg['success_rate']['mean']*100:.1f}% ± {agg['success_rate']['std']*100:.1f}%"
                mr = f"{agg['mean_reward']['mean']:.4f} ± {agg['mean_reward']['std']:.4f}"
                ms = f"{agg['mean_steps']['mean']:.1f}"
                ss = f"{agg['mean_steps']['std']:.1f}"
                lines.append(f"{agent:<12} | {sr:<18} | {mr:<18} | {ms:<15} | {ss:<10}")

    lines.append("\n")
    lines.append("=" * 100)
    lines.append("RANKING FINAL (Media Success Rate pe toate environment-urile)")
    lines.append("=" * 100)

    agent_scores = {}
    for agent in agents:
        scores = []
        for env in envs:
            agg = results['results'][env][agent]['aggregated']
            if 'error' not in agg:
                scores.append(agg['success_rate']['mean'])
        if scores:
            agent_scores[agent] = np.mean(scores)

    sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

    for idx, (agent, score) in enumerate(sorted_agents):
        medal = ["1st", "2nd", "3rd", "4th", "5th"][idx] if idx < 5 else f"{idx+1}th"
        lines.append(f"  {medal}: {agent} - {score*100:.1f}%")

    with open(f"{output_dir}/metrics_table.txt", 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Salvat: metrics_table.txt")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    table_data = []
    headers = ['Agent'] + [ENV_DISPLAY_NAMES.get(e, e) for e in envs] + ['Average']

    for agent in agents:
        row = [agent]
        scores = []
        for env in envs:
            agg = results['results'][env][agent]['aggregated']
            if 'error' not in agg:
                sr = agg['success_rate']['mean'] * 100
                std = agg['success_rate']['std'] * 100
                row.append(f'{sr:.1f}% ± {std:.1f}%')
                scores.append(sr)
            else:
                row.append('ERROR')
        avg = np.mean(scores) if scores else 0
        row.append(f'{avg:.1f}%')
        table_data.append(row)

    table_data.sort(key=lambda x: float(x[-1].replace('%', '')), reverse=True)

    table = ax.table(cellText=[headers] + table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data) + 1):
        avg_val = float(table_data[i-1][-1].replace('%', ''))
        r = max(0, min(1, 1 - avg_val/100))
        g = max(0, min(1, avg_val/100))
        for j in range(len(headers)):
            table[(i, j)].set_facecolor((r, g, 0.3, 0.3))

    ax.set_title('Tabel Complet Metrici - Success Rate per Environment',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_table.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: metrics_table.png")
    plt.close()



def plot_ppo_losses(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    ppo_agents = ['PPO', 'PPO-RND']
    ppo_agents = [a for a in ppo_agents if a in agents]

    if not ppo_agents:
        print("  Skip: Nu exista agenti PPO")
        return

    fig, axes = plt.subplots(len(envs), 3, figsize=(18, 4 * len(envs)))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        env_results = results['results'][env]

        ax1 = axes[env_idx][0]
        ax2 = axes[env_idx][1]
        ax3 = axes[env_idx][2]

        for agent in ppo_agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r]

            policy_data = [r.get('policy_losses') for r in valid_runs if r.get('policy_losses')]
            if policy_data:
                min_len = min(len(p) for p in policy_data)
                policy_data = [p[:min_len] for p in policy_data]
                mean_policy = np.mean(policy_data, axis=0)
                x_vals = np.linspace(0, 100, len(mean_policy))
                ax1.plot(x_vals, mean_policy, 'o-', label=agent,
                        color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

            value_data = [r.get('value_losses') for r in valid_runs if r.get('value_losses')]
            if value_data:
                min_len = min(len(v) for v in value_data)
                value_data = [v[:min_len] for v in value_data]
                mean_value = np.mean(value_data, axis=0)
                x_vals = np.linspace(0, 100, len(mean_value))
                ax2.plot(x_vals, mean_value, 'o-', label=agent,
                        color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

            entropy_data = [r.get('entropies') for r in valid_runs if r.get('entropies')]
            if entropy_data:
                min_len = min(len(e) for e in entropy_data)
                entropy_data = [e[:min_len] for e in entropy_data]
                mean_entropy = np.mean(entropy_data, axis=0)
                x_vals = np.linspace(0, 100, len(mean_entropy))
                ax3.plot(x_vals, mean_entropy, 'o-', label=agent,
                        color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

        ax1.set_xlabel('Training Progress (%)', fontsize=10)
        ax1.set_ylabel('Policy Loss', fontsize=10)
        ax1.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Policy Loss', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.set_xlabel('Training Progress (%)', fontsize=10)
        ax2.set_ylabel('Value Loss', fontsize=10)
        ax2.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Value Loss', fontsize=11, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(alpha=0.3)

        ax3.set_xlabel('Training Progress (%)', fontsize=10)
        ax3.set_ylabel('Entropy', fontsize=10)
        ax3.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Entropy', fontsize=11, fontweight='bold')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(alpha=0.3)

    plt.suptitle('PPO/PPO-RND Training Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ppo_losses.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: ppo_losses.png")
    plt.close()


def plot_buffer_evolution(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    buffer_agents = ['DQN', 'DQN-PER']
    buffer_agents = [a for a in buffer_agents if a in agents]

    if not buffer_agents:
        print("  Skip: Nu exista agenti cu buffer")
        return

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        for agent in buffer_agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('buffer_sizes')]

            if not valid_runs:
                continue

            all_buffers = [r['buffer_sizes'] for r in valid_runs]
            min_len = min(len(b) for b in all_buffers)
            all_buffers = [b[:min_len] for b in all_buffers]
            mean_buffer = np.mean(all_buffers, axis=0)

            episodes = range(len(mean_buffer))
            ax.plot(episodes, mean_buffer, '-', label=agent,
                   color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Buffer Size', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Replay Buffer', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Evolutia Replay Buffer (DQN, DQN-PER)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/buffer_evolution.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: buffer_evolution.png")
    plt.close()


def plot_sample_efficiency(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('training_successes')]

            if not valid_runs:
                continue

            all_successes = [r['training_successes'] for r in valid_runs]
            min_len = min(len(s) for s in all_successes)
            all_successes = [s[:min_len] for s in all_successes]

            mean_successes = np.mean(all_successes, axis=0)
            cumsum = np.cumsum(mean_successes)
            indices = np.arange(1, len(cumsum) + 1)
            cumulative_sr = cumsum / indices * 100

            ax.plot(range(len(cumulative_sr)), cumulative_sr, '-', label=agent,
                   color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

        ax.set_xlabel('Training Episodes/Segments', fontsize=10)
        ax.set_ylabel('Cumulative Success Rate (%)', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Sample Efficiency', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 105])

    plt.suptitle('Sample Efficiency: Cumulative Success Rate', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_efficiency.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: sample_efficiency.png")
    plt.close()


def plot_convergence_speed(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    thresholds = [0.5, 0.7, 0.9]
    threshold_labels = ['50%', '70%', '90%']

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 6))

    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        convergence_data = {t: [] for t in threshold_labels}

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('training_successes')]

            if not valid_runs:
                for t in threshold_labels:
                    convergence_data[t].append(np.nan)
                continue

            all_successes = [r['training_successes'] for r in valid_runs]
            min_len = min(len(s) for s in all_successes)
            all_successes = [s[:min_len] for s in all_successes]
            mean_successes = np.mean(all_successes, axis=0)

            window = min(50, len(mean_successes) // 5)
            if window < 2:
                for t in threshold_labels:
                    convergence_data[t].append(np.nan)
                continue

            running_sr = np.convolve(mean_successes, np.ones(window)/window, mode='valid')

            for thresh, label in zip(thresholds, threshold_labels):
                reached = np.where(running_sr >= thresh)[0]
                if len(reached) > 0:
                    convergence_data[label].append(reached[0] + window)
                else:
                    convergence_data[label].append(np.nan)

        x = np.arange(len(agents))
        width = 0.25

        for i, (label, data) in enumerate(convergence_data.items()):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, data, width, label=f'To {label}',
                         alpha=0.8, edgecolor='black')

        ax.set_xlabel('Agent', fontsize=10)
        ax.set_ylabel('Episodes to Converge', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - Convergence Speed', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=30, ha='right')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Viteza de Convergenta (Episoade pana la Threshold)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_speed.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: convergence_speed.png")
    plt.close()


def plot_intrinsic_rewards(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    if 'PPO-RND' not in agents:
        print("  Skip: PPO-RND nu este in lista de agenti")
        return

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))

    if len(envs) == 1:
        axes = [axes]

    has_data = False

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        runs = env_results['PPO-RND']['runs']
        valid_runs = [r for r in runs if 'error' not in r and r.get('intrinsic_rewards')]

        if valid_runs:
            has_data = True
            all_intrinsic = [r['intrinsic_rewards'] for r in valid_runs]
            min_len = min(len(ir) for ir in all_intrinsic)
            all_intrinsic = [ir[:min_len] for ir in all_intrinsic]
            mean_intrinsic = np.mean(all_intrinsic, axis=0)
            std_intrinsic = np.std(all_intrinsic, axis=0)

            x_vals = np.linspace(0, 100, len(mean_intrinsic))

            ax.plot(x_vals, mean_intrinsic, 'o-',
                   color=AGENT_COLORS.get('PPO-RND', 'purple'), linewidth=2, label='Mean')
            ax.fill_between(x_vals, mean_intrinsic - std_intrinsic, mean_intrinsic + std_intrinsic,
                           color=AGENT_COLORS.get('PPO-RND', 'purple'), alpha=0.2, label='Std')

        ax.set_xlabel('Training Progress (%)', fontsize=10)
        ax.set_ylabel('Intrinsic Reward', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)} - RND Intrinsic Rewards', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

    if not has_data:
        print("  Skip: Nu exista date de intrinsic rewards")
        plt.close()
        return

    plt.suptitle('PPO-RND: Evolutia Intrinsic Rewards', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/intrinsic_rewards.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: intrinsic_rewards.png")
    plt.close()



def plot_episode_length_evolution(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('training_steps')]

            if valid_runs:
                all_steps = [r['training_steps'] for r in valid_runs]
                min_len = min(len(s) for s in all_steps)
                all_steps = [s[:min_len] for s in all_steps]
                mean_steps = np.mean(all_steps, axis=0)

                window = max(1, len(mean_steps) // 20)
                smoothed = np.convolve(mean_steps, np.ones(window)/window, mode='valid')
                x_vals = np.linspace(0, 100, len(smoothed))

                ax.plot(x_vals, smoothed, label=agent,
                       color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

        ax.set_xlabel('Training Progress (%)', fontsize=10)
        ax.set_ylabel('Episode Length (steps)', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Evolutia Lungimii Episoadelor', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/episode_length_evolution.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: episode_length_evolution.png")
    plt.close()


def plot_action_distribution(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']
    action_names = ['Left', 'Down', 'Right', 'Up']

    fig, axes = plt.subplots(len(envs), len(agents), figsize=(4 * len(agents), 4 * len(envs)))

    if len(envs) == 1:
        axes = axes.reshape(1, -1)
    if len(agents) == 1:
        axes = axes.reshape(-1, 1)

    for env_idx, env in enumerate(envs):
        env_results = results['results'][env]

        for agent_idx, agent in enumerate(agents):
            ax = axes[env_idx, agent_idx]
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('action_counts')]

            if valid_runs:
                total_counts = {a: 0 for a in range(4)}
                for r in valid_runs:
                    ac = r['action_counts']
                    for a in range(4):
                        total_counts[a] += ac.get(str(a), ac.get(a, 0))

                counts = [total_counts[a] for a in range(4)]
                total = sum(counts) or 1
                percentages = [c / total * 100 for c in counts]

                colors = ['#ff7f7f', '#7fbf7f', '#7f7fff', '#ffbf7f']
                ax.bar(action_names, percentages, color=colors, edgecolor='black')
                ax.set_ylim(0, 60)

            ax.set_ylabel('Percentage (%)' if agent_idx == 0 else '', fontsize=9)
            ax.set_title(f'{agent}' if env_idx == 0 else '', fontsize=10, fontweight='bold')

            if agent_idx == 0:
                ax.set_ylabel(f'{ENV_DISPLAY_NAMES.get(env, env)}\nPercentage (%)', fontsize=9)

    plt.suptitle('Distributia Actiunilor', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: action_distribution.png")
    plt.close()


def plot_reward_distribution_violin(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        data_for_violin = []
        labels = []

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('training_rewards')]

            if valid_runs:
                final_rewards = []
                for r in valid_runs:
                    rewards = r['training_rewards']
                    n = max(1, len(rewards) // 5)
                    final_rewards.extend(rewards[-n:])

                data_for_violin.append(final_rewards)
                labels.append(agent)

        if data_for_violin:
            parts = ax.violinplot(data_for_violin, showmeans=True, showextrema=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(AGENT_COLORS.get(labels[i], 'gray'))
                pc.set_alpha(0.7)

            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=30, ha='right')

        ax.set_xlabel('Agent', fontsize=10)
        ax.set_ylabel('Reward', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Distributia Recompenselor (Ultimele 20% episoade)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_distribution_violin.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: reward_distribution_violin.png")
    plt.close()


def plot_learning_stability_var(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('training_rewards')]

            if valid_runs:
                all_rewards = [r['training_rewards'] for r in valid_runs]
                min_len = min(len(rw) for rw in all_rewards)
                all_rewards = [rw[:min_len] for rw in all_rewards]
                mean_rewards = np.mean(all_rewards, axis=0)

                window = max(10, len(mean_rewards) // 20)
                rolling_var = []
                for i in range(window, len(mean_rewards)):
                    rolling_var.append(np.var(mean_rewards[i-window:i]))

                x_vals = np.linspace(0, 100, len(rolling_var))
                ax.plot(x_vals, rolling_var, label=agent,
                       color=AGENT_COLORS.get(agent, 'gray'), linewidth=2)

        ax.set_xlabel('Training Progress (%)', fontsize=10)
        ax.set_ylabel('Reward Variance', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Stabilitatea Invatarii (Varianta Recompenselor)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_stability_variance.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: learning_stability_variance.png")
    plt.close()


def plot_priority_distribution(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    if 'DQN-PER' not in agents:
        print("  Skip: DQN-PER nu este in lista de agenti")
        return

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    has_data = False

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        runs = env_results['DQN-PER']['runs']
        valid_runs = [r for r in runs if 'error' not in r and r.get('priority_samples')]

        if valid_runs:
            has_data = True
            all_priorities = []
            for r in valid_runs:
                ps = r['priority_samples']
                if ps:
                    all_priorities.extend(ps[-1])

            if all_priorities:
                ax.hist(all_priorities, bins=50, color=AGENT_COLORS.get('DQN-PER', 'orange'),
                       alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(all_priorities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_priorities):.3f}')
                ax.legend(fontsize=9)

        ax.set_xlabel('Priority Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

    if not has_data:
        print("  Skip: Nu exista date de prioritati")
        plt.close()
        return

    plt.suptitle('DQN-PER: Distributia Prioritatilor in Buffer', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/priority_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: priority_distribution.png")
    plt.close()


def plot_state_visitation_heatmap(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    for env in envs:
        if env == 'easy':
            grid_size = 4
        else:
            grid_size = 8

        fig, axes = plt.subplots(1, len(agents), figsize=(4 * len(agents), 4))
        if len(agents) == 1:
            axes = [axes]

        env_results = results['results'][env]

        for agent_idx, agent in enumerate(agents):
            ax = axes[agent_idx]
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('state_visits')]

            grid = np.zeros((grid_size, grid_size))

            if valid_runs:
                for r in valid_runs:
                    sv = r['state_visits']
                    for state, count in sv.items():
                        state_int = int(state)
                        row = state_int // grid_size
                        col = state_int % grid_size
                        if row < grid_size and col < grid_size:
                            grid[row, col] += count

                if grid.max() > 0:
                    grid = grid / grid.max()

            im = ax.imshow(grid, cmap='YlOrRd', aspect='equal')
            ax.set_title(agent, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            ax.text(0, 0, 'S', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
            ax.text(grid_size-1, grid_size-1, 'G', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='green')

        plt.suptitle(f'{ENV_DISPLAY_NAMES.get(env, env)} - State Visitation Heatmap',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/state_heatmap_{env}.png", dpi=300, bbox_inches='tight')
        print(f"  Salvat: state_heatmap_{env}.png")
        plt.close()


def plot_final_performance_box(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        data_for_box = []
        labels = []
        colors = []

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('training_rewards')]

            if valid_runs:
                final_rewards = []
                for r in valid_runs:
                    rewards = r['training_rewards']
                    n = max(1, len(rewards) // 10)
                    final_rewards.extend(rewards[-n:])

                data_for_box.append(final_rewards)
                labels.append(agent)
                colors.append(AGENT_COLORS.get(agent, 'gray'))

        if data_for_box:
            bp = ax.boxplot(data_for_box, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticklabels(labels, rotation=30, ha='right')

        ax.set_xlabel('Agent', fontsize=10)
        ax.set_ylabel('Final Reward', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Comparatie Performanta Finala (Ultimele 10% episoade)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_performance_box.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: final_performance_box.png")
    plt.close()


def plot_training_time_comparison(results: Dict, output_dir: str) -> None:
    agents = results['metadata']['agents']
    envs = results['metadata']['environments']

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        env_results = results['results'][env]

        times_mean = []
        times_std = []
        labels = []
        colors = []

        for agent in agents:
            runs = env_results[agent]['runs']
            valid_runs = [r for r in runs if 'error' not in r and r.get('training_time')]

            if valid_runs:
                training_times = [r['training_time'] for r in valid_runs]
                times_mean.append(np.mean(training_times))
                times_std.append(np.std(training_times))
                labels.append(agent)
                colors.append(AGENT_COLORS.get(agent, 'gray'))
            else:
                times_mean.append(0)
                times_std.append(0)
                labels.append(agent)
                colors.append(AGENT_COLORS.get(agent, 'gray'))

        x = np.arange(len(labels))
        ax.bar(x, times_mean, yerr=times_std, color=colors, alpha=0.7,
               edgecolor='black', capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')

        ax.set_xlabel('Agent', fontsize=10)
        ax.set_ylabel('Training Time (seconds)', fontsize=10)
        ax.set_title(f'{ENV_DISPLAY_NAMES.get(env, env)}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Comparatie Timp de Antrenament', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Salvat: training_time_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Vizualizare Studiu Comparativ')
    parser.add_argument('--input', type=str, help='Fisier JSON cu rezultate')
    parser.add_argument('--output', type=str, default='../results', help='Director output')

    args = parser.parse_args()

    print("=" * 60)
    print("VIZUALIZARE STUDIU COMPARATIV")
    print("=" * 60)

    results = load_results(args.input)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10

    print("\nGenerez grafice principale...")

    plot_heatmap_success_rate(results, output_dir)
    plot_bars_per_environment(results, output_dir)
    plot_boxplots_variance(results, output_dir)
    plot_learning_curves(results, output_dir)
    plot_radar_chart(results, output_dir)
    plot_ranking_table(results, output_dir)
    plot_scalability_analysis(results, output_dir)
    generate_summary_figure(results, output_dir)

    print("\nGenerez grafice metrici extinse...")

    plot_loss_curves(results, output_dir)
    plot_q_values_evolution(results, output_dir)
    plot_epsilon_decay(results, output_dir)
    plot_td_errors(results, output_dir)
    plot_training_stability(results, output_dir)
    generate_metrics_table(results, output_dir)

    print("\nGenerez grafice noi (PPO, Buffer, Efficiency, Convergence)...")

    plot_ppo_losses(results, output_dir)
    plot_buffer_evolution(results, output_dir)
    plot_sample_efficiency(results, output_dir)
    plot_convergence_speed(results, output_dir)
    plot_intrinsic_rewards(results, output_dir)

    print("\nGenerez grafice aditionale (Behavior, Distribution, Performance)...")

    plot_episode_length_evolution(results, output_dir)
    plot_action_distribution(results, output_dir)
    plot_reward_distribution_violin(results, output_dir)
    plot_learning_stability_var(results, output_dir)
    plot_priority_distribution(results, output_dir)
    plot_state_visitation_heatmap(results, output_dir)
    plot_final_performance_box(results, output_dir)
    plot_training_time_comparison(results, output_dir)

    print("\n" + "=" * 60)
    print("TOATE GRAFICELE GENERATE!")
    print("=" * 60)
    print(f"\nGrafice salvate in: {output_dir}/")
    print("\nFisiere generate:")
    print("  Principale:")
    print("    - heatmap_success_rate.png")
    print("    - bars_per_environment.png")
    print("    - boxplots_variance.png")
    print("    - learning_curves.png")
    print("    - radar_chart.png")
    print("    - ranking_final.png")
    print("    - scalability_analysis.png")
    print("    - summary_figure.png")
    print("  Metrici extinse:")
    print("    - loss_curves.png")
    print("    - q_values_evolution.png")
    print("    - epsilon_decay.png")
    print("    - td_errors.png")
    print("    - training_stability.png")
    print("    - metrics_table.png + .txt")
    print("  Grafice noi:")
    print("    - ppo_losses.png")
    print("    - buffer_evolution.png")
    print("    - sample_efficiency.png")
    print("    - convergence_speed.png")
    print("    - intrinsic_rewards.png")
    print("  Grafice aditionale:")
    print("    - episode_length_evolution.png")
    print("    - action_distribution.png")
    print("    - reward_distribution_violin.png")
    print("    - learning_stability_variance.png")
    print("    - priority_distribution.png")
    print("    - state_heatmap_easy.png")
    print("    - state_heatmap_medium.png")
    print("    - state_heatmap_hard.png")
    print("    - final_performance_box.png")
    print("    - training_time_comparison.png")


if __name__ == "__main__":
    main()