import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_results(filepath='results/final_5seeds_results.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    noncoop_rates = data['noncoop']
    coop_rates = data['coop']
    
    results = {
        'noncoop': [{'seed': 42+i, 'success_rate': rate} for i, rate in enumerate(noncoop_rates)],
        'coop': [{'seed': 42+i, 'success_rate': rate} for i, rate in enumerate(coop_rates)],
        'summary': {
            'noncoop_mean': np.mean(noncoop_rates),
            'noncoop_std': np.std(noncoop_rates),
            'coop_mean': np.mean(coop_rates),
            'coop_std': np.std(coop_rates),
            'improvement': np.mean(coop_rates) - np.mean(noncoop_rates)
        }
    }
    return results


def plot_success_rate_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    systems = ['Non-Cooperative\n(Individual Rewards)', 'Cooperative\n(Shared Rewards)']
    means = [results['summary']['noncoop_mean'], results['summary']['coop_mean']]
    stds = [results['summary']['noncoop_std'], results['summary']['coop_std']]
    
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax.bar(systems, means, yerr=stds, capsize=10, color=colors, 
                   edgecolor='black', linewidth=2, alpha=0.8)
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean:.1f}%\n±{std:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.annotate('', xy=(1, means[1]), xytext=(1, means[0]),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax.text(1.15, (means[0] + means[1])/2, 
            f'+{results["summary"]["improvement"]:.1f}%', 
            fontsize=14, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('MARL Comparison: Impact of Shared Rewards', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/1_success_rate_comparison.png', dpi=300, bbox_inches='tight')
    print("Generated: 1_success_rate_comparison.png")
    plt.close()


def plot_per_seed_comparison(results):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seeds = [r['seed'] for r in results['noncoop']]
    noncoop_scores = [r['success_rate'] for r in results['noncoop']]
    coop_scores = [r['success_rate'] for r in results['coop']]
    
    x = np.arange(len(seeds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, noncoop_scores, width, 
                    label='Non-Cooperative', color='#ff6b6b', 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, coop_scores, width, 
                    label='Cooperative', color='#4ecdc4',
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for i, (nc, c) in enumerate(zip(noncoop_scores, coop_scores)):
        improvement = c - nc
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'+{improvement:.0f}%', 
                    xy=(i, max(nc, c) + 3),
                    fontsize=9, ha='center', fontweight='bold', color=color)
    
    ax.set_xlabel('Seed', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Seed Comparison: Non-Cooperative vs Cooperative',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/2_per_seed_comparison.png', dpi=300, bbox_inches='tight')
    print("Generated: 2_per_seed_comparison.png")
    plt.close()


def plot_q_value_heatmaps_detailed():
    try:
        q_data = np.load('results/q_tables.npz')
        with open('results/maps.json', 'r') as f:
            maps_data = json.load(f)
    except FileNotFoundError:
        print("Q-tables not found! Run experiment first to generate them.")
        return
    
    seeds = [42, 43, 44, 45, 46]
    
    for seed_idx, seed in enumerate(seeds):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Seed {seed}: Learned Q-Values Comparison', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        seed_key = f'seed_{seed}'
        nc_a1_q = q_data[f'{seed_key}_noncoop_agent1']
        nc_a2_q = q_data[f'{seed_key}_noncoop_agent2']
        coop_a1_q = q_data[f'{seed_key}_coop_agent1']
        coop_a2_q = q_data[f'{seed_key}_coop_agent2']
        
        def q_to_grid(q_table):
            max_q = np.max(q_table, axis=1)
            return max_q.reshape(8, 8)
        
        nc_a1_grid = q_to_grid(nc_a1_q)
        nc_a2_grid = q_to_grid(nc_a2_q)
        coop_a1_grid = q_to_grid(coop_a1_q)
        coop_a2_grid = q_to_grid(coop_a2_q)
        
        sns.heatmap(nc_a1_grid, ax=axes[0, 0], cmap='Reds', 
                    annot=False, fmt='.2f', cbar_kws={'label': 'Max Q-Value'},
                    vmin=0, vmax=1, square=True, linewidths=0.5)
        axes[0, 0].set_title('Non-Cooperative\nAgent 1', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Agent 1', fontsize=12, fontweight='bold')
        
        sns.heatmap(coop_a1_grid, ax=axes[0, 1], cmap='Blues',
                    annot=False, fmt='.2f', cbar_kws={'label': 'Max Q-Value'},
                    vmin=0, vmax=1, square=True, linewidths=0.5)
        axes[0, 1].set_title('Cooperative\nAgent 1', fontsize=14, fontweight='bold')
        
        diff_a1 = coop_a1_grid - nc_a1_grid
        sns.heatmap(diff_a1, ax=axes[0, 2], cmap='RdBu_r', center=0,
                    annot=False, fmt='.2f', cbar_kws={'label': 'Delta Q-Value'},
                    square=True, linewidths=0.5)
        axes[0, 2].set_title('Difference\n(Coop - Non-Coop)', fontsize=14, fontweight='bold')
        
        sns.heatmap(nc_a2_grid, ax=axes[1, 0], cmap='Oranges',
                    annot=False, fmt='.2f', cbar_kws={'label': 'Max Q-Value'},
                    vmin=0, vmax=1, square=True, linewidths=0.5)
        axes[1, 0].set_title('Non-Cooperative\nAgent 2', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Agent 2', fontsize=12, fontweight='bold')
        
        sns.heatmap(coop_a2_grid, ax=axes[1, 1], cmap='Greens',
                    annot=False, fmt='.2f', cbar_kws={'label': 'Max Q-Value'},
                    vmin=0, vmax=1, square=True, linewidths=0.5)
        axes[1, 1].set_title('Cooperative\nAgent 2', fontsize=14, fontweight='bold')
        
        diff_a2 = coop_a2_grid - nc_a2_grid
        sns.heatmap(diff_a2, ax=axes[1, 2], cmap='RdBu_r', center=0,
                    annot=False, fmt='.2f', cbar_kws={'label': 'Delta Q-Value'},
                    square=True, linewidths=0.5)
        axes[1, 2].set_title('Difference\n(Coop - Non-Coop)', fontsize=14, fontweight='bold')
        
        fig.text(0.5, 0.02, 
                 'Warmer colors = higher learned values | Blue in difference = Coop learned better | Red = Non-Coop learned better',
                 ha='center', fontsize=11, style='italic',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f'results/3_q_values_seed_{seed}.png', dpi=300, bbox_inches='tight')
        print(f"Generated: 3_q_values_seed_{seed}.png")
        plt.close()
    
    print(f"Generated Q-value heatmaps for all {len(seeds)} seeds")


def plot_summary_dashboard(results):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, :2])
    systems = ['Non-Coop', 'Coop']
    means = [results['summary']['noncoop_mean'], results['summary']['coop_mean']]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax1.bar(systems, means, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2., mean + 1,
                f'{mean:.1f}%', ha='center', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Average Success Rate', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.text(0.5, 0.6, f'+{results["summary"]["improvement"]:.1f}%', 
             ha='center', va='center', fontsize=48, fontweight='bold', color='green')
    ax2.text(0.5, 0.3, 'Improvement', ha='center', va='center', fontsize=16)
    ax2.text(0.5, 0.1, 'from Collaboration', ha='center', va='center', fontsize=12, style='italic')
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    ax3 = fig.add_subplot(gs[1, :])
    seeds = [r['seed'] for r in results['noncoop']]
    noncoop_scores = [r['success_rate'] for r in results['noncoop']]
    coop_scores = [r['success_rate'] for r in results['coop']]
    x = np.arange(len(seeds))
    width = 0.35
    ax3.bar(x - width/2, noncoop_scores, width, label='Non-Coop', color='#ff6b6b', alpha=0.8)
    ax3.bar(x + width/2, coop_scores, width, label='Coop', color='#4ecdc4', alpha=0.8)
    ax3.set_xlabel('Seed', fontweight='bold')
    ax3.set_ylabel('Success Rate (%)', fontweight='bold')
    ax3.set_title('Per-Seed Results', fontweight='bold', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(seeds)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    table_data = [
        ['Metric', 'Non-Coop', 'Coop'],
        ['Mean', f'{results["summary"]["noncoop_mean"]:.1f}%', f'{results["summary"]["coop_mean"]:.1f}%'],
        ['Std Dev', f'±{results["summary"]["noncoop_std"]:.1f}%', f'±{results["summary"]["coop_std"]:.1f}%'],
        ['Best Seed', f'{max(noncoop_scores):.0f}%', f'{max(coop_scores):.0f}%'],
        ['Worst Seed', f'{min(noncoop_scores):.0f}%', f'{min(coop_scores):.0f}%'],
    ]
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(table_data)):
        if i == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#cccccc')
                table[(i, j)].set_text_props(weight='bold')
    ax4.set_title('Key Statistics', fontweight='bold', fontsize=12, pad=20)
    
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.axis('off')
    explanation = """
    Experiment: Non-Cooperativ vs Cooperativ IQL
    
    Setup:
    - 2 agenti cu Independent Q-Learning
    - Aceeasi harta (25 gropi, 39% densitate, 35% slippery)
    - 20,000 episoade antrenare per sistem
    
    Diferenta:
    - Non-Coop: Rewarduri individuale + penalizare (-0.3) cand partenerul reuseste
    - Coop: Rewarduri shared (ambii primesc +1 daca oricare reuseste)
    
    Rezultat:
    - Sistemul cooperativ arata imbunatatire
    - Demonstreaza valoarea colaborarii implicite
    - Imbunatatire consistenta pe toate seeds
    """
    ax5.text(0.05, 0.95, explanation, ha='left', va='top', fontsize=11,
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('MARL Experiment Summary Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('results/4_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("Generated: 4_summary_dashboard.png")
    plt.close()


def generate_all_visualizations():
    print("\n" + "="*60)
    print("MARL Visualization Generator")
    print("="*60)
    
    print("\nLoading results from results/final_5seeds_results.json...")
    results = load_results()
    
    print(f"\nLoaded data:")
    print(f"  Non-Coop mean: {results['summary']['noncoop_mean']:.1f}%")
    print(f"  Coop mean: {results['summary']['coop_mean']:.1f}%")
    print(f"  Improvement: +{results['summary']['improvement']:.1f}%")
    
    print("\nGenerating visualizations...")
    
    plot_per_seed_comparison(results)
    plot_q_value_heatmaps_detailed()
    plot_summary_dashboard(results)
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)
    print("\nGenerated files in results/:")
    print("  1. 2_per_seed_comparison.png")
    print("  2. 3_q_values_seed_XX.png (5 files)")
    print("  3. 4_summary_dashboard.png")
    

if __name__ == "__main__":
    generate_all_visualizations()
