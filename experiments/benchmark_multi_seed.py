"""
Benchmark cu multiple seed-uri pentru reproducibilitate si analiza statistica.

Ruleaza toti cei 5 agenti pe EasyFrozenLake cu 5 seed-uri diferite
si calculeaza mean +/- std pentru fiecare metrica.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import torch
import random
from tqdm import tqdm
from datetime import datetime

from environments.easy_frozenlake import EasyFrozenLakeEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.dqn_per import DQN_PERAgent
from agents.ppo import PPOAgent
from agents.ppo_rnd import PPORndAgent


# Seed-uri pentru experimente
SEEDS = [42, 123, 456, 789, 1024]


def set_all_seeds(seed):
    """Seteaza seed-ul pentru toate librariile."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_env(seed):
    """Creeaza environment cu seed specificat."""
    return EasyFrozenLakeEnv(
        map_size=4,
        max_steps=50,
        slippery=0.05,
        step_penalty=-0.01,
        hole_penalty=-0.5,
        goal_reward=1.0,
        shaped_rewards=True,
        shaping_scale=0.05,
        hole_ratio=0.10,
        safe_zone_radius=1,
        regenerate_map_each_episode=False,
        seed=seed
    )


def train_q_learning(env, n_episodes=500, seed=42):
    """Train Q-Learning."""
    set_all_seeds(seed)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )

    for _ in range(n_episodes):
        agent.train_episode(env)

    eval_stats = agent.evaluate(env, n_episodes=100)
    return eval_stats


def train_dqn(env, n_episodes=500, seed=42):
    """Train DQN."""
    set_all_seeds(seed)

    agent = DQNAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=5000,
        batch_size=32,
        target_update_freq=10,
        hidden_dim=64
    )

    for _ in range(n_episodes):
        agent.train_episode(env)

    eval_stats = agent.evaluate(env, n_episodes=100)
    return eval_stats


def train_dqn_per(env, n_episodes=500, seed=42):
    """Train DQN + PER."""
    set_all_seeds(seed)

    agent = DQN_PERAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=5000,
        batch_size=32,
        target_update_freq=10,
        hidden_dim=64,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_anneal_steps=50000,
        seed=seed
    )

    for _ in range(n_episodes):
        agent.train_episode(env)

    eval_stats = agent.evaluate(env, n_episodes=100)
    return eval_stats


def train_ppo(env, total_timesteps=25000, seed=42):
    """Train PPO."""
    set_all_seeds(seed)

    agent = PPOAgent(
        env=env,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0
    )

    agent.train(total_timesteps=total_timesteps, progress_bar=False)
    eval_stats = agent.evaluate(env, n_episodes=100)
    return eval_stats


def train_ppo_rnd(env, total_timesteps=25000, seed=42):
    """Train PPO + RND."""
    set_all_seeds(seed)

    agent = PPORndAgent(
        env=env,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        beta_int=0.02,
        rnd_lr=1e-4,
        normalize_int_reward=True,
        seed=seed,
        verbose=0
    )

    agent.train(total_timesteps=total_timesteps, progress_bar=False)
    eval_stats = agent.evaluate(env, n_episodes=100)
    return eval_stats


def run_multi_seed_experiment():
    """Ruleaza toate experimentele cu multiple seed-uri."""

    print("="*70)
    print("BENCHMARK MULTI-SEED - 5 ALGORITMI x 5 SEED-URI")
    print("="*70)
    print(f"Seeds: {SEEDS}")
    print()

    algorithms = {
        'Q-Learning': train_q_learning,
        'DQN': train_dqn,
        'DQN+PER': train_dqn_per,
        'PPO': train_ppo,
        'PPO+RND': train_ppo_rnd
    }

    # Rezultate pentru fiecare algoritm
    all_results = {}

    for algo_name, train_func in algorithms.items():
        print(f"\n{'='*70}")
        print(f"ALGORITMUL: {algo_name}")
        print(f"{'='*70}")

        seed_results = {
            'success_rates': [],
            'mean_rewards': [],
            'mean_steps': []
        }

        for seed in tqdm(SEEDS, desc=f"{algo_name}"):
            # Creeaza environment nou pentru fiecare seed
            env = create_env(seed)

            # Determina parametrii in functie de algoritm
            if algo_name in ['PPO', 'PPO+RND']:
                eval_stats = train_func(env, total_timesteps=25000, seed=seed)
            else:
                eval_stats = train_func(env, n_episodes=500, seed=seed)

            seed_results['success_rates'].append(eval_stats['success_rate'])
            seed_results['mean_rewards'].append(eval_stats['mean_reward'])
            seed_results['mean_steps'].append(eval_stats['mean_steps'])

            print(f"  Seed {seed}: Success={eval_stats['success_rate']:.2%}, "
                  f"Reward={eval_stats['mean_reward']:.4f}, Steps={eval_stats['mean_steps']:.2f}")

        # Calculeaza statistici
        all_results[algo_name] = {
            'success_rate': {
                'mean': np.mean(seed_results['success_rates']),
                'std': np.std(seed_results['success_rates']),
                'values': seed_results['success_rates']
            },
            'mean_reward': {
                'mean': np.mean(seed_results['mean_rewards']),
                'std': np.std(seed_results['mean_rewards']),
                'values': seed_results['mean_rewards']
            },
            'mean_steps': {
                'mean': np.mean(seed_results['mean_steps']),
                'std': np.std(seed_results['mean_steps']),
                'values': seed_results['mean_steps']
            }
        }

        print(f"\n  STATISTICI {algo_name}:")
        print(f"    Success Rate: {all_results[algo_name]['success_rate']['mean']:.2%} "
              f"+/- {all_results[algo_name]['success_rate']['std']:.2%}")
        print(f"    Mean Reward:  {all_results[algo_name]['mean_reward']['mean']:.4f} "
              f"+/- {all_results[algo_name]['mean_reward']['std']:.4f}")
        print(f"    Mean Steps:   {all_results[algo_name]['mean_steps']['mean']:.2f} "
              f"+/- {all_results[algo_name]['mean_steps']['std']:.2f}")

    return all_results


def print_final_table(results):
    """Afiseaza tabelul final cu rezultatele."""
    print("\n" + "="*90)
    print("REZULTATE FINALE - MULTI-SEED BENCHMARK")
    print("="*90)
    print(f"\n{'Algorithm':<15} {'Success Rate':<20} {'Mean Reward':<20} {'Mean Steps':<20}")
    print("-"*90)

    for algo_name, stats in results.items():
        sr = f"{stats['success_rate']['mean']:.2%} +/- {stats['success_rate']['std']:.2%}"
        mr = f"{stats['mean_reward']['mean']:.4f} +/- {stats['mean_reward']['std']:.4f}"
        ms = f"{stats['mean_steps']['mean']:.2f} +/- {stats['mean_steps']['std']:.2f}"
        print(f"{algo_name:<15} {sr:<20} {mr:<20} {ms:<20}")

    print("-"*90)

    # Gaseste cel mai bun algoritm
    best_algo = max(results.items(),
                    key=lambda x: (x[1]['success_rate']['mean'], -x[1]['mean_steps']['mean']))
    print(f"\nCel mai bun algoritm: {best_algo[0]}")
    print(f"  Success Rate: {best_algo[1]['success_rate']['mean']:.2%} +/- {best_algo[1]['success_rate']['std']:.2%}")


def save_results(results, filename):
    """Salveaza rezultatele in format JSON."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    output = {
        'experiment_info': {
            'type': 'multi_seed_benchmark',
            'seeds': SEEDS,
            'n_seeds': len(SEEDS),
            'timestamp': datetime.now().isoformat(),
            'environment': 'EasyFrozenLake 4x4'
        },
        'results': convert(results)
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nRezultatele au fost salvate in: {filename}")


def main():
    # Ruleaza experimentele
    results = run_multi_seed_experiment()

    # Afiseaza tabelul final
    print_final_table(results)

    # Salveaza rezultatele
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"benchmark_multi_seed_{timestamp}.json")
    save_results(results, filename)

    print("\n" + "="*90)
    print("EXPERIMENT COMPLET!")
    print("="*90)


if __name__ == "__main__":
    main()
