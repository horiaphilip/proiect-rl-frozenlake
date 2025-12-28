"""
Script pentru rularea experimentelor cu toți agenții RL.

Antrenează Q-Learning, DQN, PPO și PPO+RND pe mediul DynamicFrozenLake și salvează rezultatele.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from environments.dynamic_frozenlake import DynamicFrozenLakeEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.ppo_rnd import PPORndAgent


def create_results_dir():
    """Creează directorul pentru rezultate cu timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../results/experiment_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def run_q_learning_experiment(env, n_episodes=1000, n_runs=5):
    """
    Rulează experimente cu Q-Learning.
    """
    print("\n" + "="*60)
    print("ANTRENARE Q-LEARNING")
    print("="*60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )

        episode_rewards = []
        episode_steps = []
        epsilons = []
        td_errors = []

        for episode in tqdm(range(n_episodes), desc=f"Q-Learning Run {run+1}"):
            stats = agent.train_episode(env)
            episode_rewards.append(stats['total_reward'])
            episode_steps.append(stats['steps'])
            epsilons.append(stats['epsilon'])
            td_errors.append(stats['avg_td_error'])

        eval_stats = agent.evaluate(env, n_episodes=100)

        all_results.append({
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'epsilons': epsilons,
            'td_errors': td_errors,
            'final_eval': eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {
        'algorithm': 'Q-Learning',
        'runs': all_results,
        'hyperparameters': {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995
        }
    }


def run_dqn_experiment(env, n_episodes=1000, n_runs=5):
    """
    Rulează experimente cu DQN.
    """
    print("\n" + "="*60)
    print("ANTRENARE DQN")
    print("="*60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = DQNAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=0.001,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_capacity=10000,
            batch_size=64,
            target_update_freq=10,
            hidden_dim=128
        )

        episode_rewards = []
        episode_steps = []
        epsilons = []
        losses = []
        buffer_sizes = []

        for episode in tqdm(range(n_episodes), desc=f"DQN Run {run+1}"):
            stats = agent.train_episode(env)
            episode_rewards.append(stats['total_reward'])
            episode_steps.append(stats['steps'])
            epsilons.append(stats['epsilon'])
            losses.append(stats['avg_loss'])
            buffer_sizes.append(stats['buffer_size'])

        eval_stats = agent.evaluate(env, n_episodes=100)

        all_results.append({
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'epsilons': epsilons,
            'losses': losses,
            'buffer_sizes': buffer_sizes,
            'final_eval': eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {
        'algorithm': 'DQN',
        'runs': all_results,
        'hyperparameters': {
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'hidden_dim': 128
        }
    }


def run_ppo_experiment(env, total_timesteps=100000, n_runs=5):
    """
    Rulează experimente cu PPO (SB3 wrapper-ul tău).
    """
    print("\n" + "="*60)
    print("ANTRENARE PPO")
    print("="*60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = PPOAgent(
            env=env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0
        )

        print(f"Training PPO for {total_timesteps} timesteps...")
        stats = agent.train(total_timesteps=total_timesteps, progress_bar=True)

        eval_stats = agent.evaluate(env, n_episodes=100)

        all_results.append({
            'training_stats': stats,
            'final_eval': eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {
        'algorithm': 'PPO',
        'runs': all_results,
        'hyperparameters': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'total_timesteps': total_timesteps
        }
    }


def run_ppo_rnd_experiment(env, total_timesteps=100000, n_runs=5):
    """
    Rulează experimente cu PPO + RND (agent custom, PyTorch).
    """
    print("\n" + "="*60)
    print("ANTRENARE PPO + RND")
    print("="*60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = PPORndAgent(
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,

            beta_int=0.05,             # dacă success rate rămâne 0, încearcă 0.1
            rnd_lr=1e-4,
            rnd_out_dim=64,
            rnd_update_proportion=0.25,
            normalize_int_reward=True,

            verbose=0,
        )

        print(f"Training PPO+RND for {total_timesteps} timesteps...")
        stats = agent.train(total_timesteps=total_timesteps, progress_bar=True)

        eval_stats = agent.evaluate(env, n_episodes=100)

        all_results.append({
            'training_stats': stats,
            'final_eval': eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {
        'algorithm': 'PPO+RND',
        'runs': all_results,
        'hyperparameters': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'beta_int': 0.05,
            'rnd_lr': 1e-4,
            'rnd_out_dim': 64,
            'rnd_update_proportion': 0.25,
            'normalize_int_reward': True,
            'total_timesteps': total_timesteps
        }
    }


def save_results(results, results_dir):
    """
    Salvează rezultatele experimentelor.
    """
    results_file = os.path.join(results_dir, 'results.json')

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nRezultate salvate în: {results_file}")


def main():
    """Funcția principală pentru rularea experimentelor."""
    print("="*60)
    print("EXPERIMENTE REINFORCEMENT LEARNING")
    print("Mediu: Dynamic FrozenLake")
    print("="*60)

    MAP_SIZE = 8
    N_EPISODES = 500
    PPO_TIMESTEPS = 50000
    N_RUNS = 5

    results_dir = create_results_dir()
    print(f"\nRezultate vor fi salvate în: {results_dir}")

    env = DynamicFrozenLakeEnv(
        map_size=MAP_SIZE,
        max_steps=100,
        slippery_start=0.1,
        slippery_end=0.4,
        step_penalty=-0.01,
        ice_melting=True,
        melting_rate=0.01
    )

    print(f"\nMediu: DynamicFrozenLake {MAP_SIZE}x{MAP_SIZE}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    all_results = {}

    # 1. Q-Learning
    q_learning_results = run_q_learning_experiment(env, n_episodes=N_EPISODES, n_runs=N_RUNS)
    all_results['q_learning'] = q_learning_results

    # 2. DQN
    dqn_results = run_dqn_experiment(env, n_episodes=N_EPISODES, n_runs=N_RUNS)
    all_results['dqn'] = dqn_results

    # 3. PPO
    ppo_results = run_ppo_experiment(env, total_timesteps=PPO_TIMESTEPS, n_runs=N_RUNS)
    all_results['ppo'] = ppo_results

    # 4. PPO + RND
    ppo_rnd_results = run_ppo_rnd_experiment(env, total_timesteps=PPO_TIMESTEPS, n_runs=N_RUNS)
    all_results['ppo_rnd'] = ppo_rnd_results

    save_results(all_results, results_dir)

    print("\n" + "="*60)
    print("REZUMAT FINAL")
    print("="*60)

    for algo_name, algo_results in all_results.items():
        print(f"\n{algo_name.upper()}:")
        final_evals = [run['final_eval'] for run in algo_results['runs']]
        mean_rewards = [e['mean_reward'] for e in final_evals]
        success_rates = [e['success_rate'] for e in final_evals]

        print(f"  Mean Reward: {np.mean(mean_rewards):.4f} ± {np.std(mean_rewards):.4f}")
        print(f"  Success Rate: {np.mean(success_rates):.2%} ± {np.std(success_rates):.2%}")

    print(f"\nRezultate complete salvate în: {results_dir}")
    print("Rulați experiments/visualize.py pentru a genera grafice.")


if __name__ == "__main__":
    main()
