"""
Benchmark complet: Toti cei 5 agenti pe EasyFrozenLake

Testeaza:
1. Q-Learning
2. DQN
3. DQN + PER
4. PPO
5. PPO + RND

Salveaza rezultatele pentru comparatie.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

from environments.easy_frozenlake import EasyFrozenLakeEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.dqn_per import DQN_PERAgent
from agents.ppo import PPOAgent
from agents.ppo_rnd import PPORndAgent


def test_q_learning(env, n_episodes=500, seed=42):
    """Test Q-Learning."""
    print("\n" + "="*60)
    print("Q-LEARNING")
    print("="*60)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )

    rewards = []
    for episode in tqdm(range(n_episodes), desc="Q-Learning"):
        stats = agent.train_episode(env)
        rewards.append(stats['total_reward'])

    eval_stats = agent.evaluate(env, n_episodes=100)

    print(f"Mean Reward: {eval_stats['mean_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

    return {
        'algorithm': 'Q-Learning',
        'training_rewards': rewards,
        'eval': eval_stats
    }


def test_dqn(env, n_episodes=500, seed=42):
    """Test DQN."""
    print("\n" + "="*60)
    print("DQN")
    print("="*60)

    np.random.seed(seed)

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

    rewards = []
    for episode in tqdm(range(n_episodes), desc="DQN"):
        stats = agent.train_episode(env)
        rewards.append(stats['total_reward'])

    eval_stats = agent.evaluate(env, n_episodes=100)

    print(f"Mean Reward: {eval_stats['mean_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

    return {
        'algorithm': 'DQN',
        'training_rewards': rewards,
        'eval': eval_stats
    }


def test_dqn_per(env, n_episodes=500, seed=42):
    """Test DQN + PER."""
    print("\n" + "="*60)
    print("DQN + PER")
    print("="*60)

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
        seed=seed  # DQN_PER accepta seed
    )

    rewards = []
    for episode in tqdm(range(n_episodes), desc="DQN+PER"):
        stats = agent.train_episode(env)
        rewards.append(stats['total_reward'])

    eval_stats = agent.evaluate(env, n_episodes=100)

    print(f"Mean Reward: {eval_stats['mean_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

    return {
        'algorithm': 'DQN+PER',
        'training_rewards': rewards,
        'eval': eval_stats
    }


def test_ppo(env, total_timesteps=25000, seed=42):
    """Test PPO."""
    print("\n" + "="*60)
    print("PPO")
    print("="*60)

    np.random.seed(seed)

    agent = PPOAgent(
        env=env,
        learning_rate=0.0003,
        n_steps=512,  # mai mic pentru env simplu
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

    print(f"Training for {total_timesteps} timesteps...")
    stats = agent.train(total_timesteps=total_timesteps, progress_bar=False)

    eval_stats = agent.evaluate(env, n_episodes=100)

    print(f"Mean Reward: {eval_stats['mean_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

    return {
        'algorithm': 'PPO',
        'training_stats': stats,
        'eval': eval_stats
    }


def test_ppo_rnd(env, total_timesteps=25000, seed=42):
    """Test PPO + RND."""
    print("\n" + "="*60)
    print("PPO + RND")
    print("="*60)

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
        beta_int=0.02,  # mai mic pentru env simplu
        rnd_lr=1e-4,
        normalize_int_reward=True,
        seed=seed,  # PPO+RND accepta seed
        verbose=0
    )

    print(f"Training for {total_timesteps} timesteps...")
    stats = agent.train(total_timesteps=total_timesteps, progress_bar=False)

    eval_stats = agent.evaluate(env, n_episodes=100)

    print(f"Mean Reward: {eval_stats['mean_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

    return {
        'algorithm': 'PPO+RND',
        'training_stats': stats,
        'eval': eval_stats
    }


def save_results(results, filename):
    """Salveaza rezultatele."""
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

    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    print("="*60)
    print("BENCHMARK ALL AGENTS ON EASY FROZENLAKE")
    print("="*60)

    SEED = 42

    # Creeaza environment
    env = EasyFrozenLakeEnv(
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
        seed=SEED
    )

    print(f"\nEnvironment: EasyFrozenLake {env.map_size}x{env.map_size}")
    print(f"Slippery: {env.slippery}")
    print(f"Hole ratio: {env.hole_ratio}")
    print(f"Max steps: {env.max_steps}")

    # Ruleaza toti agentii
    all_results = []

    # 1. Q-Learning
    q_results = test_q_learning(env, n_episodes=500, seed=SEED)
    all_results.append(q_results)

    # 2. DQN
    dqn_results = test_dqn(env, n_episodes=500, seed=SEED)
    all_results.append(dqn_results)

    # 3. DQN + PER
    dqn_per_results = test_dqn_per(env, n_episodes=500, seed=SEED)
    all_results.append(dqn_per_results)

    # 4. PPO
    ppo_results = test_ppo(env, total_timesteps=25000, seed=SEED)
    all_results.append(ppo_results)

    # 5. PPO + RND
    ppo_rnd_results = test_ppo_rnd(env, total_timesteps=25000, seed=SEED)
    all_results.append(ppo_rnd_results)

    # Salvare rezultate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/benchmark_easy_{timestamp}.json"
    save_results(all_results, filename)

    # Comparatie finala
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'Algorithm':<15} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Steps':<15}")
    print("-"*60)

    for result in all_results:
        algo = result['algorithm']
        eval_stats = result['eval']
        print(f"{algo:<15} {eval_stats['success_rate']:<15.2%} "
              f"{eval_stats['mean_reward']:<15.4f} {eval_stats['mean_steps']:<15.2f}")

    # Gaseste cel mai bun
    best_algo = max(all_results, key=lambda x: x['eval']['success_rate'])
    print(f"\nBest algorithm: {best_algo['algorithm']} "
          f"(Success Rate: {best_algo['eval']['success_rate']:.2%})")


if __name__ == "__main__":
    main()
