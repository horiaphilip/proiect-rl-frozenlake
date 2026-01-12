"""
Script de test pentru EasyFrozenLake environment.

Rulează rapid TOȚI cei 5 algoritmi pe mediul easy pentru a verifica că pot învăța.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from environments.easy_frozenlake import EasyFrozenLakeEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.dqn_per import DQN_PERAgent
from agents.ppo import PPOAgent
from agents.ppo_rnd import PPORNDAgent


def test_q_learning(env, n_episodes=300):
    """Test rapid Q-Learning."""
    print("\n" + "="*60)
    print("TEST 1/5: Q-LEARNING")
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

    # Training
    for episode in tqdm(range(n_episodes), desc="Q-Learning Training"):
        agent.train_episode(env)

    # Evaluation
    eval_stats = agent.evaluate(env, n_episodes=50)

    print(f"✓ Success Rate: {eval_stats['success_rate']:.1%}")
    print(f"✓ Mean Steps: {eval_stats['mean_steps']:.2f}")
    print(f"✓ Mean Reward: {eval_stats['mean_reward']:.4f}")

    return eval_stats


def test_dqn(env, n_episodes=300):
    """Test rapid DQN."""
    print("\n" + "="*60)
    print("TEST 2/5: DQN")
    print("="*60)

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

    # Training
    for episode in tqdm(range(n_episodes), desc="DQN Training"):
        agent.train_episode(env)

    # Evaluation
    eval_stats = agent.evaluate(env, n_episodes=50)

    print(f"✓ Success Rate: {eval_stats['success_rate']:.1%}")
    print(f"✓ Mean Steps: {eval_stats['mean_steps']:.2f}")
    print(f"✓ Mean Reward: {eval_stats['mean_reward']:.4f}")

    return eval_stats


def test_dqn_per(env, n_episodes=300):
    """Test rapid DQN+PER."""
    print("\n" + "="*60)
    print("TEST 3/5: DQN+PER")
    print("="*60)

    agent = DQN_PERAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10,
        hidden_dim=128,
        per_alpha=0.6,
        per_beta_start=0.4
    )

    # Training
    for episode in tqdm(range(n_episodes), desc="DQN+PER Training"):
        agent.train_episode(env)

    # Evaluation
    eval_stats = agent.evaluate(env, n_episodes=50)

    print(f"✓ Success Rate: {eval_stats['success_rate']:.1%}")
    print(f"✓ Mean Steps: {eval_stats['mean_steps']:.2f}")
    print(f"✓ Mean Reward: {eval_stats['mean_reward']:.4f}")

    return eval_stats


def test_ppo(env, total_timesteps=15000):
    """Test rapid PPO."""
    print("\n" + "="*60)
    print("TEST 4/5: PPO")
    print("="*60)

    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        verbose=0
    )

    # Training
    print(f"Training PPO pentru {total_timesteps} timesteps...")
    agent.train(total_timesteps=total_timesteps)

    # Evaluation
    eval_stats = agent.evaluate(env, n_episodes=50)

    print(f"✓ Success Rate: {eval_stats['success_rate']:.1%}")
    print(f"✓ Mean Steps: {eval_stats['mean_steps']:.2f}")
    print(f"✓ Mean Reward: {eval_stats['mean_reward']:.4f}")

    return eval_stats


def test_ppo_rnd(env, total_timesteps=15000):
    """Test rapid PPO+RND."""
    print("\n" + "="*60)
    print("TEST 5/5: PPO+RND")
    print("="*60)

    agent = PPORNDAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        intrinsic_reward_weight=0.01,
        verbose=0
    )

    # Training
    print(f"Training PPO+RND pentru {total_timesteps} timesteps...")
    agent.train(total_timesteps=total_timesteps)

    # Evaluation
    eval_stats = agent.evaluate(env, n_episodes=50)

    print(f"✓ Success Rate: {eval_stats['success_rate']:.1%}")
    print(f"✓ Mean Steps: {eval_stats['mean_steps']:.2f}")
    print(f"✓ Mean Reward: {eval_stats['mean_reward']:.4f}")

    return eval_stats


def main():
    print("="*60)
    print("TEST COMPLET - EASYFROZENLAKE (TOȚI ALGORITMII)")
    print("="*60)

    # Creează environment easy
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
        seed=42
    )

    print(f"\nMediu: EasyFrozenLake {env.map_size}x{env.map_size}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max steps: {env.max_steps}")
    print(f"Slippery prob: {env.slippery}")

    # Afiseaza harta
    print("\nHarta initiala:")
    env.reset()
    print(env.render())

    # Rulează toți algoritmii
    results = {}

    results['Q-Learning'] = test_q_learning(env, n_episodes=300)
    results['DQN'] = test_dqn(env, n_episodes=300)
    results['DQN+PER'] = test_dqn_per(env, n_episodes=300)
    results['PPO'] = test_ppo(env, total_timesteps=15000)
    results['PPO+RND'] = test_ppo_rnd(env, total_timesteps=15000)

    # Comparatie finala
    print("\n" + "="*60)
    print("COMPARATIE FINALA - TOȚI ALGORITMII")
    print("="*60)
    print(f"\n{'Algorithm':<15} {'Success Rate':<15} {'Mean Steps':<12} {'Mean Reward':<12}")
    print("-"*60)

    # Sortează după success rate
    sorted_results = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)

    for algo_name, stats in sorted_results:
        success_marker = "🏆" if stats['success_rate'] == max(r['success_rate'] for r in results.values()) else "  "
        print(f"{algo_name:<15} {stats['success_rate']:<14.1%} {success_marker} "
              f"{stats['mean_steps']:<12.2f} {stats['mean_reward']:<12.4f}")

    print("\n" + "="*60)

    # Statistici finale
    success_count = sum(1 for stats in results.values() if stats['success_rate'] >= 80)
    print(f"\n✓ {success_count}/5 algoritmi au atins ≥80% success rate")

    best_algo = max(results.items(), key=lambda x: x[1]['success_rate'])
    print(f"🏆 Câștigător: {best_algo[0]} cu {best_algo[1]['success_rate']:.1%} success rate")


if __name__ == "__main__":
    main()
