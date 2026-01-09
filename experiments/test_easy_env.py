"""
Script de test pentru EasyFrozenLake environment.

Rulează rapid Q-Learning și DQN pe mediul easy pentru a verifica că pot învăța.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from environments.easy_frozenlake import EasyFrozenLakeEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent


def test_q_learning(env, n_episodes=500):
    """Test rapid Q-Learning pe environment easy."""
    print("\n" + "="*60)
    print("TEST Q-LEARNING PE EASY ENVIRONMENT")
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
    rewards_window = []
    for episode in tqdm(range(n_episodes), desc="Q-Learning Training"):
        stats = agent.train_episode(env)
        rewards_window.append(stats['total_reward'])

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            mean_reward = np.mean(rewards_window[-100:])
            print(f"Episode {episode+1}: Mean Reward (last 100) = {mean_reward:.4f}, Epsilon = {stats['epsilon']:.3f}")

    # Evaluation
    print("\nEvaluare finală...")
    eval_stats = agent.evaluate(env, n_episodes=100)

    print(f"\n{'='*60}")
    print("REZULTATE Q-LEARNING:")
    print(f"{'='*60}")
    print(f"Mean Reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

    return eval_stats


def test_dqn(env, n_episodes=500):
    """Test rapid DQN pe environment easy."""
    print("\n" + "="*60)
    print("TEST DQN PE EASY ENVIRONMENT")
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
        hidden_dim=64  # mai mic pentru environment simplu
    )

    # Training
    rewards_window = []
    for episode in tqdm(range(n_episodes), desc="DQN Training"):
        stats = agent.train_episode(env)
        rewards_window.append(stats['total_reward'])

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            mean_reward = np.mean(rewards_window[-100:])
            print(f"Episode {episode+1}: Mean Reward (last 100) = {mean_reward:.4f}, "
                  f"Epsilon = {stats['epsilon']:.3f}, Loss = {stats['avg_loss']:.4f}")

    # Evaluation
    print("\nEvaluare finală...")
    eval_stats = agent.evaluate(env, n_episodes=100)

    print(f"\n{'='*60}")
    print("REZULTATE DQN:")
    print(f"{'='*60}")
    print(f"Mean Reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

    return eval_stats


def main():
    print("="*60)
    print("TEST EASY FROZENLAKE ENVIRONMENT")
    print("="*60)

    # Creează environment easy
    env = EasyFrozenLakeEnv(
        map_size=4,
        max_steps=50,
        slippery=0.05,  # foarte mic
        step_penalty=-0.01,
        hole_penalty=-0.5,
        goal_reward=1.0,
        shaped_rewards=True,
        shaping_scale=0.05,
        hole_ratio=0.10,  # doar 10% găuri
        safe_zone_radius=1,
        regenerate_map_each_episode=False,
        seed=42
    )

    print(f"\nMediu: EasyFrozenLake {env.map_size}x{env.map_size}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max steps: {env.max_steps}")
    print(f"Slippery prob: {env.slippery}")
    print(f"Hole ratio: {env.hole_ratio}")

    # Afiseaza harta
    print("\nHarta initiala:")
    env.reset()
    print(env.render())

    # Test Q-Learning
    q_results = test_q_learning(env, n_episodes=500)

    # Test DQN
    dqn_results = test_dqn(env, n_episodes=500)

    # Comparatie finala
    print("\n" + "="*60)
    print("COMPARATIE FINALA")
    print("="*60)
    print(f"\nQ-Learning Success Rate: {q_results['success_rate']:.2%}")
    print(f"DQN Success Rate: {dqn_results['success_rate']:.2%}")

    if q_results['success_rate'] > 0 or dqn_results['success_rate'] > 0:
        print("\nSUCCESS! Cel putin un agent a invatat sa ajunga la goal!")
    else:
        print("\nWARNING: Success rate inca 0. Incearca sa reduci si mai mult dificultatea.")


if __name__ == "__main__":
    main()