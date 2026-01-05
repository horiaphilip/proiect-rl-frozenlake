"""
Test rapid DQN+PER pe DynamicFrozenLake (8x8, dificil)

Compara performanta cu EasyFrozenLake (4x4, usor).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

from environments.dynamic_frozenlake import DynamicFrozenLakeEnv
from agents.dqn_per import DQN_PERAgent


def test_dqn_per_dynamic(env, n_episodes=500, seed=42):
    """Test DQN+PER on DynamicFrozenLake."""
    print("\n" + "="*60)
    print("DQN+PER on DynamicFrozenLake (8x8)")
    print("="*60)

    agent = DQN_PERAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,  # mai mare pentru env complex
        batch_size=64,
        target_update_freq=10,
        hidden_dim=128,  # mai mare pentru env complex
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_anneal_steps=50000,
        seed=seed
    )

    rewards = []
    successes = []

    print(f"\nTraining for {n_episodes} episodes...")
    for episode in tqdm(range(n_episodes), desc="Training"):
        stats = agent.train_episode(env)
        rewards.append(stats['total_reward'])
        successes.append(1 if stats['total_reward'] >= 0.9 else 0)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            recent_success = np.mean(successes[-100:]) * 100
            recent_reward = np.mean(rewards[-100:])
            print(f"\nEpisode {episode+1}: Recent Success Rate = {recent_success:.1f}%, "
                  f"Recent Mean Reward = {recent_reward:.3f}")

    # Final evaluation
    print("\n" + "-"*60)
    print("FINAL EVALUATION (100 episodes)")
    print("-"*60)

    eval_stats = agent.evaluate(env, n_episodes=100)

    print(f"\nMean Reward: {eval_stats['mean_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

    return {
        'algorithm': 'DQN+PER',
        'environment': 'DynamicFrozenLake',
        'training_rewards': rewards,
        'training_successes': successes,
        'eval': eval_stats
    }


def save_results(results, filename):
    """Save results to JSON."""
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


def compare_with_easy():
    """Load and compare with EasyFrozenLake results."""
    from pathlib import Path

    results_dir = Path("../results")
    easy_files = list(results_dir.glob("benchmark_easy_*.json"))

    if not easy_files:
        print("\nNo EasyFrozenLake results found for comparison.")
        return

    latest_easy = max(easy_files, key=lambda p: p.stat().st_mtime)

    with open(latest_easy, 'r') as f:
        easy_results = json.load(f)

    # Find DQN+PER results
    dqn_per_easy = next((r for r in easy_results if r['algorithm'] == 'DQN+PER'), None)

    if dqn_per_easy:
        print("\n" + "="*60)
        print("COMPARISON: EasyFrozenLake vs DynamicFrozenLake")
        print("="*60)
        print(f"\n{'Environment':<25} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Steps':<15}")
        print("-"*70)

        print(f"{'EasyFrozenLake (4x4)':<25} "
              f"{dqn_per_easy['eval']['success_rate']:<15.2%} "
              f"{dqn_per_easy['eval']['mean_reward']:<15.4f} "
              f"{dqn_per_easy['eval']['mean_steps']:<15.2f}")


def main():
    print("="*60)
    print("QUICK TEST: DQN+PER on DynamicFrozenLake")
    print("="*60)

    SEED = 42
    np.random.seed(SEED)

    # Create DynamicFrozenLake environment
    env = DynamicFrozenLakeEnv(
        map_size=8,
        max_steps=120,
        slippery_start=0.08,
        slippery_end=0.25,
        step_penalty=-0.01,
        hole_penalty=-1.0,
        ice_melting=True,
        melting_rate=0.003,
        shaped_rewards=True,
        shaping_scale=0.02,
        hole_ratio=0.18,
        protect_safe_zone_from_melting=True,
        melt_cells_per_step=1,
        regenerate_map_each_episode=False,
        max_map_tries=200
    )

    print(f"\nEnvironment: DynamicFrozenLake {env.map_size}x{env.map_size}")
    print(f"Slippery: {env.slippery_start} -> {env.slippery_end}")
    print(f"Hole ratio: {env.hole_ratio}")
    print(f"Ice melting: {env.ice_melting}")
    print(f"Max steps: {env.max_steps}")

    # Test DQN+PER
    results = test_dqn_per_dynamic(env, n_episodes=500, seed=SEED)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/dqn_per_dynamic_{timestamp}.json"
    save_results(results, filename)

    # Compare with Easy environment
    compare_with_easy()

    print("\n" + "="*60)
    print("TEST COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
