import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

from environments.dynamic_frozenlake import DynamicFrozenLakeEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.ppo_rnd import PPORndAgent
from agents.dqn_per import DQN_PERAgent
from agents.ppo import PPOAgent


def create_results_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../results/experiment_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_results(results, results_dir):
    results_file = os.path.join(results_dir, "results.json")

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

    with open(results_file, "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nRezultate salvate în: {results_file}")


# =========================================================
# Q-Learning
# =========================================================
def run_q_learning_experiment(env, n_episodes=20000, n_runs=1):
    print("\n" + "=" * 60)
    print("ANTRENARE Q-LEARNING")
    print("=" * 60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=0.20,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.10,
            epsilon_decay=0.9997
        )

        episode_rewards, episode_steps, epsilons, td_errors = [], [], [], []

        for episode in tqdm(range(n_episodes), desc=f"Q-Learning Run {run+1}"):
            stats = agent.train_episode(env)
            episode_rewards.append(stats["total_reward"])
            episode_steps.append(stats["steps"])
            epsilons.append(stats["epsilon"])
            td_errors.append(stats["avg_td_error"])

        eval_stats = agent.evaluate(env, n_episodes=500)

        all_results.append({
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "epsilons": epsilons,
            "td_errors": td_errors,
            "final_eval": eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {"algorithm": "Q-Learning", "runs": all_results}


# =========================================================
# DQN
# =========================================================
def run_dqn_experiment(env, n_episodes=6000, n_runs=1):
    print("\n" + "=" * 60)
    print("ANTRENARE DQN (stabilized)")
    print("=" * 60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = DQNAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,

            learning_rate=3e-4,
            discount_factor=0.99,

            epsilon_start=1.0,
            epsilon_end=0.10,
            epsilon_decay=0.999,

            buffer_capacity=50000,
            batch_size=64,

            hidden_dim=256,

            # stabilitate
            min_replay_size=2000,
            train_freq=1,

            # soft target update (recomandat)
            tau=0.01,

            # Double DQN
            use_double_dqn=True,

            max_grad_norm=10.0,
        )

        episode_rewards, episode_steps, epsilons, losses, buffer_sizes = [], [], [], [], []

        for episode in tqdm(range(n_episodes), desc=f"DQN Run {run+1}"):
            stats = agent.train_episode(env)
            episode_rewards.append(stats["total_reward"])
            episode_steps.append(stats["steps"])
            epsilons.append(stats["epsilon"])
            losses.append(stats["avg_loss"])
            buffer_sizes.append(stats["buffer_size"])

        eval_stats = agent.evaluate(env, n_episodes=500)

        all_results.append({
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "epsilons": epsilons,
            "losses": losses,
            "buffer_sizes": buffer_sizes,
            "final_eval": eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {"algorithm": "DQN", "runs": all_results}


# =========================================================
# PPO + RND
# =========================================================
def run_ppo_rnd_experiment(env, total_timesteps=250000, n_runs=1):
    print("\n" + "=" * 60)
    print("ANTRENARE PPO + RND")
    print("=" * 60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = PPORndAgent(env=env, verbose=0)

        print(f"Training PPO+RND for {total_timesteps} timesteps...")
        stats = agent.train(total_timesteps=total_timesteps, progress_bar=True)

        eval_stats = agent.evaluate(env, n_episodes=500)

        all_results.append({
            "training_stats": stats,
            "final_eval": eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {"algorithm": "PPO+RND", "runs": all_results}

def run_dqn_per_experiment(env, n_episodes=6000, n_runs=1):
    print("\n" + "=" * 60)
    print("ANTRENARE DQN + PER")
    print("=" * 60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = DQN_PERAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=3e-4,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.10,
            epsilon_decay=0.999,
            buffer_capacity=50000,
            batch_size=64,
            target_update_freq=10,
            hidden_dim=256,

            # PER params (default-uri ok)
            per_alpha=0.6,
            per_beta_start=0.4,
            per_beta_end=1.0,
            per_beta_anneal_steps=100000,
            per_eps=1e-6
        )

        episode_rewards, episode_steps, epsilons, losses, buffer_sizes = [], [], [], [], []

        for episode in tqdm(range(n_episodes), desc=f"DQN+PER Run {run+1}"):
            stats = agent.train_episode(env)
            episode_rewards.append(stats["total_reward"])
            episode_steps.append(stats["steps"])
            epsilons.append(stats["epsilon"])
            losses.append(stats["avg_loss"])
            buffer_sizes.append(stats["buffer_size"])

        eval_stats = agent.evaluate(env, n_episodes=500)

        all_results.append({
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "epsilons": epsilons,
            "losses": losses,
            "buffer_sizes": buffer_sizes,
            "final_eval": eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {"algorithm": "DQN+PER", "runs": all_results}


def run_ppo_experiment(env, total_timesteps=250000, n_runs=1):
    print("\n" + "=" * 60)
    print("ANTRENARE PPO")
    print("=" * 60)

    all_results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        agent = PPOAgent(
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.03,
            verbose=0
        )

        print(f"Training PPO for {total_timesteps} timesteps...")
        stats = agent.train(total_timesteps=600000, progress_bar=True)

        eval_stats = agent.evaluate(env, n_episodes=500)

        all_results.append({
            "training_stats": stats,
            "final_eval": eval_stats
        })

        print(f"Final Evaluation - Mean Reward: {eval_stats['mean_reward']:.4f}, "
              f"Success Rate: {eval_stats['success_rate']:.2%}")

    return {"algorithm": "PPO", "runs": all_results}



# =========================================================
# Main
# =========================================================
def main():
    print("=" * 60)
    print("EXPERIMENTE REINFORCEMENT LEARNING")
    print("Mediu: Dynamic FrozenLake (melting ON, solvable, time-aware)")
    print("=" * 60)

    MAP_SIZE = 8
    results_dir = create_results_dir()
    print(f"\nRezultate vor fi salvate în: {results_dir}")

    # ✅ mediu “hard dar learnable”
    env = DynamicFrozenLakeEnv(
        map_size=MAP_SIZE,

        max_steps=160,
        slippery_start=0.02,
        slippery_end=0.12,
        step_penalty=-0.001,

        ice_melting=True,
        melting_rate=0.002,
        melt_cells_per_step=1,
        melt_delay_steps=25,

        hole_ratio=0.10,

        shaped_rewards=True,
        shaping_scale=0.02,
        hole_penalty=-1.0,

        protect_safe_zone_from_melting=True,
        protect_solution_path_from_melting=True,
        regenerate_map_each_episode=False,

        time_buckets=2,  # ✅ mic pentru Q-learning, ok și pentru DQN
    )

    print(f"\nMediu: DynamicFrozenLake {MAP_SIZE}x{MAP_SIZE}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    all_results = {}

    # 1) Q-Learning
    all_results["q_learning"] = run_q_learning_experiment(env, n_episodes=20000, n_runs=1)

    # 2) DQN
    all_results["dqn"] = run_dqn_experiment(env, n_episodes=6000, n_runs=1)

    # 3) PPO + RND
    all_results["ppo_rnd"] = run_ppo_rnd_experiment(env, total_timesteps=250000, n_runs=1)

    # #4) PPO
    all_results["ppo"] = run_ppo_experiment(env, total_timesteps=250000, n_runs=1)

    # 5) DQN+PER
    all_results["dqn_per"] = run_dqn_per_experiment(env, n_episodes=6000, n_runs=1)

    save_results(all_results, results_dir)

    print("\n" + "=" * 60)
    print("REZUMAT FINAL")
    print("=" * 60)

    for algo_name, algo_results in all_results.items():
        print(f"\n{algo_name.upper()}:")
        final_evals = [run["final_eval"] for run in algo_results["runs"]]
        mean_rewards = [e["mean_reward"] for e in final_evals]
        success_rates = [e["success_rate"] for e in final_evals]
        print(f"  Mean Reward: {np.mean(mean_rewards):.4f} ± {np.std(mean_rewards):.4f}")
        print(f"  Success Rate: {np.mean(success_rates):.2%} ± {np.std(success_rates):.2%}")

    print(f"\nRezultate complete salvate în: {results_dir}")
    print("Rulați experiments/visualize.py pentru a genera grafice.")


if __name__ == "__main__":
    main()