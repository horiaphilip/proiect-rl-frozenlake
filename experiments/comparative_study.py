import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch

from environments.easy_frozenlake import EasyFrozenLakeEnv
from environments.dynamic_frozenlake_medium_env import DynamicFrozenLakeEnv as MediumEnv
from environments.dynamic_frozenlake import DynamicFrozenLakeEnv as HardEnv

from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.dqn_per import DQN_PERAgent
from agents.ppo import PPOAgent
from agents.ppo_rnd import PPORndAgent


def create_easy_env(seed: int = 42) -> EasyFrozenLakeEnv:
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


def create_medium_env(seed: int = 42) -> MediumEnv:
    np.random.seed(seed)
    return MediumEnv(
        map_size=8,
        max_steps=140,
        slippery_start=0.02,
        slippery_end=0.15,
        step_penalty=-0.001,
        ice_melting=True,
        melting_rate=0.003,
        melt_cells_per_step=1,
        melt_delay_steps=10,
        shaped_rewards=True,
        shaping_scale=0.02,
        hole_penalty=-1.0,
        hole_ratio=0.12,
        protect_safe_zone_from_melting=True,
        protect_solution_path_from_melting=True,
        regenerate_map_each_episode=False,
    )


def create_hard_env(seed: int = 42) -> HardEnv:
    np.random.seed(seed)
    return HardEnv(
        map_size=8,
        max_steps=120,
        slippery_start=0.10,
        slippery_end=0.40,
        step_penalty=-0.01,
        ice_melting=True,
        melting_rate=0.005,
        melt_cells_per_step=1,
        shaped_rewards=True,
        shaping_scale=0.02,
        hole_penalty=-1.0,
        hole_ratio=0.20,
        protect_safe_zone_from_melting=True,
        regenerate_map_each_episode=False,
    )



EPISODES_CONFIG = {
    'easy': {'episodic': 500, 'ppo_timesteps': 25000},
    'medium': {'episodic': 2000, 'ppo_timesteps': 100000},
    'hard': {'episodic': 3000, 'ppo_timesteps': 150000},
}

QUICK_EPISODES_CONFIG = {
    'easy': {'episodic': 100, 'ppo_timesteps': 5000},
    'medium': {'episodic': 200, 'ppo_timesteps': 10000},
    'hard': {'episodic': 300, 'ppo_timesteps': 15000},
}


def get_agent_config(env_name: str, env, seed: int) -> Dict[str, Dict]:

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    configs = {
        'Q-Learning': {
            'class': QLearningAgent,
            'params': {
                'n_states': n_states,
                'n_actions': n_actions,
                'learning_rate': 0.1,
                'discount_factor': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01 if env_name != 'easy' else 0.05,
                'epsilon_decay': 0.995,
            },
            'type': 'episodic'
        },
        'DQN': {
            'class': DQNAgent,
            'params': {
                'n_states': n_states,
                'n_actions': n_actions,
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01 if env_name != 'easy' else 0.05,
                'epsilon_decay': 0.995,
                'buffer_capacity': 5000 if env_name == 'easy' else 20000,
                'batch_size': 32 if env_name == 'easy' else 64,
                'target_update_freq': 10,
                'hidden_dim': 64 if env_name == 'easy' else 128,
            },
            'type': 'episodic'
        },
        'DQN-PER': {
            'class': DQN_PERAgent,
            'params': {
                'n_states': n_states,
                'n_actions': n_actions,
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01 if env_name != 'easy' else 0.05,
                'epsilon_decay': 0.995,
                'buffer_capacity': 5000 if env_name == 'easy' else 20000,
                'batch_size': 32 if env_name == 'easy' else 64,
                'target_update_freq': 10,
                'hidden_dim': 64 if env_name == 'easy' else 128,
                'per_alpha': 0.6,
                'per_beta_start': 0.4,
                'per_beta_end': 1.0,
                'per_beta_anneal_steps': 50000,
                'seed': seed,
            },
            'type': 'episodic'
        },
        'PPO': {
            'class': PPOAgent,
            'params': {
                'env': env,
                'learning_rate': 0.0003,
                'n_steps': 512 if env_name == 'easy' else 1024,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'verbose': 0,
            },
            'type': 'ppo'
        },
        'PPO-RND': {
            'class': PPORndAgent,
            'params': {
                'env': env,
                'learning_rate': 0.0003,
                'n_steps': 512 if env_name == 'easy' else 1024,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'beta_int': 0.02 if env_name == 'easy' else 0.05,
                'rnd_lr': 1e-4,
                'normalize_int_reward': True,
                'seed': seed,
                'verbose': 0,
            },
            'type': 'ppo'
        },
    }

    return configs


def train_episodic_agent(agent, env, n_episodes: int, agent_name: str) -> Dict[str, Any]:
    import time
    start_time = time.time()

    rewards = []
    steps_list = []
    successes = []

    epsilons = []
    losses = []
    td_errors = []
    buffer_sizes = []
    q_values_mean = []
    q_values_max = []

    action_counts = {a: 0 for a in range(env.action_space.n)}
    state_visits = {}
    priority_samples = []

    for episode in tqdm(range(n_episodes), desc=f"{agent_name}", leave=False):
        stats = agent.train_episode(env)

        rewards.append(stats['total_reward'])
        steps_list.append(stats['steps'])
        successes.append(1 if stats['total_reward'] > 0.5 else 0)

        epsilons.append(stats.get('epsilon', 0.0))

        if 'avg_loss' in stats:
            losses.append(stats['avg_loss'])

        if 'avg_td_error' in stats:
            td_errors.append(stats['avg_td_error'])

        if 'buffer_size' in stats:
            buffer_sizes.append(stats['buffer_size'])

        if episode % max(1, n_episodes // 10) == 0:
            q_stats = _sample_q_values(agent, env)
            q_values_mean.append(q_stats['mean'])
            q_values_max.append(q_stats['max'])

        if hasattr(agent, 'buffer') and hasattr(agent.buffer, 'tree'):
            if episode % max(1, n_episodes // 5) == 0:
                priorities = _sample_priorities(agent)
                if priorities:
                    priority_samples.append(priorities)

    training_time = time.time() - start_time

    action_counts, state_visits = _collect_behavior_data(agent, env, n_episodes=50)

    eval_stats = agent.evaluate(env, n_episodes=100)

    return {
        'training_rewards': rewards,
        'training_steps': steps_list,
        'training_successes': successes,
        'epsilons': epsilons,
        'losses': losses if losses else None,
        'td_errors': td_errors if td_errors else None,
        'buffer_sizes': buffer_sizes if buffer_sizes else None,
        'q_values_mean': q_values_mean,
        'q_values_max': q_values_max,
        'training_time': training_time,
        'action_counts': action_counts,
        'state_visits': state_visits,
        'priority_samples': priority_samples if priority_samples else None,
        'eval': eval_stats,
    }


def _sample_q_values(agent, env, n_samples: int = 20) -> Dict[str, float]:
    q_values = []

    try:
        if hasattr(agent, 'q_table'):
            q_values = agent.q_table.flatten().tolist()

        elif hasattr(agent, 'policy_net'):
            import torch
            agent.policy_net.eval()
            n_states = env.observation_space.n

            sample_states = np.random.choice(n_states, min(n_samples, n_states), replace=False)

            with torch.no_grad():
                for state in sample_states:
                    state_tensor = torch.zeros(n_states, device=agent.device)
                    state_tensor[state] = 1.0
                    q_vals = agent.policy_net(state_tensor.unsqueeze(0))
                    q_values.extend(q_vals.cpu().numpy().flatten().tolist())

            agent.policy_net.train()
    except Exception:
        pass

    if not q_values:
        return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}

    return {
        'mean': float(np.mean(q_values)),
        'max': float(np.max(q_values)),
        'min': float(np.min(q_values)),
        'std': float(np.std(q_values)),
    }


def _sample_priorities(agent, n_samples: int = 100) -> List[float]:
    try:
        if hasattr(agent, 'buffer') and hasattr(agent.buffer, 'tree'):
            tree = agent.buffer.tree
            capacity = tree.capacity
            n_entries = tree.n_entries

            if n_entries == 0:
                return []

            leaf_start = capacity - 1
            priorities = []
            for i in range(min(n_samples, n_entries)):
                idx = leaf_start + (i % n_entries)
                if tree.tree[idx] > 0:
                    priorities.append(float(tree.tree[idx]))

            return priorities
    except Exception:
        pass
    return []


def _collect_behavior_data(agent, env, n_episodes: int = 50) -> tuple:
    action_counts = {a: 0 for a in range(env.action_space.n)}
    state_visits = {}

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            state_visits[state] = state_visits.get(state, 0) + 1

            action = agent.select_action(state, training=False)
            action_counts[action] = action_counts.get(action, 0) + 1

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

    return action_counts, state_visits


def _collect_ppo_behavior_data(agent, env, n_episodes: int = 50) -> tuple:
    action_counts = {a: 0 for a in range(env.action_space.n)}
    state_visits = {}

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            state_visits[state] = state_visits.get(state, 0) + 1

            action = agent.select_action(state, training=False)
            action_counts[action] = action_counts.get(action, 0) + 1

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

    return action_counts, state_visits


def train_ppo_agent(agent, env, total_timesteps: int, agent_name: str) -> Dict[str, Any]:
    import time
    start_time = time.time()

    print(f"  Training {agent_name} for {total_timesteps} timesteps...")

    rewards_history = []
    steps_history = []
    successes_history = []
    policy_losses = []
    value_losses = []
    entropies = []
    intrinsic_rewards = []

    segment_size = max(1, total_timesteps // 10)
    timesteps_trained = 0

    for segment in range(10):
        current_segment_steps = min(segment_size, total_timesteps - timesteps_trained)
        if current_segment_steps <= 0:
            break

        stats = agent.train(total_timesteps=current_segment_steps, progress_bar=False)
        timesteps_trained += current_segment_steps

        rewards_history.append(stats.get('mean_reward', 0.0))
        steps_history.append(stats.get('mean_length', 0.0))

        if 'policy_loss' in stats:
            policy_losses.append(stats['policy_loss'])
        if 'value_loss' in stats:
            value_losses.append(stats['value_loss'])
        if 'entropy' in stats:
            entropies.append(stats['entropy'])
        if 'intrinsic_reward_mean' in stats:
            intrinsic_rewards.append(stats['intrinsic_reward_mean'])

        quick_eval = agent.evaluate(env, n_episodes=20)
        successes_history.append(quick_eval['success_rate'])

        print(f"    Segment {segment+1}/10: {timesteps_trained}/{total_timesteps} steps, "
              f"SR={quick_eval['success_rate']:.1%}")

    training_time = time.time() - start_time

    action_counts, state_visits = _collect_ppo_behavior_data(agent, env, n_episodes=50)

    eval_stats = agent.evaluate(env, n_episodes=100)

    return {
        'training_rewards': rewards_history,
        'training_steps': steps_history,
        'training_successes': successes_history,
        'policy_losses': policy_losses if policy_losses else None,
        'value_losses': value_losses if value_losses else None,
        'entropies': entropies if entropies else None,
        'intrinsic_rewards': intrinsic_rewards if intrinsic_rewards else None,
        'training_time': training_time,
        'action_counts': action_counts,
        'state_visits': state_visits,
        'eval': eval_stats,
    }



def run_single_experiment(
    env_name: str,
    agent_name: str,
    seed: int,
    quick_mode: bool = False
) -> Dict[str, Any]:

    np.random.seed(seed)
    torch.manual_seed(seed)

    if env_name == 'easy':
        env = create_easy_env(seed)
    elif env_name == 'medium':
        env = create_medium_env(seed)
    else:
        env = create_hard_env(seed)

    config = QUICK_EPISODES_CONFIG if quick_mode else EPISODES_CONFIG
    episodes_config = config[env_name]

    agent_configs = get_agent_config(env_name, env, seed)
    agent_config = agent_configs[agent_name]

    agent = agent_config['class'](**agent_config['params'])

    if agent_config['type'] == 'episodic':
        results = train_episodic_agent(
            agent, env,
            n_episodes=episodes_config['episodic'],
            agent_name=agent_name
        )
    else:
        results = train_ppo_agent(
            agent, env,
            total_timesteps=episodes_config['ppo_timesteps'],
            agent_name=agent_name
        )

    env.close()

    return results


def run_comparative_study(
    n_seeds: int = 3,
    quick_mode: bool = False,
    agents: Optional[List[str]] = None,
    envs: Optional[List[str]] = None,
) -> Dict[str, Any]:

    if agents is None:
        agents = ['Q-Learning', 'DQN', 'DQN-PER', 'PPO', 'PPO-RND']
    if envs is None:
        envs = ['easy', 'medium', 'hard']

    seeds = list(range(42, 42 + n_seeds))

    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_seeds': n_seeds,
            'seeds': seeds,
            'agents': agents,
            'environments': envs,
            'quick_mode': quick_mode,
        },
        'results': {}
    }

    total_experiments = len(agents) * len(envs) * n_seeds
    print(f"\n{'='*60}")
    print(f"STUDIU COMPARATIV: {len(agents)} agenti x {len(envs)} env-uri x {n_seeds} seeds")
    print(f"Total experimente: {total_experiments}")
    print(f"{'='*60}\n")

    experiment_count = 0

    for env_name in envs:
        all_results['results'][env_name] = {}

        for agent_name in agents:
            all_results['results'][env_name][agent_name] = {
                'runs': [],
                'aggregated': {}
            }

            print(f"\n[{env_name.upper()}] {agent_name}")
            print("-" * 40)

            for seed in seeds:
                experiment_count += 1
                print(f"  Seed {seed} ({experiment_count}/{total_experiments})")

                try:
                    results = run_single_experiment(
                        env_name, agent_name, seed, quick_mode
                    )
                    all_results['results'][env_name][agent_name]['runs'].append({
                        'seed': seed,
                        **results
                    })

                    sr = results['eval']['success_rate']
                    mr = results['eval']['mean_reward']
                    print(f"    -> Success Rate: {sr:.1%}, Mean Reward: {mr:.4f}")

                except Exception as e:
                    print(f"    -> EROARE: {e}")
                    all_results['results'][env_name][agent_name]['runs'].append({
                        'seed': seed,
                        'error': str(e)
                    })

            aggregate_results(all_results['results'][env_name][agent_name])

    return all_results


def aggregate_results(agent_results: Dict) -> None:
    runs = agent_results['runs']
    valid_runs = [r for r in runs if 'error' not in r]

    if not valid_runs:
        agent_results['aggregated'] = {'error': 'No valid runs'}
        return

    success_rates = [r['eval']['success_rate'] for r in valid_runs]
    mean_rewards = [r['eval']['mean_reward'] for r in valid_runs]
    mean_steps = [r['eval']['mean_steps'] for r in valid_runs]

    agent_results['aggregated'] = {
        'n_valid_runs': len(valid_runs),
        'success_rate': {
            'mean': float(np.mean(success_rates)),
            'std': float(np.std(success_rates)),
            'min': float(np.min(success_rates)),
            'max': float(np.max(success_rates)),
        },
        'mean_reward': {
            'mean': float(np.mean(mean_rewards)),
            'std': float(np.std(mean_rewards)),
            'min': float(np.min(mean_rewards)),
            'max': float(np.max(mean_rewards)),
        },
        'mean_steps': {
            'mean': float(np.mean(mean_steps)),
            'std': float(np.std(mean_steps)),
            'min': float(np.min(mean_steps)),
            'max': float(np.max(mean_steps)),
        },
    }



def save_results(results: Dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"comparative_study_{timestamp}.json")

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

    print(f"\nRezultate salvate in: {filename}")
    return filename


def print_summary(results: Dict) -> None:
    print("\n" + "=" * 80)
    print("SUMAR REZULTATE")
    print("=" * 80)

    envs = results['metadata']['environments']
    agents = results['metadata']['agents']

    for env_name in envs:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name.upper()}")
        print(f"{'='*60}")
        print(f"\n{'Agent':<12} {'Success Rate':<20} {'Mean Reward':<20} {'Mean Steps':<15}")
        print("-" * 67)

        env_results = results['results'][env_name]

        for agent_name in agents:
            agg = env_results[agent_name]['aggregated']
            if 'error' in agg:
                print(f"{agent_name:<12} ERROR")
                continue

            sr = agg['success_rate']
            mr = agg['mean_reward']
            ms = agg['mean_steps']

            print(f"{agent_name:<12} "
                  f"{sr['mean']*100:>5.1f}% +/- {sr['std']*100:>4.1f}%    "
                  f"{mr['mean']:>6.4f} +/- {mr['std']:>5.4f}    "
                  f"{ms['mean']:>5.1f} +/- {ms['std']:>4.1f}")

        best_agent = None
        best_sr = -1
        for agent_name in agents:
            agg = env_results[agent_name]['aggregated']
            if 'error' not in agg and agg['success_rate']['mean'] > best_sr:
                best_sr = agg['success_rate']['mean']
                best_agent = agent_name

        if best_agent:
            print(f"\n  -> Cel mai bun: {best_agent} ({best_sr*100:.1f}% success rate)")



def main():
    parser = argparse.ArgumentParser(description='Studiu Comparativ RL')
    parser.add_argument('--seeds', type=int, default=3, help='Numar de seed-uri (default: 3)')
    parser.add_argument('--quick', action='store_true', help='Mod rapid pentru testare')
    parser.add_argument('--agents', nargs='+', help='Lista de agenti (default: toti)')
    parser.add_argument('--envs', nargs='+', help='Lista de env-uri (default: toate)')
    parser.add_argument('--output', type=str, default='../results', help='Director output')

    args = parser.parse_args()

    print("=" * 60)
    print("STUDIU COMPARATIV REINFORCEMENT LEARNING")
    print("5 Agenti x 3 Environment-uri")
    print("=" * 60)

    if args.quick:
        print("\n*** MOD RAPID ACTIVAT - rezultate orientative ***\n")

    results = run_comparative_study(
        n_seeds=args.seeds,
        quick_mode=args.quick,
        agents=args.agents,
        envs=args.envs,
    )

    output_file = save_results(results, args.output)

    print_summary(results)

    print(f"\n{'='*60}")
    print("STUDIU COMPLET!")
    print(f"{'='*60}")
    print(f"\nRezultate salvate in: {output_file}")
    print("Ruleaza: python experiments/visualize_comparative.py")
    print("pentru a genera graficele comparative.")


if __name__ == "__main__":
    main()