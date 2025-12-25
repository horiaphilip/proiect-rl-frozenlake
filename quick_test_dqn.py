"""Quick test - antrenare si evaluare rapida DQN"""

from environments.dynamic_frozenlake import DynamicFrozenLakeEnv
from agents.dqn import DQNAgent

print("="*60)
print("TEST RAPID - DQN PE DYNAMIC FROZENLAKE")
print("="*60)

# Creaza mediul
env = DynamicFrozenLakeEnv(map_size=4, max_steps=50)

print(f"\nMediu: DynamicFrozenLake 4x4")
print(f"States: {env.observation_space.n}")
print(f"Actions: {env.action_space.n}")

# Creaza agent
agent = DQNAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.001,
    batch_size=32,
    buffer_capacity=5000,
    epsilon_decay=0.99
)

# Antrenare
print(f"\nAntrenare pe 200 episoade...")
for episode in range(200):
    stats = agent.train_episode(env)
    if episode % 50 == 0:
        loss = stats.get('avg_loss', 0)
        print(f"  Episode {episode}: reward={stats['total_reward']:.3f}, loss={loss:.4f}, buffer={stats['buffer_size']}")

print("\nAntrenare completa!")

# Evaluare
print("\nEvaluare pe 20 episoade...")
eval_stats = agent.evaluate(env, n_episodes=20)

print("\n" + "="*60)
print("REZULTATE FINALE")
print("="*60)
print(f"Mean Reward: {eval_stats['mean_reward']:.4f} +/- {eval_stats['std_reward']:.4f}")
print(f"Success Rate: {eval_stats['success_rate']:.1%}")
print(f"Mean Steps: {eval_stats['mean_steps']:.1f}")
print("="*60)

env.close()
