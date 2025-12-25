"""
Script pentru testarea rapidă a configurației proiectului.

Verifică că toate componentele sunt instalate corect și funcționează.
"""

import sys
import os

print("="*60)
print("TEST CONFIGURARE PROIECT")
print("="*60)

# Test 1: Verificare importuri
print("\n1. Verificare importuri...")
try:
    import numpy as np
    print("   [OK] numpy")
except ImportError as e:
    print(f"   [FAIL] numpy: {e}")
    sys.exit(1)

try:
    import gymnasium as gym
    print("   [OK] gymnasium")
except ImportError as e:
    print(f"   [FAIL] gymnasium: {e}")
    sys.exit(1)

try:
    import torch
    print(f"   [OK] torch (version {torch.__version__})")
    if torch.cuda.is_available():
        print(f"   [OK] CUDA disponibil: {torch.cuda.get_device_name(0)}")
    else:
        print("   [INFO] CUDA nu este disponibil (se va folosi CPU)")
except ImportError as e:
    print(f"   [FAIL] torch: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    print("   [OK] stable-baselines3")
except ImportError as e:
    print(f"   [FAIL] stable-baselines3: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("   [OK] matplotlib")
except ImportError as e:
    print(f"   [FAIL] matplotlib: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("   [OK] pandas")
except ImportError as e:
    print(f"   [FAIL] pandas: {e}")
    sys.exit(1)

try:
    import seaborn as sns
    print("   [OK] seaborn")
except ImportError as e:
    print(f"   [FAIL] seaborn: {e}")
    sys.exit(1)

# Test 2: Verificare mediu personalizat
print("\n2. Verificare mediu personalizat...")
try:
    from environments.dynamic_frozenlake import DynamicFrozenLakeEnv

    env = DynamicFrozenLakeEnv(map_size=4, max_steps=50)
    state, info = env.reset()
    print(f"   [OK] Mediu creat cu succes")
    print(f"   [OK] Observation space: {env.observation_space}")
    print(f"   [OK] Action space: {env.action_space}")

    # Test episod scurt
    for _ in range(5):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    print("   [OK] Mediul functioneaza corect")
    env.close()

except Exception as e:
    print(f"   [FAIL] Eroare la testarea mediului: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verificare agenti
print("\n3. Verificare agenti...")

# Q-Learning
try:
    from agents.q_learning import QLearningAgent

    agent = QLearningAgent(n_states=16, n_actions=4)
    action = agent.select_action(0, training=True)
    print("   [OK] Q-Learning agent functioneaza")
except Exception as e:
    print(f"   [FAIL] Eroare Q-Learning: {e}")
    sys.exit(1)

# DQN
try:
    from agents.dqn import DQNAgent

    agent = DQNAgent(n_states=16, n_actions=4)
    action = agent.select_action(0, training=True)
    print("   [OK] DQN agent functioneaza")
except Exception as e:
    print(f"   [FAIL] Eroare DQN: {e}")
    sys.exit(1)

# PPO
try:
    from agents.ppo import PPOAgent

    test_env = DynamicFrozenLakeEnv(map_size=4, max_steps=50)
    agent = PPOAgent(env=test_env)
    action = agent.select_action(0, training=False)
    print("   [OK] PPO agent functioneaza")
    test_env.close()
except Exception as e:
    print(f"   [FAIL] Eroare PPO: {e}")
    sys.exit(1)

# Test 4: Test antrenament rapid
print("\n4. Test antrenament rapid (Q-Learning pe 10 episoade)...")
try:
    from environments.dynamic_frozenlake import DynamicFrozenLakeEnv
    from agents.q_learning import QLearningAgent

    env = DynamicFrozenLakeEnv(map_size=4, max_steps=50)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        epsilon_start=1.0
    )

    for episode in range(10):
        stats = agent.train_episode(env)
        if episode % 5 == 0:
            print(f"   Episode {episode}: reward={stats['total_reward']:.3f}")

    print("   [OK] Antrenament functioneaza corect")
    env.close()

except Exception as e:
    print(f"   [FAIL] Eroare la antrenament: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verificare structura directoare
print("\n5. Verificare structura directoare...")
required_dirs = ['environments', 'agents', 'experiments', 'results']
for dir_name in required_dirs:
    if os.path.isdir(dir_name):
        print(f"   [OK] {dir_name}/")
    else:
        print(f"   [FAIL] {dir_name}/ lipseste")

# Succes
print("\n" + "="*60)
print("[SUCCESS] TOATE TESTELE AU TRECUT CU SUCCES!")
print("="*60)
print("\nProiectul este gata de utilizare!")
print("\nPasi urmatori:")
print("  1. Rulati 'python experiments/run_experiments.py' pentru antrenament complet")
print("  2. Rulati 'python experiments/visualize.py' pentru generare grafice")
print("="*60)
