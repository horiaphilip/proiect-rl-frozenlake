"""
Demo Script - Demonstrație vizuală a mediului DynamicFrozenLake

Rulează un agent aleator pe mediu și afișează starea la fiecare pas.
"""

from environments.dynamic_frozenlake import DynamicFrozenLakeEnv
import time


def run_random_agent_demo(map_size=8, n_episodes=3, delay=0.5):
    """
    Demonstrație cu agent aleator.

    Args:
        map_size: Dimensiunea hărții
        n_episodes: Număr de episoade de demonstrat
        delay: Întârziere între pași (secunde)
    """
    print("="*60)
    print("DEMO: DYNAMIC FROZENLAKE")
    print("="*60)
    print(f"\nHartă: {map_size}x{map_size}")
    print("Agent: Random (acțiuni aleatorii)")
    print("\nLegendă:")
    print("  S = Start")
    print("  F = Frozen (gheață)")
    print("  H = Hole (gaură)")
    print("  G = Goal (destinație)")
    print("  X = Poziția agentului")
    print("\nAcțiuni: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")
    print("="*60)

    # Creează mediul
    env = DynamicFrozenLakeEnv(
        map_size=map_size,
        max_steps=100,
        render_mode="ansi"
    )

    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    for episode in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"EPISOD {episode + 1}/{n_episodes}")
        print(f"{'='*60}")

        state, info = env.reset()
        total_reward = 0
        step_count = 0

        print(env.render())
        time.sleep(delay * 2)  # Pauză mai lungă la început

        while True:
            # Selectează acțiune aleatorie
            action = env.action_space.sample()

            # Execută acțiune
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_count += 1
            total_reward += reward

            # Afișează starea
            print(f"\nPas {step_count}:")
            print(f"  Acțiune: {action_names[action]} ({action})")
            print(f"  Reward: {reward:.3f}")
            print(f"  Slippery prob: {info['slippery_prob']:.2f}")
            print(env.render())

            time.sleep(delay)

            if done:
                print(f"\n{'─'*60}")
                if terminated and reward > 0:
                    print("✓ SUCCES! Agentul a ajuns la destinație!")
                elif terminated:
                    print("✗ EȘEC! Agentul a căzut într-o gaură!")
                else:
                    print("⏱ TIMEOUT! Agentul a depășit limita de pași!")

                print(f"Total Reward: {total_reward:.3f}")
                print(f"Total Steps: {step_count}")
                print(f"{'─'*60}")
                break

            state = next_state

        if episode < n_episodes - 1:
            print("\nApasă Enter pentru următorul episod...")
            input()

    env.close()

    print(f"\n{'='*60}")
    print("DEMO COMPLETĂ")
    print(f"{'='*60}")


def run_trained_agent_demo():
    """Demo cu un agent antrenat (exemplu rapid Q-Learning)."""
    from agents.q_learning import QLearningAgent

    print("="*60)
    print("DEMO: AGENT Q-LEARNING ANTRENAT")
    print("="*60)

    # Creează mediul
    env = DynamicFrozenLakeEnv(map_size=4, max_steps=50)  # Hartă mică pentru antrenament rapid

    # Creează agent
    print("\nAntrenare agent Q-Learning pe 200 episoade...")
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.2,
        epsilon_decay=0.99
    )

    # Antrenament rapid
    for episode in range(200):
        agent.train_episode(env)
        if episode % 50 == 0:
            print(f"  Episode {episode}/200")

    print("✓ Antrenare completă!\n")

    # Demo cu agentul antrenat
    print("="*60)
    print("DEMONSTRAȚIE CU AGENT ANTRENAT")
    print("="*60)

    env_render = DynamicFrozenLakeEnv(map_size=4, max_steps=50, render_mode="ansi")
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    state, _ = env_render.reset()
    total_reward = 0
    step_count = 0

    print(env_render.render())
    time.sleep(1)

    while True:
        # Agent selectează acțiune (fără explorare)
        action = agent.select_action(state, training=False)

        next_state, reward, terminated, truncated, info = env_render.step(action)
        done = terminated or truncated

        step_count += 1
        total_reward += reward

        print(f"\nPas {step_count}:")
        print(f"  Acțiune: {action_names[action]} (Q-value: {agent.get_q_values(state)[action]:.3f})")
        print(f"  Reward: {reward:.3f}")
        print(env_render.render())

        time.sleep(0.8)

        if done:
            print(f"\n{'─'*60}")
            if terminated and reward > 0:
                print("✓ SUCCES!")
            elif terminated:
                print("✗ EȘEC!")
            else:
                print("⏱ TIMEOUT!")

            print(f"Total Reward: {total_reward:.3f}")
            print(f"Total Steps: {step_count}")
            print(f"{'─'*60}")
            break

        state = next_state

    env_render.close()

    # Evaluează performanța
    print("\nEvaluare finală pe 20 episoade...")
    eval_stats = agent.evaluate(env, n_episodes=20)
    print(f"Mean Reward: {eval_stats['mean_reward']:.4f}")
    print(f"Success Rate: {eval_stats['success_rate']:.1%}")

    env.close()


def main():
    """Funcția principală."""
    print("\nAlegeți demo:")
    print("1. Agent aleator (vizualizare mediu)")
    print("2. Agent Q-Learning antrenat")
    print("3. Ambele")

    choice = input("\nAlegeți opțiunea (1/2/3): ").strip()

    if choice == "1":
        run_random_agent_demo(map_size=8, n_episodes=2, delay=0.3)
    elif choice == "2":
        run_trained_agent_demo()
    elif choice == "3":
        run_random_agent_demo(map_size=8, n_episodes=1, delay=0.3)
        print("\n" * 3)
        run_trained_agent_demo()
    else:
        print("Opțiune invalidă. Rulare demo implicit (agent aleator)...")
        run_random_agent_demo(map_size=8, n_episodes=2, delay=0.3)


if __name__ == "__main__":
    main()
