"""
Q-Learning Agent (Tabular Method)

Algoritm clasic de Reinforcement Learning care învață o tabelă Q pentru fiecare
pereche (stare, acțiune).
"""

import numpy as np
from typing import Optional, Dict, Any
import pickle


class QLearningAgent:
    """
    Agent Q-Learning (Tabular).

    Învață o tabelă Q(s, a) care estimează reward-ul cumulativ așteptat pentru
    fiecare pereche (stare, acțiune).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

        self.training_steps = 0

    def select_action(self, state: int, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[int(state)]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        state = int(state)
        next_state = int(next_state)
        action = int(action)
        reward = float(reward)

        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * float(np.max(self.q_table[next_state]))

        td_error = td_target - float(self.q_table[state, action])
        self.q_table[state, action] += self.learning_rate * td_error

        self.training_steps += 1
        return float(abs(td_error))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        td_errors = []

        while True:
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            td_error = self.update(state, action, reward, next_state, done)
            td_errors.append(td_error)

            total_reward += float(reward)
            steps += 1
            state = next_state

            if done or (max_steps is not None and steps >= max_steps):
                break

        self.decay_epsilon()

        return {
            "total_reward": float(total_reward),
            "steps": int(steps),
            "epsilon": float(self.epsilon),
            "avg_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
            "max_td_error": float(np.max(td_errors)) if td_errors else 0.0,
        }

    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        rewards = []
        steps_list = []
        success_count = 0

        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            last_reward = 0.0
            done = False

            while True:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                last_reward = reward
                steps += 1
                state = next_state

                if done:
                    break

            rewards.append(total_reward)
            steps_list.append(steps)

            # ✅ SUCCES = ai terminat episodul cu reward final 1.0 (Goal)
            if done and last_reward >= 1.0:
                success_count += 1

        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_steps': float(np.mean(steps_list)),
            'success_rate': success_count / n_episodes
        }

    def save(self, filepath: str):
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "hyperparameters": {
                "n_states": self.n_states,
                "n_actions": self.n_actions,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        self.training_steps = data["training_steps"]

        hp = data["hyperparameters"]
        self.n_states = hp["n_states"]
        self.n_actions = hp["n_actions"]
        self.learning_rate = hp["learning_rate"]
        self.discount_factor = hp["discount_factor"]
        self.epsilon_end = hp["epsilon_end"]
        self.epsilon_decay = hp["epsilon_decay"]

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.q_table, axis=1)

    def get_q_values(self, state: int) -> np.ndarray:
        return self.q_table[int(state)]
