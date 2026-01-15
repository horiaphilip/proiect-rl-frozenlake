import numpy as np
from typing import Tuple, Optional, Dict, Any
import pickle


class QLearningAgent:
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
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((n_states, n_actions))

        self.training_steps = 0

    def select_action(self, state: int, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            # Explorare: acțiune aleatorie
            return np.random.randint(self.n_actions)
        else:
            # Exploatare: acțiunea cu Q-value maxim
            return np.argmax(self.q_table[state])

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * np.max(self.q_table[next_state])

        td_error = td_target - self.q_table[state, action]

        self.q_table[state, action] += self.learning_rate * td_error

        self.training_steps += 1

        return abs(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        td_errors = []

        while True:
            action = self.select_action(state, training=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            td_error = self.update(state, action, reward, next_state, done)
            td_errors.append(td_error)

            total_reward += reward
            steps += 1
            state = next_state

            if done or (max_steps and steps >= max_steps):
                break

        self.decay_epsilon()

        return {
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'avg_td_error': np.mean(td_errors),
            'max_td_error': np.max(td_errors)
        }

    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        rewards = []
        steps_list = []
        success_count = 0

        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0

            while True:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            rewards.append(total_reward)
            steps_list.append(steps)
            if total_reward > 0.5:
                success_count += 1

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps_list),
            'success_rate': success_count / n_episodes
        }

    def save(self, filepath: str):
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'hyperparameters': {
                'n_states': self.n_states,
                'n_actions': self.n_actions,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']

        hp = data['hyperparameters']
        self.n_states = hp['n_states']
        self.n_actions = hp['n_actions']
        self.learning_rate = hp['learning_rate']
        self.discount_factor = hp['discount_factor']
        self.epsilon_end = hp['epsilon_end']
        self.epsilon_decay = hp['epsilon_decay']

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.q_table, axis=1)

    def get_q_values(self, state: int) -> np.ndarray:
        return self.q_table[state]
