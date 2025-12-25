"""
Q-Learning Agent (Tabular Method)

Algoritm clasic de Reinforcement Learning care învață o tabelă Q pentru fiecare
pereche (stare, acțiune).
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
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
        """
        Inițializare agent Q-Learning.

        Args:
            n_states: Numărul de stări din mediu
            n_actions: Numărul de acțiuni posibile
            learning_rate: Rata de învățare (alpha)
            discount_factor: Factorul de discount (gamma)
            epsilon_start: Valoarea inițială a epsilon pentru epsilon-greedy
            epsilon_end: Valoarea minimă a epsilon
            epsilon_decay: Rata de decay a epsilon
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Inițializare tabelă Q cu 0
        self.q_table = np.zeros((n_states, n_actions))

        # Statistici
        self.training_steps = 0

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Selectează o acțiune folosind politica epsilon-greedy.

        Args:
            state: Starea curentă
            training: Dacă este în modul training (folosește epsilon-greedy)

        Returns:
            Acțiunea selectată
        """
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
        """
        Actualizează tabelul Q folosind regula Q-Learning.

        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        Args:
            state: Starea curentă
            action: Acțiunea executată
            reward: Reward-ul primit
            next_state: Noua stare
            done: Dacă episodul s-a terminat

        Returns:
            TD error (pentru monitoring)
        """
        # Calculează TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * np.max(self.q_table[next_state])

        # Calculează TD error
        td_error = td_target - self.q_table[state, action]

        # Actualizează Q-value
        self.q_table[state, action] += self.learning_rate * td_error

        self.training_steps += 1

        return abs(td_error)

    def decay_epsilon(self):
        """Reduce epsilon pentru mai puțină explorare."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Antrenează agentul pe un episod complet.

        Args:
            env: Mediul de antrenament
            max_steps: Număr maxim de pași (None = fără limită)

        Returns:
            Dicționar cu statistici despre episod
        """
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        td_errors = []

        while True:
            # Selectează acțiune
            action = self.select_action(state, training=True)

            # Execută acțiune
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Actualizează Q-table
            td_error = self.update(state, action, reward, next_state, done)
            td_errors.append(td_error)

            total_reward += reward
            steps += 1
            state = next_state

            if done or (max_steps and steps >= max_steps):
                break

        # Decay epsilon
        self.decay_epsilon()

        return {
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'avg_td_error': np.mean(td_errors),
            'max_td_error': np.max(td_errors)
        }

    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluează agentul fără explorare.

        Args:
            env: Mediul de evaluare
            n_episodes: Număr de episoade de evaluare

        Returns:
            Dicționar cu metrici de evaluare
        """
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
            if total_reward > 0.5:  # Consider success dacă reward > 0.5
                success_count += 1

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps_list),
            'success_rate': success_count / n_episodes
        }

    def save(self, filepath: str):
        """Salvează agentul."""
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
        """Încarcă agentul."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']

        # Restaurare hiperparametri
        hp = data['hyperparameters']
        self.n_states = hp['n_states']
        self.n_actions = hp['n_actions']
        self.learning_rate = hp['learning_rate']
        self.discount_factor = hp['discount_factor']
        self.epsilon_end = hp['epsilon_end']
        self.epsilon_decay = hp['epsilon_decay']

    def get_policy(self) -> np.ndarray:
        """
        Returnează politica învățată (acțiunea optimă pentru fiecare stare).

        Returns:
            Array de dimensiune (n_states,) cu acțiunea optimă pentru fiecare stare
        """
        return np.argmax(self.q_table, axis=1)

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Returnează Q-values pentru o stare dată.

        Args:
            state: Starea pentru care se cer Q-values

        Returns:
            Array de dimensiune (n_actions,) cu Q-values
        """
        return self.q_table[state]
