"""
DQN Agent (Deep Q-Network)

Algoritm Deep RL care folosește o rețea neuronală pentru a aproxima funcția Q.
Include Experience Replay și Target Network pentru stabilitate.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional, Dict, Any


class QNetwork(nn.Module):
    """Rețea neuronală pentru aproximarea funcției Q."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Inițializare rețea Q.

        Args:
            state_dim: Dimensiunea spațiului de stări
            action_dim: Dimensiunea spațiului de acțiuni
            hidden_dim: Dimensiunea straturilor ascunse
        """
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        """Forward pass prin rețea."""
        return self.network(x)


class ReplayBuffer:
    """Buffer pentru stocarea experiențelor (Experience Replay)."""

    def __init__(self, capacity: int):
        """
        Inițializare replay buffer.

        Args:
            capacity: Capacitatea maximă a buffer-ului
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Adaugă o experiență în buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Eșantionează un batch aleator din buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        """Returnează dimensiunea curentă a buffer-ului."""
        return len(self.buffer)


class DQNAgent:
    """
    Agent DQN (Deep Q-Network).

    Folosește o rețea neuronală pentru a aproxima funcția Q și include
    Experience Replay și Target Network.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_dim: int = 128,
    ):
        """
        Inițializare agent DQN.

        Args:
            n_states: Numărul de stări din mediu
            n_actions: Numărul de acțiuni posibile
            learning_rate: Rata de învățare
            discount_factor: Factorul de discount (gamma)
            epsilon_start: Valoarea inițială a epsilon
            epsilon_end: Valoarea minimă a epsilon
            epsilon_decay: Rata de decay a epsilon
            buffer_capacity: Capacitatea replay buffer-ului
            batch_size: Dimensiunea batch-ului pentru antrenament
            target_update_freq: Frecvența de actualizare a target network
            hidden_dim: Dimensiunea straturilor ascunse
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device (CPU sau GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Rețele Q (policy și target)
        self.policy_net = QNetwork(n_states, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(n_states, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer și loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Statistici
        self.training_steps = 0
        self.episodes_trained = 0

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        """Convertește starea într-un tensor one-hot."""
        state_tensor = torch.zeros(self.n_states, device=self.device)
        state_tensor[state] = 1.0
        return state_tensor

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Selectează o acțiune folosind politica epsilon-greedy.

        Args:
            state: Starea curentă
            training: Dacă este în modul training

        Returns:
            Acțiunea selectată
        """
        if training and np.random.random() < self.epsilon:
            # Explorare
            return np.random.randint(self.n_actions)
        else:
            # Exploatare
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def update(self) -> Optional[float]:
        """
        Actualizează rețeaua folosind un batch din replay buffer.

        Returns:
            Loss-ul mediu sau None dacă buffer-ul nu are suficiente sample-uri
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Eșantionează batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convertește în tensori
        states_tensor = torch.stack([self._state_to_tensor(s) for s in states]).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.stack([self._state_to_tensor(s) for s in next_states]).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Calculează Q-values curente
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()

        # Calculează Q-values target
        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor).max(dim=1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.discount_factor * next_q_values

        # Calculează loss
        loss = self.criterion(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        # Actualizează target network
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Reduce epsilon pentru mai puțină explorare."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Antrenează agentul pe un episod complet.

        Args:
            env: Mediul de antrenament
            max_steps: Număr maxim de pași

        Returns:
            Dicționar cu statistici despre episod
        """
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        losses = []

        while True:
            # Selectează acțiune
            action = self.select_action(state, training=True)

            # Execută acțiune
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Adaugă experiența în replay buffer
            self.replay_buffer.push(state, action, reward, next_state, float(done))

            # Actualizează rețeaua
            loss = self.update()
            if loss is not None:
                losses.append(loss)

            total_reward += reward
            steps += 1
            state = next_state

            if done or (max_steps and steps >= max_steps):
                break

        # Decay epsilon
        self.decay_epsilon()
        self.episodes_trained += 1

        return {
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'buffer_size': len(self.replay_buffer)
        }

    # def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
    #     """
    #     Evaluează agentul fără explorare.
    #
    #     Args:
    #         env: Mediul de evaluare
    #         n_episodes: Număr de episoade de evaluare
    #
    #     Returns:
    #         Dicționar cu metrici de evaluare
    #     """
    #     rewards = []
    #     steps_list = []
    #     success_count = 0
    #
    #     for _ in range(n_episodes):
    #         state, _ = env.reset()
    #         total_reward = 0
    #         steps = 0
    #
    #         while True:
    #             action = self.select_action(state, training=False)
    #             next_state, reward, terminated, truncated, _ = env.step(action)
    #             done = terminated or truncated
    #
    #             total_reward += reward
    #             steps += 1
    #             state = next_state
    #
    #             if done:
    #                 break
    #
    #         rewards.append(total_reward)
    #         steps_list.append(steps)
    #         if total_reward > 0.5:
    #             success_count += 1
    #
    #     return {
    #         'mean_reward': np.mean(rewards),
    #         'std_reward': np.std(rewards),
    #         'mean_steps': np.mean(steps_list),
    #         'success_rate': success_count / n_episodes
    #     }

    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluează agentul fără explorare.
        Success = a atins GOAL (reward final == 1.0).
        """
        rewards = []
        steps_list = []
        success_count = 0

        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            steps = 0
            reached_goal = False

            while True:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += float(reward)
                steps += 1
                state = next_state

                # ✅ succes real: ultimul reward = 1.0 (adică a intrat pe G)
                if terminated and float(reward) >= 1.0:
                    reached_goal = True

                if done:
                    break

            rewards.append(total_reward)
            steps_list.append(steps)

            if reached_goal:
                success_count += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps_list)),
            "success_rate": success_count / n_episodes,
        }

    def save(self, filepath: str):
        """Salvează agentul."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes_trained': self.episodes_trained,
        }, filepath)

    def load(self, filepath: str):
        """Încarcă agentul."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episodes_trained = checkpoint['episodes_trained']
