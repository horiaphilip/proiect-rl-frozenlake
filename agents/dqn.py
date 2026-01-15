import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional, Dict, Any


class QNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
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
        return len(self.buffer)


class DQNAgent:

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

        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(n_states, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(n_states, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.training_steps = 0
        self.episodes_trained = 0

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        state_tensor = torch.zeros(self.n_states, device=self.device)
        state_tensor[state] = 1.0
        return state_tensor

    def select_action(self, state: int, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def update(self) -> Optional[tuple]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.stack([self._state_to_tensor(s) for s in states]).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.stack([self._state_to_tensor(s) for s in next_states]).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor).max(dim=1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.discount_factor * next_q_values

        with torch.no_grad():
            td_errors = torch.abs(current_q_values - target_q_values)
            avg_td_error = td_errors.mean().item()

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item(), avg_td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        losses = []
        td_errors = []

        while True:
            action = self.select_action(state, training=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.replay_buffer.push(state, action, reward, next_state, float(done))

            update_result = self.update()
            if update_result is not None:
                loss, td_error = update_result
                losses.append(loss)
                td_errors.append(td_error)

            total_reward += reward
            steps += 1
            state = next_state

            if done or (max_steps and steps >= max_steps):
                break

        self.decay_epsilon()
        self.episodes_trained += 1

        return {
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'avg_td_error': np.mean(td_errors) if td_errors else 0.0,
            'buffer_size': len(self.replay_buffer)
        }


    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
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
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes_trained': self.episodes_trained,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episodes_trained = checkpoint['episodes_trained']
