"""
DQN Agent (Deep Q-Network) - Stabilized Version

Improvements vs basic DQN:
- Replay warmup (min_replay_size)
- Huber loss (SmoothL1Loss)
- Gradient clipping
- Double DQN (optional, default True)
- Soft target updates (tau) OR periodic hard update
- Train frequency control (train_freq)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Optional, Dict, Any


# =========================================================
# Q Network
# =========================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =========================================================
# Replay Buffer
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: int, action: int, reward: float, next_state: int, done: float):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.int64),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.int64),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================
# DQN Agent
# =========================================================
class DQNAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.10,
        epsilon_decay: float = 0.999,

        buffer_capacity: int = 50000,
        batch_size: int = 64,

        # update behavior
        train_freq: int = 1,            # update every N env steps
        min_replay_size: int = 2000,    # warmup
        target_update_freq: int = 1000, # hard update period (if tau is None)
        tau: float = 0.01,              # soft update factor (if not None)

        hidden_dim: int = 256,
        max_grad_norm: float = 10.0,

        use_double_dqn: bool = True,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.discount_factor = float(discount_factor)

        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.batch_size = int(batch_size)
        self.buffer_capacity = int(buffer_capacity)

        self.train_freq = int(train_freq)
        self.min_replay_size = int(min_replay_size)

        self.target_update_freq = int(target_update_freq)
        self.tau = tau  # if None => hard update; else soft update

        self.max_grad_norm = float(max_grad_norm)
        self.use_double_dqn = bool(use_double_dqn)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(self.n_states, self.n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.n_states, self.n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss (mai stabil decât MSE)

        self.replay_buffer = ReplayBuffer(self.buffer_capacity)

        self.training_steps = 0   # gradient updates count
        self.env_steps = 0        # env transitions count
        self.episodes_trained = 0

    # ---------------------------------------------------------
    # State encoding (one-hot)
    # ---------------------------------------------------------
    def _state_to_tensor(self, state: int) -> torch.Tensor:
        x = torch.zeros(self.n_states, device=self.device)
        x[int(state)] = 1.0
        return x

    # ---------------------------------------------------------
    # Policy
    # ---------------------------------------------------------
    def select_action(self, state: int, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        with torch.no_grad():
            s = self._state_to_tensor(state).unsqueeze(0)
            q = self.policy_net(s)
            return int(torch.argmax(q, dim=1).item())

    # ---------------------------------------------------------
    # Target updates
    # ---------------------------------------------------------
    def _soft_update(self, tau: float):
        with torch.no_grad():
            for tgt, src in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tgt.data.mul_(1.0 - tau)
                tgt.data.add_(tau * src.data)

    def _hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---------------------------------------------------------
    # One training step
    # ---------------------------------------------------------
    def update(self) -> Optional[float]:
        # Warmup: nu învățăm până nu avem destule sample-uri
        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.stack([self._state_to_tensor(s) for s in states]).to(self.device)
        next_states_tensor = torch.stack([self._state_to_tensor(s) for s in next_states]).to(self.device)

        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Target Q
        with torch.no_grad():
            if self.use_double_dqn:
                # action selection from policy_net, evaluation from target_net
                next_actions = torch.argmax(self.policy_net(next_states_tensor), dim=1)
                next_q = self.target_net(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(next_states_tensor).max(dim=1)[0]

            target = rewards_tensor + (1.0 - dones_tensor) * self.discount_factor * next_q

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.training_steps += 1

        # Update target network
        if self.tau is not None:
            self._soft_update(self.tau)
        else:
            if self.training_steps % self.target_update_freq == 0:
                self._hard_update()

        return float(loss.item())

    # ---------------------------------------------------------
    # Epsilon decay
    # ---------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ---------------------------------------------------------
    # Train 1 episode
    # ---------------------------------------------------------
    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        losses = []

        while True:
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.replay_buffer.push(state, action, float(reward), next_state, float(done))

            self.env_steps += 1

            # train every train_freq steps
            if self.env_steps % self.train_freq == 0:
                loss = self.update()
                if loss is not None:
                    losses.append(loss)

            total_reward += float(reward)
            steps += 1
            state = next_state

            if done or (max_steps and steps >= max_steps):
                break

        self.decay_epsilon()
        self.episodes_trained += 1

        return {
            "total_reward": float(total_reward),
            "steps": int(steps),
            "epsilon": float(self.epsilon),
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "buffer_size": int(len(self.replay_buffer)),
        }

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------
    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        rewards = []
        steps_list = []
        success = 0

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
                success += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps_list)),
            "success_rate": float(success / n_episodes),
        }

    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------
    def save(self, filepath: str):
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "env_steps": self.env_steps,
            "episodes_trained": self.episodes_trained,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.training_steps = int(checkpoint.get("training_steps", 0))
        self.env_steps = int(checkpoint.get("env_steps", 0))
        self.episodes_trained = int(checkpoint.get("episodes_trained", 0))
