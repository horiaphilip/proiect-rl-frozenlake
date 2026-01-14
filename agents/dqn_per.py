"""
DQN + Prioritized Experience Replay (PER)

- DQN clasic + replay buffer prioritizat pe |TD error|
- Sampling: P(i) ∝ p_i^alpha
- Importance Sampling weights: w_i = (N * P(i))^-beta, normalizate
- Target network update periodic
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def one_hot(state_idx: int, n_states: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(n_states, device=device, dtype=torch.float32)
    x[state_idx] = 1.0
    return x


# -----------------------
# SumTree for PER
# -----------------------
class SumTree:
    """
    Binary tree where parent value = sum of children priorities.
    Leaves store priorities for transitions.

    Supports:
    - add(p)
    - update(idx, p)
    - sample(value) -> leaf index
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float):
        idx = self.write + self.capacity - 1
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p

        # propagate change up
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, value: float) -> int:
        """
        Traverse the tree to find a leaf index such that cumulative sum crosses 'value'.
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

    def leaf_to_data_index(self, leaf_idx: int) -> int:
        return leaf_idx - (self.capacity - 1)


@dataclass
class Transition:
    s: int
    a: int
    r: float
    s2: int
    done: bool


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps

        self.data: List[Optional[Transition]] = [None] * capacity
        self.tree = SumTree(capacity)

        self.max_priority = 1.0  # start with 1 so new transitions get sampled

    def __len__(self) -> int:
        return self.tree.n_entries

    def add(self, transition: Transition):
        self.data[self.tree.write] = transition
        p = (self.max_priority + self.eps) ** self.alpha
        self.tree.add(p)

    def sample(self, batch_size: int, beta: float) -> Tuple[List[int], List[Transition], np.ndarray]:
        """
        Returns:
        - leaf indices (tree indices) to update priorities later
        - transitions
        - IS weights (np array, shape batch_size)
        """
        assert len(self) > 0
        indices = []
        samples = []
        priorities = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            leaf = self.tree.get_leaf(s)
            data_idx = self.tree.leaf_to_data_index(leaf)

            tr = self.data[data_idx]
            # in rare cases data could be None (shouldn't happen if used correctly)
            if tr is None:
                # fallback: resample uniformly
                data_idx = random.randrange(len(self))
                tr = self.data[data_idx]
                leaf = data_idx + (self.capacity - 1)

            indices.append(leaf)
            samples.append(tr)
            priorities.append(self.tree.tree[leaf])

        probs = np.array(priorities, dtype=np.float64) / max(self.tree.total, 1e-12)
        weights = (len(self) * probs) ** (-beta)
        weights /= weights.max() + 1e-12
        return indices, samples, weights.astype(np.float32)

    def update_priorities(self, leaf_indices: List[int], td_errors: np.ndarray):
        td_errors = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, float(td_errors.max()))
        for leaf, err in zip(leaf_indices, td_errors):
            p = float(err) ** self.alpha
            self.tree.update(leaf, p)


# -----------------------
# Q Network
# -----------------------
class QNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQN_PERAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,

        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,

        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_dim: int = 128,

        # PER params
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = 50000,
        per_eps: float = 1e-6,

        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = discount_factor

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = QNetwork(n_states, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.target_net = QNetwork(n_states, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha, eps=per_eps)

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.train_steps = 0  # gradient steps
        self.env_steps = 0    # interactions with env

        # PER beta schedule
        self.beta_start = per_beta_start
        self.beta_end = per_beta_end
        self.beta_anneal_steps = max(1, per_beta_anneal_steps)

    def _beta(self) -> float:
        t = min(self.env_steps / self.beta_anneal_steps, 1.0)
        return float(self.beta_start + t * (self.beta_end - self.beta_start))

    def select_action(self, state: int, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            x = one_hot(state, self.n_states, self.device).unsqueeze(0)
            q = self.policy_net(x)
            return int(torch.argmax(q, dim=1).item())

    def _learn_one_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        beta = self._beta()
        leaf_indices, batch, weights = self.buffer.sample(self.batch_size, beta=beta)

        states = torch.stack([one_hot(tr.s, self.n_states, self.device) for tr in batch], dim=0)
        actions = torch.tensor([tr.a for tr in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([tr.r for tr in batch], device=self.device, dtype=torch.float32)
        next_states = torch.stack([one_hot(tr.s2, self.n_states, self.device) for tr in batch], dim=0)
        dones = torch.tensor([tr.done for tr in batch], device=self.device, dtype=torch.float32)
        w = torch.tensor(weights, device=self.device, dtype=torch.float32)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # target: r + gamma * max_a' Q_target(s',a')  (0 if done)
            q_next = self.target_net(next_states).max(dim=1).values
            y = rewards + self.gamma * (1.0 - dones) * q_next

        td_error = (y - q_values).detach().cpu().numpy()
        avg_td_error = float(np.abs(td_error).mean())

        # PER weighted MSE
        loss = (w * (q_values - y) ** 2).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # update priorities in buffer
        self.buffer.update_priorities(leaf_indices, td_error)

        self.train_steps += 1
        return float(loss.item()), avg_td_error

    def train_episode(self, env) -> Dict[str, Any]:
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0

        losses = []
        td_errors = []

        while True:
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            self.env_steps += 1
            total_reward += float(reward)
            steps += 1

            self.buffer.add(Transition(s=state, a=action, r=float(reward), s2=next_state, done=done))

            result = self._learn_one_step()
            if result != 0.0 and isinstance(result, tuple):
                loss, td_error = result
                losses.append(loss)
                td_errors.append(td_error)

            # update target network periodically (based on env steps)
            if self.env_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            state = next_state
            if done:
                break

        # epsilon decay per episode
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": float(self.epsilon),
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "avg_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
            "buffer_size": len(self.buffer),
        }

    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        rewards = []
        steps_list = []
        success_count = 0

        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            steps = 0

            while True:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += float(reward)
                steps += 1
                state = next_state

                if done:
                    break

            rewards.append(total_reward)
            steps_list.append(steps)
            if total_reward > 0.5:
                success_count += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps_list)),
            "success_rate": float(success_count / n_episodes),
        }
