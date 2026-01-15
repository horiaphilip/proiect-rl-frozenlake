import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


def one_hot(states: torch.Tensor, n_states: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(states.long(), num_classes=n_states).float()


class ActorCritic(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden_dim, n_actions)
        self.v = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def get_logprob_entropy_value(self, x: torch.Tensor, action: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy, value


class RNDTarget(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPORndAgent:
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,

        beta_int: float = 0.05,         # try 0.02..0.15
        rnd_lr: float = 1e-4,
        rnd_out_dim: int = 64,
        rnd_hidden_dim: int = 128,
        rnd_update_proportion: float = 0.25,  # fraction used per minibatch
        normalize_int_reward: bool = True,

        seed: Optional[int] = None,
        device: Optional[str] = None,
        verbose: int = 0,
    ):
        self.env = env
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.ac = ActorCritic(self.n_states, self.n_actions, hidden_dim=128).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=learning_rate, eps=1e-5)

        self.rnd_target = RNDTarget(self.n_states, out_dim=rnd_out_dim, hidden_dim=rnd_hidden_dim).to(self.device)
        self.rnd_pred = RNDPredictor(self.n_states, out_dim=rnd_out_dim, hidden_dim=rnd_hidden_dim).to(self.device)
        self.rnd_opt = optim.Adam(self.rnd_pred.parameters(), lr=rnd_lr, eps=1e-5)

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.beta_int = beta_int
        self.rnd_update_proportion = rnd_update_proportion
        self.normalize_int_reward = normalize_int_reward

        self._int_count = 1e-4
        self._int_mean = 0.0
        self._int_M2 = 1.0

        self.last10_rewards = deque(maxlen=10)
        self.last10_lengths = deque(maxlen=10)
        self.num_episodes = 0
        self.training_timesteps = 0

        self._obs, _ = self.env.reset()
        self._ep_return_ext = 0.0
        self._ep_len = 0

    @torch.no_grad()
    def _intrinsic_reward(self, s_onehot: torch.Tensor) -> torch.Tensor:
        t = self.rnd_target(s_onehot)
        p = self.rnd_pred(s_onehot)
        mse = torch.mean((t - p) ** 2, dim=1)
        return mse

    def _update_int_stats(self, r_int_np: np.ndarray):
        for x in r_int_np:
            self._int_count += 1.0
            delta = x - self._int_mean
            self._int_mean += delta / self._int_count
            delta2 = x - self._int_mean
            self._int_M2 += delta * delta2

    def _normalize_int(self, r_int: torch.Tensor) -> torch.Tensor:
        if not self.normalize_int_reward:
            return r_int
        var = self._int_M2 / max(self._int_count - 1.0, 1.0)
        std = float(np.sqrt(var + 1e-8))
        return r_int / std

    def _collect_rollout(self):
        states = torch.zeros(self.n_steps, dtype=torch.long, device=self.device)
        actions = torch.zeros(self.n_steps, dtype=torch.long, device=self.device)
        logprobs = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        values = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        rewards_ext = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        rewards_int = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)

        int_for_stats = []

        for t in range(self.n_steps):
            s = torch.tensor(self._obs, device=self.device).long()
            s_oh = one_hot(s.view(1), self.n_states)

            with torch.no_grad():
                a, lp, ent, v = self.ac.get_action(s_oh)
                r_int = self._intrinsic_reward(s_oh)  # (1,)

            next_obs, r_ext, terminated, truncated, _ = self.env.step(int(a.item()))
            done = bool(terminated or truncated)

            states[t] = s
            actions[t] = a.squeeze(0)
            logprobs[t] = lp.squeeze(0)
            values[t] = v.squeeze(0)
            rewards_ext[t] = float(r_ext)
            rewards_int[t] = r_int.squeeze(0)
            dones[t] = 1.0 if done else 0.0

            self._ep_return_ext += float(r_ext)
            self._ep_len += 1
            int_for_stats.append(float(r_int.item()))

            self._obs = next_obs

            if done:
                self.last10_rewards.append(self._ep_return_ext)
                self.last10_lengths.append(self._ep_len)
                self.num_episodes += 1

                self._obs, _ = self.env.reset()
                self._ep_return_ext = 0.0
                self._ep_len = 0

        if self.normalize_int_reward and len(int_for_stats) > 0:
            self._update_int_stats(np.array(int_for_stats, dtype=np.float64))

        with torch.no_grad():
            ns = torch.tensor(self._obs, device=self.device).long()
            ns_oh = one_hot(ns.view(1), self.n_states)
            _, next_value = self.ac.forward(ns_oh)
            next_value = next_value.squeeze(0)

        return states, actions, logprobs, values, rewards_ext, rewards_int, dones, next_value

    def _compute_gae(self, values, rewards_total, dones, next_value):
        T = rewards_total.shape[0]
        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
            delta = rewards_total[t] + self.gamma * next_val * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def _update(self, states, actions, old_logprobs, values, rewards_ext, rewards_int, dones, next_value):
        r_int_norm = self._normalize_int(rewards_int)
        rewards_total = rewards_ext + self.beta_int * r_int_norm

        adv, ret = self._compute_gae(values, rewards_total, dones, next_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = self.n_steps
        inds = np.arange(n)

        ppo_loss_acc = 0.0
        rnd_loss_acc = 0.0
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        steps = 0

        states_oh = one_hot(states, self.n_states)  # (T, n_states)

        for _ in range(self.n_epochs):
            np.random.shuffle(inds)
            for start in range(0, n, self.batch_size):
                mb = inds[start:start + self.batch_size]
                mb_states = states_oh[mb]
                mb_actions = actions[mb]
                mb_oldlp = old_logprobs[mb]
                mb_adv = adv[mb]
                mb_ret = ret[mb]

                newlp, entropy, newv = self.ac.get_logprob_entropy_value(mb_states, mb_actions)

                ratio = (newlp - mb_oldlp).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = torch.max(pg1, pg2).mean()

                value_loss = 0.5 * (mb_ret - newv).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()

                rnd_loss = torch.tensor(0.0, device=self.device)
                if self.rnd_update_proportion > 0:
                    mask = (torch.rand(mb_states.shape[0], device=self.device) < self.rnd_update_proportion)
                    if mask.any():
                        with torch.no_grad():
                            target = self.rnd_target(mb_states[mask])
                        pred = self.rnd_pred(mb_states[mask])
                        rnd_loss = ((pred - target) ** 2).mean()

                        self.rnd_opt.zero_grad(set_to_none=True)
                        rnd_loss.backward()
                        self.rnd_opt.step()

                ppo_loss_acc += float(loss.item())
                rnd_loss_acc += float(rnd_loss.item())
                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                entropy_acc += float(entropy_loss.item())
                steps += 1

        return {
            "ppo_loss": ppo_loss_acc / max(steps, 1),
            "rnd_loss": rnd_loss_acc / max(steps, 1),
            "policy_loss": policy_loss_acc / max(steps, 1),
            "value_loss": value_loss_acc / max(steps, 1),
            "entropy": entropy_acc / max(steps, 1),
        }

    def get_stats(self) -> Dict[str, float]:
        if len(self.last10_rewards) == 0:
            return {"mean_reward": 0.0, "mean_length": 0.0, "num_episodes": 0}
        return {
            "mean_reward": float(np.mean(self.last10_rewards)),
            "mean_length": float(np.mean(self.last10_lengths)),
            "num_episodes": int(self.num_episodes),
        }

    def train(self, total_timesteps: int, progress_bar: bool = False) -> Dict[str, Any]:
        if progress_bar:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_timesteps, desc="PPO+RND Training")
            except Exception:
                pbar = None
        else:
            pbar = None

        steps_done = 0
        all_losses = []
        all_intrinsic_rewards = []

        while steps_done < total_timesteps:
            rollout = self._collect_rollout()
            intrinsic_rewards = rollout[5].cpu().numpy()
            all_intrinsic_rewards.extend(intrinsic_rewards.tolist())

            losses = self._update(*rollout)
            all_losses.append(losses)

            steps_done += self.n_steps
            self.training_timesteps += self.n_steps
            if pbar is not None:
                pbar.update(self.n_steps)

        if pbar is not None:
            pbar.close()

        stats = self.get_stats()
        stats["total_timesteps"] = int(self.training_timesteps)

        if all_losses:
            last_losses = all_losses[-1]
            stats["ppo_loss"] = last_losses.get("ppo_loss", 0.0)
            stats["rnd_loss"] = last_losses.get("rnd_loss", 0.0)
            stats["policy_loss"] = last_losses.get("policy_loss", 0.0)
            stats["value_loss"] = last_losses.get("value_loss", 0.0)
            stats["entropy"] = last_losses.get("entropy", 0.0)

        if all_intrinsic_rewards:
            stats["intrinsic_reward_mean"] = float(np.mean(all_intrinsic_rewards))
            stats["intrinsic_reward_std"] = float(np.std(all_intrinsic_rewards))

        return stats

    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        stats = self.train(total_timesteps=self.n_steps, progress_bar=False)
        return {
            "total_reward": stats.get("mean_reward", 0.0),
            "steps": stats.get("mean_length", 0.0),
            "num_episodes": stats.get("num_episodes", 0),
            "total_timesteps": stats.get("total_timesteps", 0),
        }

    def select_action(self, state: int, training: bool = True) -> int:
        s = torch.tensor(state, device=self.device).long().view(1)
        s_oh = one_hot(s, self.n_states)
        with torch.no_grad():
            logits, _ = self.ac.forward(s_oh)
            if training:
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample().item()
            else:
                a = torch.argmax(logits, dim=-1).item()
        return int(a)

    def evaluate(self, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        rewards = []
        steps_list = []
        success = 0

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
                success += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps_list)),
            "success_rate": float(success / n_episodes),
        }
