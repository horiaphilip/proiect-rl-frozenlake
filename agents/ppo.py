import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym


class TrainingCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.approx_kls = []
        self.clip_fractions = []

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def _on_rollout_end(self) -> None:
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger = self.model.logger
                if hasattr(logger, 'name_to_value'):
                    values = logger.name_to_value
                    if 'train/policy_gradient_loss' in values:
                        self.policy_losses.append(values['train/policy_gradient_loss'])
                    if 'train/value_loss' in values:
                        self.value_losses.append(values['train/value_loss'])
                    if 'train/entropy_loss' in values:
                        self.entropy_losses.append(values['train/entropy_loss'])
                    if 'train/approx_kl' in values:
                        self.approx_kls.append(values['train/approx_kl'])
                    if 'train/clip_fraction' in values:
                        self.clip_fractions.append(values['train/clip_fraction'])
        except Exception:
            pass

    def get_stats(self) -> Dict[str, float]:
        stats = {}

        if self.episode_rewards:
            stats['mean_reward'] = float(np.mean(self.episode_rewards[-10:]))
            stats['mean_length'] = float(np.mean(self.episode_lengths[-10:]))
            stats['num_episodes'] = len(self.episode_rewards)
        else:
            stats['mean_reward'] = 0.0
            stats['mean_length'] = 0.0
            stats['num_episodes'] = 0

        if self.policy_losses:
            stats['policy_loss'] = float(np.mean(self.policy_losses[-5:]))
        if self.value_losses:
            stats['value_loss'] = float(np.mean(self.value_losses[-5:]))
        if self.entropy_losses:
            stats['entropy'] = float(np.mean(self.entropy_losses[-5:]))
        if self.approx_kls:
            stats['approx_kl'] = float(np.mean(self.approx_kls[-5:]))
        if self.clip_fractions:
            stats['clip_fraction'] = float(np.mean(self.clip_fractions[-5:]))

        return stats


class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 0,
    ):
        self.env = env

        self.model = SB3_PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
        )

        self.callback = TrainingCallback()

        self.training_timesteps = 0

    def select_action(self, state: int, training: bool = True) -> int:
        action, _ = self.model.predict(state, deterministic=not training)
        return int(action)

    def train(self, total_timesteps: int, progress_bar: bool = False) -> Dict[str, Any]:
        self.callback = TrainingCallback()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=progress_bar
        )

        self.training_timesteps += total_timesteps

        stats = self.callback.get_stats()
        stats['total_timesteps'] = self.training_timesteps

        return stats

    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        n_steps = self.model.n_steps

        stats = self.train(total_timesteps=n_steps, progress_bar=False)

        return {
            'total_reward': stats.get('mean_reward', 0.0),
            'steps': stats.get('mean_length', 0.0),
            'num_episodes': stats.get('num_episodes', 0),
            'total_timesteps': stats.get('total_timesteps', 0),
            'policy_loss': stats.get('policy_loss', 0.0),
            'value_loss': stats.get('value_loss', 0.0),
            'entropy': stats.get('entropy', 0.0),
            'approx_kl': stats.get('approx_kl', 0.0),
            'clip_fraction': stats.get('clip_fraction', 0.0),
        }

    def evaluate(self, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
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
        self.model.save(filepath)

    def load(self, filepath: str):
        self.model = SB3_PPO.load(filepath, env=self.env)

    def get_policy(self):
        return self.model.policy
