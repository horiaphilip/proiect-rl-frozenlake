"""
PPO Agent (Proximal Policy Optimization)

Agent policy-based care folosește Stable Baselines3 pentru implementarea PPO.
PPO este un algoritm modern și robust pentru RL.
"""

import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym


class TrainingCallback(BaseCallback):
    """Callback pentru colectarea statisticilor în timpul antrenamentului."""

    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Apelat la fiecare pas."""
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def get_stats(self) -> Dict[str, float]:
        """Returnează statistici despre ultimele episoade."""
        if not self.episode_rewards:
            return {
                'mean_reward': 0.0,
                'mean_length': 0.0,
                'num_episodes': 0
            }

        return {
            'mean_reward': np.mean(self.episode_rewards[-10:]),  # Ultimele 10 episoade
            'mean_length': np.mean(self.episode_lengths[-10:]),
            'num_episodes': len(self.episode_rewards)
        }


class PPOAgent:
    """
    Agent PPO (Proximal Policy Optimization).

    Wrapper peste Stable Baselines3 PPO pentru compatibilitate cu ceilalți agenți.
    """

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
        """
        Inițializare agent PPO.

        Args:
            env: Mediul de antrenament
            learning_rate: Rata de învățare
            n_steps: Număr de pași pentru colectare experiență înainte de update
            batch_size: Dimensiunea batch-ului pentru antrenament
            n_epochs: Număr de epoci de antrenament pe batch
            gamma: Factorul de discount
            gae_lambda: Lambda pentru GAE (Generalized Advantage Estimation)
            clip_range: Range pentru clipping în PPO
            ent_coef: Coeficientul pentru entropy bonus
            vf_coef: Coeficientul pentru value function loss
            max_grad_norm: Valoarea maximă pentru gradient clipping
            verbose: Nivel de verbozitate
        """
        self.env = env

        # Creează modelul PPO
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

        # Callback pentru statistici
        self.callback = TrainingCallback()

        # Statistici
        self.training_timesteps = 0

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Selectează o acțiune.

        Args:
            state: Starea curentă
            training: Dacă este în modul training (nu are efect pentru PPO)

        Returns:
            Acțiunea selectată
        """
        action, _ = self.model.predict(state, deterministic=not training)
        return int(action)

    def train(self, total_timesteps: int, progress_bar: bool = False) -> Dict[str, Any]:
        """
        Antrenează agentul.

        Args:
            total_timesteps: Număr total de pași de antrenament
            progress_bar: Dacă se afișează progress bar

        Returns:
            Dicționar cu statistici despre antrenament
        """
        # Reset callback
        self.callback = TrainingCallback()

        # Antrenează modelul
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=progress_bar
        )

        self.training_timesteps += total_timesteps

        # Returnează statistici
        stats = self.callback.get_stats()
        stats['total_timesteps'] = self.training_timesteps

        return stats

    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Antrenează agentul pe un episod complet (folosind n_steps din PPO).

        Note: PPO antrenează pe batch-uri de experiențe, nu pe episoade individuale.
        Această metodă antrenează pentru n_steps pași.

        Args:
            env: Mediul de antrenament (ignorat, se folosește self.env)
            max_steps: Număr maxim de pași (ignorat, se folosește n_steps din PPO)

        Returns:
            Dicționar cu statistici despre antrenament
        """
        # PPO antrenează pe batch-uri, nu pe episoade
        # Antrenăm pentru n_steps pași
        n_steps = self.model.n_steps

        stats = self.train(total_timesteps=n_steps, progress_bar=False)

        return {
            'total_reward': stats.get('mean_reward', 0.0),
            'steps': stats.get('mean_length', 0.0),
            'num_episodes': stats.get('num_episodes', 0),
            'total_timesteps': stats.get('total_timesteps', 0)
        }

    def evaluate(self, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
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
            if total_reward > 0.5:
                success_count += 1

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps_list),
            'success_rate': success_count / n_episodes
        }

    def save(self, filepath: str):
        """Salvează agentul."""
        self.model.save(filepath)

    def load(self, filepath: str):
        """Încarcă agentul."""
        self.model = SB3_PPO.load(filepath, env=self.env)

    def get_policy(self):
        """
        Returnează politica învățată.

        Returns:
            Modelul PPO
        """
        return self.model.policy
