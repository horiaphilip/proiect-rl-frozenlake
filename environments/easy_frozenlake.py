"""
Easy FrozenLake Environment

Versiune simplificată și learnable a DynamicFrozenLake:
- Hartă mică (4x4) pentru început
- Slippery minimal și constant (fără creștere în timp)
- Puține găuri (10-15%)
- Safe zone largă lângă start
- Fără topire de gheață
- Reward shaping puternic pentru a ghida agentul
- Penalizare mică pentru pași
- Max steps generos
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any


class EasyFrozenLakeEnv(gym.Env):
    """
    Mediu FrozenLake simplificat pentru învățare rapidă.

    Caracteristici:
    - Hartă mică (4x4 default)
    - Slippery constant și mic (0.05)
    - Puține găuri (10%)
    - Safe zone protejată
    - Reward shaping pronunțat
    - Guaranteed solvable map
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        map_size: int = 4,
        render_mode: Optional[str] = None,
        max_steps: int = 50,

        # Slippery foarte mic și constant (NU crește)
        slippery: float = 0.05,

        # Rewards
        step_penalty: float = -0.01,
        hole_penalty: float = -0.5,
        goal_reward: float = 1.0,

        # Reward shaping puternic
        shaped_rewards: bool = True,
        shaping_scale: float = 0.05,  # mai mare decât în hard mode

        # Dificultate
        hole_ratio: float = 0.10,  # doar 10% găuri

        # Safe zone mare
        safe_zone_radius: int = 1,  # protejează 2x2 lângă start

        # Mapă fixă pentru stabilitate
        regenerate_map_each_episode: bool = False,

        seed: Optional[int] = None,
    ):
        super().__init__()

        self.map_size = map_size
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.slippery = float(slippery)
        self.step_penalty = float(step_penalty)
        self.hole_penalty = float(hole_penalty)
        self.goal_reward = float(goal_reward)

        self.shaped_rewards = shaped_rewards
        self.shaping_scale = float(shaping_scale)

        self.hole_ratio = float(hole_ratio)
        self.safe_zone_radius = int(safe_zone_radius)
        self.regenerate_map_each_episode = regenerate_map_each_episode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(map_size * map_size)

        # Safe zone - zona protejată lângă start
        self.safe_zone = set()
        for i in range(min(safe_zone_radius + 1, map_size)):
            for j in range(min(safe_zone_radius + 1, map_size)):
                self.safe_zone.add((i, j))

        if seed is not None:
            np.random.seed(seed)

        self._generate_map_solvable()

        self.current_step = 0
        self.current_position = self.start_state

    def _get_state_from_pos(self, r: int, c: int) -> int:
        return r * self.map_size + c

    def _get_pos_from_state(self, s: int) -> Tuple[int, int]:
        return s // self.map_size, s % self.map_size

    def _apply_action(self, state: int, action: int) -> int:
        r, c = self._get_pos_from_state(state)
        if action == 0:   # LEFT
            c = max(c - 1, 0)
        elif action == 1: # DOWN
            r = min(r + 1, self.map_size - 1)
        elif action == 2: # RIGHT
            c = min(c + 1, self.map_size - 1)
        elif action == 3: # UP
            r = max(r - 1, 0)
        return self._get_state_from_pos(r, c)

    def _manhattan_to_goal(self, s: int) -> int:
        r, c = self._get_pos_from_state(s)
        gr, gc = self._get_pos_from_state(self.goal_state)
        return abs(r - gr) + abs(c - gc)

    def _generate_map_once(self):
        """Generează o hartă simplu cu puține găuri."""
        size = self.map_size
        self.desc = np.full((size, size), 'F', dtype='c')

        self.desc[0, 0] = b'S'
        self.start_state = 0

        self.desc[size - 1, size - 1] = b'G'
        self.goal_state = size * size - 1

        # Poziții disponibile pentru găuri (excludem safe zone + goal)
        available_positions = [
            (i, j)
            for i in range(size)
            for j in range(size)
            if (i, j) not in self.safe_zone and (i, j) != (size - 1, size - 1)
        ]

        # Număr mic de găuri
        num_holes = int((size * size - 2) * self.hole_ratio)
        num_holes = min(num_holes, len(available_positions))

        if num_holes > 0:
            hole_indices = np.random.choice(len(available_positions), num_holes, replace=False)
            for idx in hole_indices:
                r, c = available_positions[idx]
                self.desc[r, c] = b'H'

    def _is_solvable(self) -> bool:
        """BFS pentru a verifica dacă există drum de la S la G."""
        size = self.map_size
        start = (0, 0)
        goal = (size - 1, size - 1)

        q = deque([start])
        visited = set([start])

        def passable(rr, cc):
            cell = self.desc[rr, cc]
            return cell in (b'S', b'F', b'G')

        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited:
                    if passable(nr, nc):
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return False

    def _generate_map_solvable(self):
        """Generează harta până găsește una solvabilă."""
        max_tries = 100
        for _ in range(max_tries):
            self._generate_map_once()
            if self._is_solvable():
                return
        # Dacă nu găsește, creează o hartă garantat solvabilă (fără găuri)
        self.desc = np.full((self.map_size, self.map_size), 'F', dtype='c')
        self.desc[0, 0] = b'S'
        self.desc[self.map_size - 1, self.map_size - 1] = b'G'

    def step(self, action: int):
        self.current_step += 1
        old_state = self.current_position

        # Slippery constant și mic
        if np.random.random() < self.slippery:
            # Slip perpendicular
            if action in (0, 2):  # LEFT/RIGHT
                action = int(np.random.choice([1, 3]))  # DOWN/UP
            else:  # DOWN/UP
                action = int(np.random.choice([0, 2]))  # LEFT/RIGHT

        new_state = self._apply_action(self.current_position, action)
        r, c = self._get_pos_from_state(new_state)

        self.current_position = new_state

        # Calculează reward
        reward = float(self.step_penalty)

        # Reward shaping - bonus când te apropii de goal
        if self.shaped_rewards:
            old_dist = self._manhattan_to_goal(old_state)
            new_dist = self._manhattan_to_goal(new_state)
            reward += self.shaping_scale * (old_dist - new_dist)

        terminated = False
        truncated = False

        if self.desc[r, c] == b'G':
            reward = self.goal_reward
            terminated = True
        elif self.desc[r, c] == b'H':
            reward = self.hole_penalty
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            "current_step": self.current_step,
            "position": (int(r), int(c)),
            "distance_to_goal": self._manhattan_to_goal(new_state),
        }

        return int(new_state), float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.regenerate_map_each_episode or (options and options.get("regenerate_map", False)):
            self._generate_map_solvable()

        self.current_step = 0
        self.current_position = self.start_state

        info = {
            "current_step": 0,
            "position": (0, 0),
            "distance_to_goal": self._manhattan_to_goal(self.start_state),
        }
        return int(self.current_position), info

    def render(self):
        if self.render_mode in ("ansi", "human"):
            out = "\n"
            for i in range(self.map_size):
                for j in range(self.map_size):
                    s = self._get_state_from_pos(i, j)
                    if s == self.current_position:
                        out += " X "
                    else:
                        out += f" {self.desc[i, j].decode('utf-8')} "
                out += "\n"
            out += f"\nStep: {self.current_step}/{self.max_steps}\n"
            if self.render_mode == "human":
                print(out)
            return out

    def close(self):
        pass


def register_easy_frozenlake():
    """Înregistrează mediul în Gymnasium."""
    gym.register(
        id="EasyFrozenLake-v0",
        entry_point="environments.easy_frozenlake:EasyFrozenLakeEnv",
        max_episode_steps=50,
    )