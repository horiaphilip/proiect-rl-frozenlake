import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any


class DynamicFrozenLakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        map_size: int = 8,
        render_mode: Optional[str] = None,
        max_steps: int = 120,
        slippery_start: float = 0.10,
        slippery_end: float = 0.40,
        step_penalty: float = -0.01,
        ice_melting: bool = True,
        melting_rate: float = 0.005,

        shaped_rewards: bool = True,
        shaping_scale: float = 0.02,
        hole_penalty: float = -1.0,

        protect_safe_zone_from_melting: bool = True,

        hole_ratio: float = 0.20,

        melt_cells_per_step: int = 1,

        regenerate_map_each_episode: bool = False,

        max_map_tries: int = 200,
    ):
        super().__init__()

        self.map_size = map_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.slippery_start = slippery_start
        self.slippery_end = slippery_end
        self.step_penalty = step_penalty
        self.ice_melting = ice_melting
        self.melting_rate = float(melting_rate)

        self.shaped_rewards = shaped_rewards
        self.shaping_scale = float(shaping_scale)
        self.hole_penalty = float(hole_penalty)

        self.protect_safe_zone_from_melting = protect_safe_zone_from_melting
        self.hole_ratio = float(hole_ratio)
        self.melt_cells_per_step = int(melt_cells_per_step)
        self.regenerate_map_each_episode = regenerate_map_each_episode
        self.max_map_tries = int(max_map_tries)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(map_size * map_size)

        self.safe_zone = {(0, 0), (0, 1), (1, 0), (1, 1)}

        self._generate_map_solvable()

        self.current_step = 0
        self.current_position = self.start_state
        self.current_slippery = self.slippery_start

        self.hole_probabilities = np.ones(self.map_size * self.map_size, dtype=np.float32)

    def _get_state_from_pos(self, r: int, c: int) -> int:
        return r * self.map_size + c

    def _get_pos_from_state(self, s: int) -> Tuple[int, int]:
        return s // self.map_size, s % self.map_size

    def _apply_action(self, state: int, action: int) -> int:
        r, c = self._get_pos_from_state(state)
        if action == 0:
            c = max(c - 1, 0)
        elif action == 1:
            r = min(r + 1, self.map_size - 1)
        elif action == 2:
            c = min(c + 1, self.map_size - 1)
        elif action == 3:
            r = max(r - 1, 0)
        return self._get_state_from_pos(r, c)

    def _get_slippery_prob(self) -> float:
        progress = min(self.current_step / self.max_steps, 1.0)
        return self.slippery_start + progress * (self.slippery_end - self.slippery_start)

    def _manhattan_to_goal(self, s: int) -> int:
        r, c = self._get_pos_from_state(s)
        gr, gc = self._get_pos_from_state(self.goal_state)
        return abs(r - gr) + abs(c - gc)

    def _generate_map_once(self):
        size = self.map_size
        self.desc = np.full((size, size), 'F', dtype='c')

        self.desc[0, 0] = b'S'
        self.start_state = 0

        self.desc[size - 1, size - 1] = b'G'
        self.goal_state = size * size - 1

        available_positions = [
            (i, j)
            for i in range(size)
            for j in range(size)
            if (i, j) not in self.safe_zone and (i, j) != (size - 1, size - 1)
        ]

        num_holes = int((size * size - 2) * self.hole_ratio)
        num_holes = min(num_holes, len(available_positions))

        if num_holes <= 0:
            return

        hole_indices = np.random.choice(len(available_positions), num_holes, replace=False)
        for idx in hole_indices:
            r, c = available_positions[idx]
            self.desc[r, c] = b'H'

    def _is_solvable(self) -> bool:
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
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited:
                    if passable(nr, nc):
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return False

    def _generate_map_solvable(self):
        for _ in range(self.max_map_tries):
            self._generate_map_once()
            if self._is_solvable():
                return



    def _update_ice_melting(self):
        if not self.ice_melting:
            return

        candidates = []
        for s in range(self.map_size * self.map_size):
            r, c = self._get_pos_from_state(s)

            if self.protect_safe_zone_from_melting and (r, c) in self.safe_zone:
                continue
            if self.desc[r, c] in (b'S', b'G', b'H'):
                continue
            if self.desc[r, c] == b'F':
                candidates.append(s)

        if not candidates:
            return

        k = min(self.melt_cells_per_step, len(candidates))
        chosen = np.random.choice(candidates, k, replace=False)

        for s in chosen:
            self.hole_probabilities[s] -= self.melting_rate
            if self.hole_probabilities[s] < 0.0:
                self.hole_probabilities[s] = 0.0


    def step(self, action: int):
        self.current_step += 1
        old_state = self.current_position

        self.current_slippery = self._get_slippery_prob()

        if np.random.random() < self.current_slippery:
            if action in (0, 2):
                action = int(np.random.choice([1, 3]))
            else:
                action = int(np.random.choice([0, 2]))

        new_state = self._apply_action(self.current_position, action)
        r, c = self._get_pos_from_state(new_state)

        self._update_ice_melting()

        if self.desc[r, c] == b'F' and np.random.random() > self.hole_probabilities[new_state]:
            if (r, c) not in self.safe_zone:
                self.desc[r, c] = b'H'

        self.current_position = new_state

        reward = float(self.step_penalty)
        if self.shaped_rewards:
            reward += self.shaping_scale * (
                self._manhattan_to_goal(old_state) - self._manhattan_to_goal(new_state)
            )

        terminated = False
        truncated = False

        if self.desc[r, c] == b'G':
            reward = 1.0
            terminated = True
        elif self.desc[r, c] == b'H':
            reward = self.hole_penalty
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            "current_step": self.current_step,
            "slippery_prob": float(self.current_slippery),
            "position": (int(r), int(c)),
        }

        return int(new_state), float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.regenerate_map_each_episode or (options and options.get("regenerate_map", False)):
            self._generate_map_solvable()

        self.current_step = 0
        self.current_position = self.start_state
        self.current_slippery = self.slippery_start
        self.hole_probabilities[:] = 1.0

        info = {"current_step": 0, "slippery_prob": float(self.current_slippery), "position": (0, 0)}
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
            out += f"\nStep: {self.current_step}/{self.max_steps} | Slippery: {self.current_slippery:.2f}\n"
            if self.render_mode == "human":
                print(out)
            return out

    def close(self):
        pass


def register_dynamic_frozenlake():
    gym.register(
        id="DynamicFrozenLake-v0",
        entry_point="environments.dynamic_frozenlake:DynamicFrozenLakeEnv",
        max_episode_steps=100,
    )
