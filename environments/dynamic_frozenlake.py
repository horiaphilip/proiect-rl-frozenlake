"""
Dynamic FrozenLake Environment

Mediu personalizat bazat pe FrozenLake din Gymnasium cu următoarele caracteristici:
- Probabilitate de alunecare variabilă în timp (crește de la 0.1 la 0.4)
- Penalizare pentru fiecare pas: -0.01
- Gheață care se topește progresiv
- Obstacole mobile (opțional)
- Dimensiune variabilă a hărții
- Limită de max_steps=100
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class DynamicFrozenLakeEnv(gym.Env):
    """
    Mediu FrozenLake dinamic cu dificultate crescândă.

    Observație: starea curentă (int)
    Acțiuni: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        map_size: int = 8,
        render_mode: Optional[str] = None,
        max_steps: int = 100,
        slippery_start: float = 0.1,
        slippery_end: float = 0.4,
        step_penalty: float = -0.01,
        ice_melting: bool = True,
        melting_rate: float = 0.01,
    ):
        """
        Inițializare mediu DynamicFrozenLake.

        Args:
            map_size: Dimensiunea hărții (map_size x map_size)
            render_mode: Mod de randare ("human", "ansi" sau None)
            max_steps: Numărul maxim de pași per episod
            slippery_start: Probabilitatea inițială de alunecare
            slippery_end: Probabilitatea finală de alunecare
            step_penalty: Penalizare pentru fiecare pas
            ice_melting: Activează topirea gheții
            melting_rate: Rata de topire a gheții
        """
        super().__init__()

        self.map_size = map_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.slippery_start = slippery_start
        self.slippery_end = slippery_end
        self.step_penalty = step_penalty
        self.ice_melting = ice_melting
        self.melting_rate = melting_rate

        # Spații de acțiuni și observații
        self.action_space = spaces.Discrete(4)  # LEFT, DOWN, RIGHT, UP
        self.observation_space = spaces.Discrete(map_size * map_size)

        # Generare hartă
        self._generate_map()

        # Inițializare stare
        self.current_step = 0
        self.current_position = 0
        self.current_slippery = slippery_start
        self.hole_probabilities = np.ones(self.map_size * self.map_size)

    def _generate_map(self):
        """Generează harta mediului."""
        size = self.map_size

        # Creează o hartă cu gheață (F), găuri (H), start (S) și goal (G)
        self.desc = np.full((size, size), 'F', dtype='c')

        # Setează start (stânga sus)
        self.desc[0, 0] = b'S'
        self.start_state = 0

        # Setează goal (dreapta jos)
        self.desc[size-1, size-1] = b'G'
        self.goal_state = size * size - 1

        # Adaugă găuri (aproximativ 20% din celulele disponibile)
        num_holes = int((size * size - 2) * 0.2)
        available_positions = [(i, j) for i in range(size) for j in range(size)
                              if not (i == 0 and j == 0) and not (i == size-1 and j == size-1)]

        hole_positions = np.random.choice(len(available_positions), num_holes, replace=False)
        for idx in hole_positions:
            i, j = available_positions[idx]
            self.desc[i, j] = b'H'

    def _get_state_from_pos(self, row: int, col: int) -> int:
        """Convertește poziție (row, col) în stare."""
        return row * self.map_size + col

    def _get_pos_from_state(self, state: int) -> Tuple[int, int]:
        """Convertește stare în poziție (row, col)."""
        return state // self.map_size, state % self.map_size

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Verifică dacă poziția este validă."""
        return 0 <= row < self.map_size and 0 <= col < self.map_size

    def _apply_action(self, state: int, action: int) -> int:
        """Aplică acțiunea și returnează noua stare."""
        row, col = self._get_pos_from_state(state)

        # Mișcări: LEFT=0, DOWN=1, RIGHT=2, UP=3
        if action == 0:  # LEFT
            col = max(col - 1, 0)
        elif action == 1:  # DOWN
            row = min(row + 1, self.map_size - 1)
        elif action == 2:  # RIGHT
            col = min(col + 1, self.map_size - 1)
        elif action == 3:  # UP
            row = max(row - 1, 0)

        return self._get_state_from_pos(row, col)

    def _get_slippery_prob(self) -> float:
        """Calculează probabilitatea curentă de alunecare (crește liniar cu pașii)."""
        progress = min(self.current_step / self.max_steps, 1.0)
        return self.slippery_start + progress * (self.slippery_end - self.slippery_start)

    def _update_ice_melting(self):
        """Actualizează probabilitățile de topire a gheții."""
        if not self.ice_melting:
            return

        # Gheața se topește progresiv, transformând unele celule în găuri
        for state in range(self.map_size * self.map_size):
            row, col = self._get_pos_from_state(state)
            if self.desc[row, col] == b'F':  # Doar gheață normală
                # Scade probabilitatea că gheața rămâne solidă
                self.hole_probabilities[state] -= self.melting_rate
                self.hole_probabilities[state] = max(0, self.hole_probabilities[state])

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execută o acțiune în mediu.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Actualizează probabilitatea de alunecare
        self.current_slippery = self._get_slippery_prob()

        # Aplică alunecare
        if np.random.random() < self.current_slippery:
            # Alege o direcție aleatorie (perpendiculară la acțiunea dorită)
            action = np.random.choice([0, 1, 2, 3])

        # Calculează noua poziție
        new_state = self._apply_action(self.current_position, action)
        row, col = self._get_pos_from_state(new_state)

        # Verifică topirea gheții
        self._update_ice_melting()

        # Verifică dacă gheața s-a topit (devine gaură)
        if self.desc[row, col] == b'F' and np.random.random() > self.hole_probabilities[new_state]:
            self.desc[row, col] = b'H'

        self.current_position = new_state

        # Calculează reward
        reward = self.step_penalty  # Penalizare pentru fiecare pas
        terminated = False
        truncated = False

        if self.desc[row, col] == b'G':  # Goal
            reward = 1.0
            terminated = True
        elif self.desc[row, col] == b'H':  # Hole
            reward = 0.0
            terminated = True

        # Truncare la max_steps
        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            'current_step': self.current_step,
            'slippery_prob': self.current_slippery,
            'position': (row, col)
        }

        return self.current_position, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Resetează mediul la starea inițială."""
        super().reset(seed=seed)

        # Regenerare hartă (opțional, pentru variabilitate)
        if options and options.get('regenerate_map', False):
            self._generate_map()

        self.current_step = 0
        self.current_position = self.start_state
        self.current_slippery = self.slippery_start
        self.hole_probabilities = np.ones(self.map_size * self.map_size)

        info = {
            'current_step': self.current_step,
            'slippery_prob': self.current_slippery,
            'position': (0, 0)
        }

        return self.current_position, info

    def render(self):
        """Randează starea curentă a mediului."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            output = "\n"
            for i in range(self.map_size):
                for j in range(self.map_size):
                    state = self._get_state_from_pos(i, j)
                    if state == self.current_position:
                        output += " X "
                    else:
                        cell = self.desc[i, j].decode('utf-8')
                        output += f" {cell} "
                output += "\n"
            output += f"\nStep: {self.current_step}/{self.max_steps} | Slippery: {self.current_slippery:.2f}\n"

            if self.render_mode == "human":
                print(output)
            return output

    def close(self):
        """Închide mediul."""
        pass


# Funcție helper pentru înregistrare în Gymnasium
def register_dynamic_frozenlake():
    """Înregistrează mediul în Gymnasium."""
    gym.register(
        id='DynamicFrozenLake-v0',
        entry_point='environments.dynamic_frozenlake:DynamicFrozenLakeEnv',
        max_episode_steps=100,
    )
