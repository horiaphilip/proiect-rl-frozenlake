# Experiment MARL Complet - 5 Seeds care sunt posibile garantat prin verificare BFS
# DIFICULTATE FINALA: 25 Holes (39% densitate), 35% Slippery, Collab Penalty -0.3

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

from agents import SimpleQLearning, SimpleMARLSystem

class SolvableFrozenLake:
    
    def generate_solvable_map(self, seed):
        rng = np.random.RandomState(seed)
        max_attempts = 10000
        
        for attempt in range(max_attempts):
            desc = np.full((8, 8), 'F', dtype='U1')

            num_holes = 25
            holes_placed = 0
            while holes_placed < num_holes:
                r, c = rng.randint(0, 8), rng.randint(0, 8)
                if (r,c) not in [(0,0), (7,7), (0,7), (7,0)] and desc[r, c] == 'F':
                    desc[r, c] = 'H'
                    holes_placed += 1
            
            # VALIDARE
            agent1_can_reach = self._bfs_check_path(desc, (0,0), (0,7))

            agent2_can_reach = self._bfs_check_path(desc, (7,7), (0,7))

            if agent1_can_reach and agent2_can_reach:
                return desc
        
        raise RuntimeError(f"Could not generate fully solvable map for seed {seed} after {max_attempts} attempts.")

    def _bfs_check_path(self, desc, start, goal):
        return self._bfs_check_any_goal(desc, start, [goal])

    def _bfs_check_any_goal(self, desc, start, goals):
        rows, cols = desc.shape
        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            if (r, c) in goals:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited and desc[nr, nc] != 'H':
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False
    
    def _get_start_goal_positions(self):
        pass

class ProvenSoloEnv(SolvableFrozenLake):
    def __init__(self, seed=42):
        self.desc = self.generate_solvable_map(seed)
        self.desc[0, 0] = 'S'
        self.desc[7, 7] = 'G'
        
    def _get_start_goal_positions(self):
        return [(0, 0)], [(7, 7)]


def save_map_visualization(desc, seed, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Generated Map - Seed {seed}\n(Red=Ag1, Blue=Ag2)", fontsize=12, fontweight='bold')
    
    for r in range(8):
        for c in range(8):
            cell = desc[r, c]
            color = 'white'
            text = ''
            
            if cell == 'S':
                color = 'lightgray'
                text = 'Start'
            elif cell == 'G':
                color = 'gold'
                text = 'Goal'
            elif cell == '1': 
                color = '#ffcccc' # Light Red
                text = 'S1'
            elif cell == '2': 
                color = '#cce5ff' # Light Blue
                text = 'S2'
            elif cell == 'A': 
                color = '#ff9999' # Darker Red
                text = 'G1'
            elif cell == 'B': 
                color = '#99ccff' # Darker Blue
                text = 'G2'
            elif cell == 'H': 
                color = '#404040' # Dark Gray
                text = ''
            
            rect = plt.Rectangle((c, 7-r), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)
            if text:
                weight = 'bold' if 'G' in text else 'normal'
                ax.text(c+0.5, 7-r+0.5, text, ha='center', va='center', fontsize=10, fontweight=weight)
                
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class WrapperSoloEnv:
    def __init__(self, desc, slipperiness=0.30):
        self.desc = desc
        self.slippery = slipperiness
        self.start_pos = (0, 0)
        self.goal_pos = (7, 7)
        
    def reset(self):
        self.state = 0
        return self.state, {}
        
    def step(self, action):
        row, col = self.state // 8, self.state % 8
        if random.random() < self.slippery:
            action = random.randint(0, 3)
        dr, dc = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        nr = max(0, min(row + dr, 7))
        nc = max(0, min(col + dc, 7))
        
        cell = self.desc[nr, nc]
        new_state = nr * 8 + nc
        
        reward = 0
        done = False
        
        if cell == 'H':
            done = True
            reward = -0.5 # Penalty
        elif cell == 'G':
            done = True
            reward = 1
        
        self.state = new_state
        return new_state, reward, done, False, {}

class WrapperMARLEnv:
    # Shared Rewards
    def __init__(self, desc, slipperiness=0.35):
        self.desc = desc
        self.slippery = slipperiness
        self.s1 = 0
        self.s2 = 63
        
    def reset(self):
        self.s1 = 0
        self.s2 = 63
        return (self.s1, self.s2), {}
        
    def to_coords(self, s):
        return s // 8, s % 8
        
    def _move_single(self, state, action):
        r, c = self.to_coords(state)
        if random.random() < self.slippery:
            action = random.randint(0, 3)
        dr, dc = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        nr = max(0, min(r + dr, 7))
        nc = max(0, min(c + dc, 7))
        
        cell = self.desc[nr, nc]
        ns = nr * 8 + nc
        
        term = False
        rew = 0
        
        if cell == 'H':
            term = True
            rew = -0.5 # Penalty
        elif cell == 'A' or cell == 'B' or cell == 'G':
            term = True
            rew = 1
            
        return ns, rew, term

    def step(self, actions):
        a1, a2 = actions
        ns1, r1, t1 = self._move_single(self.s1, a1)
        ns2, r2, t2 = self._move_single(self.s2, a2)
        
        self.s1, self.s2 = ns1, ns2
        
        done = False
        info = {}
        
        # SHARED REWARD: If ANY succeeds -> BOTH get +1
        if (r1 > 0) or (r2 > 0):
            r1 = 1
            r2 = 1
            done = True
            info['success'] = True
        
        # Fail if BOTH die
        cell1 = self.desc[ns1 // 8, ns1 % 8]
        cell2 = self.desc[ns2 // 8, ns2 % 8]
        if cell1 == 'H' and cell2 == 'H':
            done = True
            info['success'] = False
            
        return (ns1, ns2), (r1, r2), done, False, info


class WrapperMARLNonCooperative:
    # Non Cooperative
    def __init__(self, desc, slipperiness=0.35):
        self.desc = desc
        self.slippery = slipperiness
        self.s1 = 0
        self.s2 = 63
        
    def reset(self):
        self.s1 = 0
        self.s2 = 63
        return (self.s1, self.s2), {}
        
    def to_coords(self, s):
        return s // 8, s % 8
        
    def _move_single(self, state, action):
        r, c = self.to_coords(state)
        if random.random() < self.slippery:
            action = random.randint(0, 3)
        dr, dc = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        nr = max(0, min(r + dr, 7))
        nc = max(0, min(c + dc, 7))
        
        cell = self.desc[nr, nc]
        ns = nr * 8 + nc
        
        term = False
        rew = 0
        
        if cell == 'H':
            term = True
            rew = -0.5 # Penalty
        elif cell == 'A' or cell == 'B' or cell == 'G':
            term = True
            rew = 1
            
        return ns, rew, term

    def step(self, actions):
        a1, a2 = actions
        ns1, r1, t1 = self._move_single(self.s1, a1)
        ns2, r2, t2 = self._move_single(self.s2, a2)
        
        self.s1, self.s2 = ns1, ns2
        
        done = False
        info = {}
        
        # INDIVIDUAL REWARDS: Fiecare agent cu propriile rewarduri
        # Daca agent 1 reuseste si ag. 2 nu => penalty ag2

        if (r1 > 0) or (r2 > 0):
            done = True
            # Success daca cel putin unul a ajuns
            info['success'] = (r1 > 0) or (r2 > 0)
            
            # unul reuseste, celalalt nu
            if r1 > 0 and r2 <= 0:
                r2 = -0.3  # Agent 2 fail in timp ce Agent 1 success
            elif r2 > 0 and r1 <= 0:
                r1 = -0.3  # Agent 1 fail in timp ce Agent 2 success
        
        # Fail daca ambii nu reusesc
        cell1 = self.desc[ns1 // 8, ns1 % 8]
        cell2 = self.desc[ns2 // 8, ns2 % 8]
        if cell1 == 'H' and cell2 == 'H':
            done = True
            info['success'] = False

        return (ns1, ns2), (r1, r2), done, False, info

def make_solvable_marl_noncoop(seed):
    class MarGen(SolvableFrozenLake):
        def _get_start_goal_positions(self):
            return [(0,0), (7,7)], [(0,7), (7,0)]
    gen = MarGen()
    desc = gen.generate_solvable_map(seed)
    
    desc[0, 0] = '1'  # Agent 1 start
    desc[7, 7] = '2'  # Agent 2 start
    desc[0, 7] = 'G'  # Goal unic
    
    return WrapperMARLNonCooperative(desc, slipperiness=0.35)

def make_solvable_marl_coop(seed):
    class MarGen(SolvableFrozenLake):
        def _get_start_goal_positions(self):
            return [(0,0), (7,7)], [(0,7), (7,0)]
    gen = MarGen()
    desc = gen.generate_solvable_map(seed)
    
    desc[0, 0] = '1'  # Agent 1 start
    desc[7, 7] = '2'  # Agent 2 start
    desc[0, 7] = 'G'  # Goal unic
    
    return WrapperMARLEnv(desc, slipperiness=0.35)

def run_experiment_5_seeds():
    seeds = [42, 43, 44, 45, 46]
    n_episodes = 20000
    
    print(f"Running MARL Comparison on 5 Seeds")
    print(f"Comparing: Non-Cooperative IQL vs Cooperative IQL (Shared Rewards)")
    print(f"Difficulty: 25 Holes (39% density), 35% Slippery, Collab Penalty -0.3")
    print("-" * 70)
    
    results = {'noncoop': [], 'coop': []}
    q_tables = {}  #Q-tables
    
    for seed in seeds:
        print(f"\nProcessing Seed {seed}...")

        env_noncoop = make_solvable_marl_noncoop(seed)
        env_coop = make_solvable_marl_coop(seed)

        save_map_visualization(env_coop.desc, seed, f"results/map_seed_{seed}_marl.png")
        print(f"  > Map visualization saved.")
        
        # Non-Cooperative MARL Train
        marl_noncoop = SimpleMARLSystem()
        for _ in range(n_episodes): marl_noncoop.train_episode(env_noncoop)
        noncoop_score = marl_noncoop.evaluate(env_noncoop, n_episodes=100)
        
        # Cooperative MARL Train
        marl_coop = SimpleMARLSystem()
        for _ in range(n_episodes): marl_coop.train_episode(env_coop)
        coop_score = marl_coop.evaluate(env_coop, n_episodes=100)
        
        print(f"  > Non-Cooperative Score: {noncoop_score}%")
        print(f"  > Cooperative Score:     {coop_score}%")
        print(f"  > Improvement from Shared Rewards: +{coop_score - noncoop_score:.1f}%")
        
        results['noncoop'].append(noncoop_score)
        results['coop'].append(coop_score)

        q_tables[f'seed_{seed}'] = {
            'noncoop_agent1': marl_noncoop.agent1.q_table.copy(),
            'noncoop_agent2': marl_noncoop.agent2.q_table.copy(),
            'coop_agent1': marl_coop.agent1.q_table.copy(),
            'coop_agent2': marl_coop.agent2.q_table.copy(),
            'map': env_coop.desc.copy()
        }

    avg_noncoop = np.mean(results['noncoop'])
    avg_coop = np.mean(results['coop'])
    
    print("\n" + "="*70)
    print("FINAL RESULTS (5 SEEDS)")
    print("="*70)
    print(f"Non-Cooperative (Individual Rewards): {avg_noncoop:.1f}%")
    print(f"Cooperative (Shared Rewards):          {avg_coop:.1f}%")
    print(f"Improvement from Collaboration:        +{avg_coop - avg_noncoop:.1f}%")
    print("="*70)
    print(f"\n Conclusion: Shared rewards improve success rate by {avg_coop - avg_noncoop:.1f}%")

    with open('../results/final_5seeds_results.json', 'w') as f:
        json.dump(results, f)

    q_tables_flat = {}
    for seed in seeds:
        seed_key = f'seed_{seed}'
        q_tables_flat[f'{seed_key}_noncoop_agent1'] = q_tables[seed_key]['noncoop_agent1']
        q_tables_flat[f'{seed_key}_noncoop_agent2'] = q_tables[seed_key]['noncoop_agent2']
        q_tables_flat[f'{seed_key}_coop_agent1'] = q_tables[seed_key]['coop_agent1']
        q_tables_flat[f'{seed_key}_coop_agent2'] = q_tables[seed_key]['coop_agent2']
    
    np.savez('../results/q_tables.npz', **q_tables_flat)

    maps_data = {f'seed_{seed}_map': q_tables[f'seed_{seed}']['map'].tolist() 
                 for seed in seeds}
    with open('../results/maps.json', 'w') as f:
        json.dump(maps_data, f)
    
    print("\n Q-tables and maps saved for visualization!")
        
if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    run_experiment_5_seeds()
