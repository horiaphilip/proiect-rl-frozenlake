pute# Quick Start Guide - Proiect Reinforcement Learning

Ghid rapid pentru a începe să lucrezi cu proiectul de RL.

---

## Instalare Rapidă

### Windows

```bash
# 1. Navighează în directorul proiectului
cd C:\Users\Horia\PyCharmMiscProject\proiect_irl

# 2. Activează virtual environment
.venv\Scripts\activate

# 3. Instalează dependențele (dacă nu ai făcut-o deja)
pip install -r requirements.txt
```

### Linux/Mac

```bash
# 1. Navighează în directorul proiectului
cd /path/to/proiect_irl

# 2. Activează virtual environment
source .venv/bin/activate

# 3. Instalează dependențele (dacă nu ai făcut-o deja)
pip install -r requirements.txt
```

**Verificare instalare:**
```python
python -c "import gymnasium, torch, stable_baselines3; print('✓ All packages installed!')"
```

---

## Rulare Experimente - 4 Opțiuni Principale

### Opțiunea 0: Test Rapid - Toți Algoritmii pe EasyFrozenLake (2-3 minute) ⚡

Verifică rapid că toți cei 5 algoritmi funcționează pe mediul simplu:

```bash
cd experiments
python test_easy_env.py
```

**Ce face:**
- Antrenează Q-Learning (300 episoade)
- Antrenează DQN (300 episoade)
- Antrenează DQN+PER (300 episoade)
- Antrenează PPO (15,000 timesteps)
- Antrenează PPO+RND (15,000 timesteps)
- Evaluează fiecare agent (50 episoade)
- Afișează tabel comparativ cu rezultate

**Output așteptat:**
```
COMPARATIE FINALA - TOȚI ALGORITMII
============================================================
Algorithm       Success Rate    Mean Steps   Mean Reward
------------------------------------------------------------
DQN+PER         100.0%     🏆   6.5          1.19
PPO             100.0%     🏆   6.6          1.19
PPO+RND         100.0%     🏆   6.7          1.18
Q-Learning      100.0%     🏆   6.8          1.18
DQN             35.0%           30.2         0.12

✓ 4/5 algoritmi au atins ≥80% success rate
🏆 Câștigător: DQN+PER cu 100.0% success rate
```

---

### Opțiunea 1: Benchmark Complet (5-10 minute) 🏆

Antrenează și evaluează toți cei 5 algoritmi:

```bash
cd experiments
python benchmark_all_agents.py
```

**Ce face:**
- Antrenează Q-Learning (500 episoade)
- Antrenează DQN (500 episoade)
- Antrenează DQN+PER (500 episoade) ⭐
- Antrenează PPO (25,000 timesteps)
- Antrenează PPO+RND (25,000 timesteps)
- Evaluează fiecare agent (100 episoade)
- Salvează rezultate JSON în `results/benchmark_easy_TIMESTAMP.json`

**Rezultate așteptate:**

| Algorithm | Success Rate | Mean Steps | Efficiency Score |
|-----------|--------------|------------|------------------|
| Q-Learning | 100% | 6.54 | 15.29 |
| DQN | 32% | 32.76 | 0.98 |
| **DQN+PER** 🏆 | **100%** | **6.37** | **15.70** |
| PPO | 100% | 6.38 | 15.67 |
| PPO+RND | 100% | 6.40 | 15.62 |

**Pentru test rapid (1-2 min)**, modifică în `benchmark_all_agents.py`:
```python
N_EPISODES = 100        # În loc de 500
PPO_TIMESTEPS = 5000    # În loc de 25000
N_EVAL = 20             # În loc de 100
```

---

### Opțiunea 2: Vizualizare Rezultate (10 secunde) 📊

Generează grafice din ultimul benchmark:

```bash
cd experiments
python visualize_benchmark.py
```

**Grafice generate (în `results/`):**
1. `benchmark_comparison.png` - Comparație 3 metrici (Success Rate, Reward, Steps)
2. `learning_curves.png` - Evoluția rewardurilor în timpul training-ului
3. `efficiency_scatter.png` - Scatter plot Success Rate vs Mean Steps
4. `winner_ranking.png` - Ranking cu DQN+PER ca câștigător

---

### Opțiunea 3: Multi-Seed Analysis (15-20 minute) 🔬

Testează reproducibilitatea pe 5 seed-uri diferite:

```bash
cd experiments
python benchmark_multi_seed.py
```

**Ce face:**
- Rulează toți cei 5 algoritmi pe seeds: [42, 123, 456, 789, 1024]
- Calculează mean ± std pentru fiecare metric
- Identifică seed-uri problematice (789, 1024 sunt dificile)
- Salvează `results/benchmark_multi_seed_TIMESTAMP.json`

**Apoi vizualizează:**
```bash
python visualize_multi_seed.py
```

**Grafice generate:**
1. `multi_seed_comparison.png` - Mean ± std bars pentru 3 metrici
2. `multi_seed_stability.png` - Comparație deviații standard
3. `multi_seed_distribution.png` - Distribuție rezultate per seed

**Rezultate așteptate (mean ± std):**

| Algorithm | Success Rate | Stabilitate |
|-----------|--------------|-------------|
| **PPO** 🏆 | **98.60% ± 2.33%** | Foarte stabil |
| PPO+RND | 98.60% ± 2.33% | Foarte stabil |
| DQN+PER | 80.20% ± 39.60% | Instabil |
| Q-Learning | 66.60% ± 42.22% | Instabil |
| DQN | 41.20% ± 44.93% | Foarte instabil |

---

## Training Custom - Exemple Practice

### Exemplu 1: Q-Learning pe Map Custom

```python
from environments.easy_frozenlake import EasyFrozenLakeEnv
from agents.q_learning import QLearningAgent

# Creează mediu custom (mai dificil)
env = EasyFrozenLakeEnv(
    map_size=4,
    slippery=0.10,        # Crește alunecare (default 0.05)
    hole_ratio=0.15,      # Mai multe găuri (default 0.10)
    shaped_rewards=True,
    seed=42
)

# Creează agent
agent = QLearningAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.1,
    discount_factor=0.99
)

# Training
from tqdm import tqdm
for episode in tqdm(range(500), desc="Training Q-Learning"):
    stats = agent.train_episode(env)

# Evaluare
eval_stats = agent.evaluate(env, n_episodes=100)
print(f"\nSuccess Rate: {eval_stats['success_rate']:.2%}")
print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")

# Salvare agent antrenat
agent.save("models/q_learning_custom.pkl")
```

---

### Exemplu 2: DQN+PER (Cel Mai Eficient) ⭐

```python
from agents.dqn_per import DQN_PERAgent
from tqdm import tqdm

# Creează agent DQN+PER (best performer)
agent = DQN_PERAgent(
    n_states=16,
    n_actions=4,
    learning_rate=0.001,
    per_alpha=0.6,          # Prioritizare experiențe
    per_beta_start=0.4,     # Importance sampling
    buffer_capacity=10000,
    seed=42
)

# Training cu progress bar
for episode in tqdm(range(500), desc="Training DQN+PER"):
    stats = agent.train_episode(env)

    # Logging periodic
    if (episode + 1) % 100 == 0:
        eval_stats = agent.evaluate(env, n_episodes=10)
        print(f"\nEpisode {episode+1}: Success Rate = {eval_stats['success_rate']:.1%}")

# Evaluare finală
final_stats = agent.evaluate(env, n_episodes=100)
print(f"\n=== Final Results ===")
print(f"Success Rate: {final_stats['success_rate']:.2%}")  # Expected: 100%
print(f"Mean Steps: {final_stats['mean_steps']:.2f}")     # Expected: ~6.37

# Salvare
agent.save("models/dqn_per_best.pth")
```

---

### Exemplu 3: PPO (Cel Mai Stabil) 🎯

```python
from agents.ppo import PPOAgent

# Creează agent PPO (most stable)
agent = PPOAgent(
    env=env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    verbose=1  # Progress logging
)

# Training (timesteps nu episoade)
agent.train(total_timesteps=25000)

# Evaluare
eval_stats = agent.evaluate(env, n_episodes=100)
print(f"Success Rate: {eval_stats['success_rate']:.2%}")  # Expected: 100%
print(f"Mean Steps: {eval_stats['mean_steps']:.2f}")     # Expected: ~6.38

# Salvare
agent.save("models/ppo_stable.zip")
```

---

## Încărcare și Refolosire Agenți Antrenați

### Q-Learning

```python
from agents.q_learning import QLearningAgent

# Încărcare
agent = QLearningAgent.load("models/q_learning_custom.pkl")

# Evaluare pe mediu nou
new_env = EasyFrozenLakeEnv(seed=123)
eval_stats = agent.evaluate(new_env, n_episodes=100)
print(f"Success Rate on new seed: {eval_stats['success_rate']:.2%}")
```

### DQN+PER

```python
from agents.dqn_per import DQN_PERAgent

# Încărcare
agent = DQN_PERAgent.load("models/dqn_per_best.pth")

# Evaluare
eval_stats = agent.evaluate(env, n_episodes=100)
```

### PPO

```python
from agents.ppo import PPOAgent

# Încărcare (necesită env pentru compatibilitate Stable-Baselines3)
agent = PPOAgent.load("models/ppo_stable.zip", env=env)

# Evaluare
eval_stats = agent.evaluate(env, n_episodes=100)
```

---

## Modificare Parametri Mediu

### EasyFrozenLake - Niveluri de Dificultate

**Easy (default):**
```python
env = EasyFrozenLakeEnv(
    map_size=4,
    slippery=0.05,       # 5% alunecare
    hole_ratio=0.10,     # 10% găuri
    shaped_rewards=True
)
```

**Medium:**
```python
env = EasyFrozenLakeEnv(
    map_size=4,
    slippery=0.10,       # 10% alunecare
    hole_ratio=0.15,     # 15% găuri
    shaped_rewards=True
)
```

**Hard:**
```python
env = EasyFrozenLakeEnv(
    map_size=4,
    slippery=0.15,       # 15% alunecare
    hole_ratio=0.20,     # 20% găuri
    shaped_rewards=True,
    shaping_scale=0.03   # Reduce reward shaping
)
```

### DynamicFrozenLake (Challenge) - 8×8

```python
from environments.dynamic_frozenlake import DynamicFrozenLakeEnv

# Mediu dificil cu dificultate crescândă
dynamic_env = DynamicFrozenLakeEnv(
    map_size=8,
    slippery_start=0.08,
    slippery_end=0.25,
    hole_ratio=0.18,
    ice_melting=True,
    shaped_rewards=True
)

# Recomandat: PPO cu training extins
agent = PPOAgent(dynamic_env)
agent.train(total_timesteps=100000)  # 100k timesteps
```

---

## Structura Output-urilor

După rularea experimentelor:

```
results/
├── benchmark_easy_TIMESTAMP.json        # Date benchmark complet
│   ├── Q-Learning: {training_rewards: [...], eval_stats: {...}}
│   ├── DQN: {...}
│   ├── DQN+PER: {...}
│   ├── PPO: {...}
│   └── PPO+RND: {...}
│
├── benchmark_multi_seed_TIMESTAMP.json  # Date multi-seed analysis
│
└── Grafice PNG:
    ├── benchmark_comparison.png         # 3 metrici comparative
    ├── learning_curves.png              # Training progress
    ├── efficiency_scatter.png           # Scatter plot
    ├── winner_ranking.png               # Ranking câștigător
    ├── multi_seed_comparison.png        # Mean ± std bars
    ├── multi_seed_stability.png         # Std comparison
    └── multi_seed_distribution.png      # Per-seed distribution
```

---

## Probleme Comune & Soluții

### 1. ModuleNotFoundError

**Problemă:** `ModuleNotFoundError: No module named 'environments'`

**Soluție:**
```bash
# Asigură-te că rulezi din directorul experiments/
cd experiments
python benchmark_all_agents.py
```

### 2. CUDA Out of Memory (DQN)

**Problemă:** Eroare memorie GPU pentru DQN

**Soluție:**
```python
# Forțează CPU în agents/dqn.py sau agents/dqn_per.py
self.device = torch.device("cpu")
```

### 3. Virtual Environment Nu Se Activează

**Windows:**
```bash
# Dacă .venv\Scripts\activate nu funcționează
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
# Dacă source .venv/bin/activate nu funcționează
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Dependențe Lipsă

**Problemă:** Import errors pentru pachete

**Soluție:**
```bash
# Reinstalare dependențe
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Workflow Recomandat

### Pentru Testing Rapid

```python
# Test rapid un singur algoritm (1 min)
from environments.easy_frozenlake import EasyFrozenLakeEnv
from agents.dqn_per import DQN_PERAgent

env = EasyFrozenLakeEnv()
agent = DQN_PERAgent(16, 4)

# Training rapid
for episode in range(100):
    agent.train_episode(env)

# Evaluare
stats = agent.evaluate(env, n_episodes=20)
print(f"Success Rate: {stats['success_rate']:.1%}")
```

### Pentru Evaluare Completă

```bash
# 1. Benchmark complet (toate algoritmii)
cd experiments
python benchmark_all_agents.py  # ~10 min

# 2. Vizualizare rezultate
python visualize_benchmark.py   # ~10 sec

# 3. Analiză multi-seed (opțional)
python benchmark_multi_seed.py  # ~20 min
python visualize_multi_seed.py  # ~10 sec
```

### Pentru Comparații Custom

```python
# Compară doar 2 algoritmi specifici
from environments.easy_frozenlake import EasyFrozenLakeEnv
from agents.q_learning import QLearningAgent
from agents.dqn_per import DQN_PERAgent
from tqdm import tqdm

env = EasyFrozenLakeEnv(seed=42)

# Train Q-Learning
ql_agent = QLearningAgent(16, 4)
for episode in tqdm(range(500), desc="Q-Learning"):
    ql_agent.train_episode(env)
ql_stats = ql_agent.evaluate(env, n_episodes=100)

# Train DQN+PER
dqn_agent = DQN_PERAgent(16, 4)
for episode in tqdm(range(500), desc="DQN+PER"):
    dqn_agent.train_episode(env)
dqn_stats = dqn_agent.evaluate(env, n_episodes=100)

# Compare
print(f"\nQ-Learning: {ql_stats['success_rate']:.1%} success")
print(f"DQN+PER: {dqn_stats['success_rate']:.1%} success")
```

---

## Tips & Tricks

### Monitorizare Training în Timp Real

```python
# Pentru Q-Learning/DQN/DQN+PER
from tqdm import tqdm

for episode in tqdm(range(500), desc="Training"):
    stats = agent.train_episode(env)

    if episode % 50 == 0:
        # Evaluare intermediară
        eval_stats = agent.evaluate(env, n_episodes=10)
        print(f"\nEpisode {episode}: Success Rate = {eval_stats['success_rate']:.1%}")
```

### Salvare Periodică Checkpoints

```python
# Training cu checkpoints la fiecare 100 episoade
for episode in range(500):
    agent.train_episode(env)

    # Salvare checkpoint
    if (episode + 1) % 100 == 0:
        agent.save(f"models/checkpoint_ep{episode+1}.pkl")
        print(f"✓ Checkpoint saved at episode {episode+1}")
```

### Testare Rapidă cu Parametri Reduși

Pentru debugging rapid, modifică parametrii în scripturi:

```python
# În benchmark_all_agents.py
N_EPISODES = 100        # În loc de 500 (5× mai rapid)
PPO_TIMESTEPS = 5000    # În loc de 25000 (5× mai rapid)
N_EVAL = 20             # În loc de 100 (5× mai rapid)
```

---

## Structura Proiectului

```
proiect_irl/
├── agents/                          # Implementări algoritmi RL
│   ├── q_learning.py               # Q-Learning tabular
│   ├── dqn.py                      # Deep Q-Network
│   ├── dqn_per.py                  # DQN + Prioritized Experience Replay
│   ├── ppo.py                      # Proximal Policy Optimization
│   └── ppo_rnd.py                  # PPO + Random Network Distillation
│
├── environments/                    # Medii custom
│   ├── easy_frozenlake.py          # FrozenLake 4×4 simplificat
│   └── dynamic_frozenlake.py       # FrozenLake 8×8 dinamic
│
├── experiments/                     # Scripturi experimentale
│   ├── benchmark_all_agents.py     # Benchmark complet (MAIN)
│   ├── benchmark_multi_seed.py     # Multi-seed analysis
│   ├── visualize_benchmark.py      # Generare grafice benchmark
│   └── visualize_multi_seed.py     # Generare grafice multi-seed
│
├── results/                         # Output experimente
│   ├── *.json                      # Rezultate numerice
│   └── *.png                       # Grafice generate
│
├── README.md                        # Documentație completă
├── QUICKSTART.md                    # Acest fișier
└── requirements.txt                 # Dependențe Python
```

---

## Scripturi Disponibile - Rezumat

| Script | Locație | Descriere | Timp | Output |
|--------|---------|-----------|------|--------|
| `test_easy_env.py` | `experiments/` | Test rapid 5 algoritmi | ~3 min | Tabel comparativ |
| `benchmark_all_agents.py` | `experiments/` | Benchmark complet | ~10 min | JSON + logs |
| `visualize_benchmark.py` | `experiments/` | Generare 4 grafice | < 10s | 4 PNG files |
| `benchmark_multi_seed.py` | `experiments/` | Multi-seed (5 seeds) | ~20 min | JSON + statistici |
| `visualize_multi_seed.py` | `experiments/` | Grafice multi-seed | < 10s | 3 PNG files |
| `run_experiments.py` | `experiments/` | Script general custom | Variabil | Customizabil |

---

## Next Steps

După ce ai rulat experimentele cu succes:

1. **Analizează graficele** din `results/` pentru insights
2. **Compară algoritmii** - care e cel mai bun pentru task-ul tău?
3. **Experimentează cu hiperparametri** diferiți
4. **Încearcă seed-uri noi** pentru testare robustețe
5. **Modifică mediul** pentru challenge-uri noi (map size, slippery, holes)
6. **Implementează algoritmi noi** bazat pe arhitectura existentă

---

## Resurse Utile

### Documentație Proiect
- **README.md** - Documentație completă cu toate detaliile tehnice

### Documentații Externe
- [Gymnasium Docs](https://gymnasium.farama.org/) - Framework environments
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/) - PPO implementation
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) - DQN tutorial
- [OpenAI Spinning Up](https://spinningup.openai.com/) - Deep RL educational resource

---

**Gata să începi! Succes cu experimentele! 🚀**
