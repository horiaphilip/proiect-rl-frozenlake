# Proiect Reinforcement Learning - Dynamic FrozenLake

Implementare și comparație a **5 algoritmi** de Reinforcement Learning pe medii FrozenLake custom cu dificultate variabilă.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Cuprins

- [Descriere](#descriere)
- [Algoritmi Implementați](#algoritmi-implementați)
- [Medii (Environments)](#medii-environments)
- [Rezultate](#rezultate)
- [Instalare](#instalare)
- [Utilizare](#utilizare)
- [Structura Proiectului](#structura-proiectului)
- [Referințe](#referințe)

---

## 🎯 Descriere

Acest proiect implementează și compară **5 algoritmi moderni** de Reinforcement Learning pe variante custom ale mediului FrozenLake:

1. **Q-Learning** (clasic tabular)
2. **DQN** (Deep Q-Network)
3. **DQN + PER** (DQN cu Prioritized Experience Replay) ⭐
4. **PPO** (Proximal Policy Optimization)
5. **PPO + RND** (PPO cu Random Network Distillation)

### Caracteristici Principale

✅ **Implementări complete** de la zero (PyTorch pentru deep RL)
✅ **Două medii custom** cu dificultate variabilă (Easy 4x4 și Dynamic 8x8)
✅ **Benchmark comprehensiv** cu 5 algoritmi + vizualizări
✅ **Rezultate validate**: DQN+PER câștigător cu 100% success rate
✅ **Documentație detaliată** și cod bine comentat

---

## 🤖 Algoritmi Implementați

### 1. Q-Learning
**Locație:** `agents/q_learning.py`

Algoritm clasic de Reinforcement Learning tabular.

**Caracteristici:**
- Q-table pentru stocare valori
- ε-greedy exploration
- Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

**Rezultate pe EasyFrozenLake:**
- Success Rate: **100%**
- Mean Steps: 6.54

---

### 2. DQN (Deep Q-Network)
**Locație:** `agents/dqn.py`

Extindere deep learning a Q-Learning, folosind rețele neuronale.

**Caracteristici:**
- Q-Network (neural network) pentru aproximare
- Experience Replay Buffer (10,000 capacitate)
- Target Network (update periodic)
- ε-greedy exploration

**Arhitectură rețea:**
```
Input (one-hot state) → Hidden(128) → ReLU → Hidden(128) → ReLU → Output(n_actions)
```

**Rezultate pe EasyFrozenLake:**
- Success Rate: 32% (necesită mai mult tuning)
- Mean Steps: 32.76

---

### 3. DQN + PER (Prioritized Experience Replay) 🏆
**Locație:** `agents/dqn_per.py`

**Câștigător Benchmark!**

DQN îmbunătățit cu sampling prioritizat din replay buffer.

**Caracteristici:**
- **SumTree** pentru sampling eficient O(log n)
- Prioritizare bazată pe TD-error: P(i) ∝ |δᵢ|^α
- Importance Sampling weights pentru corectare bias
- Beta annealing schedule (0.4 → 1.0)

**De ce funcționează mai bine:**
- Învață mai repede din tranziții importante (TD-error mare)
- Sample-efficiency mult mai mare vs DQN vanilla
- Convergență mai rapidă și mai stabilă

**Rezultate pe EasyFrozenLake:**
- Success Rate: **100%** ⭐
- Mean Steps: **6.37** (cel mai eficient!)
- Efficiency Score: **15.70** (best overall)

---

### 4. PPO (Proximal Policy Optimization)
**Locație:** `agents/ppo.py`

Algoritm modern policy gradient cu clipping pentru stabilitate.

**Caracteristici:**
- Actor-Critic architecture
- Clipped surrogate objective
- GAE (Generalized Advantage Estimation)
- Multiple epochs pe același batch

**Obiectiv clipat:**
```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
```

**Rezultate pe EasyFrozenLake:**
- Success Rate: **100%**
- Mean Steps: 6.38
- Foarte stabil și consistent

---

### 5. PPO + RND (Random Network Distillation)
**Locație:** `agents/ppo_rnd.py`

PPO extins cu intrinsic rewards pentru explorare mai bună.

**Caracteristici:**
- **Target Network** (fixed random)
- **Predictor Network** (trainable)
- Intrinsic reward: r_int = MSE(target(s), predictor(s))
- Total reward: r = r_ext + β * normalize(r_int)

**Când e util:**
- Medii cu sparse rewards
- Explorare dificilă
- State-space mare

**Rezultate pe EasyFrozenLake:**
- Success Rate: **100%**
- Mean Steps: 6.40
- Nu aduce beneficii pe easy env (rewards nu sunt sparse)

---

## 🏔️ Medii (Environments)

### EasyFrozenLake ⭐ (Recomandat pentru început)
**Locație:** `environments/easy_frozenlake.py`

Mediu simplificat, optimizat pentru învățare rapidă.

| Caracteristică | Valoare |
|----------------|---------|
| Map size | 4x4 (16 stări) |
| Slippery | 5% (constant) |
| Hole ratio | 10% |
| Safe zone | 2x2 lângă start |
| Ice melting | OFF |
| Reward shaping | ON |
| Max steps | 50 |

**Rezultate Benchmark:**

| Algorithm | Success Rate | Mean Steps | Efficiency |
|-----------|--------------|------------|------------|
| Q-Learning | 100% | 6.54 | 15.29 |
| DQN | 32% | 32.76 | 0.98 |
| **DQN+PER** | **100%** | **6.37** | **15.70** 🏆 |
| PPO | 100% | 6.38 | 15.67 |
| PPO+RND | 100% | 6.40 | 15.62 |

**Când să folosești:**
- Testing rapid algoritmi noi
- Debugging și proof-of-concept
- Baseline pentru comparații
- Success rate garantat > 90%

---

### DynamicFrozenLake (Challenge)
**Locație:** `environments/dynamic_frozenlake.py`

Mediu complex cu dificultate crescândă în timp.

| Caracteristică | Valoare |
|----------------|---------|
| Map size | 8x8 (64 stări) |
| Slippery | 0.08 → 0.25 (crește) |
| Hole ratio | 18-20% |
| Safe zone | Protejată |
| Ice melting | ON (controlat) |
| Reward shaping | Opțional |
| Max steps | 120-140 |

**Challenge-uri:**
- Probabilitate variabilă de alunecare (crește în timp)
- Gheață se topește progresiv (devine gaură)
- Map mai mare = explorare mai dificilă
- Necesită 1000+ episoade training

**Când să folosești:**
- Testare robustețe algoritmi
- Comparație performanță pe task dificil
- Research și experimente avansate

---

## 📊 Rezultate

### Benchmark Complet pe EasyFrozenLake (4x4)

**Setup:**
- 500 episoade training (Q-Learning, DQN, DQN+PER)
- 25,000 timesteps (PPO, PPO+RND)
- 100 episoade evaluare
- Seed: 42 (reproducibilitate)

**Tabel Rezultate:**

| Algorithm | Success Rate | Mean Reward | Mean Steps | Efficiency Score* |
|-----------|--------------|-------------|------------|-------------------|
| Q-Learning | 100.00% | 1.1946 | 6.54 | 15.29 |
| DQN | 32.00% | 0.0538 | 32.76 | 0.98 |
| **DQN+PER** | **100.00%** | **1.1963** | **6.37** | **15.70** 👑 |
| PPO | 100.00% | 1.1962 | 6.38 | 15.67 |
| PPO+RND | 100.00% | 1.1960 | 6.40 | 15.62 |

*Efficiency Score = Success Rate / Mean Steps (higher is better)

### 🏆 Câștigător: DQN + PER

**De ce câștigă DQN+PER:**
1. **100% success rate** (împreună cu Q-Learning, PPO, PPO+RND)
2. **Cel mai eficient**: doar 6.37 pași în medie
3. **Prioritized Experience Replay** face diferența enormă vs DQN vanilla
4. **Sample efficiency**: converge mai rapid

### Observații Cheie

1. **PER face diferența**: DQN (32%) → DQN+PER (100%)
2. **PPO foarte consistent**: success rate 100%, eficiență excelentă
3. **Q-Learning surprinde**: funcționează foarte bine pe medii simple
4. **RND nu ajută** pe EasyFrozenLake (rewards nu sunt sparse enough)
5. **DQN vanilla** necesită mai mult tuning sau training

### Vizualizări Generate

Vezi folder `results/` pentru grafice:

1. **benchmark_comparison.png** - Comparație 3 metrici (Success, Reward, Steps)
2. **learning_curves.png** - Evoluția rewardurilor în timp
3. **efficiency_scatter.png** - Success Rate vs Mean Steps
4. **winner_ranking.png** - Clasament final cu câștigător evidențiat

---

## 🚀 Instalare

### Cerințe

- Python 3.8+
- pip
- Virtual environment (recomandat)

### Setup Rapid

```bash
# Navigate to project directory
cd proiect_irl

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependințe Principale

```
numpy>=1.24.0
torch>=2.0.0
gymnasium>=0.29.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
stable-baselines3>=2.0.0
```

---

## 💻 Utilizare

### Quick Start - Test Rapid

```bash
cd experiments
python test_easy_env.py
```

**Output așteptat:**
```
Q-Learning Success Rate: 63%
DQN Success Rate: 64%
SUCCESS! Agents learned to reach the goal!
```

### Benchmark Complet (Toți cei 5 Agenți)

```bash
cd experiments
python benchmark_all_agents.py
```

**Ce face:**
- Rulează Q-Learning (500 ep)
- Rulează DQN (500 ep)
- Rulează DQN+PER (500 ep)
- Rulează PPO (25k timesteps)
- Rulează PPO+RND (25k timesteps)
- Salvează rezultate în `results/benchmark_easy_TIMESTAMP.json`

**Timp estimat:** ~5-10 minute

### Vizualizare Rezultate

```bash
cd experiments
python visualize_benchmark.py
```

**Output:**
- Încarcă ultimul benchmark
- Generează 4 grafice PNG
- Salvează în `results/`

### Training Custom

#### Q-Learning
```python
from environments.easy_frozenlake import EasyFrozenLakeEnv
from agents.q_learning import QLearningAgent

env = EasyFrozenLakeEnv(map_size=4)
agent = QLearningAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.1,
    discount_factor=0.99
)

# Training
for episode in range(500):
    stats = agent.train_episode(env)
    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {stats['total_reward']}")

# Evaluation
eval_stats = agent.evaluate(env, n_episodes=100)
print(f"Success Rate: {eval_stats['success_rate']:.2%}")
```

#### DQN + PER (Recomandat)
```python
from agents.dqn_per import DQN_PERAgent

agent = DQN_PERAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.001,
    per_alpha=0.6,
    per_beta_start=0.4,
    seed=42
)

for episode in range(500):
    stats = agent.train_episode(env)

eval_stats = agent.evaluate(env, n_episodes=100)
```

#### PPO
```python
from agents.ppo import PPOAgent

agent = PPOAgent(
    env=env,
    learning_rate=0.0003,
    n_steps=512,
    batch_size=64
)

agent.train(total_timesteps=25000)
eval_stats = agent.evaluate(env, n_episodes=100)
```

---

## 📁 Structura Proiectului

```
proiect_irl/
│
├── agents/                      # Implementări algoritmi RL
│   ├── __init__.py
│   ├── q_learning.py           # Q-Learning tabular
│   ├── dqn.py                  # Deep Q-Network
│   ├── dqn_per.py             # DQN + Prioritized Replay ⭐
│   ├── ppo.py                  # Proximal Policy Optimization
│   └── ppo_rnd.py             # PPO + Random Network Distillation
│
├── environments/                # Medii custom
│   ├── __init__.py
│   ├── easy_frozenlake.py     # Environment simplu (4x4) ⭐
│   ├── dynamic_frozenlake.py  # Environment dificil (8x8)
│   └── README_ENVIRONMENTS.md  # Documentație medii
│
├── experiments/                 # Scripturi pentru rulare
│   ├── test_easy_env.py       # Test rapid pe Easy
│   ├── benchmark_all_agents.py # Benchmark complet 5 algoritmi
│   ├── visualize_benchmark.py  # Generare grafice
│   └── run_experiments.py      # Training complet (toate mediile)
│
├── results/                     # Rezultate și grafice
│   ├── benchmark_easy_*.json   # Date benchmark
│   ├── benchmark_comparison.png
│   ├── learning_curves.png
│   ├── efficiency_scatter.png
│   └── winner_ranking.png
│
├── .venv/                      # Virtual environment
├── requirements.txt            # Dependințe Python
└── README.md                   # Acest fișier
```

---

## 🎓 Concluzii și Învățăminte

### Ce am învățat din benchmark:

1. **PER chiar funcționează** ⚡
   - DQN simplu: 32% success
   - DQN+PER: 100% success
   - Sample efficiency mult mai bună

2. **Environment design contează** 🏔️
   - EasyFrozenLake: success rate > 90% pentru majoritatea
   - DynamicFrozenLake: challenge real pentru algoritmi

3. **Nu întotdeauna mai complex = mai bun** 🤔
   - Q-Learning simplu bate DQN vanilla pe medii simple
   - RND nu ajută când rewards nu sunt sparse

4. **Tuning hiperparametri e esențial** 🎛️
   - Epsilon decay
   - Learning rate
   - Buffer size
   - Update frequency

### Recomandări Practice:

**Pentru medii simple (4x4, puține găuri):**
- Folosește **Q-Learning** sau **DQN+PER**
- Training rapid (< 500 episoade)
- Success rate garantat

**Pentru medii complexe (8x8+, very sparse rewards):**
- Folosește **PPO** sau **PPO+RND**
- Mai mult training (> 1000 episoade)
- Reward shaping ajută

**Pentru research:**
- **DQN+PER** = cel mai eficient overall
- **PPO** = cel mai stabil
- **PPO+RND** = best pentru explorare dificilă

---

## 📚 Referințe

### Papers

1. **Q-Learning**
   Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3), 279-292.

2. **DQN**
   Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

3. **Prioritized Experience Replay**
   Schaul, T., et al. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.

4. **PPO**
   Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

5. **RND (Random Network Distillation)**
   Burda, Y., et al. (2018). Exploration by random network distillation. arXiv preprint arXiv:1810.12894.

### Resurse Utile

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch RL Tutorials](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)

---

## 📝 Licență

MIT License - vezi fișierul LICENSE pentru detalii.
