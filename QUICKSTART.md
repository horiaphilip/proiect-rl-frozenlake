# Quick Start Guide

Ghid rapid pentru a începe să lucrezi cu proiectul.

## Instalare Rapidă (Windows)

```bash
# 1. Navighează în directorul proiectului
cd C:\Users\Horia\PyCharmMiscProject\proiect_irl

# 2. Activează virtual environment
.venv\Scripts\activate

# 3. Instalează dependențele
pip install -r requirements.txt

# 4. Testează configurarea
python test_setup.py
```

## Instalare Rapidă (Linux/Mac)

```bash
# 1. Navighează în directorul proiectului
cd /path/to/proiect_irl

# 2. Activează virtual environment
source .venv/bin/activate

# 3. Instalează dependențele
pip install -r requirements.txt

# 4. Testează configurarea
python test_setup.py
```

## Rulare Experimente

### Opțiunea 1: Experimente Complete (Recomandat)

Antrenează toți cei 3 agenți (durează ~30-60 minute):

```bash
cd experiments
python run_experiments.py
```

Apoi generează graficele:

```bash
python visualize.py
```

### Opțiunea 2: Test Rapid (Demo)

Testează doar câteva episoade pentru fiecare agent:

```python
# Rulează în Python interpreter sau creează un script
from environments.dynamic_frozenlake import DynamicFrozenLakeEnv
from agents.q_learning import QLearningAgent

# Creează mediul
env = DynamicFrozenLakeEnv(map_size=8)

# Creează agent
agent = QLearningAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n
)

# Antrenează pe 100 episoade
for episode in range(100):
    stats = agent.train_episode(env)
    if episode % 20 == 0:
        print(f"Episode {episode}: reward={stats['total_reward']:.3f}")

# Evaluează
eval_stats = agent.evaluate(env, n_episodes=10)
print(f"\nEvaluare: {eval_stats}")
```

## Structura Output-urilor

După rularea experimentelor, vei găsi în `results/experiment_TIMESTAMP/`:

```
results/experiment_20250125_143022/
├── results.json              # Date brute (JSON)
└── plots/                    # Grafice
    ├── learning_curves.png
    ├── final_comparison.png
    ├── convergence_analysis.png
    └── comparison_table.csv
```

## Modificare Parametri

### Modificare Mediu

Editează în `experiments/run_experiments.py`:

```python
env = DynamicFrozenLakeEnv(
    map_size=8,              # 4, 8, 12, 16
    max_steps=100,           # Limită pași
    slippery_start=0.1,      # Probabilitate alunecare inițială
    slippery_end=0.4,        # Probabilitate alunecare finală
    step_penalty=-0.01,      # Penalizare per pas
    ice_melting=True,        # Activează topirea gheții
    melting_rate=0.01        # Rata de topire
)
```

### Modificare Hiperparametri Agenți

**Q-Learning:**
```python
agent = QLearningAgent(
    n_states=64,
    n_actions=4,
    learning_rate=0.1,       # Alpha
    discount_factor=0.99,    # Gamma
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995
)
```

**DQN:**
```python
agent = DQNAgent(
    n_states=64,
    n_actions=4,
    learning_rate=0.001,
    batch_size=64,
    buffer_capacity=10000,
    hidden_dim=128
)
```

**PPO:**
```python
agent = PPOAgent(
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10
)
```

## Probleme Comune

### 1. ModuleNotFoundError

**Problemă:** `ModuleNotFoundError: No module named 'environments'`

**Soluție:**
```bash
# Asigură-te că ești în directorul corect
cd experiments  # Pentru run_experiments.py și visualize.py
```

### 2. CUDA Out of Memory

**Problemă:** DQN aruncă eroare de memorie GPU

**Soluție:**
```python
# În agents/dqn.py, forțează CPU
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

## Tips & Tricks

### Rulare Rapidă pentru Testare

Modifică în `run_experiments.py`:

```python
N_EPISODES = 100        # În loc de 500
PPO_TIMESTEPS = 10000   # În loc de 50000
N_RUNS = 2              # În loc de 5
```

### Monitorizare în Timp Real

Adaugă print statements în `train_episode()`:

```python
if episode % 10 == 0:
    print(f"Episode {episode}/{n_episodes}: reward={stats['total_reward']:.3f}")
```

### Salvare Agenți Antrenați

```python
# După antrenament
agent.save("models/q_learning_agent.pkl")

# Încărcare
agent.load("models/q_learning_agent.pkl")
```

## Next Steps

După ce ai rulat experimentele cu succes:

1. Analizează graficele din `results/experiment_TIMESTAMP/plots/`
2. Citește comparison_table.csv pentru metrici detaliate
3. Experimentează cu hiperparametri diferiți
4. Încearcă dimensiuni diferite ale hărții (4x4, 12x12, 16x16)
5. Modifică mecanicile mediului pentru a face problema mai ușoară/mai grea

## Resurse Utile

- [README.md](README.md) - Documentare completă
- [test_setup.py](test_setup.py) - Test configurare
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

## Support

Pentru întrebări sau probleme:
1. Verifică README.md secțiunea "Probleme Întâmpinate"
2. Rulează test_setup.py pentru a identifica probleme
3. Verifică că toate dependențele sunt instalate: `pip list`
