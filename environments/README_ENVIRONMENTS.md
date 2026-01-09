# FrozenLake Environments

Proiectul include doua versiuni ale mediului FrozenLake:

## 1. EasyFrozenLake (RECOMANDAT pentru invatare)

**Locatie:** `environments/easy_frozenlake.py`

### Caracteristici:
- **Map size:** 4x4 (default)
- **Slippery:** 0.05 (constant, nu creste)
- **Hole ratio:** 10% (foarte putine gauri)
- **Safe zone:** 2x2 langa start (protejat de gauri)
- **Ice melting:** OFF (fara topire)
- **Reward shaping:** ON (bonus cand te apropii de goal)
- **Max steps:** 50

### Rezultate:
- **Q-Learning:** 63% success rate in 300 episoade
- **DQN:** 64% success rate in 400 episoade

### Utilizare:
```python
from environments.easy_frozenlake import EasyFrozenLakeEnv

env = EasyFrozenLakeEnv(
    map_size=4,
    slippery=0.05,
    hole_ratio=0.10,
    shaped_rewards=True,
    seed=42
)
```

### De ce folosim EasyFrozenLake?
- Garantat **learnable** - agentii pot invata rapid
- Success rate > 60% dupa cateva sute de episoade
- Ideal pentru **debugging** si **testare rapida** a algoritmilor
- Baza solida pentru a creste apoi dificultatea gradual

---

## 2. DynamicFrozenLake (CHALLENGE)

**Locatie:** `environments/dynamic_frozenlake.py`

### Caracteristici:
- **Map size:** 8x8 (default)
- **Slippery:** 0.08 → 0.25 (creste in timp)
- **Hole ratio:** 18-20%
- **Ice melting:** ON (optional, controlat)
- **Reward shaping:** ON (optional)
- **Max steps:** 120-140

### Dificultate:
- Mult mai greu decat EasyFrozenLake
- Necesita mai multe episoade pentru invatare
- Slippery variabil + topire gheata = challenge complex

### Utilizare:
```python
from environments.dynamic_frozenlake import DynamicFrozenLakeEnv

env = DynamicFrozenLakeEnv(
    map_size=8,
    slippery_start=0.08,
    slippery_end=0.25,
    ice_melting=True,
    melting_rate=0.003,
    hole_ratio=0.18,
    shaped_rewards=True
)
```

---

## Cum sa alegi environment-ul potrivit?

### Foloseste **EasyFrozenLake** daca:
- Vrei sa testezi rapid un algoritm nou
- Ai nevoie de success rate > 0 garantat
- Vrei sa vezi agentul invatand rapid (< 500 episoade)
- Debugging sau prototyping

### Foloseste **DynamicFrozenLake** daca:
- Vrei un challenge mai greu
- Testezi robustetea algoritmilor
- Vrei sa compari performanta pe task dificil
- Ai timp pentru training mai lung (> 1000 episoade)

---

## Parametri comuni

Ambele environments suporta:

### Reward structure:
- `goal_reward`: 1.0 (ajunge la G)
- `hole_penalty`: -0.5 sau -1.0 (cade in H)
- `step_penalty`: -0.01 (fiecare pas)

### Reward shaping:
- `shaped_rewards=True`: bonus mic cand te apropii de goal
- `shaping_scale`: cat de mare e bonus-ul (0.02-0.05)

### Safe zone:
- Zona protejata langa start
- Nu se genereaza gauri acolo
- Previne episoade terminate instant

---

## Quick Start

### Test rapid pe EasyFrozenLake:
```bash
cd experiments
python test_easy_env.py
```

### Training complet:
```bash
cd experiments
python run_experiments.py
```

---

## Rezultate comparative

| Environment | Map | Slippery | Holes | Q-Learning SR | DQN SR |
|-------------|-----|----------|-------|---------------|--------|
| **Easy** | 4x4 | 0.05 | 10% | **63%** | **64%** |
| **Dynamic** | 8x8 | 0.08-0.25 | 18% | ~5-15% | ~10-20% |

**SR = Success Rate** (dupa 300-500 episoade)
