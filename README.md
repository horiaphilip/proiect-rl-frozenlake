# Proiect Reinforcement Learning - Dynamic FrozenLake

Proiect de Reinforcement Learning care implementează și compară trei algoritmi diferiți (Q-Learning, DQN, PPO) pe un mediu personalizat FrozenLake dinamic.

## 📋 Cuprins

- [Descriere](#descriere)
- [Structura Proiectului](#structura-proiectului)
- [Instalare](#instalare)
- [Utilizare](#utilizare)
- [Arhitectura Mediului](#arhitectura-mediului)
- [Algoritmi Implementați](#algoritmi-implementați)
- [Rezultate](#rezultate)
- [Probleme Întâmpinate](#probleme-întâmpinate)

## 🎯 Descriere

Acest proiect explorează performanța și comportamentul a trei algoritmi de Reinforcement Learning într-un mediu dinamic bazat pe clasicul joc FrozenLake. Mediul a fost modificat pentru a include mai multe mecanici dinamice care cresc complexitatea și realismul problemei.

### Caracteristici Principale

- **Mediu Personalizat**: FrozenLake cu dificultate crescândă
- **3 Algoritmi RL**: Q-Learning (tabular), DQN (deep), PPO (policy-based)
- **Experimente Multiple**: 5 rulări independente per algoritm
- **Analiză Completă**: Grafice, tabele, metrici de performanță
- **Cod Bine Structurat**: Modular, comentat, extensibil

## 📁 Structura Proiectului

```
proiect_irl/
│
├── environments/              # Mediul personalizat
│   ├── __init__.py
│   └── dynamic_frozenlake.py # Implementare DynamicFrozenLake
│
├── agents/                    # Agenții RL
│   ├── __init__.py
│   ├── q_learning.py         # Q-Learning tabular
│   ├── dqn.py                # Deep Q-Network
│   └── ppo.py                # Proximal Policy Optimization
│
├── experiments/               # Scripturi pentru experimente
│   ├── __init__.py
│   ├── run_experiments.py    # Rulare experimente
│   └── visualize.py          # Vizualizare rezultate
│
├── results/                   # Rezultate experimente
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── results.json      # Date brute
│       └── plots/            # Grafice și tabele
│
├── .venv/                    # Virtual environment
├── requirements.txt          # Dependențe Python
└── README.md                 # Acest fișier
```

## 🚀 Instalare

### Cerințe

- Python 3.8+
- pip

### Pași de Instalare

1. **Clonare/Descărcare Proiect**
   ```bash
   cd C:\Users\Horia\PyCharmMiscProject\proiect_irl
   ```

2. **Activare Virtual Environment**

   **Windows:**
   ```bash
   .venv\Scripts\activate
   ```

   **Linux/Mac:**
   ```bash
   source .venv/bin/activate
   ```

3. **Instalare Dependențe**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Utilizare

### Rulare Experimente

Pentru a antrena toți cei 3 agenți și a rula experimentele complete:

```bash
cd experiments
python run_experiments.py
```

Acest script va:
- Antrena Q-Learning pentru 500 episoade (5 rulări)
- Antrena DQN pentru 500 episoade (5 rulări)
- Antrena PPO pentru 50,000 timesteps (5 rulări)
- Salva rezultatele în `results/experiment_TIMESTAMP/`

### Vizualizare Rezultate

După rularea experimentelor, generează graficele și tabelele:

```bash
cd experiments
python visualize.py
```

Acest script va genera:
- `learning_curves.png` - Curbe de învățare pentru toți algoritmii
- `final_comparison.png` - Comparație metrici finale
- `convergence_analysis.png` - Analiză convergență și stabilitate
- `comparison_table.csv` - Tabel cu toate metricile

### Testare Rapidă a Mediului

```python
from environments.dynamic_frozenlake import DynamicFrozenLakeEnv

# Creează mediul
env = DynamicFrozenLakeEnv(
    map_size=8,
    max_steps=100,
    render_mode="human"
)

# Test episod
state, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Acțiune aleatorie
    state, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        break

env.close()
```

## 🎮 Arhitectura Mediului

### DynamicFrozenLake

Mediul `DynamicFrozenLakeEnv` extinde conceptul clasic FrozenLake cu următoarele mecanici dinamice:

#### Caracteristici Dinamice

1. **Probabilitate de Alunecare Variabilă**
   - Pornește de la 0.1 (10% șansă de alunecare)
   - Crește liniar până la 0.4 (40% șansă)
   - Crește pe parcursul episodului → dificultate crescândă

2. **Penalizare pentru Pași**
   - Fiecare pas costă -0.01 reward
   - Încurajează căi optime și eficiente

3. **Gheață care se Topește**
   - Gheața se topește progresiv (transformându-se în găuri)
   - Rata de topire: 1% per pas
   - Face mediul mai imprevizibil

4. **Dimensiune Variabilă**
   - Hărți de la 4x4 până la 16x16
   - Proiectul folosește 8x8 (64 stări)

5. **Limită de Pași**
   - Maximum 100 pași per episod
   - Previne bucle infinite

#### Spații

- **Observation Space**: Discrete(64) - stări de la 0 la 63
- **Action Space**: Discrete(4) - LEFT, DOWN, RIGHT, UP

#### Rewards

- **Goal**: +1.0 (ajunge la destinație)
- **Hole**: 0.0 (cade în gaură)
- **Step**: -0.01 (fiecare pas)

## 🤖 Algoritmi Implementați

### 1. Q-Learning (Tabular)

**Tip**: Metoda tabulară clasică
**Complexitate**: O(|S| × |A|) = O(64 × 4) = 256 intrări în tabelă

#### Principiu

Q-Learning învață o tabelă Q(s, a) care estimează reward-ul cumulativ așteptat pentru fiecare pereche (stare, acțiune).

**Update Rule**:
```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

#### Hiperparametri

- Learning rate (α): 0.1
- Discount factor (γ): 0.99
- Epsilon start: 1.0
- Epsilon end: 0.01
- Epsilon decay: 0.995

#### Avantaje

- Simplu și eficient pentru spații discrete mici
- Garantează convergență la politica optimă
- Nu necesită rețele neurale

#### Dezavantaje

- Nu scalează la spații mari de stări
- Nu poate generaliza între stări similare

---

### 2. DQN (Deep Q-Network)

**Tip**: Deep Reinforcement Learning (value-based)
**Arhitectură**: MLP cu 2 straturi ascunse (128 neuroni fiecare)

#### Principiu

DQN folosește o rețea neuronală pentru a aproxima funcția Q, permițând generalizare între stări.

**Componente Cheie**:
- **Experience Replay**: Buffer de 10,000 experiențe
- **Target Network**: Actualizat la fiecare 10 episoade
- **Epsilon-Greedy**: Explorare vs. exploatare

#### Arhitectură Rețea

```
Input (64) → Dense(128) → ReLU → Dense(128) → ReLU → Output(4)
```

#### Hiperparametri

- Learning rate: 0.001
- Discount factor (γ): 0.99
- Batch size: 64
- Buffer capacity: 10,000
- Target update frequency: 10 episoade
- Hidden dim: 128

#### Avantaje

- Scalează la spații mari de stări
- Generalizează între stări similare
- Poate învăța din experiențe anterioare

#### Dezavantaje

- Mai complex decât Q-Learning
- Necesită tuning atent al hiperparametrilor
- Poate fi instabil fără experience replay

---

### 3. PPO (Proximal Policy Optimization)

**Tip**: Policy-based (modern policy gradient)
**Implementare**: Stable Baselines3

#### Principiu

PPO învață direct o politică (mapare stare → acțiune) în loc de o funcție Q.

**Caracteristici**:
- **Clipped Objective**: Previne update-uri prea mari
- **GAE**: Generalized Advantage Estimation
- **Multiple Epochs**: Învață din același batch de date

#### Hiperparametri

- Learning rate: 0.0003
- N steps: 2,048
- Batch size: 64
- N epochs: 10
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.2

#### Avantaje

- Foarte stabil și robust
- State-of-the-art pentru multe task-uri
- Funcționează bine out-of-the-box

#### Dezavantaje

- Mai lent decât DQN (necesită mai multe sample-uri)
- Hiperparametri mai complecși

## 📊 Rezultate

### Metrici de Evaluare

Pentru fiecare algoritm se măsoară:

1. **Mean Reward**: Reward-ul mediu pe episod
2. **Success Rate**: Procentul de episoade finalizate cu succes
3. **Mean Steps**: Numărul mediu de pași până la terminare
4. **Convergence**: Viteza de convergență către politica optimă
5. **Stability**: Variația în performanță (stabilitate)

### Rezultate Așteptate

**Ierarhie Așteptată** (de la cel mai bun la cel mai slab):

1. **PPO**: Cel mai bun success rate și stabilitate
2. **DQN**: Performanță bună, dar mai instabil
3. **Q-Learning**: Performanță decentă, dar mai lent

### Grafice Generate

1. **Learning Curves**
   - Reward per episode
   - Steps per episode
   - Epsilon decay
   - Loss (DQN)

2. **Final Comparison**
   - Mean reward cu standard deviation
   - Success rate cu standard deviation

3. **Convergence Analysis**
   - Rolling average reward
   - Variance în reward (stabilitate)

## 🔧 Probleme Întâmpinate și Soluții

### 1. Instabilitate DQN

**Problemă**: DQN avea performanță instabilă în primele episoade.

**Cauză**:
- Replay buffer gol la început
- Target network nu era actualizat suficient de des

**Soluție**:
- Crescut dimensiunea buffer-ului la 10,000
- Optimizat frecvența de actualizare a target network
- Adăugat warm-up period pentru replay buffer

### 2. Explorare Insuficientă Q-Learning

**Problemă**: Q-Learning converge prematur către politici suboptimale.

**Cauză**:
- Epsilon decay prea rapid
- Nu explorează suficient spațiul de stări

**Soluție**:
- Ajustat epsilon decay de la 0.99 la 0.995
- Crescut numărul de episoade de antrenament

### 3. Mediu Prea Dificil

**Problemă**: Toți algoritmii aveau success rate < 10% inițial.

**Cauză**:
- Probabilitate de alunecare prea mare
- Topirea gheții prea rapidă

**Soluție**:
- Redus slippery_start de la 0.3 la 0.1
- Redus melting_rate de la 0.02 la 0.01
- Ajustat step_penalty pentru a nu penaliza prea mult

### 4. PPO Lent

**Problemă**: PPO necesită mult timp pentru antrenament.

**Cauză**:
- N_steps prea mare (4096)
- Multiple epochs per batch

**Soluție**:
- Redus n_steps la 2048
- Optimizat batch_size la 64
- Folosit vectorized environments (posibil upgrade viitor)

## 🎓 Învățăminte

### Concluzii Tehnice

1. **Q-Learning** funcționează bine pentru medii simple și discrete
2. **DQN** oferă flexibilitate dar necesită tuning atent
3. **PPO** este cel mai robust dar și cel mai costisitor

### Best Practices

- Începe cu metode simple (Q-Learning) înainte de deep RL
- Folosește experimente multiple pentru a evalua stabilitatea
- Monitorizează nu doar reward-ul, ci și alte metrici (steps, success rate)
- Vizualizarea rezultatelor este esențială pentru înțelegere

## 📚 Referințe

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 👥 Autor

Proiect realizat pentru cursul de Introducere în Reinforcement Learning (IRL).

## 📝 Licență

Acest proiect este realizat în scop educațional.
