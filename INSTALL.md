# Ghid de Instalare Detaliat

Acest document conține instrucțiuni complete pentru instalarea și configurarea proiectului.

## Cerințe de Sistem

### Hardware
- CPU: Orice procesor modern (Intel/AMD)
- RAM: Minim 4GB (recomandat 8GB+)
- GPU: Opțional (NVIDIA cu CUDA pentru accelerare DQN)
- Spațiu disc: ~2GB

### Software
- Python 3.8, 3.9, 3.10, sau 3.11
- pip (manager de pachete Python)
- git (opțional, pentru clonare repository)

## Instalare Pas cu Pas

### Pasul 1: Verificare Python

Verifică că ai Python instalat:

```bash
python --version
# sau
python3 --version
```

Ar trebui să vezi ceva gen: `Python 3.10.x`

Dacă nu ai Python instalat:
- **Windows**: Descarcă de la [python.org](https://www.python.org/downloads/)
- **Linux**: `sudo apt-get install python3 python3-pip`
- **Mac**: `brew install python3`

### Pasul 2: Navighează în Directorul Proiectului

```bash
cd C:\Users\Horia\PyCharmMiscProject\proiect_irl
```

### Pasul 3: Verificare Virtual Environment

Virtual environment-ul ar trebui să fie deja creat (`.venv/`). Dacă nu există, creează-l:

```bash
python -m venv .venv
```

### Pasul 4: Activare Virtual Environment

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

După activare, ar trebui să vezi `(.venv)` în prompt.

### Pasul 5: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Pasul 6: Instalare Dependențe

```bash
pip install -r requirements.txt
```

Acest pas va instala:
- gymnasium==0.29.1
- numpy==1.24.3
- torch==2.1.0
- stable-baselines3==2.2.1
- matplotlib==3.8.0
- seaborn==0.13.0
- pandas==2.1.1
- tqdm==4.66.1

**Notă**: Instalarea PyTorch poate dura câteva minute.

### Pasul 7: Verificare Instalare

```bash
python test_setup.py
```

Ar trebui să vezi:
```
✓ TOATE TESTELE AU TRECUT CU SUCCES!
```

## Probleme Comune de Instalare

### Problem 1: pip nu este recunoscut

**Soluție Windows:**
```bash
python -m pip install -r requirements.txt
```

**Soluție Linux/Mac:**
```bash
python3 -m pip install -r requirements.txt
```

### Problem 2: PowerShell nu permite rularea scripturilor

**Eroare:** `cannot be loaded because running scripts is disabled`

**Soluție:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Apoi rulează din nou:
```powershell
.venv\Scripts\Activate.ps1
```

### Problem 3: PyTorch instalare eșuată

**Soluție:** Instalează PyTorch manual pentru CPU:

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### Problem 4: CUDA/GPU erori

Dacă întâmpini probleme cu CUDA și vrei doar CPU:

```bash
pip uninstall torch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Problem 5: Gymnasium instalare eșuată

**Soluție:**
```bash
pip install gymnasium==0.29.1 --no-cache-dir
```

### Problem 6: Module Not Found după instalare

Asigură-te că:
1. Virtual environment este activat (`(.venv)` în prompt)
2. Ești în directorul corect: `proiect_irl/`
3. Pachete sunt instalate: `pip list | grep gymnasium`

## Instalare GPU (Opțional - Pentru NVIDIA)

Dacă ai placă video NVIDIA și vrei să folosești CUDA pentru DQN:

### Windows/Linux cu CUDA 11.8:

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### Verificare CUDA:

```python
import torch
print(torch.cuda.is_available())  # True dacă CUDA este disponibil
print(torch.cuda.get_device_name(0))  # Numele GPU-ului
```

## Instalare din Zero (Fără Virtual Environment Precreat)

Dacă dorești să creezi totul de la zero:

```bash
# 1. Creează directorul proiectului
mkdir proiect_irl
cd proiect_irl

# 2. Creează virtual environment
python -m venv .venv

# 3. Activează virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Instalează dependențe
pip install gymnasium==0.29.1 numpy==1.24.3 torch==2.1.0 stable-baselines3==2.2.1 matplotlib==3.8.0 seaborn==0.13.0 pandas==2.1.1 tqdm==4.66.1

# 5. Copiază fișierele proiectului în directorul curent
```

## Verificare Versiuni Pachete

Pentru a verifica că toate pachetele sunt instalate corect:

```bash
pip list
```

Ar trebui să vezi:

```
Package              Version
-------------------- -------
gymnasium            0.29.1
matplotlib           3.8.0
numpy                1.24.3
pandas               2.1.1
seaborn              0.13.0
stable-baselines3    2.2.1
torch                2.1.0
tqdm                 4.66.1
```

## Dezactivare Virtual Environment

După ce ai terminat lucrul:

```bash
deactivate
```

## Reinstalare Completă

Dacă întâmpini probleme persistente:

```bash
# 1. Dezactivează virtual environment
deactivate

# 2. Șterge virtual environment
# Windows:
rmdir /s .venv
# Linux/Mac:
rm -rf .venv

# 3. Recrează virtual environment
python -m venv .venv

# 4. Activează și reinstalează
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

pip install --upgrade pip
pip install -r requirements.txt
```

## Next Steps

După instalare cu succes:

1. Rulează test_setup.py pentru verificare finală
2. Citește [QUICKSTART.md](QUICKSTART.md) pentru început rapid
3. Citește [README.md](README.md) pentru documentare completă
4. Rulează `python demo.py` pentru o demonstrație vizuală

## Support

Pentru probleme de instalare:

1. Verifică că ai versiunea corectă de Python (3.8-3.11)
2. Asigură-te că virtual environment este activat
3. Încearcă reinstalarea completă (vezi secțiunea de mai sus)
4. Verifică logs-urile de eroare și caută pe Google/StackOverflow

## Informații Suplimentare

- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [pip Documentation](https://pip.pypa.io/en/stable/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
