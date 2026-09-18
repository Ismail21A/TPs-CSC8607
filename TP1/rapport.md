# TP1 — SLURM et environnement de travail

## 1. Utilisation de SLURM

### GPU alloué

Le GPU qui m'a été alloué est un **NVIDIA L4**.

### Annulation du job interactif

J'ai lancé un job interactif avec SLURM afin d'accéder à un nœud de calcul équipé d'un GPU.

Le job avait pour identifiant `2015`.

La commande utilisée pour l'annuler est :

```bash
scancel 2015
```

Après l'annulation, le job est passé par l'état `CG` (`COMPLETING`) avant de disparaître de la file d'attente.

### Job batch `hello.sh`

J'ai créé un script batch `hello.sh` contenant les directives SLURM nécessaires pour demander un GPU, un CPU et 8 Go de mémoire :

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -J hello-slurm
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvidia-smi || echo "nvidia-smi indisponible"
echo "Bonjour depuis SLURM !"
```

Le script a été soumis avec la commande :

```bash
sbatch hello.sh
```

Le job obtenu avait pour identifiant `2059`.

Le fichier de sortie généré est :

```text
logs/hello-slurm-2059.out
```

Le job a été exécuté sur le nœud :

```text
starfighter-slurm-node-06-1
```

Le fichier de sortie indique que le GPU utilisé était un **NVIDIA L4**.

### Informations sur l'utilisation des ressources

J'ai utilisé la commande suivante pour consulter les informations du job :

```bash
sacct -j 2059 --format=JobID,State,Elapsed,MaxRSS,ReqMem,ReqCPUS
```

Le résultat principal était :

```text
JobID             State    Elapsed     MaxRSS     ReqMem  ReqCPUS
------------ ---------- ---------- -------- -------- --------
2059          COMPLETED   00:00:01                    8G        1
2059.batch    COMPLETED   00:00:01     17984K                   1
2059.extern    COMPLETED   00:00:02                              1
```

`ReqMem` correspond à la quantité de mémoire RAM demandée à SLURM lors de la soumission du job. Dans mon cas, j'ai demandé **8 Go** avec l'option :

```bash
#SBATCH --mem=8G
```

`MaxRSS` correspond à la quantité maximale de mémoire RAM réellement utilisée par le job pendant son exécution.

Pour le job `2059.batch`, `MaxRSS` était de `17984K`, soit environ **18 Mo**.

Ainsi, `ReqMem` représente la mémoire demandée/réservée pour le job, tandis que `MaxRSS` indique la quantité maximale de mémoire effectivement utilisée pendant son exécution.

## 2. Création d'un environnement virtuel Python

### Installation de Miniforge

J'ai installé Miniforge afin de disposer de l'outil `mamba` pour gérer les environnements Python.

L'installation a été réalisée sur le cluster dans le répertoire :

```text
/mnt/hdd/homes/iabid/miniforge3
```

La version de `mamba` utilisée est :

```text
2.9.0
```

### Création de l'environnement

J'ai créé un environnement virtuel nommé `deeplearning` avec Python 3.10 :

```bash
mamba create -n deeplearning python=3.10
```

Après activation de l'environnement, j'ai vérifié la version de Python ainsi que le chemin vers l'interpréteur :

```bash
python --version
which python
```

Résultat :

```text
Python 3.10.21
/mnt/hdd/homes/iabid/miniforge3/envs/deeplearning/bin/python
```

### Installation de PyTorch avec CUDA

L'installation Conda initiale de PyTorch ne permettait pas d'obtenir le build GPU CUDA 12.1 souhaité avec Python 3.10. J'ai donc utilisé les wheels officielles de PyTorch avec CUDA 12.1 :

```bash
python -m pip install --default-timeout=120 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Les versions installées sont notamment :

```text
PyTorch : 2.5.1+cu121
torchvision : 0.20.1+cu121
torchaudio : 2.5.1+cu121
```

### Vérification du GPU

J'ai créé le script `check_gpu.py` suivant :

```python
import torch

print("PyTorch version:", torch.__version__)

gpu_available = torch.cuda.is_available()
print("CUDA available:", gpu_available)

if gpu_available:
    print("Device count:", torch.cuda.device_count())
    print("Device 0 name:", torch.cuda.get_device_name(0))
else:
    print("Attention, aucun GPU détecté !")
```

Le script a été exécuté avec :

```bash
python check_gpu.py
```

Le résultat obtenu est :

```text
PyTorch version: 2.5.1+cu121
CUDA available: True
Device count: 1
Device 0 name: NVIDIA L4
```

CUDA est donc correctement détecté par PyTorch et un GPU **NVIDIA L4** est accessible depuis l'environnement.

Si `torch.cuda.is_available()` avait retourné `False`, deux causes possibles auraient été l'absence de GPU correctement alloué au job SLURM ou une installation de PyTorch ne disposant pas du support CUDA.

### Reproductibilité de l'environnement

Afin de conserver les dépendances de l'environnement, j'ai généré le fichier `environment.yml` avec :

```bash
mamba env export --from-history -n deeplearning > environment.yml
```

Ce fichier est présent dans le dépôt du TP afin de permettre de reproduire l'environnement.

### TensorBoard

TensorBoard a également été installé dans l'environnement `deeplearning`.

La version installée est :

```text
2.20.0
```

La vérification a été effectuée avec :

```bash
tensorboard --version
```

TensorBoard fonctionne correctement dans l'environnement, avec un avertissement concernant l'utilisation de `pkg_resources` qui n'empêche pas son exécution.

