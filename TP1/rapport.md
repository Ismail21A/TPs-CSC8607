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

## 3. Exercices théoriques

### Architecture et paramètres

#### Architecture du MLP

Le modèle comporte :

- une couche d'entrée de **3 neurones** ;
- une couche cachée de **4 neurones** ;
- une couche de sortie de **2 neurones**.

Le schéma de l'architecture est le suivant :

![Architecture du MLP](schema_mlp.png)

Chaque neurone d'une couche est connecté à tous les neurones de la couche suivante.

**Nombre de paramètres sans les biais :**

Pour la première couche :

$$
3 \times 4 = 12
$$

Pour la deuxième couche :

$$
4 \times 2 = 8
$$

Donc :

$$
12 + 8 = \boxed{20}
$$

paramètres sans les biais.

**Nombre de paramètres avec les biais :**

La couche cachée possède 4 biais :

$$
4
$$

La couche de sortie possède 2 biais :

$$
2
$$

Donc le nombre total de paramètres est :

$$
20 + 4 + 2 = \boxed{26}
$$

Ainsi, le modèle possède **20 paramètres sans biais** et **26 paramètres avec biais**.

### Équations et dimensions

Pour un batch de taille $N$, l'entrée est :

$$
X \in \mathbb{R}^{N \times 3}
$$

Le forward pass est :

$$
H = \operatorname{ReLU}(X \cdot W_1^T + b_1)
$$

$$
Y = H \cdot W_2^T + b_2
$$

Les dimensions sont :

| Élément | Dimension |
|---|---|
| $X$ | $(N,3)$ |
| $W_1$ | $(4,3)$ |
| $b_1$ | $(1,4)$ → diffusé en $(N,4)$ |
| $H$ | $(N,4)$ |
| $W_2$ | $(2,4)$ |
| $b_2$ | $(1,2)$ → diffusé en $(N,2)$ |
| $Y$ | $(N,2)$ |

En effet :

$$
(N,3) \times (3,4) = (N,4)
$$

puis :

$$
(N,4) \times (4,2) = (N,2)
$$

### Graphe de calcul et rétropropagation

#### Graphe de calcul

On considère la fonction :

$$
f(x,y,z) = \frac{x}{y} + z
$$

On introduit le nœud intermédiaire :

$$
q = \frac{x}{y}
$$

Le graphe de calcul est :

![Graphe de calcul](graphe_calcul.png)

#### Forward pass

Avec :

$$
x=2,\qquad y=4,\qquad z=0
$$

On calcule d'abord :

$$
q = \frac{x}{y} = \frac{2}{4} = 0.5
$$

Puis :

$$
f = q+z = 0.5+0 = \boxed{0.5}
$$

La valeur de la fonction est donc :

$$
\boxed{f=0.5}
$$

#### Backpropagation

On commence par :

$$
f=q+z
$$

Donc :

$$
\frac{\partial f}{\partial q}=1
$$

et :

$$
\frac{\partial f}{\partial z}=1
$$

Pour :

$$
q=\frac{x}{y}
$$

on a :

$$
\frac{\partial q}{\partial x}=\frac{1}{y}
$$

et :

$$
\frac{\partial q}{\partial y}=-\frac{x}{y^2}
$$

Par la règle de la chaîne :

$$
\frac{\partial f}{\partial x}
=
\frac{\partial f}{\partial q}
\frac{\partial q}{\partial x}
$$

Donc, au point $(2,4,0)$ :

$$
\frac{\partial f}{\partial x}
=
1\times\frac{1}{4}
=
\boxed{0.25}
$$

Pour $y$ :

$$
\frac{\partial f}{\partial y}
=
\frac{\partial f}{\partial q}
\frac{\partial q}{\partial y}
$$

$$
=
1\times\left(-\frac{2}{4^2}\right)
=
-\frac{2}{16}
=
\boxed{-0.125}
$$

Enfin :

$$
\frac{\partial f}{\partial z}
=
\boxed{1}
$$

Les gradients sont donc :

$$
\boxed{
\frac{\partial f}{\partial x}=0.25,\qquad
\frac{\partial f}{\partial y}=-0.125,\qquad
\frac{\partial f}{\partial z}=1
}
$$

### Mise à jour des poids

On utilise une descente de gradient avec :

$$
\eta=1
$$

La règle de mise à jour est :

$$
x'=x-\eta\frac{\partial f}{\partial x}
$$

$$
y'=y-\eta\frac{\partial f}{\partial y}
$$

$$
z'=z-\eta\frac{\partial f}{\partial z}
$$

Pour $x$ :

$$
x'=2-1(0.25)=\boxed{1.75}
$$

Pour $y$ :

$$
y'=4-1(-0.125)=\boxed{4.125}
$$

Pour $z$ :

$$
z'=0-1(1)=\boxed{-1}
$$

On obtient donc :

$$
\boxed{x'=1.75,\qquad y'=4.125,\qquad z'=-1}
$$

La nouvelle valeur de la fonction est :

$$
f'=\frac{x'}{y'}+z'
$$

$$
f'=\frac{1.75}{4.125}-1
$$

$$
f'\approx0.4242-1
$$

$$
\boxed{f'\approx-0.5758}
$$

La valeur initiale était :

$$
f=0.5
$$

Après la mise à jour :

$$
f'\approx-0.5758
$$

On constate donc que :

$$
-0.5758 < 0.5
$$

La valeur de la fonction a bien **diminué**, comme attendu avec cette étape de descente de gradient.

## Questions de réflexion

### Pourquoi utilisons-nous la règle de la chaîne ?

La règle de la chaîne permet de calculer le gradient d'une fonction composée de nombreuses opérations successives. Dans un réseau profond, elle permet de propager l'erreur depuis la sortie vers les couches précédentes afin de calculer le gradient de chaque paramètre.

### Pourquoi utiliser des mini-batchs ?

Les mini-batchs offrent un compromis entre le coût de calcul d'un seul exemple et celui de l'ensemble des données. Ils permettent d'exploiter efficacement les calculs parallèles tout en fournissant des estimations fréquentes du gradient pour mettre à jour les paramètres.

## Association des fonctions de sortie et des fonctions de perte

| Tâche | Fonction finale (Sortie) | Fonction de perte (Loss) |
|---|---|---|
| Classification binaire | **Sigmoïde** | **Binary Cross-Entropy (BCE)** |
| Classification multi-classe | **Softmax** | **Cross-Entropy** |
| Régression pure | **Identité (aucune)** | **MSE (Mean Squared Error)** |

Ainsi :

1. Classification binaire → **Sigmoïde + Binary Cross-Entropy**
2. Classification multi-classe → **Softmax + Cross-Entropy**
3. Régression pure → **Identité + MSE**
