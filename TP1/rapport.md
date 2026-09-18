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
------------ ---------- ---------- ---------- -------- --------
2059          COMPLETED   00:00:01                    8G        1
2059.batch    COMPLETED   00:00:01     17984K                   1
2059.extern   COMPLETED   00:00:02                              1
```

`ReqMem` correspond à la quantité de mémoire RAM demandée à SLURM lors de la soumission du job. Dans mon cas, j'ai demandé **8 Go** avec l'option :

```bash
#SBATCH --mem=8G
```

`MaxRSS` correspond à la quantité maximale de mémoire RAM réellement utilisée par le job pendant son exécution.

Pour le job `2059.batch`, `MaxRSS` était de `17984K`, soit environ **18 Mo**.

Ainsi, `ReqMem` représente la mémoire demandée/réservée pour le job, tandis que `MaxRSS` indique la quantité maximale de mémoire effectivement utilisée pendant son exécution.

