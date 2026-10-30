#!/bin/bash
#SBATCH --job-name=tp4-music
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --account=csc_8607
#SBATCH --qos=normal

set -euo pipefail

cd "$HOME/TPs-CSC8607/TP4"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate deeplearning

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

EXPERIMENT="${1:?Usage: sbatch run_experiments.sh <experiment>}"

case "$EXPERIMENT" in
    rnn_noclip|rnn_clip|lstm_clip)
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        exit 1
        ;;
esac

echo "Experiment: $EXPERIMENT"
echo "Node: $(hostname)"
echo "Start: $(date)"

python -u train_music.py --experiment "$EXPERIMENT"

echo "End: $(date)"
