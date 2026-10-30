#!/bin/bash
#SBATCH --job-name=music-lstm
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-music-lstm-%j.out

set -euo pipefail

cd "$HOME/TPs-CSC8607/TP4"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate deeplearning

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi

# Extended LSTM training
python -u train_music.py --experiment lstm_long

# Generate three examples using the best checkpoint
for temperature in 0.2 0.8 2.0; do
    case "$temperature" in
        0.2) suffix="02" ;;
        0.8) suffix="08" ;;
        2.0) suffix="20" ;;
    esac

    echo "========================================"
    echo "Generating music at temperature $temperature"
    echo "========================================"

    python -u generate_music.py \
        --checkpoint checkpoints/lstm_long_best.pt \
        --temperature "$temperature" \
        --length 300 \
        --output "music_long_temp_${suffix}.abc"
done

echo "End: $(date)"

