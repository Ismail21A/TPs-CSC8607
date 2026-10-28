#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=tp3-alexnet
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd ~/TPs-CSC8607/TP3

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

nvidia-smi

python -u train_imagenet.py \
    --model alexnet \
    --batch-size 128 \
    --lr 0.0001 \
    --epochs 10 \
    --patience 5 \
    --subset-size 20000
