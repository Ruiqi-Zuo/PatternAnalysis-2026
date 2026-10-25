#!/bin/bash
# Train the 3D Improved U-Net on one Rangpur A100 (comp3710 partition).
#
# Submit from the project folder:
#     cd recognition/ImprovedUNet_HipMRI_s4983299
#     sbatch slurm/train3d.sh                         # seed 0, 20 epochs
#     EPOCHS=300 SEED=1 sbatch slurm/train3d.sh       # override via environment
#
# Re-submitting the same command resumes from runs/improved3d_seed<SEED>/last.pth.

#SBATCH --job-name=iunet3d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# no --mem: comp3710 nodes advertise almost no schedulable memory, so any --mem request is rejected
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

EPOCHS=${EPOCHS:-20}
SEED=${SEED:-0}

source "$HOME/miniconda3/bin/activate"
conda activate torch
nvidia-smi --query-gpu=name,memory.total --format=csv

python train.py \
    --model improved3d \
    --root /home/groups/comp3710/HipMRI_Study_open \
    --cache-dir "$HOME/cache/hipmri3d_crop" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --num-workers 8 \
    --out "runs/improved3d_seed${SEED}" \
    --resume
