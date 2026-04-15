#!/bin/bash
#SBATCH --job-name=lfb-strategy
#SBATCH --partition=gpu-h100-killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "MASTER_PORT=$MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

torchrun \
  --nproc_per_node=1 \
  --master_port=$MASTER_PORT \
  scripts/train.py configs/olmoe/paper_ablations/olmoe17-8x1b-lfb-strategy.yaml