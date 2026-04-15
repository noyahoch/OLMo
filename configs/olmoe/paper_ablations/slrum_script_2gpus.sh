#!/bin/bash
#SBATCH --job-name=ema-stg
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd /home/morg/students/noyahochwald/OLMo
source .venv/bin/activate

export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "MASTER_PORT=$MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

torchrun --nproc_per_node=2 scripts/train.py configs/olmoe/paper_ablations/olmoe17-8x1b-ema-strategy.yaml 