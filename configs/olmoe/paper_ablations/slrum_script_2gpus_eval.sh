#!/bin/bash
#SBATCH --job-name=eval-ema-bs4
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd /home/morg/students/noyahochwald/OLMo
source .venv/bin/activate

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "MASTER_PORT=$MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

PORT=$((8000 + RANDOM % 1000))
torchrun --nproc_per_node=2 --master_port=$PORT scripts/eval_routing.py \
    configs/olmoe/paper_ablations/olmoe17-8x1b-lfb-strategy.yaml \
    --device_eval_batch_size=64
