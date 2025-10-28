#!/bin/bash
#SBATCH --job-name=gat_training
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=160:00:00
#SBATCH --constraint=dgx

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/fungi.yaml}"
MODEL_NAME="${MODEL_NAME:-gat}"
FOLDS="${FOLDS:-0 1 2 3 4}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

pip install --user -e "$PROJECT_ROOT"

cd "$PROJECT_ROOT"

if [[ -n "$WANDB_API_KEY" ]]; then
  export WANDB_API_KEY
else
  export WANDB_MODE=offline
fi

for fold in $FOLDS; do
  echo "=== Training fold $fold with model $MODEL_NAME ==="
  gat-pipeline train-fold \
    --config "$CONFIG_PATH" \
    --fold "$fold" \
    --model "$MODEL_NAME"
  echo "=== Completed fold $fold ==="
  sleep 2
done
