#!/bin/bash
#SBATCH --job-name=gat_training_human
#SBATCH --output=logs/train_hum%j.out
#SBATCH --error=logs/train_hum%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=160:00:00
#SBATCH --constraint=dgx

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/human.yaml}"
MODEL_NAME="${MODEL_NAME:-gat}"
FOLDS="${FOLDS:-0 1 2 3 4}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

python -m pip install --user -e "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_ROOT"

if [ -z "$WANDB_API_KEY" ]; then
  echo "Set WANDB_API_KEY before submitting (e.g. WANDB_API_KEY=xxxxx sbatch scripts/run_training.sh)." >&2
  exit 1
fi
export WANDB_API_KEY

for fold in $FOLDS; do
  echo "=== Training fold $fold with model $MODEL_NAME ==="
  python -m gat_pipeline.cli train-fold \
    --config "$CONFIG_PATH" \
    --fold "$fold" \
    --model "$MODEL_NAME"
  echo "=== Completed fold $fold ==="
  sleep 2
done
