#!/bin/bash
#SBATCH --job-name=gat_hparam_sweep
#SBATCH --output=logs/hparam_%j.out
#SBATCH --error=logs/hparam_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=160:00:00
#SBATCH --constraint=dgx

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/fungi.yaml}"

# Hyperparameter grids (override via env vars if needed)
LR_GRID="${LR_GRID:-1e-5 2e-5 5e-5}"
DROP_GRID="${DROP_GRID:-0.3 0.45}"
WD_GRID="${WD_GRID:-1e-5 1e-4 5e-4}"
RATIO_GRID="${RATIO_GRID:-0.15 0.2 0.25}"
BATCH_GRID="${BATCH_GRID:-4 6}"

FOLDS="${FOLDS:-0 1 2 3 4}"
MODEL_NAME="${MODEL_NAME:-gat}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Set WANDB_API_KEY before submitting (e.g. WANDB_API_KEY=xxxxx sbatch scripts/run_hparam_sweep.sh)." >&2
  exit 1
fi
export WANDB_API_KEY

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

python -m pip install --user -e "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_ROOT"

run_idx=0
for lr in $LR_GRID; do
  for drop in $DROP_GRID; do
    for wd in $WD_GRID; do
      for ratio in $RATIO_GRID; do
        for batch in $BATCH_GRID; do
          run_idx=$((run_idx + 1))
          RUN_ID="sweep_lr${lr}_drop${drop}_wd${wd}_ratio${ratio}_bs${batch}_run${run_idx}"
          echo "=== Starting sweep run $RUN_ID ==="
          for fold in $FOLDS; do
            echo "-- Fold $fold"
            python -m gat_pipeline.cli train-fold \
              --config "$CONFIG_PATH" \
              --fold "$fold" \
              --model "$MODEL_NAME" \
              --lr "$lr" \
              --drop-prob "$drop" \
              --weight-decay "$wd" \
              --ratio "$ratio" \
              --train-batch-size "$batch" \
              --test-batch-size "$batch" \
              --wandb-run-name "${RUN_ID}_fold${fold}"
            sleep 2
          done
          echo "=== Completed sweep run $RUN_ID ==="
        done
      done
    done
  done
done

echo "Hyperparameter sweep finished."
