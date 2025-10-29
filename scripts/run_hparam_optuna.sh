#!/bin/bash
#SBATCH --job-name=gat_optuna
#SBATCH --output=logs/optuna_%j.out
#SBATCH --error=logs/optuna_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=160:00:00
#SBATCH --constraint=dgx

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/fungi.yaml}"
FOLDS="${FOLDS:-0 1 2 3 4}"
TRIALS="${TRIALS:-30}"
MODEL_NAME="${MODEL_NAME:-gat}"
SEED="${SEED:-1029}"
PRUNER="${PRUNER:-median}"
STUDY_NAME="${STUDY_NAME:-gat-optuna}"
STORAGE="${STORAGE:-}"
DIRECTION="${DIRECTION:-minimize}"
N_STARTUP="${N_STARTUP:-5}"
LOG_WANDB="${LOG_WANDB:-1}"

if [ -z "${WANDB_API_KEY:-}" ] && [ "$LOG_WANDB" = "1" ]; then
  echo "Set WANDB_API_KEY or disable WandB logging with LOG_WANDB=0." >&2
  exit 1
fi
if [ "$LOG_WANDB" = "1" ]; then
  export WANDB_API_KEY
else
  export WANDB_MODE=offline
fi

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

python -m pip install --user -e "${PROJECT_ROOT}[hpo]"
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_ROOT"

set -- \
  python -m gat_pipeline.hparam.optuna_runner \
  --config "$CONFIG_PATH" \
  --model "$MODEL_NAME" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --pruner "$PRUNER" \
  --study-name "$STUDY_NAME" \
  --direction "$DIRECTION" \
  --n-startup-trials "$N_STARTUP"

if [ -n "$STORAGE" ]; then
  set -- "$@" --storage "$STORAGE"
fi

for fold in $FOLDS; do
  set -- "$@" --folds "$fold"
done

if [ "$LOG_WANDB" = "1" ]; then
  set -- "$@" --wandb
fi

"$@"
