#!/bin/bash
#SBATCH --job-name=gat_infer_single
#SBATCH --output=logs/infer_single_%j.out
#SBATCH --error=logs/infer_single_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=30:00:00
#SBATCH --constraint=dgx

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: sbatch $0 \"SEQUENCE\" NAME CHECKPOINT_OR_FOLD [FOLD] [CONFIG_PATH]" >&2
  exit 1
fi

SEQUENCE_INPUT="$1"
SEQUENCE_NAME="$2"
ARG3="$3"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

if [[ "$ARG3" =~ ^[0-9]+$ ]]; then
  FOLD_NUMBER="$ARG3"
  CHECKPOINT_PATH="${MODEL_PATH:-$PROJECT_ROOT/experiments/fungi/gat/fold_${FOLD_NUMBER}/best_loss.pt}"
  CONFIG_PATH="${4:-configs/fungi.yaml}"
else
  CHECKPOINT_PATH="$ARG3"
  if [ $# -ge 4 ]; then
    if [[ "$4" =~ ^[0-9]+$ ]]; then
      FOLD_NUMBER="$4"
      CONFIG_PATH="${5:-configs/fungi.yaml}"
    else
      FOLD_NUMBER="0"
      CONFIG_PATH="$4"
    fi
  else
    FOLD_NUMBER="0"
    CONFIG_PATH="configs/fungi.yaml"
  fi
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

python -m pip install --user -e "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_ROOT"

export SEQUENCE_INPUT
export SEQUENCE_NAME
export CHECKPOINT_PATH
export FOLD_NUMBER
export CONFIG_PATH

python - <<'PY'
import json
import os
from pathlib import Path

from gat_pipeline.config import load_config
from gat_pipeline.inference.single import run_inference_with_outputs

sequence = os.environ["SEQUENCE_INPUT"]
name = os.environ["SEQUENCE_NAME"]
checkpoint_path = Path(os.environ["CHECKPOINT_PATH"])
fold_number = int(os.environ["FOLD_NUMBER"])
config_path = os.environ["CONFIG_PATH"]

config = load_config(config_path)
summary = run_inference_with_outputs(
    sequence,
    name,
    checkpoint_path,
    config,
    fold_number,
    Path("inference_results"),
)

print(json.dumps(summary, indent=2))
PY
