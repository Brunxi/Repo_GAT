#!/bin/bash
#SBATCH --job-name=gat_plot_attention
#SBATCH --output=logs/plot_attention_%j.out
#SBATCH --error=logs/plot_attention_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Uso: sbatch $0 \"SECUENCIA\" NOMBRE CHECKPOINT [CONFIG] [FOLD]" >&2
  exit 1
fi

SEQUENCE_INPUT="$1"
PROTEIN_NAME="$2"
CHECKPOINT_PATH="$3"
CONFIG_PATH="${4:-configs/fungi.yaml}"
FOLD_NUMBER="${5:-0}"

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
INFERENCE_DIR="${INFERENCE_DIR:-$PROJECT_ROOT/inference_results}"
EXPLAIN_DIR="${EXPLAIN_DIR:-$PROJECT_ROOT/gnn_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/graficos}"

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

python -m pip install --user -e "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

export SEQUENCE_INPUT
export PROTEIN_NAME
export CHECKPOINT_PATH
export CONFIG_PATH
export FOLD_NUMBER
export INFERENCE_DIR
export EXPLAIN_DIR
export OUTPUT_DIR

python - <<'PY'
import os
from pathlib import Path

from gat_pipeline.config import load_config
from gat_pipeline.visualization import plot_attention_and_importance

sequence = os.environ["SEQUENCE_INPUT"]
name = os.environ["PROTEIN_NAME"]
checkpoint_path = Path(os.environ["CHECKPOINT_PATH"])
config_path = os.environ["CONFIG_PATH"]
fold_number = int(os.environ["FOLD_NUMBER"])
inference_dir = Path(os.environ["INFERENCE_DIR"])
explain_dir = Path(os.environ["EXPLAIN_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])

config = load_config(config_path)

line_path, contact_path, contact_alt_path = plot_attention_and_importance(
    sequence=sequence,
    protein_name=name,
    checkpoint_path=checkpoint_path,
    config=config,
    fold_number=fold_number,
    inference_dir=inference_dir,
    explain_dir=explain_dir,
    output_dir=output_dir,
)

print(f"Saved line figure to {line_path}")
print(f"Saved contact map (teal) to {contact_path}")
print(f"Saved contact map (diverging) to {contact_alt_path}")
PY
