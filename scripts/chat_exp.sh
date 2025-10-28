#!/bin/bash
#SBATCH --job-name=gnnexp_nodes
#SBATCH --output=logs/gnnexp_nodes_%j.out
#SBATCH --error=logs/gnnexp_nodes_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=30:00:00
#SBATCH --constraint=dgx

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 \"SEQUENCE\" MODEL_CHECKPOINT OUTPUT_NAME [CONFIG_PATH]" >&2
  exit 1
fi

SEQUENCE="$1"
MODEL_PATH="$2"
OUTPUT_NAME="$3"
CONFIG_PATH="${4:-}"

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

pip install --user -e "$PROJECT_ROOT"

cd "$PROJECT_ROOT"

CMD=(
  gat-pipeline explain-nodes
  --sequence "$SEQUENCE"
  --name "$OUTPUT_NAME"
  --model-checkpoint "$MODEL_PATH"
  --output-dir "gnn_results"
)

if [[ -n "$CONFIG_PATH" ]]; then
  CMD+=(--config "$CONFIG_PATH")
fi

"${CMD[@]}"
