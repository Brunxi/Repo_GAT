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

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 \"SEQUENCE\" NAME CHECKPOINT_PATH [CONFIG_PATH]" >&2
  exit 1
fi

SEQUENCE="$1"
SEQUENCE_NAME="$2"
CHECKPOINT="$3"
CONFIG_PATH="${4:-configs/fungi.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

pip install --user -e "$PROJECT_ROOT"

cd "$PROJECT_ROOT"

gat-pipeline infer-sequence \
  --config "$CONFIG_PATH" \
  --sequence "$SEQUENCE" \
  --name "$SEQUENCE_NAME" \
  --model-checkpoint "$CHECKPOINT"
