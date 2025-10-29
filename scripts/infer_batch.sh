#!/bin/bash
#SBATCH --job-name=gat_infer_batch
#SBATCH --output=logs/infer_batch_%j.out
#SBATCH --error=logs/infer_batch_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --constraint=dgx

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: sbatch $0 FASTA_PATH CHECKPOINT_PATH OUTPUT_CSV [CONFIG_PATH]" >&2
  exit 1
fi

FASTA_PATH="$1"
CHECKPOINT="$2"
OUTPUT_CSV="$3"
CONFIG_PATH="${4:-configs/fungi.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

python -m pip install --user -e "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_ROOT"

python -m gat_pipeline.cli infer-fasta \
  --config "$CONFIG_PATH" \
  --fasta "$FASTA_PATH" \
  --model-checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_CSV"
