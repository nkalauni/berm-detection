#!/bin/bash
# SLURM batch script for training on University of Arizona HPC (Puma/Ocelote).
#
# This repo was developed on a group GPU server (no scheduler), so the items
# marked TODO below need to be filled in with your advisor/HPC-specific
# details before this will run -- I don't have your PI group's account name,
# preferred partition, or confirmation of what's pre-installed on the compute
# nodes there.
#
# Submit with:
#   sbatch scripts/train_uarizona_hpc.slurm configs/train_altarvalley.yaml
#
# UA HPC docs: https://public.confluence.arizona.edu/display/UAHPC

#SBATCH --job-name=berm-train
#SBATCH --account=TODO_your_pi_group          # e.g. `va-group` -- ask your advisor
#SBATCH --partition=gpu_standard               # gpu_standard | gpu_windfall (free, preemptible, lower priority)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00                        # 50 epochs on the full Altar Valley raster took well under this on a single GPU; pad generously the first time you run it since HPC node speed may differ from ocotillo
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# --- environment ---------------------------------------------------------
# TODO: confirm module names available on the cluster (`module spider python`,
# `module spider cuda`) -- these are guesses at typical UA HPC module names,
# not verified against the actual system.
module load python/3.11 2>/dev/null || true
module load cuda 2>/dev/null || true

# This repo uses uv (https://docs.astral.sh/uv/) for dependency management.
# If uv isn't available as a module, install it once per account:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."   # repo root
uv sync                   # creates/updates .venv -- fast no-op if already synced

CONFIG="${1:-configs/train_altarvalley.yaml}"
echo "Training with config: $CONFIG"
echo "GPUs visible to this job:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

uv run python scripts/train.py --config "$CONFIG" --device cuda

# --- after training: full evaluation + diagnostics -----------------------
CHECKPOINT_DIR=$(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output']['checkpoint_dir'])")
uv run python scripts/evaluate.py --checkpoint-dir "$CHECKPOINT_DIR" --device cuda
uv run python scripts/evaluate_centerline.py --checkpoint-dir "$CHECKPOINT_DIR"
uv run python scripts/plot_diagnostics.py --checkpoint-dir "$CHECKPOINT_DIR" --device cuda
