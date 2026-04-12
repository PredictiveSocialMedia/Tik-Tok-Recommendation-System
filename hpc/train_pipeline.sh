#!/bin/bash
#SBATCH --job-name=tiktok-pipeline
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=train_%x_%j.log

# === TikTok Recommendation System — Full Training Pipeline ===
set -euo pipefail

echo "=== Training Job ==="
echo "Hostname:   $(hostname)"
echo "Date:       $(date)"
echo "User:       $(whoami)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "Working dir: $(pwd)"
echo ""

# Show GPU info (non-fatal if running on CPU partition)
nvidia-smi || echo "No GPU available — running in CPU mode"

# Activate conda environment
source "$HOME/miniforge3/bin/activate" tiktok-rec

# Verify GPU is visible to PyTorch (informational only)
python3 -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Running CPU-only — NumPy vectorization still active')
"

# Set environment
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
# DATABASE_URL must be set before submitting: export DATABASE_URL="postgresql://..."
if [ -z "${DATABASE_URL:-}" ]; then
    echo "ERROR: DATABASE_URL not set. Export it before running sbatch."
    exit 1
fi

# Run the full training pipeline
echo ""
echo "=== Starting training pipeline at $(date) ==="
python -u scripts/train_full_pipeline.py --db-url "$DATABASE_URL"
TRAIN_EXIT=$?

echo ""
echo "=== Training finished at $(date) with exit code $TRAIN_EXIT ==="

# Copy artifacts to home (persistent storage, survives /scratch purge)
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Copying artifacts to ~/tiktok-artifacts/ ..."
    mkdir -p ~/tiktok-artifacts
    cp -r artifacts/recommender/latest ~/tiktok-artifacts/recommender_latest_$(date +%Y%m%d_%H%M%S)
    cp -r artifacts/datamart/ ~/tiktok-artifacts/datamart/ 2>/dev/null || true
    echo "Artifacts saved to ~/tiktok-artifacts/"

    # Print key metrics
    echo ""
    echo "=== Key Metrics ==="
    if [ -f artifacts/recommender/latest/metrics/objective_metrics.json ]; then
        python3 -c "
import json
with open('artifacts/recommender/latest/metrics/objective_metrics.json') as f:
    metrics = json.load(f)
for obj, vals in metrics.items():
    ndcg = vals.get('ndcg@20', vals.get('ndcg_at_20', 'N/A'))
    print(f'  {obj}: NDCG@20 = {ndcg}')
"
    else
        echo "  (metrics file not found)"
    fi
else
    echo "Training FAILED — check log for errors"
fi

exit $TRAIN_EXIT
