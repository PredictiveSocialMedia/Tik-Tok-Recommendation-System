#!/bin/bash
# One-time environment setup for IE HPC cluster
# Usage: bash hpc/setup_env.sh
set -euo pipefail

ENVNAME="tiktok-rec"
SCRATCH_PREFIX="/scratch/$USER"

echo "=== TikTok Rec System — HPC Environment Setup ==="
echo "User:    $(whoami)"
echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo ""

# 1. Create scratch workspace
mkdir -p "$SCRATCH_PREFIX"
echo "[1/5] Scratch workspace: $SCRATCH_PREFIX"

# 2. Install Miniforge if not present
if ! command -v conda &>/dev/null; then
    echo "[2/5] Installing Miniforge..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    conda init
    rm /tmp/miniforge.sh
    echo "  -> Miniforge installed. You may need to 'source ~/.bashrc' after this script."
else
    echo "[2/5] Conda already available: $(conda --version)"
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)" 2>/dev/null || true
fi

# 3. Create conda env on scratch (fast local SSD)
if conda env list | grep -q "$SCRATCH_PREFIX/$ENVNAME"; then
    echo "[3/5] Conda env already exists at $SCRATCH_PREFIX/$ENVNAME — activating"
else
    echo "[3/5] Creating conda env at $SCRATCH_PREFIX/$ENVNAME ..."
    conda create --prefix "$SCRATCH_PREFIX/$ENVNAME" python=3.11 -y
fi
conda activate "$SCRATCH_PREFIX/$ENVNAME"

# 4. Install dependencies
echo "[4/5] Installing Python dependencies..."
pip install --upgrade pip

# PyTorch with CUDA 12 for GPU-accelerated SBERT
pip install torch --index-url https://download.pytorch.org/whl/cu12

# Project dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"

# Swap faiss-cpu for faiss-gpu (GPU-accelerated similarity search)
pip uninstall -y faiss-cpu 2>/dev/null || true
pip install faiss-gpu-cu12 2>/dev/null || echo "  -> faiss-gpu not available, keeping faiss-cpu"

# 5. Verify
echo "[5/5] Verifying installation..."
python3 -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import sentence_transformers
print(f'SBERT:    {sentence_transformers.__version__}')

import lightgbm
print(f'LightGBM: {lightgbm.__version__}')

import faiss
print(f'FAISS:    OK')

import psycopg
print(f'psycopg:  OK (DB driver)')

print()
print('All dependencies verified.')
"

echo ""
echo "=== Setup complete ==="
echo "To activate in future sessions:"
echo "  conda activate $SCRATCH_PREFIX/$ENVNAME"
