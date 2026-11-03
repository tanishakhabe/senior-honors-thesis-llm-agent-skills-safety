#!/bin/bash
#SBATCH --account=skingsle
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=frontend_debug
#SBATCH --output=frontend_debug_%j.log

set -euxo pipefail
source "$HOME/transformer-lens-env/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
hostname
nvidia-smi
python -u -X faulthandler activations.py data/skill-copies \
  --model openai-community/gpt2 \
  --output-dir "results/debug-${SLURM_JOB_ID}"
