#!/bin/bash
#SBATCH --account=skingsle
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=skill_activations
#SBATCH --output=skill_activations_%j.out

set -euo pipefail

source "$HOME/transformer-lens-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -eq 0 ]; then
    echo "Usage: sbatch run_activations.slurm SKILLS_PATH [SKILLS_PATH ...] [options]" >&2
    exit 2
fi

python activations.py "$@"
