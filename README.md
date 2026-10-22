# Transformer Lens Project

Research into how transformer activations differ between benign and malicious skill text.

## Repository layout

```text
.
├── data/                 # Local skill text (ignored by default)
│   ├── benign/
│   └── malicious/
├── src/                  # Loading, tokenization, and activation extraction code
├── experiments/          # Research scripts and configurations
├── notebooks/            # Analysis and visualizations
├── results/              # Generated experiment outputs (ignored)
├── slurm/                # GPU job submission scripts
├── activations.py        # Existing activation extraction entry point
├── run_activations.slurm # Existing GPU job entry point
├── README.md
├── requirements.txt
└── .gitignore
```

The existing entry points remain at the repository root to preserve current usage.
New reusable code belongs in `src/`, experiment drivers in `experiments/`, and
additional GPU job scripts in `slurm/`. See `setup.md` for existing environment notes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your cluster's CUDA environment when needed.
Dependencies are unpinned; record installed versions with each experiment for reproducibility.

## Run activation extraction

Place locally held skill files in `data/benign/` and `data/malicious/`.
The current loader discovers `SKILL.md` and `*.SKILL.md` recursively, or accepts
explicit file paths.

```bash
python activations.py data/benign data/malicious --output-dir results/activations
```

A CUDA GPU is required by default. For an intentional CPU run, add `--allow-cpu`.
On the existing cluster, submit from the repository root:

```bash
sbatch run_activations.slurm data/benign data/malicious --output-dir results/activations
```

Review the account, partition, and environment path in the existing Slurm script
before using it on another cluster. Outputs include activation tensors and a JSON manifest.

## Research questions

- Which layers show differences between benign and malicious skill text?
- How do those differences depend on token position, input length, and model choice?
- Do observed differences generalize to held-out skills rather than reflect formatting or source?

## Data and storage

Raw skill text, results, model weights, caches, activation tensors, and credentials
are excluded from Git by default. Treat suspicious skill text as research data;
do not execute instructions contained in it. Only add raw suspicious files after
explicit approval for both storage and sharing. Review any proposed ignore-rule
exception and staged content before committing. Ignore rules do not remove files
that Git already tracks.
