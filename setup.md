# TransformerLens + CMU ORCHARD Setup

## Goal

Use:

- **VS Code / Colab** → notebook interface
- **CMU ORCHARD** → compute
- **NVIDIA H100 80GB** → GPU
- **Python 3.11 virtual environment** → isolated Python packages
- **TransformerLens** → mechanistic interpretability

Current architecture:

```
My Mac
   │
   └── VS Code / Colab notebook
            │
            ▼
       Jupyter runtime
            │
            ▼
        CMU ORCHARD
            │
            ▼
       NVIDIA H100 80GB
            │
            ▼
    Python 3.11 virtual environment
            │
       ┌────┴─────┐
       │          │
    PyTorch   TransformerLens
```

---

## 1. Log into ORCHARD

From your Mac:

```
gcloud compute ssh orchard-login-001 \
  --zone "us-central1-c" \
  --project "cmu-gpu-cloud" \
  --tunnel-through-iap
```

You successfully reached:

```
tkhabe@orchard-login-001:~$
```

**Important:** `orchard-login-001` is a login node. Don't run GPU-heavy work directly there.

---

## 2. Check available Slurm partitions

```
sinfo
```

You found:

```
bootstrap
debug
preempt
flame
general
advanced
```

You also checked GPU resources:

```
sinfo -p bootstrap -o "%P %G %l %a %D %t %N"
```

and found:

```
bootstrap gpu:8
```

However, your account doesn't have access to `bootstrap`.

---

## 3. Find your Slurm account

You ran:

```
sacctmgr show user $USER withassoc format=User,Account,Partition,QOS
```

and found:

```
User     Account    Partition    QOS
tkhabe   skingsle   general      general_qos
tkhabe   skingsle   preempt      preempt_qos
```

Therefore your ORCHARD Slurm account is:

```
skingsle
```

You have access to:

```
general
preempt
```

but not `bootstrap`.

---

## 4. Request a GPU

`preempt` does **not** allow interactive `srun --pty` jobs.
This failed:

```
srun --account=skingsle --partition=preempt \
     --gres=gpu:1 --time=00:30:00 --pty bash
```

with:

```
Only `sbatch` jobs are allowed in partition 'preempt'.
```

So you use **batch jobs**:

```
sbatch your_script.sh
```

---

# 5. Verify that the H100 works

You created:

```
test_gpu.sh
```

with:

```
#!/bin/bash
#SBATCH --account=skingsle
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=test_gpu
#SBATCH --output=test_gpu_%j.out

nvidia-smi
```

Submit:

```
sbatch test_gpu.sh
```

Check:

```
squeue -u $USER
```

Once finished:

```
cat test_gpu_<JOB_ID>.out
```

This successfully showed:

```
NVIDIA H100 80GB HBM3
CUDA Version: 13.0
```

So **you have working H100 GPU access.**

---

# 6. Check Python

Initially:

```
which python
```

returned nothing.
But:

```
which python3
```

returned:

```
/usr/bin/python3
```

You checked available Python versions:

```
apt-cache policy python3.10 python3.11 python3.12
```

and found:

```
Python 3.11.15 installed
Python 3.12.3 installed
```

We chose **Python 3.11**.

---

# 7. Create the virtual environment

You created:

```
/usr/bin/python3.11 -m venv ~/transformer-lens-env
```

Then activated it:

```
source ~/transformer-lens-env/bin/activate
```

Your prompt became:

```
(transformer-lens-env) tkhabe@orchard-login-001:~$
```

You verified:

```
python --version
```

→

```
Python 3.11.15
```

and:

```
which python
```

→

```
/home/tkhabe/transformer-lens-env/bin/python
```

### Important

From now on, whenever you want to use this environment:

```
source ~/transformer-lens-env/bin/activate
```

You **do not need to create it again**.

---

# 8. Install PyTorch

Inside the virtual environment:

```
python -m pip install --upgrade pip
```

Then:

```
pip install torch
```

You ended up with:

```
PyTorch 2.13.0+cu130
```

---

# 9. Verify PyTorch + H100

You created a Slurm batch test that activated the environment and checked CUDA.
The important output was:

```
PyTorch: 2.13.0+cu130
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
CUDA version: 13.0
```

Therefore:

```
Python ✓
PyTorch ✓
CUDA ✓
H100 ✓
```

---

# 10. Install TransformerLens

You installed:

```
python -m pip install numpy transformer-lens circuitsvis
```

This also fixed the earlier warning:

```
No module named 'numpy'
```

You verified TransformerLens with:

```
python -m pip show transformer-lens
```

and got:

```
Name: transformer-lens
Version: 3.8.0
Location:
/home/tkhabe/transformer-lens-env/lib/python3.11/site-packages
```

One command you tried was:

```
transformer_lens.__version__
```

which produced:

```
AttributeError:
module 'transformer_lens' has no attribute '__version__'
```

**This is not an installation problem.** TransformerLens simply doesn't expose its version that way. Use:

```
python -m pip show transformer-lens
```

instead.

---

# 11. Test TransformerLens + GPT-2

You created:

```
test_transformerlens.py
```

with:

```
import torch
from transformer_lens import HookedTransformer

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading GPT-2 Small...")

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    device="cuda"
)

print("Model loaded!")
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

text = "The capital of France is"

logits = model(text)

print("\nLogits shape:", logits.shape)

probs = logits[0, -1].softmax(dim=-1)
top_probs, top_tokens = probs.topk(10)

print("\nTop predictions:")
for prob, token in zip(top_probs, top_tokens):
    print(f"{model.to_string(token)!r}: {prob.item():.4f}")
```

You submitted it through Slurm with:

```
sbatch test_transformerlens.sh
```

The result confirmed:

```
PyTorch: 2.13.0+cu130
CUDA available: True
GPU: NVIDIA H100 80GB HBM3

Loading GPT-2 Small...
...
Model loaded!
Number of parameters: 163049041
```

and:

```
Logits shape: torch.Size([1, 6, 50257])
```

So:

> **TransformerLens successfully loaded GPT-2 Small and ran it on your H100.**

---

# 12. The deprecation warning

You got:

```
HookedTransformer.from_pretrained is deprecated.
Use TransformerBridge.boot_transformers(...)
```

This is **not an error**.
For now, we're deliberately using:

```
from transformer_lens import HookedTransformer
```

because the official Main Demo you're working through uses this API.
Later we can migrate to the newer `TransformerBridge` API.

---

# Where you are now

Your entire compute environment is working:

```
                ORCHARD
                   │
                   ▼
          Python 3.11.15
                   │
                   ▼
       transformer-lens-env
                   │
          ┌────────┴────────┐
          ▼                 ▼
       PyTorch         TransformerLens
          │                 │
          └────────┬────────┘
                   ▼
              CUDA 13.0
                   │
                   ▼
           NVIDIA H100 80GB
                   │
                   ▼
              GPT-2 Small
                   │
                   ▼
               logits
```

## What you have NOT set up yet

The remaining piece is:

```
Colab notebook
      │
      ▼
Jupyter server on ORCHARD
      │
      ▼
your transformer-lens-env
      │
      ▼
H100
```

That's the **next step**.
Once that's working, you won't need to keep creating `test_*.sh` files. You'll be able to open the official **Main Demo** in Colab and interactively run cells against your ORCHARD GPU.

---

## Useful commands to remember

### Activate your environment

```
source ~/transformer-lens-env/bin/activate
```

### Deactivate it

```
deactivate
```

### Check Python

```
python --version
which python
```

### Check installed TransformerLens

```
python -m pip show transformer-lens
```

### Check your Slurm jobs

```
squeue -u $USER
```

### Submit a GPU job

```
sbatch <script.sh>
```

### Check a completed job's output

```
cat <output_file>
```

### Check available partitions

```
sinfo
```

### Check your Slurm account

```
sacctmgr show user $USER withassoc format=User,Account,Partition,QOS
```

### Check GPU from a GPU job

```
nvidia-smi
```

---

## The key lesson about your Python confusion

You **do not need to worry about the Python version on your Mac** for this project.
Your relevant Python is:

```
ORCHARD
└── /home/tkhabe/transformer-lens-env/
       └── Python 3.11.15
```

That's where your ML packages live.
So when you're working on TransformerLens, the mental model is simply:

> **"My code runs on ORCHARD's Python 3.11 environment, using its H100."**

The Mac/VS Code/Colab is just how you interact with that environment.

# 13. Starting the project next time

When I come back to this project:

### Terminal 1 — connect to ORCHARD

```
gcloud compute ssh orchard-login-001 \
  --zone us-central1-c \
  --project cmu-gpu-cloud \
  --tunnel-through-iap
```

### On ORCHARD

Activate the environment:

```
source ~/transformer-lens-env/bin/activate
```

### Start a GPU Jupyter job

```
sbatch start_jupyter.sh
```

Then:

```
squeue -u $USER
```

Find the allocated node.

### On my Mac — create the tunnel

Use the node Slurm assigned:

```
gcloud compute ssh orchard-login-001 \
  --zone us-central1-c \
  --project cmu-gpu-cloud \
  --tunnel-through-iap \
  -- -L 8888:NODE_NAME:8888
```

### VS Code

Connect the notebook to:

```
http://localhost:8888/?token=...
```

Then verify:

```
import sys
print(sys.executable)
```

and:

```
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

If it says:

```
/home/tkhabe/transformer-lens-env/bin/python
True
NVIDIA H100 80GB HBM3
```

**I'm ready to work.**

---

# 14. Where we stopped

Infrastructure is **done**.
We have **not yet started the actual TransformerLens experiments**.
The next project step is to create:

```
transformer-lens/
└── notebooks/
    └── 01_first_transformerlens_experiment.ipynb
```

and learn TransformerLens incrementally rather than trying to understand the entire large `Main_Demo.ipynb` at once.
The progression should be roughly:

```
01  Load GPT-2
 ↓
02  Generate predictions
 ↓
03  Understand tokens / logits
 ↓
04  run_with_cache()
 ↓
05  Explore the activation cache
 ↓
06  Residual stream
 ↓
07  Attention heads
 ↓
08  Attention patterns
 ↓
09  MLPs
 ↓
10  Hooks
 ↓
11  Activation patching
 ↓
12  Actual mechanistic interpretability experiments
```

**Current status: infrastructure complete; ready to start Experiment 1.**