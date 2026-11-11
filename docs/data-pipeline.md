# Current data pipeline

As of November 4, 2026. The solid path below describes the completed frontend
pilot consumed by `notebooks/frontend_activation_comparison.ipynb`, checked
against its saved `debug-164876/manifest.json`. Dashed arrows describe an
implemented optional attention workflow, not outputs of that run.

```mermaid
flowchart TD
    subgraph inputs["1 · Local research inputs — skill text is read, never executed"]
        B["Benign frontend-design/SKILL.md<br/>1,751 tokens"]
        A["Adversarial copy + appended response override<br/>1,792 tokens"]
    end

    subgraph extraction["2 · ORCHARD GPU extraction"]
        S["run_activations.slurm → activations.py<br/>Slurm GPU job · Python virtual environment"]
        T["Discover files → read UTF-8 → GPT-2 tokenizer<br/>No truncation"]
        C["Split into independent contexts · maximum 1,024 tokens<br/>Benign: 1,024 + 727 · Adversarial: 1,024 + 768<br/>Each chunk starts fresh; no context passes between chunks"]
        M["TransformerBridge + openai-community/gpt2<br/>CUDA · float32 · inference mode"]
        R["Capture residual stream before and after each layer<br/>12 layers · cache tensors on CPU"]
        AT["Optional: --capture-attention<br/>Capture hook_pattern; validate shapes,<br/>finite nonnegative values and row sums"]
        S --> T --> C --> M --> R
        M -.-> AT
    end

    subgraph storage["3 · Saved artifacts"]
        P["4 .pt files · one per input chunk<br/>Token IDs + readable tokens + activations<br/>Model, source path, chunk index and token offset"]
        J["manifest.json<br/>Input lengths, chunk boundaries, artifact paths,<br/>model, dtype, device and package versions<br/>Completed run: capture_attention = false"]
        L["Pilot artifacts available locally<br/>experiments/orchard-pilot-20261103/<br/>results/debug-164876/"]
        R --> P
        R --> J
        P --> L
        J --> L
    end

    subgraph analysis["4 · CPU analysis — no model download or GPU required"]
        N["frontend_activation_comparison.ipynb<br/>Read manifest + load tensors with weights_only=True<br/>Check finite activations and matching chunk boundaries"]
        Q["Input-length chart<br/>Shared-prefix activation sanity check<br/>Inspect tokens at divergence<br/>Final-token cosine and relative L2 distances by layer"]
        F["Current result: descriptive comparison of one pair<br/>Length, wording and token position are confounds<br/>No behavioral attack-success or generalization evaluation"]
        N --> Q --> F
    end

    O["Separate attention-enabled extraction run<br/>frontend_skill_pilot.ipynb<br/>Saves residual_endpoint_norms.png<br/>and final_token_attention.png"]

    B --> S
    A --> S
    L --> N
    AT -.-> O

    classDef optional fill:#fff7e6,stroke:#b7791f,stroke-dasharray:5 5;
    class AT,O optional;
```

The comparison notebook displays its charts inline. The optional attention
notebook saves PNGs in `results/frontend-pilot/with-attention/` after a separate
extraction run. Raw inputs and generated results are ignored by Git and must be
transferred separately when moving between ORCHARD and the local machine.

Implementation references:

- [Extraction entry point](../activations.py)
- [Slurm launcher](../run_activations.slurm)
- [Current comparison notebook](../notebooks/frontend_activation_comparison.ipynb)
- [Optional attention analysis](../notebooks/frontend_skill_pilot.ipynb)
- [Pilot input and run instructions](../experiments/frontend_pilot.md)

`setup.md` contains historical setup milestones; the extraction code, saved run
manifest and analysis notebook establish the pipeline state shown here. Dataset
assembly is currently manual; there is no automated dataset-building stage in
the inspected pipeline.
