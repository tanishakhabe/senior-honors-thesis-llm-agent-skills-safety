"""Capture TransformerLens residual-stream activations for SKILL.md files."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from transformer_lens.model_bridge import TransformerBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a model over Skills files.")
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="SKILL.md files, or directories recursively containing them",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Tokens per chunk (default: the model context length)",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32",
        help="Model and saved activation dtype",
    )
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Run without CUDA; by default a missing GPU is an error",
    )
    return parser.parse_args()


def find_skill_files(inputs: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate.resolve() for candidate in path.rglob("SKILL.md"))
            files.update(candidate.resolve() for candidate in path.rglob("*.SKILL.md"))
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
    if not files:
        raise FileNotFoundError("No SKILL.md or *.SKILL.md files found in the inputs")
    return sorted(files)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def output_stem(path: Path) -> str:
    safe_name = "_".join(part.replace(" ", "_") for part in path.parts[-3:])
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:10]
    return f"{safe_name}_{digest}"


def main() -> None:
    args = parse_args()
    skill_files = find_skill_files(args.inputs)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Submit through Slurm on a GPU node, or pass "
            "--allow-cpu intentionally."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"TransformerLens: {package_version('transformer-lens')}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {args.model} ({args.dtype})")

    model = TransformerBridge.boot_transformers(
        args.model, device=device, dtype=dtype
    )
    model.eval()
    model_context = int(model.cfg.n_ctx)
    chunk_size = args.max_tokens or model_context
    if chunk_size < 1 or chunk_size > model_context:
        raise ValueError(
            f"--max-tokens must be between 1 and the model context length "
            f"({model_context})"
        )

    def is_residual_hook(name: str) -> bool:
        return (
            name.endswith("hook_resid_pre")
            or name.endswith("hook_resid_post")
        )

    manifest: list[dict[str, object]] = []
    for file_index, skill_path in enumerate(skill_files, start=1):
        text = skill_path.read_text(encoding="utf-8")
        input_ids = model.to_tokens(text, truncate=False).squeeze(0).cpu()
        chunks = input_ids.split(chunk_size)
        stem = output_stem(skill_path)
        print(
            f"[{file_index}/{len(skill_files)}] {skill_path}: "
            f"{len(input_ids)} tokens in {len(chunks)} chunk(s)"
        )

        chunk_records: list[dict[str, object]] = []
        for chunk_index, chunk in enumerate(chunks):
            chunk_input = chunk.unsqueeze(0).to(device)
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    chunk_input, names_filter=is_residual_hook, device="cpu"
                )
            activations = {
                name: tensor.detach().to(device="cpu", dtype=dtype)
                for name, tensor in cache.items()
            }
            if not activations:
                raise RuntimeError("No residual-stream hooks were captured")

            output_file = args.output_dir / f"{stem}.chunk-{chunk_index:04d}.pt"
            torch.save(
                {
                    "model_name": args.model,
                    "skill_file": str(skill_path),
                    "chunk_index": chunk_index,
                    "token_offset": chunk_index * chunk_size,
                    "input_ids": chunk,
                    "tokens": model.to_str_tokens(chunk),
                    "activations": activations,
                },
                output_file,
            )
            chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "token_offset": chunk_index * chunk_size,
                    "token_count": len(chunk),
                    "output_file": str(output_file),
                    "activation_names": sorted(activations),
                }
            )
            print(f"  saved {output_file}")
            del cache, activations

        manifest.append(
            {
                "skill_file": str(skill_path),
                "character_count": len(text),
                "token_count": len(input_ids),
                "chunks": chunk_records,
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_name": args.model,
                "device": device,
                "dtype": args.dtype,
                "model_context": model_context,
                "chunk_size": chunk_size,
                "transformer_lens_version": package_version("transformer-lens"),
                "files": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
