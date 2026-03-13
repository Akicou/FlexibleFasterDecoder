# FlexibleFasterDecoder

A modern, GPU-first CLI for training speculative decoding heads (multi-token prediction / MTP, à la Eagle3) on Mixture-of-Experts LLMs. FlexibleFasterDecoder (FFD) orchestrates dataset hydration, tokenization, head fine-tuning, artifact/report generation, and optional Hugging Face publishing in one workflow.

## Highlights

- **Real speculative heads** → `.safetensors` ready for vLLM-style speculators.
- **Deterministic data** → repeatable sampling across Ultrachat, CodeAlpaca-20k, Scale-SWE with `--seed`.
- **Artifacts that tell the story** → per-run README, stats, head configs/metrics, banner, and exact invocation.
- **GPU-first** → fails fast on CPU-only boxes; designed for CUDA builds of PyTorch.

## Quickstart

```bash
# 1) GPU-ready venv
python -m venv .venv
. .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) Install FFD
pip install -e .

# 3) Smoke test (tiny slice)
ffd run unsloth/gemma-3-270m \
  --dataset ultrachat \
  --mtp "[1]" \
  --max-samples-per-dataset 4 \
  --seed 42 \
  --epochs 1 \
  --batch-size 1

# 4) Bigger job + multiple heads + upload
ffd run unsloth/gemma-3-270m \
  --dataset ultrachat --dataset codealpaca-20k \
  --mtp "[1,2,3]" \
  --max-samples-per-dataset 256 \
  --seed 13 \
  --upload-to-hf --push-repo my-user/gemma-mtp
```

Per-head Rich progress bars show live steps/ETA. Outputs land under `runs/<timestamp>-<model>-<datasets>-<heads>/`.

## Requirements

- NVIDIA GPU with recent drivers.
- CUDA-enabled PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu121` or matching your CUDA).
- Internet access for Hugging Face pulls.

FFD exits early if no CUDA device is available.

## CLI Reference

```bash
ffd run BASE_MODEL \
  [--dataset {ultrachat,codealpaca-20k,scale-swe}]... \
  [--mtp "[1,2,3]" | --mtp 1 --mtp 2 ...] \
  [--max-samples-per-dataset 512] [--seed 42] \
  [--epochs 1] [--batch-size 1] [--learning-rate 5e-4] \
  [--output-dir runs] [--notes "experiment"] \
  [--upload-to-hf] [--push-repo org/repo] [--hf-token <token>] \
  [--dry-run]
```

| Option | Description |
| --- | --- |
| `BASE_MODEL` | HF checkpoint (e.g., `unsloth/gemma-3-270m`). |
| `--dataset` | Repeatable dataset keys; concatenated and shuffled. |
| `--mtp` | Heads list (repeated flags or JSON string). |
| `--max-samples-per-dataset` | Cap examples per dataset split (deterministic). |
| `--seed` | Seed for sampling/shuffling. |
| `--epochs`, `--batch-size`, `--learning-rate` | Training hyperparameters. |
| `--upload-to-hf`, `--push-repo`, `--hf-token` | Publishing controls. |
| `--notes`, `--dry-run` | Run metadata and upload skip. |

## Outputs

- `artifacts/head-*/spec_head-*.safetensors` – trained speculative heads.
- `artifacts/head-*/head_config.json` – head architecture metadata.
- `artifacts/head-*/metrics.json` – per-head training metrics.
- `README.md` + `stats.json` in each run directory – human/machine summaries.

## Project Layout

```
FlexibleFasterDecoder/
├── pyproject.toml
├── README.md
├── LICENSE
└── src/ffd/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── config.py
    ├── data.py
    ├── datasets.py
    ├── trainer.py
    ├── banner.py
    ├── hardware.py
    ├── hf_uploader.py
    ├── spec_head.py
    ├── report.py
    └── utils.py
```

## Environment

- `HF_TOKEN` – required when `--upload-to-hf` is set (or pass `--hf-token`).

## Notes

- Tokenization length is capped (512) to reduce GPU OOM risk; tune `--max-samples-per-dataset` and `--batch-size` to fit your GPU.
- For vLLM speculative serving, point `--speculative-model` at the emitted `.safetensors` and use the paired `head_config.json`.

Contributions via issues or PRs are welcome once the repository is initialized under git.
