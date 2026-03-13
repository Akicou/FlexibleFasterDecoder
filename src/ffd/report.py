"""Generate human-readable and machine-readable run reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from dataclasses import asdict, dataclass

from .config import RunConfig
from .hardware import HardwareSnapshot


class ReportWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def write(
        self,
        spec: RunConfig,
        metrics: Iterable["HeadMetrics"],
        hardware: HardwareSnapshot,
    ) -> tuple[Path, Path]:
        markdown_path = self.run_dir / "README.md"
        json_path = self.run_dir / "stats.json"

        metric_dicts = [asdict(metric) for metric in metrics]
        dataset_lines = [
            f"- {card.name} ({card.hf_repo})" for card in spec.dataset_cards
        ]

        head_list = [str(m["head"]) for m in metric_dicts]
        invocation = spec.invocation or "ffd run ..."
        example_path = "artifacts/head-<n>/spec_head-<n>.safetensors"
        if metric_dicts:
            first_head = metric_dicts[0]["head"]
            example_path = (
                self.run_dir
                / "artifacts"
                / f"head-{first_head}"
                / f"spec_head-{first_head}.safetensors"
            ).as_posix()

        markdown_lines = [
            "# FlexibleFasterDecoder Run",
            "",
            f"**Base model:** {spec.base_model}",
            f"**Datasets:**",
            *dataset_lines,
            "",
            f"**Heads:** {', '.join(head_list)}",
            f"**Seed:** {spec.seed}",
            f"**Notes:** {spec.notes or '—'}",
            "",
            "## Metrics",
            "| Head | Loss | Tokens/s | Steps | Wall Time (min) | Dataset Tokens | Samples |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for metric in metric_dicts:
            markdown_lines.append(
                "| {head} | {final_loss:.4f} | {tokens_per_second:.0f} | {steps} | "
                "{wall_time_minutes:.1f} | {dataset_tokens:,} | {samples:,} |".format(
                    **metric
                )
            )
        markdown_lines.extend(
            [
                "",
                "## Usage",
                "```bash",
                invocation,
                "```",
                f"Each head artifact is stored under `{example_path}`.",
                "Example vLLM invocation:",
                "```bash",
                "python -m vllm.entrypoints.api_server \\",
                f"  --model {spec.base_model} \\",
                f"  --speculative-model {example_path}",
                "```",
                "",
                "## Hardware",
                f"- CPU: {hardware.cpu} ({hardware.cores} cores)",
                f"- RAM: {hardware.memory_gb} GB",
                f"- GPU: {hardware.gpu or 'n/a'}",
                f"- GPU Memory: {hardware.gpu_memory_gb or 'n/a'} GB",
                f"- OS: {hardware.os}",
            ]
        )

        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        json_payload = {
            "model": spec.base_model,
            "datasets": [card.key for card in spec.dataset_cards],
            "heads": metric_dicts,
            "hardware": hardware.to_dict(),
            "notes": spec.notes,
            "seed": spec.seed,
        }
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        return markdown_path, json_path


@dataclass(slots=True)
class HeadMetrics:
    head: int
    final_loss: float
    steps: int
    wall_time_minutes: float
    tokens_per_second: float
    dataset_tokens: int
    samples: int
