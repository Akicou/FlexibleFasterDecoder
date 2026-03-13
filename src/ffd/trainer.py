"""Training orchestration for Flexible Faster Decoder."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .banner import BannerFactory
from .config import RunConfig
from .data import build_tokenized_dataset, load_dataset_cards
from .hardware import HardwareSnapshot, gather_hardware_snapshot
from .report import HeadMetrics, ReportWriter
from .spec_head import HeadConfig, SpeculativeHead
from .utils import current_run_id, ensure_dir


@dataclass(slots=True)
class TrainingResult:
    run_dir: Path
    artifacts_dir: Path
    metrics: List[HeadMetrics]
    report_path: Path
    stats_path: Path
    banner_path: Path
    hardware: HardwareSnapshot


class TrainingOrchestrator:
    """High-level pipeline invoked by the CLI."""

    def __init__(self, config: RunConfig, *, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self) -> TrainingResult:
        self._ensure_gpu()
        run_dir = ensure_dir(
            self.config.output_dir / f"{current_run_id()}-{self.config.slug}"
        )
        artifacts_dir = ensure_dir(run_dir / "artifacts")
        cards = load_dataset_cards(self.config.datasets)

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(self.config.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        base_model.to(self.device)
        base_model.eval()

        dataset = build_tokenized_dataset(
            cards=cards,
            tokenizer=tokenizer,
            max_samples_per_dataset=self.config.max_samples_per_dataset,
            seed=self.config.seed,
        )
        dataloader = self._create_dataloader(dataset)
        seq_length = dataset[0]["input_ids"].shape[-1]
        dataset_tokens = len(dataset) * seq_length

        metrics: list[HeadMetrics] = []
        config_payload = {
            "base_model": self.config.base_model,
            "datasets": self.config.datasets,
            "heads": self.config.mtp,
            "notes": self.config.notes,
            "seed": self.config.seed,
            "max_samples_per_dataset": self.config.max_samples_per_dataset,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "invocation": self.config.invocation,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2), encoding="utf-8"
        )

        with Progress(console=self.console) as progress:
            task = progress.add_task("Training heads", total=len(self.config.mtp))
            for head in self.config.mtp:
                head_dir = ensure_dir(artifacts_dir / f"head-{head}")
                head_config = HeadConfig(
                    head=head,
                    hidden_size=config.hidden_size,
                    vocab_size=config.vocab_size,
                    base_model=self.config.base_model,
                    tokenizer=tokenizer.name_or_path,
                    pad_token_id=tokenizer.pad_token_id,
                    max_position_embeddings=getattr(
                        config, "max_position_embeddings", None
                    ),
                )
                metrics.append(
                    self._train_head(
                        head_config=head_config,
                        base_model=base_model,
                        dataloader=dataloader,
                        dataset_tokens=dataset_tokens,
                        head_dir=head_dir,
                    )
                )
                progress.advance(task)

        hardware = gather_hardware_snapshot()
        report_writer = ReportWriter(run_dir)
        report_path, stats_path = report_writer.write(self.config, metrics, hardware)
        banner_path = BannerFactory().build(
            self.config, [asdict(metric) for metric in metrics], artifacts_dir
        )

        return TrainingResult(
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            metrics=metrics,
            report_path=report_path,
            stats_path=stats_path,
            banner_path=banner_path,
            hardware=hardware,
        )

    def _create_dataloader(self, dataset) -> DataLoader:  # type: ignore[no-untyped-def]
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def _train_head(
        self,
        *,
        head_config: HeadConfig,
        base_model: AutoModelForCausalLM,
        dataloader: DataLoader,
        dataset_tokens: int,
        head_dir: Path,
    ) -> HeadMetrics:
        total_steps_est = max(len(dataloader) * self.config.epochs, 1)
        head_model = SpeculativeHead(head_config).to(self.device)
        optimizer = torch.optim.AdamW(
            head_model.parameters(), lr=self.config.learning_rate
        )
        total_loss = 0.0
        total_steps = 0
        tokens_processed = 0
        start_time = time.perf_counter()

        with Progress(
            TextColumn(f"Head {head_config.head} Training"),
            BarColumn(),
            "{task.completed}/{task.total}",
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("train", total=total_steps_est)

            for epoch in range(self.config.epochs):
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    with torch.no_grad():
                        outputs = base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )
                        hidden_states = outputs.hidden_states[-1].float()

                    head_model.train()
                    optimizer.zero_grad()
                    loss = head_model(hidden_states, input_ids)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_steps += 1
                    tokens_processed += input_ids.numel()
                    progress.advance(task_id, 1)

        duration = max(time.perf_counter() - start_time, 1e-9)
        tps = tokens_processed / duration
        wall_time_minutes = duration / 60
        average_loss = total_loss / max(total_steps, 1)

        weights_path = head_dir / f"spec_head-{head_config.head}.safetensors"
        head_model.save(str(weights_path))
        (head_dir / "head_config.json").write_text(
            json.dumps(head_config.to_dict(), indent=2), encoding="utf-8"
        )
        (head_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "loss": average_loss,
                    "steps": total_steps,
                    "tokens_per_second": tps,
                    "tokens_processed": tokens_processed,
                    "wall_time_minutes": wall_time_minutes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return HeadMetrics(
            head=head_config.head,
            final_loss=average_loss,
            steps=total_steps,
            wall_time_minutes=wall_time_minutes,
            tokens_per_second=tps,
            dataset_tokens=dataset_tokens,
            samples=len(dataloader.dataset),  # type: ignore[arg-type]
        )

    def _ensure_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FlexibleFasterDecoder requires a CUDA-capable GPU. Install torch with CUDA support."
            )
