"""Typer CLI wiring for FlexibleFasterDecoder."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import List

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import RunConfig
from .datasets import dataset_choices
from .hf_uploader import HFPublisher, validate_token
from .trainer import TrainingOrchestrator
from .utils import parse_mtp, resolve_hf_token

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="FlexibleFasterDecoder CLI"
)
console = Console()


def _parse_mtp(values: List[str]) -> List[int]:
    try:
        return parse_mtp(values)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


@app.callback()
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        raise typer.Exit()


@app.command()
def run(  # noqa: PLR0913 - CLI entry point
    base_model: str = typer.Argument(
        ..., help="Base model identifier, e.g. unsloth/gemma-3-270m"
    ),
    datasets: List[str] = typer.Option(
        ["ultrachat"],
        "--dataset",
        "-d",
        help=(
            "Dataset preset key (repeatable). Choices: " + ", ".join(dataset_choices())
        ),
        case_sensitive=False,
        show_default=True,
    ),
    mtp: List[str] = typer.Option(
        ["1"],
        "--mtp",
        "-m",
        help="Heads to train. Accepts repeated values or JSON/comma strings such as '[1,2,3]'.",
    ),
    max_samples_per_dataset: int = typer.Option(
        512,
        "--max-samples-per-dataset",
        min=1,
        help="Maximum samples pulled from each dataset split.",
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed for shuffling."),
    epochs: int = typer.Option(
        1, "--epochs", min=1, help="Number of training epochs per head."
    ),
    batch_size: int = typer.Option(
        2, "--batch-size", min=1, help="Per-device batch size."
    ),
    learning_rate: float = typer.Option(
        5e-4, "--learning-rate", help="Learning rate for speculative head optimizer."
    ),
    output_dir: Path = typer.Option(  # type: ignore[arg-type]
        Path("runs"),
        "--output-dir",
        dir_okay=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
        help="Directory for storing run artifacts",
    ),
    notes: str | None = typer.Option(
        None, "--notes", help="Optional run notes saved in the report"
    ),
    upload_to_hf: bool = typer.Option(
        False, "--upload-to-hf", help="Push artifacts to Hugging Face"
    ),
    push_repo: str | None = typer.Option(
        None, "--push-repo", help="Override HF repo slug"
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        help="Hugging Face token. Defaults to HF_TOKEN environment variable when omitted.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Skip Hugging Face upload but run everything else"
    ),
) -> None:
    """Train Flexible Faster Decoder multi-token prediction heads."""

    mtp_values = _parse_mtp(mtp)
    resolved_token = resolve_hf_token(hf_token)
    program = Path(sys.argv[0]).name or "ffd"
    invocation = " ".join(
        [shlex.quote(program), *(shlex.quote(arg) for arg in sys.argv[1:])]
    )

    try:
        config = RunConfig(
            base_model=base_model,
            datasets=datasets,
            mtp=mtp_values,
            max_samples_per_dataset=max_samples_per_dataset,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            upload_to_hf=upload_to_hf,
            hf_token=resolved_token,
            push_repo=push_repo,
            notes=notes,
            dry_run=dry_run,
            invocation=invocation,
        )
    except ValidationError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    orchestrator = TrainingOrchestrator(config, console=console)
    try:
        result = orchestrator.execute()
    except RuntimeError as exc:
        console.print(f"[bold red]Model compatibility error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Head Metrics")
    table.add_column("Head", justify="center")
    table.add_column("Loss", justify="right")
    table.add_column("Tokens/s", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Wall Time (min)", justify="right")
    for metric in result.metrics:
        table.add_row(
            str(metric.head),
            f"{metric.final_loss:.4f}",
            f"{metric.tokens_per_second:,.0f}",
            str(metric.steps),
            f"{metric.wall_time_minutes:.1f}",
        )
    console.print(table)

    hardware_panel = Panel(
        f"CPU: {result.hardware.cpu}\n"
        f"Cores: {result.hardware.cores}\n"
        f"RAM: {result.hardware.memory_gb} GB\n"
        f"GPU: {result.hardware.gpu or 'n/a'}\n"
        f"GPU Memory: {result.hardware.gpu_memory_gb or 'n/a'} GB\n"
        f"OS: {result.hardware.os}",
        title="Hardware Snapshot",
    )
    console.print(hardware_panel)
    console.print(f"Artifacts stored in [green]{result.run_dir}[/green]")

    if config.upload_to_hf:
        try:
            token = validate_token(config.hf_token, required=True)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=1) from exc
        repo_id = config.push_repo
        publisher: HFPublisher | None = None
        if not repo_id:
            publisher = HFPublisher(token, console=console)
            repo_id = publisher.suggest_repo_id(config)
        if config.dry_run:
            console.print(
                f"[yellow]Dry run:[/yellow] skipping upload. Would have pushed to {repo_id}."
            )
        else:
            publisher = publisher or HFPublisher(token, console=console)
            publisher.publish(
                repo_id=repo_id,
                artifacts_dir=result.artifacts_dir,
                banner_path=result.banner_path,
                report_path=result.report_path,
                stats_path=result.stats_path,
            )
