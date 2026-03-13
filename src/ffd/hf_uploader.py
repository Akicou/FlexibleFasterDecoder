"""Hugging Face publishing helpers."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi
from rich.console import Console

from .config import RunConfig


class HFPublisher:
    def __init__(self, token: str, *, console: Console | None = None) -> None:
        self.token = token
        self.api = HfApi(token=token)
        self.console = console or Console()

    def suggest_repo_id(self, spec: RunConfig) -> str:
        try:
            who = self.api.whoami(token=self.token)
            handle = who.get("name") or who.get("email", "ffd-user").split("@")[0]
        except Exception:  # pragma: no cover - network failure resilience
            handle = "ffd-user"
        return f"{handle}/{spec.slug}"

    def publish(
        self,
        *,
        repo_id: str,
        artifacts_dir: Path,
        banner_path: Path,
        report_path: Path,
        stats_path: Path,
    ) -> None:
        self.console.print(
            f"[cyan]Pushing artifacts to https://huggingface.co/{repo_id}[/cyan]"
        )
        self.api.create_repo(repo_id=repo_id, exist_ok=True, token=self.token)
        self.api.upload_folder(
            folder_path=str(artifacts_dir),
            repo_id=repo_id,
            token=self.token,
            commit_message="Add FlexibleFasterDecoder artifacts",
        )
        self.api.upload_file(
            path_or_fileobj=str(banner_path),
            repo_id=repo_id,
            path_in_repo="assets/banner.png",
            token=self.token,
            commit_message="Update banner",
        )
        self.api.upload_file(
            path_or_fileobj=str(report_path),
            repo_id=repo_id,
            path_in_repo="README.md",
            token=self.token,
            commit_message="Update report",
        )
        self.api.upload_file(
            path_or_fileobj=str(stats_path),
            repo_id=repo_id,
            path_in_repo="stats.json",
            token=self.token,
            commit_message="Update stats",
        )


def validate_token(token: str | None, *, required: bool) -> str:
    if not token:
        if required:
            raise ValueError(
                "--upload-to-hf requires a token. Provide --hf-token or set HF_TOKEN in the environment."
            )
        raise ValueError("Missing Hugging Face token")
    return token
