"""Pydantic-backed run configuration objects."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator

from .datasets import DATASETS, DatasetCard
from .utils import slugify


class RunConfig(BaseModel):
    """Configuration derived from CLI arguments."""

    base_model: str = Field(
        ..., description="Base HF checkpoint, e.g. unsloth/gemma-3-270m"
    )
    datasets: List[str] = Field(
        default_factory=lambda: ["ultrachat"], description="Dataset keys"
    )
    mtp: List[int] = Field(..., description="Distinct head counts to train (1-3)")
    max_samples_per_dataset: int = Field(512, ge=1)
    seed: int = Field(42)
    epochs: int = Field(1, ge=1)
    batch_size: int = Field(2, ge=1)
    learning_rate: float = Field(5e-4, gt=0)
    output_dir: Path = Field(default_factory=lambda: Path("runs"))
    upload_to_hf: bool = False
    hf_token: str | None = None
    push_repo: str | None = None
    notes: str | None = None
    dry_run: bool = False
    invocation: str = ""

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("datasets")
    @classmethod
    def validate_dataset(cls, values: List[str]) -> List[str]:
        normalized: list[str] = []
        for value in values:
            key = value.lower()
            if key not in DATASETS:
                allowed = ", ".join(sorted(DATASETS))
                raise ValueError(
                    f"Unsupported dataset '{value}'. Choose from: {allowed}"
                )
            normalized.append(key)
        return normalized

    @field_validator("mtp")
    @classmethod
    def validate_mtp(cls, values: List[int]) -> List[int]:
        if not values:
            raise ValueError("At least one --mtp value is required.")
        invalid = [v for v in values if v not in {1, 2, 3}]
        if invalid:
            joined = ", ".join(str(v) for v in sorted(set(invalid)))
            raise ValueError(f"MTP heads must be between 1 and 3. Invalid: {joined}")
        return sorted(set(values))

    @property
    def dataset_cards(self) -> List[DatasetCard]:
        return [DATASETS[key] for key in self.datasets]

    @property
    def slug(self) -> str:
        head_part = "-".join(str(m) for m in self.mtp)
        dataset_part = "-".join(self.datasets)
        return slugify(f"{self.base_model}-{dataset_part}-{head_part}")
