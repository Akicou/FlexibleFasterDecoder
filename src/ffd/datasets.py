"""Dataset presets used by Flexible Faster Decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class DatasetCard:
    key: str
    name: str
    hf_repo: str
    description: str
    task_types: str
    recommended_sequence_length: int
    split: str = "train"


DATASETS: dict[str, DatasetCard] = {
    "ultrachat": DatasetCard(
        key="ultrachat",
        name="Ultrachat",
        hf_repo="HuggingFaceH4/ultrachat_200k",
        description="Conversational pairs collected for instruction/alignment tuning.",
        task_types="chat, alignment",
        recommended_sequence_length=4096,
        split="train_sft",
    ),
    "codealpaca-20k": DatasetCard(
        key="codealpaca-20k",
        name="CodeAlpaca 20k",
        hf_repo="yahma/alpaca-cleaned",
        description="20k instruction/code exemplars derived from the Alpaca recipe.",
        task_types="code generation, instruction following",
        recommended_sequence_length=3072,
        split="train",
    ),
    "scale-swe": DatasetCard(
        key="scale-swe",
        name="Scale SWE",
        hf_repo="scaleai/ScaleCodeSWE-instruct",
        description="Scale AI curated software engineering instruction set.",
        task_types="software engineering, reasoning",
        recommended_sequence_length=4096,
        split="train",
    ),
}


def dataset_choices() -> list[str]:
    """Return allowed dataset keys for CLI usage."""

    return sorted(DATASETS.keys())


def extract_text_from_record(record: dict[str, Any]) -> str:
    """Flatten any record structure into a newline-delimited text block."""

    fragments: list[str] = []

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                fragments.append(stripped)
        elif isinstance(value, dict):
            for inner in value.values():
                _collect(inner)
        elif isinstance(value, Iterable):
            for item in value:
                _collect(item)

    _collect(record)
    return "\n".join(fragments)
