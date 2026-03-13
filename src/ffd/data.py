"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

from functools import partial
from typing import Iterable, List

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from .datasets import DATASETS, DatasetCard, extract_text_from_record


def load_dataset_cards(keys: Iterable[str]) -> List[DatasetCard]:
    cards: list[DatasetCard] = []
    for key in keys:
        normalized = key.lower()
        if normalized not in DATASETS:
            allowed = ", ".join(sorted(DATASETS))
            raise ValueError(f"Unknown dataset '{key}'. Expected one of: {allowed}")
        cards.append(DATASETS[normalized])
    return cards


def build_tokenized_dataset(
    *,
    cards: List[DatasetCard],
    tokenizer: AutoTokenizer,
    max_samples_per_dataset: int,
    seed: int,
) -> "datasets.Dataset":
    prepared = []
    for card in cards:
        ds = load_dataset(card.hf_repo, split=card.split)
        ds = ds.shuffle(seed=seed)
        if max_samples_per_dataset:
            limited = min(max_samples_per_dataset, len(ds))
            ds = ds.select(range(limited))
        ds = ds.map(lambda example: {"text": extract_text_from_record(example)})
        prepared.append(
            ds.remove_columns([col for col in ds.column_names if col != "text"])
        )

    merged = concatenate_datasets(prepared) if len(prepared) > 1 else prepared[0]
    merged = merged.shuffle(seed=seed)

    max_length = min(card.recommended_sequence_length for card in cards)
    # Constrain sequence length to keep memory manageable during head training.
    max_length = min(max_length, 512)
    tokenize = partial(
        tokenizer,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )

    def tokenize_batch(batch: dict[str, List[str]]) -> dict[str, list[list[int]]]:
        tokens = tokenize(batch["text"])
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    tokenized = merged.map(
        tokenize_batch, batched=True, remove_columns=["text"], num_proc=1
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized
