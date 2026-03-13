"""Speculative head model definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class HeadConfig:
    head: int
    hidden_size: int
    vocab_size: int
    base_model: str
    tokenizer: str
    pad_token_id: int
    max_position_embeddings: int | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


class SpeculativeHead(nn.Module):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.projections = nn.ModuleList(
            nn.Linear(config.hidden_size, config.vocab_size, bias=True)
            for _ in range(config.head)
        )

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        losses = []
        for offset, projection in enumerate(self.projections, start=1):
            trimmed_hidden = hidden_states[:, :-offset, :]
            trimmed_labels = labels[:, offset:]
            logits = projection(trimmed_hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                trimmed_labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
            losses.append(loss)
        return torch.stack(losses).mean()

    def save(self, path: str) -> None:
        tensors = {
            name: tensor.detach().cpu() for name, tensor in self.state_dict().items()
        }
        save_file(tensors, path)
