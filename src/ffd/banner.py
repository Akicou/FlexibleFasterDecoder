"""Generate Hugging Face friendly banner assets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .config import RunConfig
from .utils import ensure_dir


class BannerFactory:
    """Creates a gradient banner summarizing a run."""

    def __init__(self, width: int = 1600, height: int = 400) -> None:
        self.width = width
        self.height = height

    def _palette(self, dataset_key: str) -> tuple[str, str]:
        palettes = {
            "ultrachat": ("#6C2BD9", "#F97316"),
            "codealpaca-20k": ("#0F172A", "#22D3EE"),
            "scale-swe": ("#2563EB", "#FDE047"),
        }
        return palettes.get(dataset_key, ("#111827", "#6EE7B7"))

    def build(
        self,
        spec: RunConfig,
        metrics: Iterable[dict[str, float | int]],
        output_dir: Path,
    ) -> Path:
        ensure_dir(output_dir)
        banner_path = output_dir / "banner.png"
        primary_dataset = spec.datasets[0]
        left, right = self._palette(primary_dataset)

        image = Image.new("RGB", (self.width, self.height), color=left)
        draw = ImageDraw.Draw(image)
        for x in range(self.width):
            blend = x / self.width
            color = tuple(
                int(
                    int(left[i : i + 2], 16) * (1 - blend)
                    + int(right[i : i + 2], 16) * blend
                )
                for i in (1, 3, 5)
            )
            draw.line([(x, 0), (x, self.height)], fill=color)

        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

        head_line = ", ".join(f"{m['head']}x" for m in metrics)
        draw.text((48, 64), "FlexibleFasterDecoder", font=font_title, fill="#FFFFFF")
        draw.text((48, 140), spec.base_model, font=font_body, fill="#F1F5F9")
        dataset_names = ", ".join(card.name for card in spec.dataset_cards)
        draw.text(
            (48, 200), f"Datasets · {dataset_names}", font=font_body, fill="#E2E8F0"
        )
        draw.text((48, 260), f"Heads · {head_line}", font=font_body, fill="#E2E8F0")
        draw.text(
            (48, 320), "Speculative decoding ready", font=font_body, fill="#E2E8F0"
        )

        watermark = "github.com/Akicou/FlexibleFasterDecoder"
        bbox = draw.textbbox((0, 0), watermark, font=font_body)
        wm_width = bbox[2] - bbox[0]
        wm_height = bbox[3] - bbox[1]
        margin = 32
        draw.text(
            (self.width - wm_width - margin, self.height - wm_height - margin),
            watermark,
            font=font_body,
            fill="#CBD5E1",
        )

        image.save(banner_path, format="PNG")
        return banner_path
