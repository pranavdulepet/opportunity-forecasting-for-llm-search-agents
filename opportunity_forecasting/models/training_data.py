"""Load labeled states and render the shared forecasting prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opportunity_forecasting.models.prompted_forecaster import (
    build_stopping_prompt,
    format_seen_products_for_prompt,
)


MAX_INPUT_LENGTH = 6144


def build_forecast_prompt(example: dict[str, Any], *, top_k_seen: int = 15) -> str:
    input_data = example["input"]
    seen = input_data.get("seen_products", {}) or {}
    seen_text = (
        format_seen_products_for_prompt(seen, None, top_n=int(top_k_seen))
        if seen
        else "No items seen yet."
    )
    metadata = example.get("metadata", {}) or {}
    return build_stopping_prompt(
        goal=input_data["goal"],
        seen_products_text=seen_text,
        best_reward_seen=input_data.get("best_reward_seen", 0.0),
        top_k=int(top_k_seen),
        checkpoint_step=example.get(
            "checkpoint_step", input_data.get("checkpoint_step")
        ),
        total_horizon_steps=metadata.get("total_horizon_steps", 60),
        trigger=metadata.get("trigger"),
        observation=input_data.get("observation"),
    )


def load_labeled_data(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
