"""Train and evaluate the zero-and-one-inflated beta forecast head.

The head uses the last hidden state of the forecasting backbone to predict:

  p0        = P(Delta = 0)
  p1        = P(Delta = 1)
  m_pos     = E[Delta | 0 < Delta < 1]
  k_pos     = interior positive-tail Beta concentration

Predictions use the shared schema consumed by the evaluation pipeline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from opportunity_forecasting import REPO_ROOT
from opportunity_forecasting.manifest import resolve_model_reference

os.environ.setdefault(
    "WEBSHOP_DATA_DIR",
    str(REPO_ROOT / "third_party" / "WebShop" / "data"),
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from opportunity_forecasting.models.distributions import (
    HURDLE_BETA_BETA_EPS,
    HURDLE_BETA_CONCENTRATION_MAX,
    HURDLE_BETA_FAMILY,
    HURDLE_BETA_ZERO_TOL,
    MAX_REWARD_DELTA,
    compute_hurdle_beta_targets,
    forecast_from_fields,
    forecast_implied_moments,
    forecast_mean_log_likelihood,
    forecast_numeric_domain_ok,
)
from opportunity_forecasting.models.training_data import (
    MAX_INPUT_LENGTH,
    build_forecast_prompt,
    load_labeled_data,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_DTYPE = "bf16"
SUPPORT_MODES = ("raw", "remaining")


def _json_dump_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as wf:
        json.dump(payload, wf, indent=2)
    tmp.replace(path)


def _maybe_init_wandb(args: argparse.Namespace, out_dir: Path) -> Any:
    mode = str(getattr(args, "wandb_mode", "") or os.environ.get("WANDB_MODE", "disabled")).strip()
    if mode.lower() in {"", "disabled", "disable", "off", "false", "0"}:
        return None
    project = str(getattr(args, "wandb_project", "") or os.environ.get("WANDB_PROJECT", "")).strip()
    if not project:
        print("[wandb] WANDB_MODE is set but no project was provided; skipping W&B logging.", flush=True)
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[wandb] wandb import failed ({exc}); continuing without W&B logging.", flush=True)
        return None

    entity = str(getattr(args, "wandb_entity", "") or os.environ.get("WANDB_ENTITY", "")).strip() or None
    run_name = str(getattr(args, "wandb_run_name", "") or os.environ.get("WANDB_RUN_NAME", "")).strip() or out_dir.name
    wandb_dir = str(getattr(args, "wandb_dir", "") or os.environ.get("WANDB_DIR", "")).strip() or None
    if wandb_dir:
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    config = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        dir=wandb_dir,
        mode=mode,
        config=config,
    )


def _wandb_log(run: Any, metrics: Dict[str, Any], *, step: int, prefix: str, extra: Optional[Dict[str, Any]] = None) -> None:
    if run is None:
        return
    payload: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[f"{prefix}/{key}"] = float(value)
    if extra:
        payload.update(extra)
    if payload:
        run.log(payload, step=int(step))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _load_dtype(name: str) -> Optional[torch.dtype]:
    raw = str(name or "").strip().lower()
    if raw in {"", "auto", "none"}:
        return None
    if raw == "bf16":
        return torch.bfloat16
    if raw in {"fp16", "float16"}:
        return torch.float16
    if raw in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype {name!r}")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _example_goal_id(ex: Dict[str, Any]) -> int:
    for key in ("goal_idx", "goal_id"):
        if key in ex:
            return int(ex[key])
    input_data = ex.get("input", {}) or {}
    return int(input_data.get("goal_idx", input_data.get("goal_id", 0)) or 0)


def _example_step(ex: Dict[str, Any]) -> int:
    input_data = ex.get("input", {}) or {}
    return int(ex.get("checkpoint_step", input_data.get("checkpoint_step", 0)) or 0)


def _example_goal_text(ex: Dict[str, Any]) -> str:
    input_data = ex.get("input", {}) or {}
    return str(ex.get("goal_text", input_data.get("goal", "")) or "")


def _example_best_reward(ex: Dict[str, Any]) -> float:
    input_data = ex.get("input", {}) or {}
    return _safe_float(input_data.get("best_reward_seen", ex.get("best_reward_seen", 0.0)), 0.0)


def _example_visited_candidate(ex: Dict[str, Any]) -> bool:
    input_data = ex.get("input", {}) or {}
    metadata = ex.get("metadata", {}) or {}
    if "visited_product_page" in input_data:
        return bool(input_data.get("visited_product_page"))
    if "visited_product_page" in metadata:
        return bool(metadata.get("visited_product_page"))
    trigger = str(metadata.get("trigger", input_data.get("trigger", "")) or "").lower()
    if trigger in {"product_page", "paper_page", "item_page", "new_paper_page"}:
        return True
    obs = str(input_data.get("observation", ex.get("observation", "")) or "").lower()
    return ("buy now" in obs) or ("paper page" in obs) or ("current_paper_id:" in obs)


def _example_trigger(ex: Dict[str, Any]) -> str:
    input_data = ex.get("input", {}) or {}
    metadata = ex.get("metadata", {}) or {}
    return str(metadata.get("trigger", input_data.get("trigger", "")) or "")


def _clean_deltas(raw: Sequence[Any], *, max_k: Optional[int] = None) -> List[float]:
    vals: List[float] = []
    for x in raw or []:
        if x is None:
            continue
        v = _safe_float(x, 0.0)
        vals.append(max(0.0, min(float(MAX_REWARD_DELTA), v)))
    if max_k is not None and int(max_k) > 0:
        vals = vals[: int(max_k)]
    return vals


def _support_transform(
    ex: Dict[str, Any],
    deltas: Sequence[float],
    *,
    support_mode: str,
) -> Tuple[List[float], float]:
    mode = str(support_mode)
    if mode == "raw":
        return [float(x) for x in deltas], 1.0
    if mode != "remaining":
        raise ValueError(f"Unknown support_mode={support_mode!r}")
    current_best = max(0.0, min(1.0, _example_best_reward(ex)))
    scale = max(0.0, 1.0 - current_best)
    if scale <= HURDLE_BETA_ZERO_TOL:
        return [0.0 for _ in deltas], 0.0
    return [max(0.0, min(1.0, float(x) / scale)) for x in deltas], float(scale)


def _delta_std(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    mean_val = float(sum(vals) / len(vals))
    var = float(sum((float(v) - mean_val) ** 2 for v in vals) / len(vals))
    return float(math.sqrt(max(0.0, var)))


def _endpoint_summary(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    zeros = 0
    ones = 0
    positive = 0
    for ex in examples:
        for d in _clean_deltas(ex.get("continuation_deltas", []) or []):
            total += 1
            if d <= HURDLE_BETA_ZERO_TOL:
                zeros += 1
            elif d >= 1.0 - HURDLE_BETA_BETA_EPS:
                ones += 1
                positive += 1
            else:
                positive += 1
    return {
        "num_deltas": int(total),
        "num_exact_zero": int(zeros),
        "num_exact_or_clipped_one": int(ones),
        "num_positive": int(positive),
        "exact_zero_fraction": float(zeros / total) if total else 0.0,
        "exact_or_clipped_one_fraction": float(ones / total) if total else 0.0,
        "endpoint_handling": (
            "Zero and one deltas are modeled by explicit point masses; the "
            "Beta likelihood is used only for interior deltas in (0, 1)."
        ),
    }


def _example_total_horizon_steps(ex: Dict[str, Any]) -> int:
    input_data = ex.get("input", {}) or {}
    metadata = ex.get("metadata", {}) or {}
    return int(metadata.get("total_horizon_steps", input_data.get("total_horizon_steps", 60)) or 60)


def _prompt_from_example(ex: Dict[str, Any], *, top_k_seen: int) -> str:
    return build_forecast_prompt(ex, top_k_seen=int(top_k_seen))


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    deltas: torch.Tensor
    delta_mask: torch.Tensor
    mean_delta: torch.Tensor
    std_delta: torch.Tensor
    pos_rate: torch.Tensor
    p0_target: torch.Tensor
    p1_target: torch.Tensor
    m_pos_target: torch.Tensor
    k_pos_target: torch.Tensor
    target_scale: torch.Tensor
    checkpoint_step: torch.Tensor
    total_horizon_steps: torch.Tensor
    meta: Dict[str, Any]


class HurdleHeadDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        tokenizer,
        *,
        top_k_seen: int,
        max_input_length: int,
        max_continuations: Optional[int] = None,
        include_meta: bool = False,
        support_mode: str = "raw",
    ) -> None:
        self.rows: List[EncodedExample] = []
        for idx, ex in enumerate(examples):
            raw_deltas = _clean_deltas(ex.get("continuation_deltas", []) or [], max_k=max_continuations)
            if not raw_deltas:
                continue
            deltas, target_scale = _support_transform(ex, raw_deltas, support_mode=support_mode)
            prompt = _prompt_from_example(ex, top_k_seen=top_k_seen)
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=int(max_input_length),
                add_special_tokens=True,
                return_tensors="pt",
            )
            targets = compute_hurdle_beta_targets(deltas)
            delta_tensor = torch.tensor(deltas, dtype=torch.float32)
            meta = {}
            if include_meta:
                meta = {
                    "row_idx": int(idx),
                    "goal_idx": _example_goal_id(ex),
                    "goal_text": _example_goal_text(ex),
                    "checkpoint_step": _example_step(ex),
                    "visited_product_page": _example_visited_candidate(ex),
                    "best_reward_seen": _example_best_reward(ex),
                    "trigger": _example_trigger(ex),
                }
            self.rows.append(
                EncodedExample(
                    input_ids=enc["input_ids"][0].long(),
                    attention_mask=enc["attention_mask"][0].long(),
                    deltas=delta_tensor,
                    delta_mask=torch.ones_like(delta_tensor),
                    mean_delta=torch.tensor(float(sum(deltas) / len(deltas)), dtype=torch.float32),
                    std_delta=torch.tensor(_delta_std(deltas), dtype=torch.float32),
                    pos_rate=torch.tensor(
                        float(sum(1 for d in deltas if d > HURDLE_BETA_ZERO_TOL)) / float(len(deltas)),
                        dtype=torch.float32,
                    ),
                    p0_target=torch.tensor(float(targets["delta_zero_prob_target"]), dtype=torch.float32),
                    p1_target=torch.tensor(float(targets.get("delta_one_prob_target", 0.0)), dtype=torch.float32),
                    m_pos_target=torch.tensor(float(targets["delta_pos_mean_target"]), dtype=torch.float32),
                    k_pos_target=torch.tensor(float(targets["delta_pos_concentration_target"]), dtype=torch.float32),
                    target_scale=torch.tensor(float(target_scale), dtype=torch.float32),
                    checkpoint_step=torch.tensor(float(_example_step(ex)), dtype=torch.float32),
                    total_horizon_steps=torch.tensor(float(_example_total_horizon_steps(ex)), dtype=torch.float32),
                    meta=meta,
                )
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.rows[idx]


def collate_hurdle_head(batch: Sequence[EncodedExample], pad_token_id: int) -> Dict[str, Any]:
    max_len = max(x.input_ids.numel() for x in batch)
    max_k = max(x.deltas.numel() for x in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    deltas = torch.zeros((len(batch), max_k), dtype=torch.float32)
    delta_mask = torch.zeros((len(batch), max_k), dtype=torch.float32)
    for i, row in enumerate(batch):
        L = row.input_ids.numel()
        K = row.deltas.numel()
        input_ids[i, :L] = row.input_ids
        attention_mask[i, :L] = row.attention_mask
        deltas[i, :K] = row.deltas
        delta_mask[i, :K] = row.delta_mask
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "deltas": deltas,
        "delta_mask": delta_mask,
        "mean_delta": torch.stack([x.mean_delta for x in batch]),
        "std_delta": torch.stack([x.std_delta for x in batch]),
        "pos_rate": torch.stack([x.pos_rate for x in batch]),
        "p0_target": torch.stack([x.p0_target for x in batch]),
        "p1_target": torch.stack([x.p1_target for x in batch]),
        "m_pos_target": torch.stack([x.m_pos_target for x in batch]),
        "k_pos_target": torch.stack([x.k_pos_target for x in batch]),
        "target_scale": torch.stack([x.target_scale for x in batch]),
        "checkpoint_step": torch.stack([x.checkpoint_step for x in batch]),
        "total_horizon_steps": torch.stack([x.total_horizon_steps for x in batch]),
        "meta": [x.meta for x in batch],
    }


class HurdleBetaHeadModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, *, output_dim: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dim = int(output_dim)
        if self.output_dim != 4:
            raise ValueError(
                "The zero-one-inflated Beta head requires four outputs"
            )
        self.head = nn.Linear(int(hidden_size), self.output_dim, dtype=torch.float32)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)

    @staticmethod
    def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.long().sum(dim=1).clamp(min=1) - 1

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = out.last_hidden_state
        idx = self.last_token_index(attention_mask)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_idx, idx].to(self.head.weight.dtype)
        raw = self.head(pooled)
        p0 = torch.sigmoid(raw[:, 0])
        p1 = (1.0 - p0) * torch.sigmoid(raw[:, 1])
        m_raw = raw[:, 2]
        k_raw = raw[:, 3]
        interior_mass = (1.0 - p0 - p1).clamp_min(0.0)
        m_pos = HURDLE_BETA_BETA_EPS + (1.0 - 2.0 * HURDLE_BETA_BETA_EPS) * torch.sigmoid(m_raw)
        k_pos = 2.0 + (float(HURDLE_BETA_CONCENTRATION_MAX) - 2.0) * torch.sigmoid(k_raw)
        return {
            "raw": raw,
            "delta_zero_prob": p0,
            "delta_one_prob": p1,
            "delta_interior_prob": interior_mass,
            "delta_pos_mean": m_pos,
            "delta_pos_concentration": k_pos,
            "expected_delta": p1 + interior_mass * m_pos,
        }


def hurdle_beta_moments(preds: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    p0 = preds["delta_zero_prob"].float().clamp(HURDLE_BETA_BETA_EPS, 1.0 - HURDLE_BETA_BETA_EPS)
    p1 = preds.get("delta_one_prob", torch.zeros_like(p0)).float().clamp(0.0, 1.0 - HURDLE_BETA_BETA_EPS)
    interior_prob = (1.0 - p0 - p1).clamp_min(0.0)
    m = preds["delta_pos_mean"].float().clamp(HURDLE_BETA_BETA_EPS, 1.0 - HURDLE_BETA_BETA_EPS)
    k = preds["delta_pos_concentration"].float().clamp(2.0, float(HURDLE_BETA_CONCENTRATION_MAX))
    alpha = (m * k).clamp_min(HURDLE_BETA_BETA_EPS)
    beta = ((1.0 - m) * k).clamp_min(HURDLE_BETA_BETA_EPS)
    ev = p1 + interior_prob * m
    beta_second = alpha * (alpha + 1.0) / ((alpha + beta) * (alpha + beta + 1.0)).clamp_min(1e-8)
    second = p1 + interior_prob * beta_second
    var = (second - ev.pow(2)).clamp_min(0.0)
    return ev, torch.sqrt(var)


def hurdle_beta_distribution_loss(
    preds: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    ev_huber_weight: float = 0.0,
    event_brier_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    p0 = preds["delta_zero_prob"].float().clamp(HURDLE_BETA_BETA_EPS, 1.0 - HURDLE_BETA_BETA_EPS)
    p1 = preds.get("delta_one_prob", torch.zeros_like(p0)).float().clamp(HURDLE_BETA_BETA_EPS, 1.0 - HURDLE_BETA_BETA_EPS)
    interior_prob = (1.0 - p0 - p1).clamp_min(HURDLE_BETA_BETA_EPS)
    m = preds["delta_pos_mean"].float().clamp(HURDLE_BETA_BETA_EPS, 1.0 - HURDLE_BETA_BETA_EPS)
    k = preds["delta_pos_concentration"].float().clamp(1e-4, float(HURDLE_BETA_CONCENTRATION_MAX))
    alpha = (m * k).clamp_min(HURDLE_BETA_BETA_EPS)
    beta = ((1.0 - m) * k).clamp_min(HURDLE_BETA_BETA_EPS)

    deltas = batch["deltas"].to(p0.device).float().clamp(0.0, 1.0)
    mask = batch["delta_mask"].to(p0.device).float()
    zero = (deltas <= float(HURDLE_BETA_ZERO_TOL)).float()
    one = (deltas >= 1.0 - float(HURDLE_BETA_BETA_EPS)).float()
    interior = ((1.0 - zero) * (1.0 - one)).float()
    x = deltas.clamp(HURDLE_BETA_BETA_EPS, 1.0 - HURDLE_BETA_BETA_EPS)

    log_beta_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    log_pdf = (alpha[:, None] - 1.0) * torch.log(x) + (beta[:, None] - 1.0) * torch.log(1.0 - x) - log_beta_norm[:, None]
    log_prob_zero = torch.log(p0)[:, None].expand_as(deltas)
    log_prob_one = torch.log(p1)[:, None].expand_as(deltas)
    log_prob_pos = torch.log(interior_prob)[:, None] + log_pdf
    log_prob = zero * log_prob_zero + one * log_prob_one + interior * log_prob_pos
    nll = -((log_prob * mask).sum() / mask.sum().clamp_min(1.0))
    loss = nll

    ev_loss = torch.tensor(0.0, device=p0.device)
    if float(ev_huber_weight) > 0.0:
        target_ev = batch["mean_delta"].to(p0.device).float()
        ev_loss = F.smooth_l1_loss(preds["expected_delta"].float(), target_ev, beta=0.05)
        loss = loss + float(ev_huber_weight) * ev_loss

    brier_loss = torch.tensor(0.0, device=p0.device)
    if float(event_brier_weight) > 0.0:
        target_pos = batch["pos_rate"].to(p0.device).float()
        brier_loss = ((1.0 - p0) - target_pos).pow(2).mean()
        loss = loss + float(event_brier_weight) * brier_loss

    with torch.no_grad():
        ev, _ = hurdle_beta_moments(preds)
        ev = ev.float()
        target_ev = batch["mean_delta"].to(p0.device).float()
        event_pred = 1.0 - p0
        event_target = batch["pos_rate"].to(p0.device).float()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "ev_huber": float(ev_loss.detach().cpu()),
            "event_brier": float(((event_pred - event_target).pow(2).mean()).detach().cpu()),
            "ev_mae": float((ev - target_ev).abs().mean().detach().cpu()),
            "mean_pred_ev": float(ev.mean().detach().cpu()),
            "mean_target_ev": float(target_ev.mean().detach().cpu()),
            "mean_pred_pos_rate": float(event_pred.mean().detach().cpu()),
            "mean_target_pos_rate": float(event_target.mean().detach().cpu()),
            "mean_pred_one_prob": float(p1.mean().detach().cpu()),
            "mean_target_one_prob": float(batch["p1_target"].to(p0.device).float().mean().detach().cpu()),
        }
    return loss, metrics


def _average_metric_dict(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if vals:
            out[key] = float(sum(vals) / len(vals))
    return out


def build_backbone_and_tokenizer(
    model_name_or_path: str,
    *,
    dtype_name: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
    gradient_checkpointing: bool,
) -> Tuple[nn.Module, Any, int]:
    from transformers import AutoModel, AutoTokenizer

    dtype = _load_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    backbone = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
    if getattr(backbone, "config", None) is not None:
        backbone.config.use_cache = False
    hidden_size = int(backbone.config.hidden_size)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        modules = [x.strip() for x in str(target_modules).split(",") if x.strip()]
        cfg = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        backbone = get_peft_model(backbone, cfg)
        backbone.print_trainable_parameters()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
    else:
        for p in backbone.parameters():
            p.requires_grad_(True)

    if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    return backbone, tokenizer, hidden_size


def save_hurdle_head_model(
    out_dir: Path,
    model: HurdleBetaHeadModel,
    *,
    base_model: str,
    args: argparse.Namespace,
    best_metrics: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out_dir / "lora")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    meta = {
        "model_kind": "zoib_head",
        "forecast_family": HURDLE_BETA_FAMILY,
        "distribution_family_detail": "zero_one_inflated_beta",
        "support_mode": str(args.support_mode),
        "target_support": "[0, 1-current_best]" if str(args.support_mode) == "remaining" else "[0, 1]",
        "output_dim": int(model.output_dim),
        "base_model": str(base_model),
        "init_model_dir": str(getattr(args, "init_model_dir", "") or ""),
        "top_k_seen": int(args.top_k_seen),
        "max_input_length": int(args.max_input_length),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "target_modules": str(args.target_modules),
        "ev_huber_weight": float(args.ev_huber_weight),
        "event_brier_weight": float(args.event_brier_weight),
        "best_metrics": best_metrics,
    }
    with (out_dir / "hurdle_head_config.json").open("w", encoding="utf-8") as wf:
        json.dump(meta, wf, indent=2)


def load_hurdle_head_model(
    model_dir: Path,
    *,
    dtype_name: str,
    device: torch.device,
    trainable: bool = False,
    gradient_checkpointing: bool = False,
) -> Tuple[HurdleBetaHeadModel, Any, Dict[str, Any]]:
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    cfg_path = model_dir / "hurdle_head_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing hurdle_head_config.json in {model_dir}")
    with cfg_path.open("r", encoding="utf-8") as rf:
        cfg = json.load(rf)
    base_model = resolve_model_reference(
        os.environ.get("OPPORTUNITY_BASE_MODEL", str(cfg["base_model"]))
    )
    dtype = _load_dtype(dtype_name)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    backbone = AutoModel.from_pretrained(base_model, **load_kwargs)
    try:
        backbone = PeftModel.from_pretrained(backbone, model_dir / "lora", is_trainable=bool(trainable))
    except TypeError:
        backbone = PeftModel.from_pretrained(backbone, model_dir / "lora")
        if bool(trainable):
            for name, param in backbone.named_parameters():
                param.requires_grad_(("lora_" in name) or ("modules_to_save" in name))
    if getattr(backbone, "config", None) is not None:
        backbone.config.use_cache = False
    if bool(trainable) and hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    if bool(gradient_checkpointing) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    hidden_size = int(backbone.config.hidden_size)
    head_state = torch.load(
        model_dir / "head.pt",
        map_location="cpu",
        weights_only=True,
    )
    output_dim = int(cfg.get("output_dim") or int(head_state["weight"].shape[0]))
    if output_dim != 4:
        raise ValueError(f"Expected a four-output ZOIB head, found {output_dim}")
    model = HurdleBetaHeadModel(backbone, hidden_size, output_dim=output_dim)
    model.head.load_state_dict(head_state)
    model.to(device)
    if bool(trainable):
        model.train()
        for p in model.head.parameters():
            p.requires_grad_(True)
    else:
        model.eval()
    return model, tok, cfg


@torch.no_grad()
def evaluate_model(
    model: HurdleBetaHeadModel,
    loader: DataLoader,
    device: torch.device,
    *,
    ev_huber_weight: float,
    event_brier_weight: float,
) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []
    all_pred_ev: List[torch.Tensor] = []
    all_target_ev: List[torch.Tensor] = []
    all_pred_pos: List[torch.Tensor] = []
    all_target_pos: List[torch.Tensor] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        _loss, metrics = hurdle_beta_distribution_loss(
            preds,
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
            ev_huber_weight=ev_huber_weight,
            event_brier_weight=event_brier_weight,
        )
        rows.append(metrics)
        all_pred_ev.append(preds["expected_delta"].detach().float().cpu())
        all_target_ev.append(batch["mean_delta"].detach().float().cpu())
        all_pred_pos.append((1.0 - preds["delta_zero_prob"]).detach().float().cpu())
        all_target_pos.append(batch["pos_rate"].detach().float().cpu())
    out = _average_metric_dict(rows)
    if all_pred_ev:
        pred_ev = torch.cat(all_pred_ev)
        target_ev = torch.cat(all_target_ev)
        pred_pos = torch.cat(all_pred_pos)
        target_pos = torch.cat(all_target_pos)
        out["ev_rmse"] = float(torch.sqrt(torch.mean((pred_ev - target_ev).pow(2))).item())
        out["ev_mae"] = float(torch.mean(torch.abs(pred_ev - target_ev)).item())
        out["event_brier"] = float(torch.mean((pred_pos - target_pos).pow(2)).item())
    model.train()
    return out


def should_select_checkpoint(
    *, epoch: int, metric: float, best_metric: Optional[float]
) -> bool:
    return (
        int(epoch) >= 1
        and math.isfinite(float(metric))
        and (best_metric is None or float(metric) < float(best_metric))
    )


def run_train(args: argparse.Namespace) -> None:
    if int(args.epochs) < 1:
        raise ValueError("--epochs must be at least 1")
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(args, out_dir)

    print("Loading labeled data...", flush=True)
    train_examples = load_labeled_data(Path(args.train_data))
    val_examples: List[Dict[str, Any]] = []
    if args.val_data:
        val_examples = load_labeled_data(Path(args.val_data))
    if int(args.max_train_examples) > 0:
        rng = random.Random(int(args.seed))
        train_examples = rng.sample(train_examples, min(len(train_examples), int(args.max_train_examples)))
    if int(args.max_val_examples) > 0 and val_examples:
        val_examples = val_examples[: int(args.max_val_examples)]
    print(f"train_examples={len(train_examples)} val_examples={len(val_examples)}", flush=True)
    endpoint_summary = {
        "train": _endpoint_summary(train_examples),
        "val": _endpoint_summary(val_examples) if val_examples else None,
    }
    print(f"[endpoint_summary] {json.dumps(endpoint_summary, sort_keys=True)}", flush=True)
    base_model_for_save = str(args.base_model)
    if str(args.init_model_dir or "").strip():
        if bool(args.full_finetune):
            raise ValueError("--init_model_dir continues a PEFT hurdle head; do not combine it with --full_finetune.")
        print(f"Initializing from existing hurdle head: {args.init_model_dir}", flush=True)
        model, tok, init_cfg = load_hurdle_head_model(
            Path(args.init_model_dir),
            dtype_name=args.dtype,
            device=device,
            trainable=True,
            gradient_checkpointing=bool(args.gradient_checkpointing),
        )
        base_model_for_save = str(init_cfg.get("base_model", args.base_model))
    else:
        backbone, tok, hidden_size = build_backbone_and_tokenizer(
            args.base_model,
            dtype_name=args.dtype,
            use_lora=not bool(args.full_finetune),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            gradient_checkpointing=bool(args.gradient_checkpointing),
        )
        model = HurdleBetaHeadModel(backbone, hidden_size).to(device)

    train_ds = HurdleHeadDataset(
        train_examples,
        tok,
        top_k_seen=args.top_k_seen,
        max_input_length=args.max_input_length,
        max_continuations=args.max_continuations,
        support_mode=str(args.support_mode),
    )
    val_ds = (
        HurdleHeadDataset(
            val_examples,
            tok,
            top_k_seen=args.top_k_seen,
            max_input_length=args.max_input_length,
            max_continuations=args.max_continuations,
            support_mode=str(args.support_mode),
        )
        if val_examples
        else None
    )
    if not len(train_ds):
        raise ValueError("No trainable rows after loading continuation deltas.")
    print(
        f"encoded_examples train={len(train_ds)}"
        + (f" val={len(val_ds)}" if val_ds is not None else ""),
        flush=True,
    )

    collate = lambda b: collate_hurdle_head(b, tok.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
            collate_fn=collate,
        )
        if val_ds is not None
        else None
    )

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=float(args.learning_rate), betas=(0.9, 0.95), weight_decay=float(args.weight_decay))

    best_metric: Optional[float] = None
    best_state: Optional[Dict[str, Any]] = None
    best_metrics: Dict[str, Any] = {}
    step = 0
    t0 = time.time()
    accum = max(1, int(args.gradient_accumulation_steps))
    opt.zero_grad(set_to_none=True)

    def _capture_state() -> Dict[str, Any]:
        return {
            "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
            "lora": {
                k: v.detach().cpu().clone()
                for k, v in model.backbone.named_parameters()
                if v.requires_grad
            },
        }

    if val_loader is not None:
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            ev_huber_weight=float(args.ev_huber_weight),
            event_brier_weight=float(args.event_brier_weight),
        )
        print(f"[val] epoch=0 {json.dumps(val_metrics, sort_keys=True)}", flush=True)
        _wandb_log(wandb_run, val_metrics, step=0, prefix="val", extra={"epoch": 0})

    for epoch in range(int(args.epochs)):
        model.train()
        window: List[Dict[str, float]] = []
        for micro, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_dev = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, metrics = hurdle_beta_distribution_loss(
                preds,
                batch_dev,
                ev_huber_weight=float(args.ev_huber_weight),
                event_brier_weight=float(args.event_brier_weight),
            )
            (loss / accum).backward()
            window.append(metrics)
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % int(args.log_every) == 0:
                    avg = _average_metric_dict(window)
                    window.clear()
                    elapsed = time.time() - t0
                    print(
                        f"[train] epoch={epoch+1} step={step} "
                        f"loss={avg.get('loss', 0.0):.4f} nll={avg.get('nll', 0.0):.4f} "
                        f"ev_mae={avg.get('ev_mae', 0.0):.4f} "
                        f"pred_ev={avg.get('mean_pred_ev', 0.0):.4f} "
                        f"target_ev={avg.get('mean_target_ev', 0.0):.4f} "
                        f"elapsed={elapsed/60.0:.1f}m",
                        flush=True,
                    )
                    _wandb_log(wandb_run, avg, step=step, prefix="train", extra={"epoch": epoch + 1})
                if int(args.max_steps) > 0 and step >= int(args.max_steps):
                    break
        if val_loader is not None:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                ev_huber_weight=float(args.ev_huber_weight),
                event_brier_weight=float(args.event_brier_weight),
            )
            metric = float(val_metrics.get(str(args.early_stop_metric), val_metrics.get("loss", 0.0)))
            print(f"[val] epoch={epoch+1} {json.dumps(val_metrics, sort_keys=True)}", flush=True)
            _wandb_log(wandb_run, val_metrics, step=step, prefix="val", extra={"epoch": epoch + 1})
            if should_select_checkpoint(
                epoch=epoch + 1,
                metric=metric,
                best_metric=best_metric,
            ):
                best_metric = metric
                best_metrics = {"epoch": epoch + 1, "step": step, **val_metrics}
                best_state = _capture_state()
                save_hurdle_head_model(out_dir / "best", model, base_model=base_model_for_save, args=args, best_metrics=best_metrics)
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            break

    if best_state is not None:
        model.head.load_state_dict(best_state["head"])
        with torch.no_grad():
            for k, p in model.backbone.named_parameters():
                if k in best_state["lora"]:
                    p.copy_(best_state["lora"][k].to(p.device, p.dtype))
    else:
        best_metrics = {"epoch": int(args.epochs), "step": step}
        save_hurdle_head_model(
            out_dir / "best",
            model,
            base_model=base_model_for_save,
            args=args,
            best_metrics=best_metrics,
        )
    save_hurdle_head_model(out_dir / "final", model, base_model=base_model_for_save, args=args, best_metrics=best_metrics)
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as wf:
        json.dump(
            {
                "best_metrics": best_metrics,
                "total_steps": step,
                "train_time_sec": time.time() - t0,
                "endpoint_summary": endpoint_summary,
            },
            wf,
            indent=2,
        )
    _wandb_log(
        wandb_run,
        {k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
        step=step,
        prefix="best",
        extra={"total_steps": int(step), "train_time_sec": float(time.time() - t0)},
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved hurdle-head model to {out_dir / 'final'}", flush=True)


@torch.no_grad()
def predict_examples(
    model: HurdleBetaHeadModel,
    tokenizer,
    examples: Sequence[Dict[str, Any]],
    *,
    top_k_seen: int,
    max_input_length: int,
    batch_size: int,
    device: torch.device,
    support_mode: str = "raw",
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    ds = HurdleHeadDataset(
        examples,
        tokenizer,
        top_k_seen=top_k_seen,
        max_input_length=max_input_length,
        include_meta=True,
        support_mode=support_mode,
    )
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_hurdle_head(b, tokenizer.pad_token_id),
    )
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    model.eval()
    t0 = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        preds = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        p0 = preds["delta_zero_prob"].detach().float().cpu().tolist()
        p1 = preds["delta_one_prob"].detach().float().cpu().tolist()
        m = preds["delta_pos_mean"].detach().float().cpu().tolist()
        k = preds["delta_pos_concentration"].detach().float().cpu().tolist()
        unit_ev, unit_std = hurdle_beta_moments(preds)
        unit_ev_rows = unit_ev.detach().float().cpu().tolist()
        unit_std_rows = unit_std.detach().float().cpu().tolist()
        for meta, p0_i, p1_i, m_i, k_i, unit_ev_i, unit_std_i in zip(
            batch["meta"], p0, p1, m, k, unit_ev_rows, unit_std_rows
        ):
            fields = {
                "forecast_family": (
                    "explicit_residual_zoib_remaining_support"
                    if str(support_mode) == "remaining"
                    else HURDLE_BETA_FAMILY
                ),
                "delta_zero_prob": float(p0_i),
                "delta_one_prob": float(p1_i),
                "delta_pos_mean": float(m_i),
                "delta_pos_concentration": float(k_i),
            }
            scale = 1.0
            if str(support_mode) == "remaining":
                current_best = max(0.0, min(1.0, _safe_float(meta.get("best_reward_seen", 0.0), 0.0)))
                scale = max(0.0, 1.0 - current_best)
                fields.update(
                    {
                        "remaining_gain_scale": float(scale),
                        "zoib_target": "remaining_gain_fraction",
                        "expected_unit_delta": float(unit_ev_i),
                        "expected_unit_std_delta": float(unit_std_i),
                        "event_probability": float(1.0 - p0_i),
                    }
                )
            fields["expected_delta"] = float(unit_ev_i) * float(scale)
            fields["expected_std_delta"] = float(unit_std_i) * float(scale)
            out.append((meta, fields))
        if batch_idx == 1 or batch_idx % 25 == 0:
            done = min(len(out), len(ds))
            elapsed = max(1e-6, time.time() - t0)
            rate = float(done) / elapsed
            eta_min = (float(len(ds) - done) / max(rate, 1e-6)) / 60.0
            print(
                f"[predict] {done}/{len(ds)} rows "
                f"({rate:.2f} rows/s, eta={eta_min:.1f}m)",
                flush=True,
            )
    return out


def build_cache_from_labeled_examples(
    *,
    examples: Sequence[Dict[str, Any]],
    predictions: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    model_dir: Path,
    output_path: Path,
    split_name: str,
    label_source: Path,
    model_label: str | None = None,
) -> Dict[str, Any]:
    grouped: Dict[int, List[Tuple[Dict[str, Any], Dict[str, float]]]] = {}
    for meta, fields in predictions:
        gid = int(meta["goal_idx"])
        grouped.setdefault(gid, []).append((meta, fields))
    goal_ids = sorted(grouped.keys())
    health = {
        "num_goals": len(goal_ids),
        "num_checkpoints_total": 0,
        "num_numeric_parsed": 0,
        "num_valid_forecasts": 0,
        "num_parse_failures": 0,
        "num_sigma_recovered_or_clamped": 0,
        "num_failure_examples": 0,
    }
    goal_predictions: Dict[str, Any] = {}
    for gid in goal_ids:
        rows = sorted(grouped[gid], key=lambda x: (int(x[0].get("checkpoint_step", 0)), int(x[0].get("row_idx", 0))))
        ckpt_rows = []
        goal_text = str(rows[0][0].get("goal_text", ""))
        seen_candidate = False
        best_so_far = 0.0
        for ckpt_idx, (meta, fields) in enumerate(rows):
            seen_candidate = bool(seen_candidate or bool(meta.get("visited_product_page", False)))
            best_so_far = max(best_so_far, _safe_float(meta.get("best_reward_seen", 0.0), 0.0))
            explicit = str(fields.get("forecast_family", "")).startswith("explicit_")
            if not explicit and (
                fields.get("expected_delta") is None
                or fields.get("expected_std_delta") is None
            ):
                implied_mean, implied_std = forecast_implied_moments(
                    forecast_from_fields(fields)
                )
                fields = dict(fields)
                fields.setdefault("expected_delta", float(implied_mean))
                fields.setdefault("expected_std_delta", float(implied_std))
            ok = (
                math.isfinite(_safe_float(fields.get("expected_delta"), float("nan")))
                and math.isfinite(_safe_float(fields.get("expected_std_delta"), float("nan")))
            ) if explicit else bool(forecast_numeric_domain_ok(forecast_from_fields(fields)))
            health["num_checkpoints_total"] += 1
            health["num_numeric_parsed"] += 1
            health["num_valid_forecasts"] += int(ok)
            ckpt_rows.append(
                {
                    "checkpoint_idx": int(ckpt_idx),
                    "checkpoint_step": int(meta.get("checkpoint_step", 0) or 0),
                    "candidate_available": bool(seen_candidate),
                    "best_reward_seen": float(best_so_far),
                    "forecast_family": str(fields.get("forecast_family", HURDLE_BETA_FAMILY)),
                    "delta_zero_prob": float(fields["delta_zero_prob"]),
                    "delta_one_prob": float(fields.get("delta_one_prob", 0.0)),
                    "delta_pos_mean": float(fields["delta_pos_mean"]),
                    "delta_pos_concentration": float(fields["delta_pos_concentration"]),
                    "expected_delta": float(fields.get("expected_delta", 0.0)),
                    "expected_std_delta": float(fields.get("expected_std_delta", 0.0)),
                    "event_probability": float(
                        1.0 - float(fields["delta_zero_prob"])
                    ),
                    "remaining_gain_scale": fields.get("remaining_gain_scale"),
                    "zoib_target": fields.get("zoib_target"),
                    "expected_unit_delta": fields.get("expected_unit_delta"),
                    "expected_unit_std_delta": fields.get("expected_unit_std_delta"),
                    "forecast_numeric_domain_ok": ok,
                }
            )
        goal_predictions[str(gid)] = {
            "goal_idx": int(gid),
            "goal_text": goal_text,
            "checkpoints": ckpt_rows,
        }
    total = max(1, int(health["num_checkpoints_total"]))
    payload = {
        "checkpoint_path": None,
        "label_source": str(label_source),
        "split": str(split_name),
        "goal_ids_path": None,
        "goal_ids": goal_ids,
        "model_path": str(model_label or model_dir),
        "model_type": "zoib_head",
        "engine": "hurdle_head",
        "reward_mode": None,
        "top_k_seen": None,
        "sigma_floor": None,
        "sigma_support": "",
        "goal_predictions": goal_predictions,
        "prediction_health": {
            "model_path": str(model_label or model_dir),
            "num_goals": int(health["num_goals"]),
            "num_checkpoints_total": int(health["num_checkpoints_total"]),
            "numeric_parse_rate": float(health["num_numeric_parsed"]) / total,
            "valid_forecast_rate": float(health["num_valid_forecasts"]) / total,
            "num_sigma_recovered_or_clamped": 0,
            "num_parse_failures": 0,
            "num_failure_examples": 0,
        },
        "failure_examples": [],
        "progress": {
            "processed_goals": int(health["num_goals"]),
            "total_goals": int(health["num_goals"]),
            "current_goal_idx": None,
            "complete": True,
            "updated_at_unix": time.time(),
        },
    }
    _json_dump_atomic(output_path, payload)
    return payload


def run_predict_labels(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_dir = Path(args.model_dir)
    model, tok, cfg = load_hurdle_head_model(model_dir, dtype_name=args.dtype, device=device)
    examples = load_labeled_data(Path(args.data))
    if int(args.max_examples) > 0:
        examples = examples[: int(args.max_examples)]
    if args.shard_idx is not None or args.num_shards is not None:
        if args.shard_idx is None or args.num_shards is None:
            raise ValueError("Must set both --shard_idx and --num_shards, or neither.")
        num_shards = int(args.num_shards)
        shard_idx = int(args.shard_idx)
        if num_shards <= 0:
            raise ValueError("--num_shards must be > 0")
        if not (0 <= shard_idx < num_shards):
            raise ValueError("--shard_idx must be in [0, num_shards)")
        goal_ids = sorted({_example_goal_id(ex) for ex in examples})
        shard_goal_ids = {gid for pos, gid in enumerate(goal_ids) if pos % num_shards == shard_idx}
        before_n = len(examples)
        examples = [ex for ex in examples if _example_goal_id(ex) in shard_goal_ids]
        print(
            f"Sharding enabled: shard_idx={shard_idx} num_shards={num_shards} "
            f"goals={len(shard_goal_ids)}/{len(goal_ids)} rows={len(examples)}/{before_n}",
            flush=True,
        )
    top_k = int(args.top_k_seen if args.top_k_seen is not None else cfg.get("top_k_seen", 15))
    max_len = int(args.max_input_length if args.max_input_length is not None else cfg.get("max_input_length", MAX_INPUT_LENGTH))
    support_mode = str(cfg.get("support_mode", "raw"))
    print(f"Predicting {len(examples)} labeled rows from {args.data}", flush=True)
    preds = predict_examples(
        model,
        tok,
        examples,
        top_k_seen=top_k,
        max_input_length=max_len,
        batch_size=int(args.batch_size),
        device=device,
        support_mode=support_mode,
    )
    build_cache_from_labeled_examples(
        examples=examples,
        predictions=preds,
        model_dir=model_dir,
        output_path=Path(args.output_path),
        split_name=str(args.split),
        label_source=Path(args.data),
        model_label=args.model_label,
    )
    metrics = evaluate_predictions_on_labeled_examples(examples, preds)
    metrics_path = Path(args.output_path).with_suffix(".metrics.json")
    _json_dump_atomic(metrics_path, metrics)
    print(f"Wrote cache: {args.output_path}", flush=True)
    print(f"Wrote metrics: {metrics_path}", flush=True)


def evaluate_predictions_on_labeled_examples(
    examples: Sequence[Dict[str, Any]],
    predictions: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    for ex, (_meta, fields) in zip(examples, predictions):
        deltas = _clean_deltas(ex.get("continuation_deltas", []) or [])
        support_mode = "remaining" if str(fields.get("forecast_family", "")).startswith("explicit_residual_zoib_remaining") else "raw"
        if support_mode == "remaining":
            scale = max(0.0, 1.0 - max(0.0, min(1.0, _example_best_reward(ex))))
            unit_deltas = [0.0 for _ in deltas] if scale <= HURDLE_BETA_ZERO_TOL else [max(0.0, min(1.0, d / scale)) for d in deltas]
            unit_forecast = forecast_from_fields({**fields, "forecast_family": HURDLE_BETA_FAMILY})
            mean_delta = _safe_float(fields.get("expected_delta"), 0.0)
            mean_log_likelihood = forecast_mean_log_likelihood(unit_deltas, unit_forecast)
        else:
            forecast = forecast_from_fields(fields)
            mean_delta, _std = forecast_implied_moments(forecast)
            mean_log_likelihood = forecast_mean_log_likelihood(deltas, forecast)
        target_mean = float(sum(deltas) / max(1, len(deltas)))
        target_pos = float(sum(1 for d in deltas if d > HURDLE_BETA_ZERO_TOL)) / float(max(1, len(deltas)))
        pred_pos = 1.0 - float(fields["delta_zero_prob"])
        rows.append(
            {
                "pred_ev": float(mean_delta if mean_delta is not None else 0.0),
                "target_ev": target_mean,
                "pred_pos": pred_pos,
                "target_pos": target_pos,
                "event_brier": (pred_pos - target_pos) ** 2,
                "mean_log_likelihood": mean_log_likelihood,
            }
        )
    if not rows:
        return {"num_rows": 0}
    pred_ev = [r["pred_ev"] for r in rows]
    target_ev = [r["target_ev"] for r in rows]
    pred_pos = [r["pred_pos"] for r in rows]
    target_pos = [r["target_pos"] for r in rows]
    return {
        "num_rows": int(len(rows)),
        "mean_predicted_ev": float(sum(pred_ev) / len(pred_ev)),
        "mean_realized_remaining_upside": float(sum(target_ev) / len(target_ev)),
        "positive_upside_rate": float(sum(target_pos) / len(target_pos)),
        "mean_predicted_positive_rate": float(sum(pred_pos) / len(pred_pos)),
        "ev_mae": float(sum(abs(a - b) for a, b in zip(pred_ev, target_ev)) / len(rows)),
        "ev_rmse": float(math.sqrt(sum((a - b) ** 2 for a, b in zip(pred_ev, target_ev)) / len(rows))),
        "event_brier": float(sum(r["event_brier"] for r in rows) / len(rows)),
        "mean_log_likelihood": float(sum(r["mean_log_likelihood"] for r in rows) / len(rows)),
    }


def add_common_train_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", default="")
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--init_model_dir", default="", help="Optional existing hurdle-head checkpoint to continue from.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_k_seen", type=int, default=15)
    ap.add_argument("--max_input_length", type=int, default=MAX_INPUT_LENGTH)
    ap.add_argument("--max_continuations", type=int, default=0)
    ap.add_argument(
        "--support_mode",
        default="raw",
        choices=SUPPORT_MODES,
        help="Use raw [0,1] deltas or normalize by the feasible remaining gain 1-current_best.",
    )
    ap.add_argument("--max_train_examples", type=int, default=0)
    ap.add_argument("--max_val_examples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--ev_huber_weight", type=float, default=0.2)
    ap.add_argument("--event_brier_weight", type=float, default=0.1)
    ap.add_argument(
        "--early_stop_metric",
        default="loss",
        choices=["loss", "nll", "ev_rmse", "ev_mae", "event_brier"],
    )
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default=DEFAULT_TARGET_MODULES)
    ap.add_argument("--full_finetune", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["bf16", "fp16", "fp32", "auto"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "disabled"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", ""))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    ap.add_argument("--wandb_run_name", default=os.environ.get("WANDB_RUN_NAME", ""))
    ap.add_argument("--wandb_dir", default=os.environ.get("WANDB_DIR", ""))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    train_ap = sub.add_parser("train", help="Train a LoRA-backed hurdle-Beta regression head.")
    add_common_train_args(train_ap)

    pred_ap = sub.add_parser("predict-labels", help="Export a prediction cache from labeled decision points.")
    pred_ap.add_argument("--model_dir", required=True)
    pred_ap.add_argument("--data", required=True)
    pred_ap.add_argument("--output_path", required=True)
    pred_ap.add_argument("--split", default="test")
    pred_ap.add_argument("--model_label")
    pred_ap.add_argument("--top_k_seen", type=int, default=None)
    pred_ap.add_argument("--max_input_length", type=int, default=None)
    pred_ap.add_argument("--batch_size", type=int, default=8)
    pred_ap.add_argument("--max_examples", type=int, default=0)
    pred_ap.add_argument("--shard_idx", type=int, default=None)
    pred_ap.add_argument("--num_shards", type=int, default=None)
    pred_ap.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["bf16", "fp16", "fp32", "auto"])
    pred_ap.add_argument("--cpu", action="store_true")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.cmd == "train":
        if int(args.max_continuations) <= 0:
            args.max_continuations = None
        run_train(args)
        return
    if args.cmd == "predict-labels":
        run_predict_labels(args)
        return
    raise ValueError(f"Unknown command {args.cmd!r}")


if __name__ == "__main__":
    main()
