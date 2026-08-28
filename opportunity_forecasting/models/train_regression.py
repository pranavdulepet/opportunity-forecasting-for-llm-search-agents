"""Train and evaluate full-horizon residual scalar or Gaussian heads."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from opportunity_forecasting import REPO_ROOT
from opportunity_forecasting.manifest import resolve_model_reference

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from opportunity_forecasting.models.distributions import (
    HURDLE_BETA_BETA_EPS,
    HURDLE_BETA_ZERO_TOL,
    MAX_REWARD_DELTA,
)
from opportunity_forecasting.models.training_data import MAX_INPUT_LENGTH


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_DTYPE = "bf16"
DEFAULT_GAUSSIAN_STD_FLOOR = 0.01


def _clean_deltas(raw: Sequence[Any]) -> List[float]:
    vals: List[float] = []
    for x in raw or []:
        vals.append(max(0.0, min(float(MAX_REWARD_DELTA), _safe_float(x, 0.0))))
    return vals


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(val):
        return float(default)
    return float(val)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _example_goal_id(ex: Dict[str, Any]) -> int:
    input_data = ex.get("input", {}) or {}
    return int(ex.get("goal_idx", input_data.get("goal_idx", 0)) or 0)


def _example_goal_text(ex: Dict[str, Any]) -> str:
    input_data = ex.get("input", {}) or {}
    return str(ex.get("goal_text", input_data.get("goal", "")) or "")


def _example_step(ex: Dict[str, Any]) -> int:
    input_data = ex.get("input", {}) or {}
    return int(ex.get("checkpoint_step", input_data.get("checkpoint_step", 0)) or 0)


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
    if bool(input_data.get("has_opened_paper", metadata.get("has_opened_paper", False))):
        return True
    trigger = str(metadata.get("trigger", input_data.get("trigger", "")) or "").lower()
    if trigger in {"product_page", "paper_page", "item_page", "new_paper_page"}:
        return True
    obs = str(input_data.get("observation", ex.get("observation", "")) or "").lower()
    return ("buy now" in obs) or ("paper page" in obs) or ("current_paper_id:" in obs)


def _example_trigger(ex: Dict[str, Any]) -> str:
    metadata = ex.get("metadata", {}) or {}
    input_data = ex.get("input", {}) or {}
    return str(metadata.get("trigger", input_data.get("trigger", "")) or "")


def _prompt_from_example(ex: Dict[str, Any], *, top_k_seen: int) -> str:
    from opportunity_forecasting.models.train_zoib import _prompt_from_example as _impl

    return _impl(ex, top_k_seen=top_k_seen)


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


def _maybe_init_wandb(args: argparse.Namespace, out_dir: Path) -> Any:
    from opportunity_forecasting.models.train_zoib import _maybe_init_wandb as _impl

    return _impl(args, out_dir)


def _wandb_log(run: Any, metrics: Dict[str, Any], *, step: int, prefix: str, extra: Optional[Dict[str, Any]] = None) -> None:
    from opportunity_forecasting.models.train_zoib import _wandb_log as _impl

    return _impl(run, metrics, step=step, prefix=prefix, extra=extra)


def build_backbone_and_tokenizer(*args: Any, **kwargs: Any) -> Tuple[nn.Module, Any, int]:
    from opportunity_forecasting.models.train_zoib import build_backbone_and_tokenizer as _impl

    return _impl(*args, **kwargs)


def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _population_std(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    mean = _mean(vals)
    return float(math.sqrt(sum((float(v) - mean) ** 2 for v in vals) / len(vals)))


@dataclass
class EncodedResidualExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_gain: torch.Tensor
    meta: Dict[str, Any]


class ResidualHeadDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        tokenizer,
        *,
        target_family: str,
        top_k_seen: int,
        max_input_length: int,
        include_meta: bool = False,
    ) -> None:
        self.target_family = str(target_family)
        self.rows: List[EncodedResidualExample] = []
        for idx, ex in enumerate(examples):
            target_gain = self._targets_from_example(ex)
            if target_gain is None:
                continue
            prompt = _prompt_from_example(ex, top_k_seen=int(top_k_seen))
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=int(max_input_length),
                add_special_tokens=True,
                return_tensors="pt",
            )
            meta: Dict[str, Any] = {}
            if include_meta:
                metadata = ex.get("metadata", {}) or {}
                prompt_best = _example_best_reward(ex)
                meta = {
                    "row_idx": int(idx),
                    "goal_idx": _example_goal_id(ex),
                    "goal_text": _example_goal_text(ex),
                    "checkpoint_step": _example_step(ex),
                    "visited_product_page": _example_visited_candidate(ex),
                    "best_reward_seen": prompt_best,
                    "trigger": _example_trigger(ex),
                }
            self.rows.append(
                EncodedResidualExample(
                    input_ids=enc["input_ids"][0].long(),
                    attention_mask=enc["attention_mask"][0].long(),
                    target_gain=torch.tensor(target_gain, dtype=torch.float32),
                    meta=meta,
                )
            )

    def _targets_from_example(
        self,
        ex: Dict[str, Any],
    ) -> Optional[List[float]]:
        deltas = _clean_deltas(ex.get("continuation_deltas", []) or [])
        if not deltas:
            return None
        if self.target_family == "residual_scalar":
            return [_mean(deltas)]
        if self.target_family == "residual_gaussian":
            return [_mean(deltas), _population_std(deltas)]
        raise ValueError(f"Unknown target_family={self.target_family!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> EncodedResidualExample:
        return self.rows[idx]


def collate_residual(batch: Sequence[EncodedResidualExample], pad_token_id: int) -> Dict[str, Any]:
    max_len = max(x.input_ids.numel() for x in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, row in enumerate(batch):
        L = row.input_ids.numel()
        input_ids[i, :L] = row.input_ids
        attention_mask[i, :L] = row.attention_mask
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_gain": torch.stack([x.target_gain for x in batch]),
        "meta": [x.meta for x in batch],
    }


class ResidualHeadModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        *,
        output_dim: int,
        target_family: str,
        head_init_std: float = 0.02,
        head_bias_init: Optional[float] = None,
        gaussian_std_floor: float = DEFAULT_GAUSSIAN_STD_FLOOR,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dim = int(output_dim)
        self.target_family = str(target_family)
        self.gaussian_std_floor = float(gaussian_std_floor)
        if not 0.0 < self.gaussian_std_floor < float(MAX_REWARD_DELTA):
            raise ValueError("gaussian_std_floor must be between zero and MAX_REWARD_DELTA")
        if self.target_family == "residual_scalar":
            self.head = nn.Linear(hidden_size, 1)
        elif self.target_family == "residual_gaussian":
            self.head = nn.Linear(hidden_size, 2)
        else:
            raise ValueError(f"Unknown target_family={target_family!r}")
        if head_bias_init is None:
            nn.init.zeros_(self.head.bias)
        else:
            nn.init.constant_(self.head.bias, float(head_bias_init))
        nn.init.normal_(self.head.weight, std=float(head_init_std))

    @staticmethod
    def _last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.long().sum(dim=1).clamp(min=1) - 1

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        h = out.last_hidden_state
        idx = self._last_token_index(attention_mask)
        last = h[torch.arange(h.size(0), device=h.device), idx].to(self.head.weight.dtype)
        raw = self.head(last).float()
        if self.target_family == "residual_scalar":
            expected = torch.sigmoid(raw)
            return {
                "expected_gain": expected,
                "std_gain": torch.zeros_like(expected),
            }
        if self.target_family == "residual_gaussian":
            expected = torch.sigmoid(raw[:, :1])
            std = self.gaussian_std_floor + (float(MAX_REWARD_DELTA) - self.gaussian_std_floor) * torch.sigmoid(raw[:, 1:2])
            z = (float(HURDLE_BETA_ZERO_TOL) - expected) / std.clamp_min(float(HURDLE_BETA_BETA_EPS))
            event_prob = (1.0 - 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))).clamp(0.0, 1.0)
            return {
                "expected_gain": expected,
                "std_gain": std,
                "event_prob": event_prob,
            }
        raise ValueError(f"Unknown target_family={self.target_family!r}")


def residual_loss(
    preds: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    target_family: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_abs = batch["target_gain"].to(preds["expected_gain"].device).float()
    if str(target_family) == "residual_scalar":
        ev_loss = F.smooth_l1_loss(preds["expected_gain"], target_abs)
        pred_ev = preds["expected_gain"].detach()
        mae = torch.mean(torch.abs(pred_ev - target_abs.detach()))
        rmse = torch.sqrt(torch.mean((pred_ev - target_abs.detach()).pow(2)))
        return ev_loss, {
            "loss": float(ev_loss.detach().cpu()),
            "ev_loss": float(ev_loss.detach().cpu()),
            "ev_mae": float(mae.cpu()),
            "ev_rmse": float(rmse.cpu()),
            "mean_pred_ev": float(pred_ev.mean().cpu()),
            "mean_target_ev": float(target_abs.detach().mean().cpu()),
        }
    if str(target_family) == "residual_gaussian":
        target_mean = target_abs[:, :1]
        target_std = target_abs[:, 1:2].clamp_min(0.0)
        pred_std = preds["std_gain"].clamp_min(float(HURDLE_BETA_BETA_EPS))
        ev_loss = F.smooth_l1_loss(preds["expected_gain"], target_mean)
        std_loss = F.smooth_l1_loss(pred_std, target_std)
        squared_error = (target_mean - preds["expected_gain"]).pow(2) + target_std.pow(2)
        nll_raw = 0.5 * math.log(2.0 * math.pi) + torch.log(pred_std) + 0.5 * squared_error / pred_std.pow(2)
        loss = nll_raw.mean()
        pred_ev = preds["expected_gain"].detach()
        target_for_metrics = target_mean.detach()
        mae = torch.mean(torch.abs(pred_ev - target_for_metrics))
        rmse = torch.sqrt(torch.mean((pred_ev - target_for_metrics).pow(2)))
        return loss, {
            "loss": float(loss.detach().cpu()),
            "ev_loss": float(ev_loss.detach().cpu()),
            "conditional_loss": float(std_loss.detach().cpu()),
            "nll": float(loss.detach().cpu()),
            "ev_mae": float(mae.cpu()),
            "ev_rmse": float(rmse.cpu()),
            "mean_pred_ev": float(pred_ev.mean().cpu()),
            "mean_target_ev": float(target_for_metrics.mean().cpu()),
            "mean_pred_std": float(pred_std.detach().mean().cpu()),
            "mean_target_std": float(target_std.detach().mean().cpu()),
        }
    raise ValueError(f"Unknown target_family={target_family!r}")


def _avg_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    return {k: float(sum(row.get(k, 0.0) for row in rows) / len(rows)) for k in keys}


@torch.no_grad()
def evaluate(model: ResidualHeadModel, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        _loss, metrics = residual_loss(
            preds,
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
            target_family=args.target_family,
        )
        rows.append(metrics)
    model.train()
    return _avg_metrics(rows)


def _capture_state(model: ResidualHeadModel) -> Dict[str, Any]:
    return {
        "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
        "lora": {k: v.detach().cpu().clone() for k, v in model.backbone.named_parameters() if v.requires_grad},
    }


def _is_better_trained_checkpoint(*, epoch: int, metric: float, best_metric: float) -> bool:
    """Select only checkpoints produced after at least one training epoch."""
    if int(epoch) < 1:
        raise ValueError("Checkpoint selection requires a trained epoch (epoch >= 1)")
    return float(metric) < float(best_metric)


def _restore_state(model: ResidualHeadModel, state: Dict[str, Any]) -> None:
    model.head.load_state_dict(state["head"])
    with torch.no_grad():
        for k, p in model.backbone.named_parameters():
            if k in state["lora"]:
                p.copy_(state["lora"][k].to(p.device, p.dtype))


def save_residual_head_model(
    out_dir: Path,
    model: ResidualHeadModel,
    *,
    base_model: str,
    args: argparse.Namespace,
    best_metrics: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out_dir / "lora")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    meta = {
        "model_kind": "residual_regression_head",
        "base_model": str(base_model),
        "target_family": str(args.target_family),
        "target": "full_horizon_remaining_upside",
        "output_dim": int(model.output_dim),
        "top_k_seen": int(args.top_k_seen),
        "max_input_length": int(args.max_input_length),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "target_modules": str(args.target_modules),
        "objective": (
            "continuation_sample_gaussian_nll"
            if str(args.target_family) == "residual_gaussian"
            else "huber_loss_on_mean_gain"
        ),
        "gaussian_std_floor": float(args.gaussian_std_floor) if str(args.target_family) == "residual_gaussian" else None,
        "head_init_std": float(args.head_init_std),
        "head_bias_init": None if args.head_bias_init is None else float(args.head_bias_init),
        "best_metrics": best_metrics,
    }
    with (out_dir / "regression_head_config.json").open("w", encoding="utf-8") as wf:
        json.dump(meta, wf, indent=2)


def load_residual_head_model(
    model_dir: Path,
    *,
    dtype_name: str,
    device: torch.device,
    trainable: bool = False,
) -> Tuple[ResidualHeadModel, Any, Dict[str, Any]]:
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    config_path = model_dir / "regression_head_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing regression_head_config.json in {model_dir}")
    with config_path.open("r", encoding="utf-8") as rf:
        cfg = json.load(rf)
    dtype = _load_dtype(dtype_name)
    base_model = resolve_model_reference(
        os.environ.get("OPPORTUNITY_BASE_MODEL", str(cfg["base_model"]))
    )
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
    if getattr(backbone, "config", None) is not None:
        backbone.config.use_cache = False
    hidden_size = int(backbone.config.hidden_size)
    model = ResidualHeadModel(
        backbone,
        hidden_size,
        output_dim=int(cfg["output_dim"]),
        target_family=str(cfg["target_family"]),
        gaussian_std_floor=_safe_float(
            cfg.get("gaussian_std_floor"),
            DEFAULT_GAUSSIAN_STD_FLOOR,
        ),
    )
    model.head.load_state_dict(
        torch.load(
            model_dir / "head.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.to(device)
    if trainable:
        model.train()
        for p in model.head.parameters():
            p.requires_grad_(True)
    else:
        model.eval()
    return model, tok, cfg


def make_dataset(examples: Sequence[Dict[str, Any]], tokenizer, args: argparse.Namespace, *, include_meta: bool) -> ResidualHeadDataset:
    return ResidualHeadDataset(
        examples,
        tokenizer,
        target_family=str(args.target_family),
        top_k_seen=int(args.top_k_seen),
        max_input_length=int(args.max_input_length),
        include_meta=include_meta,
    )


def run_train(args: argparse.Namespace) -> None:
    from opportunity_forecasting.models.training_data import load_labeled_data

    _set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(args, out_dir)
    train_examples = load_labeled_data(Path(args.train_data))
    val_examples = load_labeled_data(Path(args.val_data)) if args.val_data else []
    if int(args.max_train_examples) > 0:
        rng = random.Random(int(args.seed))
        train_examples = rng.sample(train_examples, min(len(train_examples), int(args.max_train_examples)))
    if int(args.max_val_examples) > 0:
        val_examples = val_examples[: int(args.max_val_examples)]
    print(f"train_examples={len(train_examples)} val_examples={len(val_examples)}", flush=True)

    backbone, tok, hidden = build_backbone_and_tokenizer(
        args.base_model,
        dtype_name=args.dtype,
        use_lora=not bool(args.full_finetune),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    output_dim = 1 if str(args.target_family) == "residual_scalar" else 2
    model = ResidualHeadModel(
        backbone,
        hidden,
        output_dim=output_dim,
        target_family=str(args.target_family),
        head_init_std=float(args.head_init_std),
        head_bias_init=args.head_bias_init,
        gaussian_std_floor=float(args.gaussian_std_floor),
    ).to(device)
    train_ds = make_dataset(train_examples, tok, args, include_meta=False)
    val_ds = make_dataset(val_examples, tok, args, include_meta=False) if val_examples else None
    if not len(train_ds):
        raise ValueError("No trainable full-horizon residual rows")
    print(f"encoded_examples train={len(train_ds)}" + (f" val={len(val_ds)}" if val_ds else ""), flush=True)

    collate = lambda b: collate_residual(b, tok.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(args.eval_batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True, collate_fn=collate) if val_ds else None
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=float(args.learning_rate), betas=(0.9, 0.95), weight_decay=float(args.weight_decay))
    accum = max(1, int(args.gradient_accumulation_steps))
    step = 0
    t0 = time.time()
    best_metric = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    best_metrics: Dict[str, Any] = {}
    initial_metrics: Dict[str, Any] = {}
    opt.zero_grad(set_to_none=True)
    if val_loader is not None:
        metrics = evaluate(model, val_loader, device, args)
        initial_metrics = {"epoch": 0, "step": 0, **metrics}
        print(f"[val] epoch=0 {json.dumps(metrics, sort_keys=True)}", flush=True)
        _wandb_log(wandb_run, metrics, step=0, prefix="val", extra={"epoch": 0})
    for epoch in range(int(args.epochs)):
        window: List[Dict[str, float]] = []
        for micro, batch in enumerate(train_loader, start=1):
            batch_dev = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            preds = model(batch_dev["input_ids"], batch_dev["attention_mask"])
            loss, metrics = residual_loss(
                preds,
                batch_dev,
                target_family=args.target_family,
            )
            (loss / accum).backward()
            window.append(metrics)
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % int(args.log_every) == 0:
                    avg = _avg_metrics(window)
                    window.clear()
                    print(
                        f"[train] epoch={epoch+1} step={step} loss={avg.get('loss', 0):.4f} "
                        f"ev_mae={avg.get('ev_mae', 0):.4f} pred={avg.get('mean_pred_ev', 0):.4f} "
                        f"target={avg.get('mean_target_ev', 0):.4f} elapsed={(time.time()-t0)/60:.1f}m",
                        flush=True,
                    )
                    _wandb_log(wandb_run, avg, step=step, prefix="train", extra={"epoch": epoch + 1})
                if int(args.max_steps) > 0 and step >= int(args.max_steps):
                    break
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, args)
            metric = float(metrics.get(str(args.early_stop_metric), metrics["loss"]))
            print(f"[val] epoch={epoch+1} {json.dumps(metrics, sort_keys=True)}", flush=True)
            _wandb_log(wandb_run, metrics, step=step, prefix="val", extra={"epoch": epoch + 1})
            if _is_better_trained_checkpoint(epoch=epoch + 1, metric=metric, best_metric=best_metric):
                best_metric = metric
                best_metrics = {"epoch": epoch + 1, "step": step, **metrics}
                best_state = _capture_state(model)
                save_residual_head_model(out_dir / "best", model, base_model=str(args.base_model), args=args, best_metrics=best_metrics)
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            break
    if best_state is not None:
        _restore_state(model, best_state)
    else:
        best_metrics = {"epoch": int(args.epochs), "step": step}
    save_residual_head_model(out_dir / "final", model, base_model=str(args.base_model), args=args, best_metrics=best_metrics)
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as wf:
        json.dump(
            {
                "initial_metrics": initial_metrics,
                "best_metrics": best_metrics,
                "total_steps": step,
                "train_time_sec": time.time() - t0,
            },
            wf,
            indent=2,
        )
    _wandb_log(wandb_run, {k: v for k, v in best_metrics.items() if isinstance(v, (int, float))}, step=step, prefix="best")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved residual-head model to {out_dir / 'final'}", flush=True)


@torch.no_grad()
def predict_examples(
    model: ResidualHeadModel,
    tokenizer,
    examples: Sequence[Dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    ds = make_dataset(examples, tokenizer, args, include_meta=True)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=lambda b: collate_residual(b, tokenizer.pad_token_id))
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    model.eval()
    t0 = time.time()
    progress_every = int(getattr(args, "progress_every", 0) or 0)
    seen_rows = 0
    for batch_idx, batch in enumerate(loader, start=1):
        preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        ev = preds["expected_gain"].detach().float().cpu()
        std = preds.get("std_gain", torch.zeros_like(preds["expected_gain"])).detach().float().cpu()
        event = preds.get("event_prob", torch.zeros_like(preds["expected_gain"])).detach().float().cpu()
        for meta, ev_row, std_row, event_row in zip(batch["meta"], ev, std, event):
            is_gaussian = str(model.target_family) == "residual_gaussian"
            expected_delta = float(ev_row.reshape(-1)[0])
            expected_std_delta = (
                float(std_row.reshape(-1)[0]) if is_gaussian else 0.0
            )
            fields = {
                "forecast_family": f"explicit_{model.target_family}",
                "expected_delta": expected_delta,
                "expected_std_delta": expected_std_delta,
                "event_probability": (
                    float(event_row.reshape(-1)[0]) if is_gaussian else None
                ),
            }
            out.append((meta, fields))
        seen_rows += len(batch["meta"])
        if progress_every > 0 and (batch_idx % progress_every == 0 or seen_rows >= len(ds)):
            elapsed = max(1e-6, time.time() - t0)
            print(
                f"[predict] rows={seen_rows}/{len(ds)} batches={batch_idx}/{len(loader)} "
                f"rate={seen_rows / elapsed:.2f} rows/s",
                flush=True,
            )
    return out


def build_cache(
    *,
    examples: Sequence[Dict[str, Any]],
    predictions: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    model_dir: Path,
    output_path: Path,
    split_name: str,
    label_source: Path,
    model_label: str | None = None,
) -> Dict[str, Any]:
    grouped: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for meta, fields in predictions:
        grouped.setdefault(int(meta["goal_idx"]), []).append((meta, fields))
    model_path_for_cache = str(model_label or model_dir)
    goal_predictions: Dict[str, Any] = {}
    health = {
        "num_goals": len(grouped),
        "num_checkpoints_total": 0,
        "num_numeric_parsed": 0,
        "num_valid_forecasts": 0,
        "num_parse_failures": 0,
        "num_sigma_recovered_or_clamped": 0,
        "num_failure_examples": 0,
    }
    for gid in sorted(grouped):
        rows = sorted(grouped[gid], key=lambda x: (int(x[0].get("checkpoint_step", 0)), int(x[0].get("row_idx", 0))))
        ckpts = []
        best_so_far = 0.0
        seen_candidate = False
        for ckpt_idx, (meta, fields) in enumerate(rows):
            best_so_far = max(best_so_far, _safe_float(meta.get("best_reward_seen", 0.0), 0.0))
            seen_candidate = bool(seen_candidate or bool(meta.get("visited_product_page", False)))
            ckpts.append(
                {
                    "checkpoint_idx": int(ckpt_idx),
                    "checkpoint_step": int(meta.get("checkpoint_step", 0) or 0),
                    "candidate_available": bool(seen_candidate),
                    "best_reward_seen": float(best_so_far),
                    "forecast_family": fields["forecast_family"],
                    "expected_delta": float(fields["expected_delta"]),
                    "expected_std_delta": float(fields.get("expected_std_delta", 0.0)),
                    "event_probability": fields.get("event_probability"),
                    "forecast_numeric_domain_ok": True,
                }
            )
            health["num_checkpoints_total"] += 1
            health["num_numeric_parsed"] += 1
            health["num_valid_forecasts"] += 1
        goal_predictions[str(gid)] = {"goal_idx": int(gid), "goal_text": str(rows[0][0].get("goal_text", "")), "checkpoints": ckpts}
    payload = {
        "checkpoint_path": None,
        "label_source": str(label_source),
        "split": str(split_name),
        "goal_ids_path": None,
        "goal_ids": sorted(grouped),
        "model_path": model_path_for_cache,
        "model_type": "regression_head",
        "engine": "regression_head",
        "reward_mode": None,
        "top_k_seen": None,
        "sigma_floor": None,
        "sigma_support": "",
        "health": health,
        "goal_predictions": goal_predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as wf:
        json.dump(payload, wf)
    return payload


def run_predict(args: argparse.Namespace) -> None:
    from opportunity_forecasting.models.training_data import load_labeled_data

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(
        f"[predict] start model_dir={args.model_dir} data={args.data} output={args.output_path} "
        f"split={args.split} batch_size={args.batch_size} device={device}",
        flush=True,
    )
    model, tok, cfg = load_residual_head_model(Path(args.model_dir), dtype_name=str(args.dtype), device=device)
    print(
        f"[predict] loaded model target_family={cfg.get('target_family')} "
        f"max_input_length={cfg.get('max_input_length')}",
        flush=True,
    )
    args.target_family = str(cfg["target_family"])
    args.top_k_seen = int(args.top_k_seen if args.top_k_seen is not None else cfg.get("top_k_seen", 15))
    args.max_input_length = int(args.max_input_length if args.max_input_length is not None else cfg.get("max_input_length", MAX_INPUT_LENGTH))
    examples = load_labeled_data(Path(args.data))
    if int(args.max_examples) > 0:
        examples = examples[: int(args.max_examples)]
    if args.shard_idx is not None or args.num_shards is not None:
        if args.shard_idx is None or args.num_shards is None:
            raise ValueError("Set both --shard_idx and --num_shards.")
        if bool(args.shard_by_goal):
            examples = [
                ex
                for ex in examples
                if _example_goal_id(ex) % int(args.num_shards) == int(args.shard_idx)
            ]
        else:
            examples = [ex for i, ex in enumerate(examples) if i % int(args.num_shards) == int(args.shard_idx)]
    print(f"[predict] examples_after_filter={len(examples)}", flush=True)
    preds = predict_examples(model, tok, examples, args=args, device=device)
    build_cache(
        examples=examples,
        predictions=preds,
        model_dir=Path(args.model_dir),
        output_path=Path(args.output_path),
        split_name=str(args.split),
        label_source=Path(args.data),
        model_label=args.model_label,
    )
    print(f"Wrote residual-head cache to {args.output_path}", flush=True)


def add_common_train_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", default="")
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--target_family",
        default="residual_scalar",
        choices=["residual_scalar", "residual_gaussian"],
    )
    ap.add_argument("--top_k_seen", type=int, default=15)
    ap.add_argument("--max_input_length", type=int, default=MAX_INPUT_LENGTH)
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
    ap.add_argument("--head_init_std", type=float, default=0.02)
    ap.add_argument("--head_bias_init", type=float, default=None)
    ap.add_argument("--gaussian_std_floor", type=float, default=DEFAULT_GAUSSIAN_STD_FLOOR)
    ap.add_argument(
        "--early_stop_metric",
        default="ev_mae",
        choices=["loss", "ev_loss", "ev_rmse", "ev_mae"],
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
    train_ap = sub.add_parser("train")
    add_common_train_args(train_ap)
    pred_ap = sub.add_parser("predict-labels")
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
    pred_ap.add_argument("--shard_by_goal", action="store_true")
    pred_ap.add_argument("--progress_every", type=int, default=0)
    pred_ap.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["bf16", "fp16", "fp32", "auto"])
    pred_ap.add_argument("--cpu", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "predict-labels":
        run_predict(args)
    else:
        raise ValueError(f"Unknown command {args.cmd!r}")


if __name__ == "__main__":
    main()
