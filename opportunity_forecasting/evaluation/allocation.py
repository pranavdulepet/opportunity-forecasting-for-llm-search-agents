"""
Build budgeted-expansion frontiers from cached decision-point forecasts.

This is a complementary evaluation to the global stop-threshold Pareto sweep.
Instead of selecting one threshold and replaying each search independently, we
simulate a fixed pool of active search threads. Each thread starts once it has
opened at least one candidate page. A scheduler repeatedly chooses one thread to
expand to its next decision point; the x-axis is the mean final step cost if all
threads were stopped at their current decision point, and the y-axis is the mean
best reward currently seen.

The forecast scheduler ranks active threads by predicted full-horizon remaining
upside at the current decision point. Baselines:
- fixed: static round-robin expansion by goal id
- random: average over random active-thread choices
- fixed-replay hindsight upper bound: dynamic-programming upper bound over each
  thread's true decision-point reward/cost options
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from opportunity_forecasting import REPO_ROOT
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-budgeted-expansion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-budgeted-expansion")

from opportunity_forecasting.models.distributions import forecast_from_fields, forecast_implied_moments
from opportunity_forecasting.figures.style import (
    BASE_PROMPT,
    GRID,
    NEUTRAL,
    ORACLE,
    REGRESSION_HEAD,
    apply_paper_style,
)


DEFAULT_RANDOM_SEEDS = 64
ORACLE_SERIES = "Fixed-replay hindsight upper bound"


@dataclass(frozen=True)
class CacheSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class ThreadCheckpoint:
    goal_id: int
    checkpoint_idx: int
    checkpoint_step: int
    stop_step_cost: int
    best_reward_seen: float
    predicted_mean_delta: Optional[float]
    forecast_valid: bool
    num_seen_candidates: int = 0
    num_opened_candidates: int = 0
    best_retrieved_rank_score: Optional[float] = None
    stagnation_count: int = 0


@dataclass(frozen=True)
class CurvePoint:
    domain: str
    split: str
    series: str
    kind: str
    mean_final_steps: float
    mean_final_reward: float
    expansion_count: int


@dataclass(frozen=True)
class BootstrapBandPoint:
    series: str
    kind: str
    mean_final_steps: float
    reward_p025: float
    reward_p500: float
    reward_p975: float
    num_bootstrap_values: int


@dataclass(frozen=True)
class BootstrapOracleGapBandPoint:
    series: str
    kind: str
    mean_final_steps: float
    gap_p025: float
    gap_p500: float
    gap_p975: float
    num_bootstrap_values: int


def _parse_cache_spec(raw: str) -> CacheSpec:
    if "=" not in raw:
        path = Path(raw)
        return CacheSpec(label=path.stem.replace("_predictions", ""), path=path)
    label, path = raw.rsplit("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Bad --cache spec {raw!r}; expected LABEL=PATH.")
    return CacheSpec(label=label, path=Path(path))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(val):
        return float(default)
    return float(val)


def _load_cache(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_checkpoint_metadata_by_goal(cache: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Load richer checkpoint rows referenced by a prediction cache when present."""
    raw_path = cache.get("checkpoint_path")
    if not raw_path:
        return {}
    path = Path(str(raw_path))
    if not path.exists():
        return {}
    out: Dict[int, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            goal_id = row.get("goal_idx", row.get("goal_id"))
            if goal_id is None:
                continue
            out.setdefault(int(goal_id), []).append(row)
    return out


def _visited_candidate_page(ckpt: Dict[str, Any]) -> bool:
    if "candidate_available" in ckpt:
        return bool(ckpt.get("candidate_available", False))
    if "visited_product_page" in ckpt:
        return bool(ckpt.get("visited_product_page", False))
    trigger = str(ckpt.get("trigger", "") or "").lower()
    if trigger in {"product_page", "paper_page", "item_page"}:
        return True
    obs = str(ckpt.get("observation", "") or "").lower()
    return ("buy now" in obs) or ("current_paper_id:" in obs) or ("paper page" in obs)


def _stop_step_cost(ckpt: Dict[str, Any], *, offline_back_steps: int, offline_to_product_steps: int) -> int:
    checkpoint_step = int(ckpt.get("checkpoint_step", 0) or 0)
    spent_search_steps = max(0, checkpoint_step - 1)
    return int(spent_search_steps + max(0, offline_back_steps) + max(0, offline_to_product_steps))


def _predicted_mean_delta(ckpt: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    explicit = ckpt.get("expected_delta")
    if explicit is not None:
        val = _safe_float(explicit, default=float("nan"))
        if math.isfinite(val):
            return float(max(0.0, val)), bool(ckpt.get("forecast_numeric_domain_ok", True))
    fields = {
        "forecast_family": ckpt.get("forecast_family"),
        "delta_zero_prob": ckpt.get("delta_zero_prob"),
        "delta_one_prob": ckpt.get("delta_one_prob"),
        "delta_pos_mean": ckpt.get("delta_pos_mean"),
        "delta_pos_concentration": ckpt.get("delta_pos_concentration"),
    }
    try:
        forecast = forecast_from_fields(fields)
        mean_delta, _ = forecast_implied_moments(forecast)
    except Exception:
        return None, False
    if mean_delta is None or not math.isfinite(float(mean_delta)):
        return None, False
    return float(max(0.0, mean_delta)), True


def _candidate_count(meta: Dict[str, Any], ckpt: Dict[str, Any]) -> int:
    for key in ("num_seen_papers", "num_seen_products", "num_seen_candidates"):
        if key in meta:
            return int(meta.get(key) or 0)
        if key in ckpt:
            return int(ckpt.get(key) or 0)
    seen = meta.get("seen_products", ckpt.get("seen_products", {}))
    if isinstance(seen, dict):
        return len(seen)
    if isinstance(seen, list):
        return len(seen)
    return 0


def _opened_candidate_count(meta: Dict[str, Any], ckpt: Dict[str, Any]) -> int:
    for key in ("num_opened_papers", "num_opened_products", "num_opened_candidates"):
        if key in meta:
            return int(meta.get(key) or 0)
        if key in ckpt:
            return int(ckpt.get(key) or 0)
    opened = meta.get("opened_paper_ids", ckpt.get("opened_paper_ids", []))
    if isinstance(opened, list):
        return len(opened)
    actions = meta.get("prefix_actions", ckpt.get("prefix_actions", []))
    if isinstance(actions, list):
        return sum(1 for action in actions if str(action).lower().startswith("click["))
    return 0


def _best_retrieved_rank_score(meta: Dict[str, Any], ckpt: Dict[str, Any]) -> Optional[float]:
    seen = meta.get("seen_products", ckpt.get("seen_products", {}))
    ranks: List[float] = []
    if isinstance(seen, dict):
        values = seen.values()
    elif isinstance(seen, list):
        values = seen
    else:
        values = []
    for val in values:
        if not isinstance(val, dict):
            continue
        rank = val.get("FirstSeenRank", val.get("first_seen_rank", val.get("rank")))
        if rank is None:
            continue
        rank_f = _safe_float(rank, default=float("nan"))
        if math.isfinite(rank_f) and rank_f >= 0:
            ranks.append(rank_f)
    if not ranks:
        return None

    return 1.0 / (1.0 + min(ranks))


def _goal_ids_in_cache(cache: Dict[str, Any]) -> List[int]:
    return [int(x) for x in cache.get("goal_ids", [])]


def shared_goal_ids(caches: Sequence[Dict[str, Any]]) -> List[int]:
    if not caches:
        return []
    ordered = _goal_ids_in_cache(caches[0])
    expected = set(ordered)
    for index, cache in enumerate(caches):
        ids = _goal_ids_in_cache(cache)
        prediction_ids = {
            int(x) for x in (cache.get("goal_predictions", {}) or {}).keys()
        }
        if ids != ordered or prediction_ids != expected:
            raise ValueError(
                f"Prediction cache {index} does not use the shared ordered goal set"
            )
    return ordered


def extract_threads(
    cache: Dict[str, Any],
    goal_ids: Sequence[int],
    *,
    offline_back_steps: int,
    offline_to_product_steps: int,
    metadata_by_goal: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> List[List[ThreadCheckpoint]]:
    """Return schedulable decision-point streams, aligned to stop-policy replay.

    A stream starts at the first decision point where a candidate page has been
    visited. After that point, every later decision point is schedulable because the
    replay policy has already unlocked stopping/committing to the best seen item.
    """
    out: List[List[ThreadCheckpoint]] = []
    blobs = cache.get("goal_predictions", {}) or {}
    metadata_by_goal = metadata_by_goal or {}
    for goal_id in goal_ids:
        raw_ckpts = (blobs.get(str(int(goal_id))) or {}).get("checkpoints", []) or []
        meta_rows = metadata_by_goal.get(int(goal_id), [])
        first_schedulable: Optional[int] = None
        for idx, ckpt in enumerate(raw_ckpts):
            if _visited_candidate_page(ckpt):
                first_schedulable = idx
                break
        if first_schedulable is None:
            continue

        stream: List[ThreadCheckpoint] = []
        last_step_cost = -1
        for idx, ckpt in enumerate(raw_ckpts[first_schedulable:], start=first_schedulable):
            meta = meta_rows[idx] if idx < len(meta_rows) else {}
            step_cost = _stop_step_cost(
                ckpt,
                offline_back_steps=offline_back_steps,
                offline_to_product_steps=offline_to_product_steps,
            )
            if step_cost < last_step_cost:
                continue
            last_step_cost = step_cost
            mean_delta, valid = _predicted_mean_delta(ckpt)
            stream.append(
                ThreadCheckpoint(
                    goal_id=int(goal_id),
                    checkpoint_idx=int(idx),
                    checkpoint_step=int(ckpt.get("checkpoint_step", 0) or 0),
                    stop_step_cost=int(step_cost),
                    best_reward_seen=_safe_float(ckpt.get("best_reward_seen", 0.0)),
                    predicted_mean_delta=mean_delta,
                    forecast_valid=bool(valid),
                    num_seen_candidates=_candidate_count(meta, ckpt),
                    num_opened_candidates=_opened_candidate_count(meta, ckpt),
                    best_retrieved_rank_score=_best_retrieved_rank_score(meta, ckpt),
                )
            )
        if stream:
            with_stagnation: List[ThreadCheckpoint] = []
            best_so_far = -float("inf")
            stale = 0
            for ckpt in stream:
                if ckpt.best_reward_seen > best_so_far + 1e-12:
                    stale = 0
                    best_so_far = ckpt.best_reward_seen
                else:
                    stale += 1
                with_stagnation.append(replace(ckpt, stagnation_count=stale))
            out.append(with_stagnation)
    return out


def _state_to_point(
    *,
    domain: str,
    split: str,
    series: str,
    kind: str,
    streams: Sequence[Sequence[ThreadCheckpoint]],
    indices: Sequence[int],
    expansion_count: int,
) -> CurvePoint:
    n = max(1, len(streams))
    total_steps = sum(streams[i][indices[i]].stop_step_cost for i in range(len(streams)))
    total_reward = sum(streams[i][indices[i]].best_reward_seen for i in range(len(streams)))
    return CurvePoint(
        domain=domain,
        split=split,
        series=series,
        kind=kind,
        mean_final_steps=float(total_steps) / float(n),
        mean_final_reward=float(total_reward) / float(n),
        expansion_count=int(expansion_count),
    )


def _totals_to_point(
    *,
    domain: str,
    split: str,
    series: str,
    kind: str,
    n: int,
    total_steps: float,
    total_reward: float,
    expansion_count: int,
) -> CurvePoint:
    denom = float(max(1, int(n)))
    return CurvePoint(
        domain=domain,
        split=split,
        series=series,
        kind=kind,
        mean_final_steps=float(total_steps) / denom,
        mean_final_reward=float(total_reward) / denom,
        expansion_count=int(expansion_count),
    )


def _priority_for_stream(
    stream: Sequence[ThreadCheckpoint],
    idx: int,
    *,
    priority_mode: str,
) -> float:
    ckpt = stream[idx]
    pred = ckpt.predicted_mean_delta
    if pred is None:
        pred = 0.0
    if priority_mode == "predicted_mean_delta":
        return float(pred)
    if priority_mode == "predicted_mean_delta_per_step":
        if idx >= len(stream) - 1:
            return float("-inf")
        step_gap = max(1, stream[idx + 1].stop_step_cost - ckpt.stop_step_cost)
        return float(pred) / float(step_gap)
    raise ValueError(f"Unknown priority_mode: {priority_mode}")


def forecast_greedy_curve(
    *,
    domain: str,
    split: str,
    label: str,
    streams: Sequence[Sequence[ThreadCheckpoint]],
    priority_mode: str,
) -> List[CurvePoint]:
    if not streams:
        return []
    indices = [0 for _ in streams]
    total_steps = sum(float(stream[0].stop_step_cost) for stream in streams)
    total_reward = sum(float(stream[0].best_reward_seen) for stream in streams)
    points = [
        _totals_to_point(
            domain=domain,
            split=split,
            series=label,
            kind="forecast_expected_remaining",
            n=len(streams),
            total_steps=total_steps,
            total_reward=total_reward,
            expansion_count=0,
        )
    ]
    heap: List[Tuple[float, int, int]] = []
    for i, stream in enumerate(streams):
        if len(stream) > 1:
            priority = _priority_for_stream(stream, 0, priority_mode=priority_mode)
            heapq.heappush(heap, (-priority, int(stream[0].goal_id), i))

    expansion_count = 0
    while heap:
        _, _, stream_idx = heapq.heappop(heap)
        if indices[stream_idx] >= len(streams[stream_idx]) - 1:
            continue
        old = streams[stream_idx][indices[stream_idx]]
        indices[stream_idx] += 1
        new = streams[stream_idx][indices[stream_idx]]
        total_steps += float(new.stop_step_cost) - float(old.stop_step_cost)
        total_reward += float(new.best_reward_seen) - float(old.best_reward_seen)
        expansion_count += 1
        points.append(
            _totals_to_point(
                domain=domain,
                split=split,
                series=label,
                kind="forecast_expected_remaining",
                n=len(streams),
                total_steps=total_steps,
                total_reward=total_reward,
                expansion_count=expansion_count,
            )
        )
        if indices[stream_idx] < len(streams[stream_idx]) - 1:
            priority = _priority_for_stream(
                streams[stream_idx],
                indices[stream_idx],
                priority_mode=priority_mode,
            )
            heapq.heappush(heap, (-priority, int(streams[stream_idx][0].goal_id), stream_idx))
    return points


HEURISTIC_LABELS = {
    "step_early": "Heuristic: earliest step",
    "low_current_best": "Heuristic: low current best",
    "few_seen": "Heuristic: few seen candidates",
    "low_stagnation": "Heuristic: low stagnation",
    "retrieved_rank": "Heuristic: best retrieved rank",
}


def _heuristic_priority(stream: Sequence[ThreadCheckpoint], idx: int, mode: str) -> float:
    ckpt = stream[idx]
    if mode == "step_early":
        return -float(ckpt.stop_step_cost)
    if mode == "low_current_best":
        return 1.0 - float(ckpt.best_reward_seen)
    if mode == "few_seen":
        return -float(ckpt.num_seen_candidates)
    if mode == "low_stagnation":
        return -float(ckpt.stagnation_count)
    if mode == "retrieved_rank":
        if ckpt.best_retrieved_rank_score is None:
            return float("-inf")
        return float(ckpt.best_retrieved_rank_score)
    raise ValueError(f"Unknown heuristic mode: {mode}")


def heuristic_greedy_curve(
    *,
    domain: str,
    split: str,
    streams: Sequence[Sequence[ThreadCheckpoint]],
    mode: str,
) -> List[CurvePoint]:
    if not streams:
        return []
    label = HEURISTIC_LABELS[mode]
    indices = [0 for _ in streams]
    total_steps = sum(float(stream[0].stop_step_cost) for stream in streams)
    total_reward = sum(float(stream[0].best_reward_seen) for stream in streams)
    points = [
        _totals_to_point(
            domain=domain,
            split=split,
            series=label,
            kind=f"heuristic_{mode}",
            n=len(streams),
            total_steps=total_steps,
            total_reward=total_reward,
            expansion_count=0,
        )
    ]
    heap: List[Tuple[float, int, int]] = []
    for i, stream in enumerate(streams):
        if len(stream) > 1:
            priority = _heuristic_priority(stream, 0, mode)
            heapq.heappush(heap, (-priority, int(stream[0].goal_id), i))

    expansion_count = 0
    while heap:
        _, _, stream_idx = heapq.heappop(heap)
        if indices[stream_idx] >= len(streams[stream_idx]) - 1:
            continue
        old = streams[stream_idx][indices[stream_idx]]
        indices[stream_idx] += 1
        new = streams[stream_idx][indices[stream_idx]]
        total_steps += float(new.stop_step_cost) - float(old.stop_step_cost)
        total_reward += float(new.best_reward_seen) - float(old.best_reward_seen)
        expansion_count += 1
        points.append(
            _totals_to_point(
                domain=domain,
                split=split,
                series=label,
                kind=f"heuristic_{mode}",
                n=len(streams),
                total_steps=total_steps,
                total_reward=total_reward,
                expansion_count=expansion_count,
            )
        )
        if indices[stream_idx] < len(streams[stream_idx]) - 1:
            priority = _heuristic_priority(streams[stream_idx], indices[stream_idx], mode)
            heapq.heappush(heap, (-priority, int(streams[stream_idx][0].goal_id), stream_idx))
    return points


def fixed_round_robin_curve(
    *,
    domain: str,
    split: str,
    streams: Sequence[Sequence[ThreadCheckpoint]],
) -> List[CurvePoint]:
    if not streams:
        return []
    order = sorted(range(len(streams)), key=lambda i: int(streams[i][0].goal_id))
    indices = [0 for _ in streams]
    total_steps = sum(float(stream[0].stop_step_cost) for stream in streams)
    total_reward = sum(float(stream[0].best_reward_seen) for stream in streams)
    points = [
        _totals_to_point(
            domain=domain,
            split=split,
            series="Fixed round-robin",
            kind="fixed_baseline",
            n=len(streams),
            total_steps=total_steps,
            total_reward=total_reward,
            expansion_count=0,
        )
    ]
    expansion_count = 0
    while True:
        progressed = False
        for i in order:
            if indices[i] >= len(streams[i]) - 1:
                continue
            old = streams[i][indices[i]]
            indices[i] += 1
            new = streams[i][indices[i]]
            total_steps += float(new.stop_step_cost) - float(old.stop_step_cost)
            total_reward += float(new.best_reward_seen) - float(old.best_reward_seen)
            expansion_count += 1
            progressed = True
            points.append(
                _totals_to_point(
                    domain=domain,
                    split=split,
                    series="Fixed round-robin",
                    kind="fixed_baseline",
                    n=len(streams),
                    total_steps=total_steps,
                    total_reward=total_reward,
                    expansion_count=expansion_count,
                )
            )
        if not progressed:
            break
    return points


def random_curve_once(
    *,
    domain: str,
    split: str,
    streams: Sequence[Sequence[ThreadCheckpoint]],
    seed: int,
) -> List[CurvePoint]:
    rng = random.Random(int(seed))
    if not streams:
        return []
    indices = [0 for _ in streams]
    active = [i for i, stream in enumerate(streams) if len(stream) > 1]
    total_steps = sum(float(stream[0].stop_step_cost) for stream in streams)
    total_reward = sum(float(stream[0].best_reward_seen) for stream in streams)
    points = [
        _totals_to_point(
            domain=domain,
            split=split,
            series=f"Random seed {seed}",
            kind="random_single_seed",
            n=len(streams),
            total_steps=total_steps,
            total_reward=total_reward,
            expansion_count=0,
        )
    ]
    expansion_count = 0
    while active:
        active_pos = rng.randrange(len(active))
        i = active[active_pos]
        old = streams[i][indices[i]]
        indices[i] += 1
        new = streams[i][indices[i]]
        total_steps += float(new.stop_step_cost) - float(old.stop_step_cost)
        total_reward += float(new.best_reward_seen) - float(old.best_reward_seen)
        expansion_count += 1
        points.append(
            _totals_to_point(
                domain=domain,
                split=split,
                series=f"Random seed {seed}",
                kind="random_single_seed",
                n=len(streams),
                total_steps=total_steps,
                total_reward=total_reward,
                expansion_count=expansion_count,
            )
        )
        if indices[i] >= len(streams[i]) - 1:
            active.pop(active_pos)
    return points


def _make_budget_grid(points: Sequence[CurvePoint], *, step: float) -> List[float]:
    if not points:
        return []
    lo = min(float(p.mean_final_steps) for p in points)
    hi = max(float(p.mean_final_steps) for p in points)
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    out: List[float] = []
    cur = start
    while cur <= end + step / 2.0:
        out.append(round(cur, 10))
        cur += step
    return out


def _reward_at_budget(points: Sequence[CurvePoint], budget: float) -> Optional[float]:
    feasible = [p.mean_final_reward for p in points if p.mean_final_steps <= budget + 1e-12]
    if not feasible:
        return None
    return float(max(feasible))


def _gap_to_oracle_at_budget(
    *,
    model_points: Sequence[CurvePoint],
    oracle_points: Sequence[CurvePoint],
    budget: float,
) -> Optional[float]:
    model_reward = _reward_at_budget(model_points, budget)
    oracle_reward = _reward_at_budget(oracle_points, budget)
    if model_reward is None or oracle_reward is None:
        return None
    gap = float(model_reward) - float(oracle_reward)


    if 0.0 < gap < 1e-10:
        gap = 0.0
    return gap


def build_oracle_gap_points(
    points_by_series: Dict[str, List[CurvePoint]],
    *,
    budget_step: float,
) -> Dict[str, List[CurvePoint]]:
    """Convert reward frontiers into model-minus-oracle gap curves.

    The oracle and the model are compared at the same mean-step budget using the
    reward attainable by each curve at or before that budget. This is the same
    budget interpolation used for the summary table and avoids comparing a model
    bootstrap sample against an unrelated fixed oracle curve.
    """
    oracle_points = points_by_series.get(ORACLE_SERIES, [])
    if not oracle_points:
        return {}
    all_points = [p for points in points_by_series.values() for p in points]
    budget_grid = _make_budget_grid(all_points, step=float(budget_step))
    out: Dict[str, List[CurvePoint]] = {}
    for label, points in points_by_series.items():
        if not points:
            continue
        kind = points[0].kind
        gap_points: List[CurvePoint] = []
        for budget in budget_grid:
            if label == ORACLE_SERIES:
                gap = 0.0 if _reward_at_budget(oracle_points, budget) is not None else None
            else:
                gap = _gap_to_oracle_at_budget(
                    model_points=points,
                    oracle_points=oracle_points,
                    budget=float(budget),
                )
            if gap is None:
                continue
            gap_points.append(
                CurvePoint(
                    domain=points[0].domain,
                    split=points[0].split,
                    series=label,
                    kind=kind,
                    mean_final_steps=float(budget),
                    mean_final_reward=float(gap),
                    expansion_count=-1,
                )
            )
        if gap_points:
            out[label] = gap_points
    return out


def averaged_random_curve(
    *,
    domain: str,
    split: str,
    streams: Sequence[Sequence[ThreadCheckpoint]],
    seeds: int,
    budget_step: float,
) -> List[CurvePoint]:
    seed_curves = [
        random_curve_once(domain=domain, split=split, streams=streams, seed=i)
        for i in range(int(seeds))
    ]
    all_points = [p for curve in seed_curves for p in curve]
    grid = _make_budget_grid(all_points, step=float(budget_step))
    averaged: List[CurvePoint] = []
    for budget in grid:
        vals = [_reward_at_budget(curve, budget) for curve in seed_curves]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        averaged.append(
            CurvePoint(
                domain=domain,
                split=split,
                series=f"Random avg ({seeds} seeds)",
                kind="random_baseline",
                mean_final_steps=float(budget),
                mean_final_reward=sum(vals) / float(len(vals)),
                expansion_count=-1,
            )
        )
    return averaged


def oracle_frontier(
    *,
    domain: str,
    split: str,
    streams: Sequence[Sequence[ThreadCheckpoint]],
) -> List[CurvePoint]:
    if not streams:
        return []

    def pareto_states(step_to_reward: Dict[int, float]) -> List[Tuple[int, float]]:
        states: List[Tuple[int, float]] = []
        best_reward = -1e18
        for steps, reward in sorted(step_to_reward.items()):
            if float(reward) > best_reward + 1e-12:
                states.append((int(steps), float(reward)))
                best_reward = float(reward)
        return states

    states: List[Tuple[int, float]] = [(0, 0.0)]
    for stream in streams:
        opts_by_step: Dict[int, float] = {}
        for ckpt in stream:
            opts_by_step[int(ckpt.stop_step_cost)] = max(
                float(ckpt.best_reward_seen),
                opts_by_step.get(int(ckpt.stop_step_cost), -1e18),
            )
        opts = pareto_states(opts_by_step)
        next_by_step: Dict[int, float] = {}
        for total_steps, total_reward in states:
            for step_cost, reward in opts:
                ns = int(total_steps) + int(step_cost)
                nr = float(total_reward) + float(reward)
                if nr > next_by_step.get(ns, -1e18):
                    next_by_step[ns] = nr
        states = pareto_states(next_by_step)

    n = float(len(streams))
    points: List[CurvePoint] = []
    for total_steps, total_reward in states:
        points.append(
            CurvePoint(
                domain=domain,
                split=split,
                series=ORACLE_SERIES,
                kind="oracle",
                mean_final_steps=float(total_steps) / n,
                mean_final_reward=float(total_reward) / n,
                expansion_count=-1,
            )
        )
    return points


def validate_shared_reward_streams(
    streams_by_label: Dict[str, List[List[ThreadCheckpoint]]],
) -> Dict[str, List[List[ThreadCheckpoint]]]:
    """Require identical fixed replay streams for every forecast method."""
    if not streams_by_label:
        return {}
    labels = list(streams_by_label.keys())
    reference_label = labels[0]
    reference = streams_by_label[reference_label]
    reference_signatures = [
        [
            (
                point.goal_id,
                point.checkpoint_step,
                point.stop_step_cost,
                round(point.best_reward_seen, 12),
            )
            for point in stream
        ]
        for stream in reference
    ]
    for label in labels[1:]:
        candidate = streams_by_label[label]
        signatures = [
            [
                (
                    point.goal_id,
                    point.checkpoint_step,
                    point.stop_step_cost,
                    round(point.best_reward_seen, 12),
                )
                for point in stream
            ]
            for stream in candidate
        ]
        if signatures != reference_signatures:
            raise ValueError(
                f"{label!r} does not use the same fixed replay reward stream as "
                f"{reference_label!r}"
            )
    return streams_by_label


def write_curve_csv(path: Path, points: Sequence[CurvePoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "domain",
        "split",
        "series",
        "kind",
        "mean_final_steps",
        "mean_final_reward",
        "expansion_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "domain": p.domain,
                    "split": p.split,
                    "series": p.series,
                    "kind": p.kind,
                    "mean_final_steps": f"{p.mean_final_steps:.10g}",
                    "mean_final_reward": f"{p.mean_final_reward:.10g}",
                    "expansion_count": p.expansion_count,
                }
            )


def _summarize_curve(points: Sequence[CurvePoint]) -> Dict[str, Any]:
    if not points:
        return {}
    return {
        "num_points": len(points),
        "min_steps": min(p.mean_final_steps for p in points),
        "max_steps": max(p.mean_final_steps for p in points),
        "start_reward": points[0].mean_final_reward,
        "max_reward": max(p.mean_final_reward for p in points),
    }


def _series_style(label: str, model_color_idx: int) -> Tuple[Dict[str, Any], int]:
    if label == ORACLE_SERIES:
        return (
            {"color": ORACLE, "linestyle": "--", "linewidth": 2.2, "zorder": 5},
            model_color_idx,
        )
    if label == "Base Prompt":
        return (
            {"color": BASE_PROMPT, "linestyle": "-", "linewidth": 2.0, "zorder": 3},
            model_color_idx,
        )
    if label == "Regression head":
        return (
            {"color": REGRESSION_HEAD, "linestyle": "-", "linewidth": 2.2, "zorder": 4},
            model_color_idx,
        )
    if label == "SFT Hurdle-Beta":
        return (
            {"color": "#2a9d8f", "linestyle": "-", "linewidth": 2.2, "zorder": 3},
            model_color_idx,
        )
    if label.startswith("Random"):
        return (
            {"color": "#A2AAB6", "linestyle": ":", "linewidth": 1.8, "zorder": 1},
            model_color_idx,
        )
    if label.startswith("Fixed"):
        return (
            {"color": "#8a7a43", "linestyle": "-.", "linewidth": 2.2, "zorder": 2},
            model_color_idx,
        )
    if label.startswith("Heuristic"):
        heuristic_colors = {
            "Heuristic: earliest step": "#6b7280",
            "Heuristic: low current best": "#b45309",
            "Heuristic: few seen candidates": "#7c3aed",
            "Heuristic: low stagnation": "#be123c",
            "Heuristic: best retrieved rank": "#475569",
        }
        return (
            {
                "color": heuristic_colors.get(label, NEUTRAL),
                "linestyle": "-.",
                "linewidth": 1.7,
                "alpha": 0.82,
                "zorder": 2,
            },
            model_color_idx,
        )
    model_colors = [BASE_PROMPT, "#2A9D8F", REGRESSION_HEAD, "#6C5CE7", "#7F5539", "#0F766E"]
    style = {
        "color": model_colors[model_color_idx % len(model_colors)],
        "linestyle": "-",
        "linewidth": 2.1,
        "zorder": 3,
    }
    return style, model_color_idx + 1


def _extend_curve_to_step(points: List[CurvePoint], max_step: float) -> None:
    if not points:
        return
    last = points[-1]
    if float(last.mean_final_steps) >= float(max_step) - 1e-12:
        return
    points.append(
        CurvePoint(
            domain=last.domain,
            split=last.split,
            series=last.series,
            kind=last.kind,
            mean_final_steps=float(max_step),
            mean_final_reward=float(last.mean_final_reward),
            expansion_count=last.expansion_count,
        )
    )


def compute_points_by_series(
    *,
    domain: str,
    split: str,
    cache_specs: Sequence[CacheSpec],
    streams_by_label: Dict[str, List[List[ThreadCheckpoint]]],
    random_seeds: int,
    budget_step: float,
    priority_mode: str,
    include_fixed_baseline: bool,
    heuristic_baselines: Sequence[str],
    include_oracle: bool = True,
    include_random: bool = True,
) -> Dict[str, List[CurvePoint]]:
    points_by_series: Dict[str, List[CurvePoint]] = {}
    if not cache_specs:
        return points_by_series

    base_streams = streams_by_label[cache_specs[0].label]
    baseline_points = {}
    if include_oracle:
        baseline_points[ORACLE_SERIES] = oracle_frontier(
            domain=domain,
            split=split,
            streams=base_streams,
        )
    if include_random:
        baseline_points[f"Random avg ({random_seeds} seeds)"] = averaged_random_curve(
            domain=domain,
            split=split,
            streams=base_streams,
            seeds=random_seeds,
            budget_step=budget_step,
        )
    if include_fixed_baseline:
        baseline_points["Fixed round-robin"] = fixed_round_robin_curve(
            domain=domain,
            split=split,
            streams=base_streams,
        )
    for mode in heuristic_baselines:
        if mode == "retrieved_rank" and all(
            ckpt.best_retrieved_rank_score is None for stream in base_streams for ckpt in stream
        ):
            continue
        baseline_points[HEURISTIC_LABELS[mode]] = heuristic_greedy_curve(
            domain=domain,
            split=split,
            streams=base_streams,
            mode=mode,
        )
    points_by_series.update(baseline_points)

    for spec in cache_specs:
        streams = streams_by_label[spec.label]
        points_by_series[spec.label] = forecast_greedy_curve(
            domain=domain,
            split=split,
            label=spec.label,
            streams=streams,
            priority_mode=priority_mode,
        )

    all_points = [p for points in points_by_series.values() for p in points]
    max_step = max((p.mean_final_steps for p in all_points), default=0.0)
    if include_oracle:
        oracle_points = points_by_series.get(ORACLE_SERIES, [])
        _extend_curve_to_step(oracle_points, max_step)
    return points_by_series


def _resample_streams_by_label(
    streams_by_label: Dict[str, List[List[ThreadCheckpoint]]],
    sample_indices: Sequence[int],
) -> Dict[str, List[List[ThreadCheckpoint]]]:
    return {
        label: [streams[int(i)] for i in sample_indices]
        for label, streams in streams_by_label.items()
    }


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def build_bootstrap_bands(
    *,
    domain: str,
    split: str,
    cache_specs: Sequence[CacheSpec],
    streams_by_label: Dict[str, List[List[ThreadCheckpoint]]],
    original_points_by_series: Dict[str, List[CurvePoint]],
    random_seeds: int,
    budget_step: float,
    priority_mode: str,
    include_fixed_baseline: bool,
    heuristic_baselines: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, List[BootstrapBandPoint]]:
    """Percentile bootstrap bands over held-out tasks.

    The resampling unit is the search task/thread. Each bootstrap draw samples
    whole streams with replacement and applies the same sampled task indices to
    every scheduler, preserving paired comparisons across models.
    """
    if int(bootstrap_samples) <= 0 or not cache_specs:
        return {}
    n = len(streams_by_label[cache_specs[0].label])
    if n <= 1:
        return {}

    all_original = [p for points in original_points_by_series.values() for p in points]
    budget_grid = _make_budget_grid(all_original, step=float(budget_step))
    if not budget_grid:
        return {}

    values: Dict[str, Dict[float, List[float]]] = {
        label: {float(budget): [] for budget in budget_grid}
        for label in original_points_by_series
        if label != ORACLE_SERIES
        and not label.startswith("Random")
        and not label.startswith("Fixed")
        and not label.startswith("Heuristic")
    }
    rng = random.Random(int(bootstrap_seed))
    for _ in range(int(bootstrap_samples)):
        sample_indices = [rng.randrange(n) for _ in range(n)]
        boot_streams = _resample_streams_by_label(streams_by_label, sample_indices)
        boot_points = compute_points_by_series(
            domain=domain,
            split=split,
            cache_specs=cache_specs,
            streams_by_label=boot_streams,
            random_seeds=random_seeds,
            budget_step=budget_step,
            priority_mode=priority_mode,
            include_fixed_baseline=False,
            heuristic_baselines=[],
            include_oracle=False,
            include_random=False,
        )
        for label, grid_values in values.items():
            points = boot_points.get(label, [])
            for budget in budget_grid:
                reward = _reward_at_budget(points, float(budget))
                if reward is not None:
                    grid_values[float(budget)].append(float(reward))

    bands: Dict[str, List[BootstrapBandPoint]] = {}
    for label, grid_values in values.items():
        kind = original_points_by_series[label][0].kind if original_points_by_series[label] else ""
        label_bands: List[BootstrapBandPoint] = []
        for budget in budget_grid:
            vals = grid_values[float(budget)]
            if len(vals) < max(10, int(0.8 * int(bootstrap_samples))):
                continue
            label_bands.append(
                BootstrapBandPoint(
                    series=label,
                    kind=kind,
                    mean_final_steps=float(budget),
                    reward_p025=_percentile(vals, 0.025),
                    reward_p500=_percentile(vals, 0.5),
                    reward_p975=_percentile(vals, 0.975),
                    num_bootstrap_values=len(vals),
                )
            )
        if label_bands:
            bands[label] = label_bands
    return bands


def build_bootstrap_oracle_gap_bands(
    *,
    domain: str,
    split: str,
    cache_specs: Sequence[CacheSpec],
    streams_by_label: Dict[str, List[List[ThreadCheckpoint]]],
    original_points_by_series: Dict[str, List[CurvePoint]],
    random_seeds: int,
    budget_step: float,
    priority_mode: str,
    include_fixed_baseline: bool,
    heuristic_baselines: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, List[BootstrapOracleGapBandPoint]]:
    """Paired percentile bands for model reward minus oracle reward.

    Each bootstrap draw resamples tasks once, recomputes both the scheduler and
    oracle curves on that same resample, and stores the paired gap at each
    budget. This keeps the upper reference at zero by construction.
    """
    if int(bootstrap_samples) <= 0 or not cache_specs:
        return {}
    n = len(streams_by_label[cache_specs[0].label])
    if n <= 1:
        return {}

    all_original = [p for points in original_points_by_series.values() for p in points]
    budget_grid = _make_budget_grid(all_original, step=float(budget_step))
    if not budget_grid:
        return {}

    values: Dict[str, Dict[float, List[float]]] = {
        label: {float(budget): [] for budget in budget_grid}
        for label in original_points_by_series
        if label != ORACLE_SERIES
        and not label.startswith("Random")
        and not label.startswith("Fixed")
        and not label.startswith("Heuristic")
    }
    rng = random.Random(int(bootstrap_seed))
    for _ in range(int(bootstrap_samples)):
        sample_indices = [rng.randrange(n) for _ in range(n)]
        boot_streams = _resample_streams_by_label(streams_by_label, sample_indices)
        boot_points = compute_points_by_series(
            domain=domain,
            split=split,
            cache_specs=cache_specs,
            streams_by_label=boot_streams,
            random_seeds=random_seeds,
            budget_step=budget_step,
            priority_mode=priority_mode,
            include_fixed_baseline=False,
            heuristic_baselines=[],
            include_oracle=True,
            include_random=False,
        )
        oracle_points = boot_points.get(ORACLE_SERIES, [])
        if not oracle_points:
            continue
        for label, grid_values in values.items():
            points = boot_points.get(label, [])
            for budget in budget_grid:
                gap = _gap_to_oracle_at_budget(
                    model_points=points,
                    oracle_points=oracle_points,
                    budget=float(budget),
                )
                if gap is not None:
                    grid_values[float(budget)].append(float(gap))

    bands: Dict[str, List[BootstrapOracleGapBandPoint]] = {}
    for label, grid_values in values.items():
        kind = original_points_by_series[label][0].kind if original_points_by_series[label] else ""
        label_bands: List[BootstrapOracleGapBandPoint] = []
        for budget in budget_grid:
            vals = grid_values[float(budget)]
            if len(vals) < max(10, int(0.8 * int(bootstrap_samples))):
                continue
            label_bands.append(
                BootstrapOracleGapBandPoint(
                    series=label,
                    kind=kind,
                    mean_final_steps=float(budget),
                    gap_p025=_percentile(vals, 0.025),
                    gap_p500=_percentile(vals, 0.5),
                    gap_p975=_percentile(vals, 0.975),
                    num_bootstrap_values=len(vals),
                )
            )
        if label_bands:
            bands[label] = label_bands
    return bands


def write_bootstrap_bands_csv(path: Path, bands_by_series: Dict[str, List[BootstrapBandPoint]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "series",
        "kind",
        "mean_final_steps",
        "reward_p025",
        "reward_p500",
        "reward_p975",
        "num_bootstrap_values",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for bands in bands_by_series.values():
            for b in bands:
                writer.writerow(
                    {
                        "series": b.series,
                        "kind": b.kind,
                        "mean_final_steps": f"{b.mean_final_steps:.10g}",
                        "reward_p025": f"{b.reward_p025:.10g}",
                        "reward_p500": f"{b.reward_p500:.10g}",
                        "reward_p975": f"{b.reward_p975:.10g}",
                        "num_bootstrap_values": b.num_bootstrap_values,
                    }
                )


def write_bootstrap_oracle_gap_bands_csv(
    path: Path,
    bands_by_series: Dict[str, List[BootstrapOracleGapBandPoint]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "series",
        "kind",
        "mean_final_steps",
        "gap_p025",
        "gap_p500",
        "gap_p975",
        "num_bootstrap_values",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for bands in bands_by_series.values():
            for b in bands:
                writer.writerow(
                    {
                        "series": b.series,
                        "kind": b.kind,
                        "mean_final_steps": f"{b.mean_final_steps:.10g}",
                        "gap_p025": f"{b.gap_p025:.10g}",
                        "gap_p500": f"{b.gap_p500:.10g}",
                        "gap_p975": f"{b.gap_p975:.10g}",
                        "num_bootstrap_values": b.num_bootstrap_values,
                    }
                )


def write_summary_json(
    path: Path,
    *,
    domain: str,
    split: str,
    cache_specs: Sequence[CacheSpec],
    goal_ids: Sequence[int],
    points_by_series: Dict[str, List[CurvePoint]],
    random_seeds: int,
    budget_step: float,
    priority_mode: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    heuristic_baselines: Sequence[str],
) -> None:
    payload = {
        "domain": domain,
        "split": split,
        "num_goal_ids": len(goal_ids),
        "goal_ids_head": list(goal_ids[:10]),
        "priority_mode": priority_mode,
        "random_seeds": int(random_seeds),
        "budget_step": float(budget_step),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "heuristic_baselines": list(heuristic_baselines),
        "caches": [{"label": spec.label, "path": str(spec.path)} for spec in cache_specs],
        "series": {label: _summarize_curve(points) for label, points in points_by_series.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_curves(
    path_prefix: Path,
    *,
    title: str,
    points_by_series: Dict[str, List[CurvePoint]],
    bootstrap_bands: Optional[Dict[str, List[BootstrapBandPoint]]] = None,
    bootstrap_oracle_gap_bands: Optional[Dict[str, List[BootstrapOracleGapBandPoint]]] = None,
    plot_series: Optional[Sequence[str]] = None,
    plot_metric: str = "reward",
    budget_step: float = 0.25,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style(plt, base_font_size=9.6)

    if plot_series:
        allowed = set(plot_series)
        points_by_series = {
            label: points for label, points in points_by_series.items() if label in allowed
        }
        if bootstrap_bands:
            bootstrap_bands = {
                label: bands for label, bands in bootstrap_bands.items() if label in allowed
            }
        if bootstrap_oracle_gap_bands:
            bootstrap_oracle_gap_bands = {
                label: bands for label, bands in bootstrap_oracle_gap_bands.items() if label in allowed
            }

    if plot_metric == "oracle_gap":
        points_by_series = build_oracle_gap_points(points_by_series, budget_step=budget_step)
        bootstrap_bands = None
    elif plot_metric != "reward":
        raise ValueError(f"Unknown plot_metric: {plot_metric}")

    fig, ax = plt.subplots(figsize=(4.65, 3.25), dpi=220)
    styles: Dict[str, Dict[str, Any]] = {}
    color_idx = 0
    for label, points in points_by_series.items():
        if not points:
            continue
        styles[label], color_idx = _series_style(label, color_idx)

    if plot_metric == "oracle_gap" and bootstrap_oracle_gap_bands:
        for label, bands in bootstrap_oracle_gap_bands.items():
            if label == ORACLE_SERIES or not bands or label not in styles:
                continue
            xs = [b.mean_final_steps for b in bands]
            lows = [b.gap_p025 for b in bands]
            highs = [b.gap_p975 for b in bands]
            color = str(styles[label]["color"])
            ax.fill_between(xs, lows, highs, color=color, alpha=0.12, linewidth=0, zorder=0)
    elif bootstrap_bands:
        for label, bands in bootstrap_bands.items():
            if label == ORACLE_SERIES or not bands or label not in styles:
                continue
            xs = [b.mean_final_steps for b in bands]
            lows = [b.reward_p025 for b in bands]
            highs = [b.reward_p975 for b in bands]
            color = str(styles[label]["color"])
            ax.fill_between(xs, lows, highs, color=color, alpha=0.12, linewidth=0, zorder=0)

    for label, points in points_by_series.items():
        if not points:
            continue
        xs = [p.mean_final_steps for p in points]
        ys = [p.mean_final_reward for p in points]
        kw = styles[label]
        ax.plot(xs, ys, label=label, **kw)

    ax.set_title(title)
    ax.set_xlabel("Mean final replay steps")
    if plot_metric == "oracle_gap":
        ax.set_ylabel("Reward gap to oracle")
    else:
        ax.set_ylabel("Mean final reward")
    ax.grid(True, color=GRID, alpha=0.48, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.9)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"))
    fig.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(fig)


def build_budgeted_expansion(
    *,
    domain: str,
    split: str,
    cache_specs: Sequence[CacheSpec],
    out_dir: Path,
    title: str,
    offline_back_steps: int,
    offline_to_product_steps: int,
    random_seeds: int,
    budget_step: float,
    priority_mode: str,
    include_fixed_baseline: bool,
    heuristic_baselines: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
    plot_series: Sequence[str],
    plot_metric: str,
) -> None:
    caches = [_load_cache(spec.path) for spec in cache_specs]
    goal_ids = shared_goal_ids(caches)
    if not goal_ids:
        raise ValueError("Prediction caches contain no goals")

    raw_streams_by_label: Dict[str, List[List[ThreadCheckpoint]]] = {}
    for spec, cache in zip(cache_specs, caches):
        metadata_by_goal = _load_checkpoint_metadata_by_goal(cache)
        raw_streams_by_label[spec.label] = extract_threads(
            cache,
            goal_ids,
            offline_back_steps=offline_back_steps,
            offline_to_product_steps=offline_to_product_steps,
            metadata_by_goal=metadata_by_goal,
        )
    streams_by_label = validate_shared_reward_streams(raw_streams_by_label)
    points_by_series = compute_points_by_series(
        domain=domain,
        split=split,
        cache_specs=cache_specs,
        streams_by_label=streams_by_label,
        random_seeds=random_seeds,
        budget_step=budget_step,
        priority_mode=priority_mode,
        include_fixed_baseline=include_fixed_baseline,
        heuristic_baselines=heuristic_baselines,
    )
    all_points = [p for points in points_by_series.values() for p in points]
    bootstrap_bands = build_bootstrap_bands(
        domain=domain,
        split=split,
        cache_specs=cache_specs,
        streams_by_label=streams_by_label,
        original_points_by_series=points_by_series,
        random_seeds=random_seeds,
        budget_step=budget_step,
        priority_mode=priority_mode,
        include_fixed_baseline=include_fixed_baseline,
        heuristic_baselines=heuristic_baselines,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    bootstrap_oracle_gap_bands = build_bootstrap_oracle_gap_bands(
        domain=domain,
        split=split,
        cache_specs=cache_specs,
        streams_by_label=streams_by_label,
        original_points_by_series=points_by_series,
        random_seeds=random_seeds,
        budget_step=budget_step,
        priority_mode=priority_mode,
        include_fixed_baseline=include_fixed_baseline,
        heuristic_baselines=heuristic_baselines,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    oracle_gap_points_by_series = build_oracle_gap_points(
        points_by_series,
        budget_step=budget_step,
    )
    oracle_gap_points = [p for points in oracle_gap_points_by_series.values() for p in points]

    write_curve_csv(out_dir / "budgeted_expansion_curves.csv", all_points)
    if oracle_gap_points:
        write_curve_csv(out_dir / "budgeted_expansion_oracle_gap_curves.csv", oracle_gap_points)
    if bootstrap_bands:
        write_bootstrap_bands_csv(out_dir / "budgeted_expansion_bootstrap_bands.csv", bootstrap_bands)
    if bootstrap_oracle_gap_bands:
        write_bootstrap_oracle_gap_bands_csv(
            out_dir / "budgeted_expansion_oracle_gap_bootstrap_bands.csv",
            bootstrap_oracle_gap_bands,
        )
    write_summary_json(
        out_dir / "budgeted_expansion_summary.json",
        domain=domain,
        split=split,
        cache_specs=cache_specs,
        goal_ids=goal_ids,
        points_by_series=points_by_series,
        random_seeds=random_seeds,
        budget_step=budget_step,
        priority_mode=priority_mode,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        heuristic_baselines=heuristic_baselines,
    )
    plot_curves(
        out_dir / "budgeted_expansion_frontier",
        title=title,
        points_by_series=points_by_series,
        bootstrap_bands=bootstrap_bands,
        bootstrap_oracle_gap_bands=bootstrap_oracle_gap_bands,
        plot_series=plot_series,
        plot_metric=plot_metric,
        budget_step=budget_step,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build budgeted expansion frontiers from forecast caches.")
    ap.add_argument("--domain", required=True, help="Domain label, e.g. WebShop or Paper Search.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache", action="append", required=True, help="Repeatable LABEL=PATH prediction cache spec.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--offline_back_steps", type=int, default=1)
    ap.add_argument("--offline_to_product_steps", type=int, default=2)
    ap.add_argument("--random_seeds", type=int, default=DEFAULT_RANDOM_SEEDS)
    ap.add_argument("--budget_step", type=float, default=0.25)
    ap.add_argument(
        "--bootstrap_samples",
        type=int,
        default=0,
        help="If >0, draw task-level bootstrap resamples and render 95%% percentile bands.",
    )
    ap.add_argument("--bootstrap_seed", type=int, default=123)
    ap.add_argument(
        "--priority_mode",
        choices=["predicted_mean_delta", "predicted_mean_delta_per_step"],
        default="predicted_mean_delta",
        help="Use raw expected remaining upside by default, matching the method description.",
    )
    ap.add_argument(
        "--include_fixed_baseline",
        action="store_true",
        help="Include deterministic round-robin expansion as an extra diagnostic baseline.",
    )
    ap.add_argument(
        "--heuristic_baseline",
        action="append",
        default=[],
        choices=sorted(HEURISTIC_LABELS),
        help=(
            "Repeatable oracle-free heuristic scheduler baseline. "
            "Available choices use only prefix features stored in checkpoint rows."
        ),
    )
    ap.add_argument(
        "--plot_series",
        action="append",
        default=[],
        help=(
            "Optional repeatable series allowlist for the rendered plot. "
            "CSV and JSON outputs still contain all computed series."
        ),
    )
    ap.add_argument(
        "--plot_metric",
        choices=["reward", "oracle_gap"],
        default="reward",
        help=(
            "What to render in budgeted_expansion_frontier.{pdf,png}. "
            "reward plots absolute mean reward; oracle_gap plots model reward "
            "minus the paired oracle reward at the same budget."
        ),
    )
    args = ap.parse_args()

    cache_specs = [_parse_cache_spec(raw) for raw in args.cache]
    title = str(args.title or f"{args.domain} Budgeted Expansion Frontier")
    build_budgeted_expansion(
        domain=str(args.domain),
        split=str(args.split),
        cache_specs=cache_specs,
        out_dir=Path(args.out_dir),
        title=title,
        offline_back_steps=int(args.offline_back_steps),
        offline_to_product_steps=int(args.offline_to_product_steps),
        random_seeds=int(args.random_seeds),
        budget_step=float(args.budget_step),
        priority_mode=str(args.priority_mode),
        include_fixed_baseline=bool(args.include_fixed_baseline),
        heuristic_baselines=list(args.heuristic_baseline or []),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        plot_series=list(args.plot_series or []),
        plot_metric=str(args.plot_metric),
    )


if __name__ == "__main__":
    main()
