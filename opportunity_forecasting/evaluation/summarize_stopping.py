"""Summarize stopping-policy sweeps as CSV, Markdown, and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _float_slug(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def _resolve_output_path(path: Path) -> Path:
    if path.exists():
        return path
    txt = str(path)
    alt_txt = txt.replace("_c0_l", f"_c{_float_slug(0.0)}_l")
    alt = Path(alt_txt)
    if alt.exists():
        return alt
    return path


def parse_model_label_map(spec: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            mapping[key] = value
    return mapping


def read_manifest(path: Path) -> List[dict]:
    def split_fields(line: str) -> List[str]:
        txt = line.rstrip("\n")
        if "\t" in txt:
            return txt.split("\t")
        if "\\t" in txt:
            return txt.split("\\t")
        return [txt]

    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        header = split_fields(f.readline())
        for line in f:
            parts = split_fields(line)
            row = {header[i]: parts[i] if i < len(parts) else "" for i in range(len(header))}
            rows.append(row)
    return rows


def model_label(model_path: str, mapping: Dict[str, str]) -> str:
    if model_path in mapping:
        return mapping[model_path]
    txt = str(model_path)
    if txt == "Qwen/Qwen2.5-7B-Instruct" or "models--Qwen--Qwen2.5-7B-Instruct" in txt:
        return "base_qwen"
    if txt.endswith("/merged"):
        return Path(txt).parent.name
    return Path(txt).name or txt


def _resolve_numeric_parse_rate(payload: dict) -> float:
    if "numeric_parse_rate" in payload and payload.get("numeric_parse_rate") not in (None, ""):
        return float(payload["numeric_parse_rate"])
    return float(payload.get("valid_score_rate", 0.0))


def _resolve_valid_forecast_rate(payload: dict) -> float:
    if "valid_forecast_rate" in payload and payload.get("valid_forecast_rate") not in (None, ""):
        return float(payload["valid_forecast_rate"])
    return _resolve_numeric_parse_rate(payload)


def flatten_results(eval_path: Path, mapping: Dict[str, str]) -> List[dict]:
    payload = json.loads(eval_path.read_text())
    summaries = payload.get("summaries", []) or []
    diagnostics = payload.get("diagnostics", {}) or {}
    rows: List[dict] = []
    for summary in summaries:
        model_path = str(summary.get("model_path"))
        diag = diagnostics.get(model_path, {}) or {}
        c_step = float(summary.get("c_step"))
        mean_reward = float(summary.get("mean_final_reward"))
        mean_steps = float(summary.get("mean_final_steps"))
        row = {
            "eval_path": str(eval_path),
            "model_path": model_path,
            "model_label": model_label(model_path, mapping),
            "policy": str(summary.get("stop_formulation")),
            "c_step": c_step,
            "lambda_risk": float(summary.get("lambda_risk", 0.0)),
            "mean_final_reward": mean_reward,
            "mean_final_steps": mean_steps,
            "stop_rate": float(summary.get("stop_rate")),
            "utility_proxy": mean_reward - c_step * mean_steps,
            "cost_unit": str(summary.get("cost_unit", "replay_step")),
            "num_goals": int(summary.get("num_goals", 0)),
            "valid_score_rate": float(diag.get("valid_score_rate", 0.0)),
            "valid_forecast_rate": _resolve_valid_forecast_rate(diag),
            "numeric_parse_rate": _resolve_numeric_parse_rate(diag),
            "stops_triggered": int(diag.get("stops_triggered", 0)),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def best_rows(rows: Sequence[dict], *, policy: str, metric: str) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row["policy"] != policy:
            continue
        grouped[row["model_label"]].append(row)
    out: List[dict] = []
    for model, model_rows in grouped.items():
        ranked = sorted(
            model_rows,
            key=lambda r: (
                float(r[metric]),
                float(r["mean_final_reward"]),
                float(r.get("valid_score_rate", 0.0)),
                float(r.get("valid_forecast_rate", 0.0)),
                float(r.get("numeric_parse_rate", 0.0)),
                -float(r["mean_final_steps"]),
            ),
            reverse=True,
        )
        if ranked:
            out.append(ranked[0])
    return sorted(out, key=lambda r: r["model_label"])


def parse_health_rows(rows: Sequence[dict], *, policy: str) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row["policy"] != policy:
            continue
        grouped[row["model_label"]].append(row)
    out: List[dict] = []
    for model, model_rows in grouped.items():
        out.append(
            {
                "model_label": model,
                "min_valid_score_rate": min(float(r.get("valid_score_rate", 0.0)) for r in model_rows),
                "max_valid_score_rate": max(float(r.get("valid_score_rate", 0.0)) for r in model_rows),
                "min_valid_forecast_rate": min(float(r.get("valid_forecast_rate", 0.0)) for r in model_rows),
                "min_numeric_parse_rate": min(float(r.get("numeric_parse_rate", 0.0)) for r in model_rows),
            }
        )
    return sorted(out, key=lambda r: r["model_label"])


def dominates_reward_steps(
    lhs: dict,
    rhs: dict,
    *,
    reward_key: str = "mean_final_reward",
    steps_key: str = "mean_final_steps",
) -> bool:
    lhs_reward = float(lhs[reward_key])
    rhs_reward = float(rhs[reward_key])
    lhs_steps = float(lhs[steps_key])
    rhs_steps = float(rhs[steps_key])
    return (
        lhs_reward >= rhs_reward
        and lhs_steps <= rhs_steps
        and (lhs_reward > rhs_reward or lhs_steps < rhs_steps)
    )


def _frontier_coord_key(row: dict, *, reward_key: str, steps_key: str) -> Tuple[float, float]:
    return (round(float(row[reward_key]), 12), round(float(row[steps_key]), 12))


def _collapse_frontier_duplicates(
    rows: Sequence[dict],
    *,
    reward_key: str,
    steps_key: str,
) -> List[dict]:
    grouped: Dict[Tuple[float, float], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[_frontier_coord_key(row, reward_key=reward_key, steps_key=steps_key)].append(row)

    out: List[dict] = []
    for _, dup_rows in grouped.items():
        def _policy_priority(row: dict) -> int:
            return 0 if str(row.get("policy", "")) == "policy1" else 1

        ranked = sorted(
            dup_rows,
            key=lambda r: (
                -float(r.get("valid_score_rate", 0.0)),
                -float(r.get("valid_forecast_rate", 0.0)),
                -float(r.get("numeric_parse_rate", 0.0)),
                _policy_priority(r),
                float(r.get("stop_rate", 0.0)),
                float(r.get("lambda_risk", 0.0)),
                float(r.get("c_step", 0.0)),
                str(r.get("policy", "")),
            ),
        )
        keep = dict(ranked[0])
        keep["num_equivalent_settings"] = len(dup_rows)
        keep["equivalent_settings"] = "; ".join(
            sorted(
                f"{str(r.get('policy', ''))}(c={float(r.get('c_step', 0.0)):.4f},lambda={float(r.get('lambda_risk', 0.0)):.2f})"
                for r in dup_rows
            )
        )
        out.append(keep)
    return sorted(out, key=lambda r: (float(r[steps_key]), -float(r[reward_key])))


def pareto_frontier_rows(
    rows: Sequence[dict],
    *,
    reward_key: str = "mean_final_reward",
    steps_key: str = "mean_final_steps",
) -> List[dict]:
    frontier: List[dict] = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if dominates_reward_steps(other, row, reward_key=reward_key, steps_key=steps_key):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return _collapse_frontier_duplicates(frontier, reward_key=reward_key, steps_key=steps_key)


def common_support_frontier_rows(rows: Sequence[dict]) -> List[dict]:
    """Clip per-model Pareto curves to their shared observed cost support."""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_label"])].append(row)
    frontiers = {label: pareto_frontier_rows(model_rows) for label, model_rows in grouped.items()}
    if not frontiers:
        return []
    common_min = max(float(frontier[0]["mean_final_steps"]) for frontier in frontiers.values())
    common_max = min(float(frontier[-1]["mean_final_steps"]) for frontier in frontiers.values())
    if common_min > common_max + 1e-12:
        raise ValueError(f"Per-model frontiers have no shared cost support: {common_min} > {common_max}")

    metric_keys = (
        "mean_final_reward",
        "stop_rate",
        "utility_proxy",
        "valid_score_rate",
        "valid_forecast_rate",
        "numeric_parse_rate",
    )

    def at_x(frontier: Sequence[dict], x_value: float) -> dict:
        for row in frontier:
            if abs(float(row["mean_final_steps"]) - x_value) <= 1e-12:
                return dict(row)
        for left, right in zip(frontier, frontier[1:]):
            left_x = float(left["mean_final_steps"])
            right_x = float(right["mean_final_steps"])
            if left_x <= x_value <= right_x:
                weight = (x_value - left_x) / (right_x - left_x)
                out = dict(left)
                out["policy"] = "common_support_interpolation"
                out["mean_final_steps"] = float(x_value)
                for key in metric_keys:
                    if left.get(key) is not None and right.get(key) is not None:
                        out[key] = float(left[key]) + weight * (float(right[key]) - float(left[key]))
                return out
        raise ValueError(f"Cannot interpolate frontier at x={x_value}")

    out: List[dict] = []
    for label, frontier in frontiers.items():
        selected = [
            dict(row)
            for row in frontier
            if common_min - 1e-12 <= float(row["mean_final_steps"]) <= common_max + 1e-12
        ]
        selected.extend((at_x(frontier, common_min), at_x(frontier, common_max)))
        for row in _collapse_frontier_duplicates(
            selected,
            reward_key="mean_final_reward",
            steps_key="mean_final_steps",
        ):
            row["model_label"] = label
            out.append(row)
    return sorted(out, key=lambda row: (str(row["model_label"]), float(row["mean_final_steps"])))


def build_shared_setting_rows(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, float, float], List[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["policy"]), float(row["c_step"]), float(row["lambda_risk"]))
        grouped[key].append(row)

    out: List[dict] = []
    for (policy, c_step, lambda_risk), vals in grouped.items():
        out.append(
            {
                "policy": policy,
                "c_step": float(c_step),
                "lambda_risk": float(lambda_risk),
                "mean_final_reward": sum(float(v["mean_final_reward"]) for v in vals) / float(len(vals)),
                "mean_final_steps": sum(float(v["mean_final_steps"]) for v in vals) / float(len(vals)),
                "stop_rate": sum(float(v["stop_rate"]) for v in vals) / float(len(vals)),
                "utility_proxy": sum(float(v["utility_proxy"]) for v in vals) / float(len(vals)),
                "valid_score_rate": min(float(v.get("valid_score_rate", 0.0)) for v in vals),
                "valid_forecast_rate": min(float(v.get("valid_forecast_rate", 0.0)) for v in vals),
                "numeric_parse_rate": min(float(v.get("numeric_parse_rate", 0.0)) for v in vals),
                "num_models": len(vals),
                "min_model_reward": min(float(v["mean_final_reward"]) for v in vals),
                "max_model_reward": max(float(v["mean_final_reward"]) for v in vals),
                "mean_model_stop_rate": sum(float(v["stop_rate"]) for v in vals) / float(len(vals)),
                "model_labels": ",".join(sorted(str(v["model_label"]) for v in vals)),
            }
        )
    return sorted(out, key=lambda r: (float(r["mean_final_steps"]), -float(r["mean_final_reward"])))


def per_model_frontier_rows(rows: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    for frontier_row in common_support_frontier_rows(rows):
        row = dict(frontier_row)
        row["frontier_scope"] = "per_model_common_support"
        out.append(row)
    return sorted(out, key=lambda r: (str(r["model_label"]), float(r["mean_final_steps"]), -float(r["mean_final_reward"])))


def build_markdown(rows: Sequence[dict]) -> str:
    lines: List[str] = []
    shared_rows = build_shared_setting_rows(rows)
    shared_frontier = pareto_frontier_rows(shared_rows)
    model_frontier = per_model_frontier_rows(rows)

    lines.append("# Stop Policy Sweep Summary")
    lines.append("")
    lines.append(f"Total rows: {len(rows)}")
    lines.append("")
    for policy in ("policy1", "policy2"):
        lines.append(f"## {policy}")
        for metric in ("mean_final_reward", "utility_proxy"):
            lines.append(f"### Best by {metric}")
            lines.append("")
            lines.append("| model | c_step | lambda_risk | mean_reward | mean_steps | stop_rate | utility_proxy | valid_score_rate | valid_forecast_rate | numeric_parse_rate |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for row in best_rows(rows, policy=policy, metric=metric):
                lines.append(
                    "| {model_label} | {c_step:.4f} | {lambda_risk:.2f} | {mean_final_reward:.6f} | "
                    "{mean_final_steps:.3f} | {stop_rate:.3f} | {utility_proxy:.6f} | {valid_score_rate:.3f} | "
                    "{valid_forecast_rate:.3f} | {numeric_parse_rate:.3f} |".format(**row)
                )
            lines.append("")
        lines.append("### Parse and Score Health")
        lines.append("")
        lines.append("| model | min_valid_score_rate | max_valid_score_rate | min_valid_forecast_rate | min_numeric_parse_rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in parse_health_rows(rows, policy=policy):
            lines.append(
                "| {model_label} | {min_valid_score_rate:.3f} | {max_valid_score_rate:.3f} | "
                "{min_valid_forecast_rate:.3f} | {min_numeric_parse_rate:.3f} |".format(**row)
            )
        lines.append("")
    lines.append("## Shared Pareto Frontier")
    lines.append("")
    lines.append("| policy | c_step | lambda_risk | avg_reward | avg_steps | avg_stop_rate | min_model_reward | valid_score_rate | equivalent_settings |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in shared_frontier:
        lines.append(
            "| {policy} | {c_step:.4f} | {lambda_risk:.2f} | {mean_final_reward:.6f} | "
            "{mean_final_steps:.3f} | {stop_rate:.3f} | {min_model_reward:.6f} | "
            "{valid_score_rate:.3f} | {equivalent_settings} |".format(**row)
        )
    lines.append("")
    lines.append("## Per-Model Pareto Frontiers")
    lines.append("")
    lines.append("| model | policy | c_step | lambda_risk | reward | steps | stop_rate | valid_score_rate | equivalent_settings |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in model_frontier:
        lines.append(
            "| {model_label} | {policy} | {c_step:.4f} | {lambda_risk:.2f} | "
            "{mean_final_reward:.6f} | {mean_final_steps:.3f} | {stop_rate:.3f} | "
            "{valid_score_rate:.3f} | {equivalent_settings} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plot_policy1_tradeoff(rows: Sequence[dict], out_path: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in common_support_frontier_rows([row for row in rows if row["policy"] == "policy1"]):
        if row["policy"] == "policy1" or row["policy"] == "common_support_interpolation":
            grouped[row["model_label"]].append(row)
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model, model_rows in sorted(grouped.items()):
        ordered = sorted(model_rows, key=lambda row: float(row["mean_final_steps"]))
        ax.plot(
            [float(r["mean_final_steps"]) for r in ordered],
            [float(r["mean_final_reward"]) for r in ordered],
            marker="o",
            label=model,
        )
    ax.set_xlabel("Mean final replay steps")
    ax.set_ylabel("Mean Final Reward")
    ax.set_title("Policy 1 Dev Sweep: Reward/Steps Tradeoff")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_policy2_heatmaps(rows: Sequence[dict], out_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row["policy"] == "policy2":
            grouped[row["model_label"]].append(row)
    for model, model_rows in sorted(grouped.items()):
        c_vals = sorted({float(r["c_step"]) for r in model_rows})
        l_vals = sorted({float(r["lambda_risk"]) for r in model_rows})
        if not c_vals or not l_vals:
            continue
        reward_grid = [[math.nan for _ in c_vals] for _ in l_vals]
        util_grid = [[math.nan for _ in c_vals] for _ in l_vals]
        for row in model_rows:
            i = l_vals.index(float(row["lambda_risk"]))
            j = c_vals.index(float(row["c_step"]))
            reward_grid[i][j] = float(row["mean_final_reward"])
            util_grid[i][j] = float(row["utility_proxy"])
        for grid_name, grid in (("reward", reward_grid), ("utility", util_grid)):
            fig, ax = plt.subplots(figsize=(6.5, 5.2))
            im = ax.imshow(grid, aspect="auto", origin="lower")
            ax.set_xticks(range(len(c_vals)))
            ax.set_xticklabels([f"{v:.4f}" for v in c_vals], rotation=45, ha="right")
            ax.set_yticks(range(len(l_vals)))
            ax.set_yticklabels([f"{v:.2f}" for v in l_vals])
            ax.set_xlabel("c_step")
            ax.set_ylabel("lambda_risk")
            ax.set_title(f"Policy 2 Dev Sweep: {model} {grid_name}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            out_path = out_dir / f"policy2_{grid_name}_{model}.png"
            fig.savefig(out_path)
            fig.savefig(out_path.with_suffix(".pdf"))
            plt.close(fig)


def plot_pareto_frontiers(rows: Sequence[dict], out_path: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    model_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in common_support_frontier_rows(rows):
        model_groups[str(row["model_label"])].append(row)

    shared_rows = build_shared_setting_rows(rows)
    shared_frontier = pareto_frontier_rows(shared_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ax_model, ax_shared = axes

    color_map = {
        "base_qwen": "#1b4965",
        "rlcr_hotpot_brier": "#ca6702",
        "sft": "#6a994e",
        "rl": "#9b2226",
    }

    for model_label, model_rows in sorted(model_groups.items()):
        color = color_map.get(model_label, None)
        ax_model.scatter(
            [float(r["mean_final_steps"]) for r in model_rows],
            [float(r["mean_final_reward"]) for r in model_rows],
            s=14,
            alpha=0.18,
            color=color,
        )
        frontier = sorted(model_rows, key=lambda row: float(row["mean_final_steps"]))
        ax_model.plot(
            [float(r["mean_final_steps"]) for r in frontier],
            [float(r["mean_final_reward"]) for r in frontier],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=model_label,
            color=color,
        )

    ax_model.set_title("Per-Model Pareto Frontiers")
    ax_model.set_xlabel("Mean final replay steps")
    ax_model.set_ylabel("Mean Final Reward")
    ax_model.grid(alpha=0.25)
    ax_model.legend(frameon=False, fontsize=8)

    ax_shared.scatter(
        [float(r["mean_final_steps"]) for r in shared_rows],
        [float(r["mean_final_reward"]) for r in shared_rows],
        s=20,
        alpha=0.15,
        color="#4d4d4d",
    )
    ax_shared.plot(
        [float(r["mean_final_steps"]) for r in shared_frontier],
        [float(r["mean_final_reward"]) for r in shared_frontier],
        marker="o",
        linewidth=2.2,
        markersize=4.8,
        color="#0d3b66",
    )
    for row in shared_frontier:
        label = (
            f"P1 c={float(row['c_step']):.3f}"
            if str(row["policy"]) == "policy1"
            else f"P2 l={float(row['lambda_risk']):.2f}"
        )
        ax_shared.annotate(
            label,
            (float(row["mean_final_steps"]), float(row["mean_final_reward"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )

    ax_shared.set_title("Shared-Setting Pareto Frontier")
    ax_shared.set_xlabel("Mean final replay steps")
    ax_shared.set_ylabel("Average Final Reward")
    ax_shared.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path)
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize stop-policy sweep eval outputs.")
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--manifest_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_label_map", type=str, default="")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_map = parse_model_label_map(args.model_label_map)
    manifest_rows = read_manifest(manifest_path)

    rows: List[dict] = []
    missing_outputs: List[str] = []
    for row in manifest_rows:
        output_path = _resolve_output_path(Path(row["output_path"]))
        if not output_path.exists():
            missing_outputs.append(str(output_path))
            continue
        rows.extend(flatten_results(output_path, model_map))

    rows = sorted(
        rows,
        key=lambda r: (str(r["policy"]), str(r["model_label"]), float(r["c_step"]), float(r["lambda_risk"])),
    )

    fieldnames = [
        "policy",
        "model_label",
        "model_path",
        "c_step",
        "lambda_risk",
        "mean_final_reward",
        "mean_final_steps",
        "stop_rate",
        "utility_proxy",
        "cost_unit",
        "valid_score_rate",
        "valid_forecast_rate",
        "numeric_parse_rate",
        "stops_triggered",
        "num_goals",
        "eval_path",
    ]
    write_csv(output_dir / "results_long.csv", rows, fieldnames)
    (output_dir / "results_long.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    model_frontier = per_model_frontier_rows(rows)
    write_csv(
        output_dir / "pareto_frontier_by_model.csv",
        model_frontier,
        [
            "model_label",
            "policy",
            "c_step",
            "lambda_risk",
            "mean_final_reward",
            "mean_final_steps",
            "stop_rate",
            "valid_score_rate",
            "valid_forecast_rate",
            "numeric_parse_rate",
            "num_equivalent_settings",
            "equivalent_settings",
            "eval_path",
        ],
    )

    shared_rows = build_shared_setting_rows(rows)
    shared_frontier = pareto_frontier_rows(shared_rows)
    write_csv(
        output_dir / "pareto_frontier_shared.csv",
        shared_frontier,
        [
            "policy",
            "c_step",
            "lambda_risk",
            "mean_final_reward",
            "mean_final_steps",
            "stop_rate",
            "valid_score_rate",
            "valid_forecast_rate",
            "numeric_parse_rate",
            "min_model_reward",
            "max_model_reward",
            "mean_model_stop_rate",
            "num_models",
            "num_equivalent_settings",
            "equivalent_settings",
            "model_labels",
        ],
    )

    for policy in ("policy1", "policy2"):
        for metric in ("mean_final_reward", "utility_proxy"):
            best = best_rows(rows, policy=policy, metric=metric)
            write_csv(
                output_dir / f"{policy}_best_by_{metric}.csv",
                best,
                ["policy", "model_label", "c_step", "lambda_risk", "mean_final_reward", "mean_final_steps", "stop_rate", "utility_proxy", "valid_score_rate", "eval_path"],
            )

    (output_dir / "summary.md").write_text(build_markdown(rows), encoding="utf-8")
    (output_dir / "summary_meta.json").write_text(
        json.dumps(
            {
                "sweep_root": str(sweep_root),
                "manifest_path": str(manifest_path),
                "num_manifest_rows": len(manifest_rows),
                "num_result_rows": len(rows),
                "missing_outputs": missing_outputs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_policy1_tradeoff(rows, output_dir / "policy1_tradeoff.png")
    plot_policy2_heatmaps(rows, output_dir)
    plot_pareto_frontiers(rows, output_dir / "pareto_frontier_dev.png")


if __name__ == "__main__":
    main()
