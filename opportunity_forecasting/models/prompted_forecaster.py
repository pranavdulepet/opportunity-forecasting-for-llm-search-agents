"""
Lightweight prompt-formatting helpers for WebShop stopping/forecasting.

These utilities intentionally avoid importing the full WebShop runtime so they
can be used inside offline training jobs without pulling in Java/Pyserini.
"""

from __future__ import annotations

from typing import Dict, Optional

from opportunity_forecasting.models.distributions import (
    HURDLE_BETA_FAMILY,
    canonical_prompt_guidance,
)


COUNTERFACTUAL_REWARD_KEY = "buy_now_reward"
PAPER_RELEVANCE_REWARD_KEY = "paper_relevance_reward"


STOPPING_PROMPT = """You are forecasting the upside from continuing to search for a search goal.

Search Goal: {goal}

Candidates seen so far (top {top_k} + summary):
{seen_products}

Current decision point:
{state_context}

Current best reward from visited candidate pages seen so far: {best_reward_seen}

Important semantics:
- The normalized reward signal depends on the domain:
  BuyNowReward for WebShop and RelevanceReward for paper search.
- "Candidate pages" means product pages in WebShop and paper pages in LitSearch.
- Forecast reward upside, not price improvement.
- Delta is defined in reward units with 0 <= Delta <= 1.
- Treat zero residual upside explicitly: some checkpoints have no remaining improvement at all.
- All forecast numerics are about normalized reward upside rather than dollars, prices, or rank positions.

Task:
- Forecast the remaining upside from continuing the search.
- Let Δ = max_future(best reward seen so far) - best reward seen now, with Δ >= 0.
- Output a bounded, zero-inflated forecast for Δ.

{prompt_guidance}
"""


def format_best_reward_seen(best_reward_seen: Optional[float]) -> str:
    if best_reward_seen is None:
        return "N/A"
    try:
        best_val = float(best_reward_seen)
    except Exception:
        return "N/A"
    return f"{best_val:.3f}" if best_val >= 0 else "N/A"


def _infer_page_type_for_prompt(observation: Optional[str]) -> str:
    obs = str(observation or "").lower()
    if not obs:
        return "unknown"
    if "total results" in obs or "next >" in obs:
        return "search_results"
    if "paper page" in obs or "current_paper_id:" in obs:
        return "paper_page"
    if "search literature" in obs:
        return "search_home"
    if "buy now" in obs or "description" in obs or "features" in obs:
        return "product_page"
    if "instruction:" in obs:
        return "instruction_context"
    return "unknown"


def _summarize_observation_for_prompt(observation: Optional[str], *, max_chars: int = 240) -> str:
    parts = [p.strip() for p in str(observation or "").split("[SEP]") if p.strip()]
    if len(parts) <= 1:
        return "N/A"
    cues = parts[1:]
    if cues:
        first_lower = cues[0].lower()
        if (
            len(cues) > 1
            and "instruction" not in first_lower
            and not any(
                tok in first_lower
                for tok in [
                    "page ",
                    "buy now",
                    "back to search",
                    "next >",
                    "description",
                    "features",
                    "results:",
                    "paper page",
                    "current_paper_id:",
                ]
            )
            and len(first_lower.split()) >= 4
        ):
            cues = cues[1:]
    cues = cues[:4]
    txt = " | ".join(cues).strip()
    if not txt:
        return "N/A"
    if len(txt) > int(max_chars):
        txt = txt[: max(32, int(max_chars) - 16)].rstrip() + " ..."
    return txt


def format_state_context_for_prompt(
    *,
    checkpoint_step: Optional[int] = None,
    total_horizon_steps: Optional[int] = None,
    trigger: Optional[str] = None,
    observation: Optional[str] = None,
) -> str:
    step_txt = "N/A"
    remaining_txt = "N/A"
    try:
        if checkpoint_step is not None:
            step_val = max(1, int(checkpoint_step))
            step_txt = str(step_val)
            if total_horizon_steps is not None:
                horizon_val = max(step_val, int(total_horizon_steps))
                step_txt = f"{step_val}/{horizon_val}"
                remaining_txt = str(max(1, horizon_val - step_val + 1))
    except Exception:
        pass
    page_type = _infer_page_type_for_prompt(observation)
    obs_summary = _summarize_observation_for_prompt(observation)
    trigger_txt = str(trigger or "N/A").replace("_", " ")
    return "\n".join(
        [
            f"- Checkpoint step: {step_txt}",
            f"- Remaining search steps including current step: {remaining_txt}",
            f"- Trigger: {trigger_txt}",
            f"- Current page type: {page_type}",
            f"- Current page cues: {obs_summary}",
        ]
    )


def build_stopping_prompt(
    *,
    goal: str,
    seen_products_text: str,
    best_reward_seen: Optional[float],
    top_k: int,
    checkpoint_step: Optional[int] = None,
    total_horizon_steps: Optional[int] = None,
    trigger: Optional[str] = None,
    observation: Optional[str] = None,
) -> str:
    return STOPPING_PROMPT.format(
        goal=goal,
        seen_products=seen_products_text,
        state_context=format_state_context_for_prompt(
            checkpoint_step=checkpoint_step,
            total_horizon_steps=total_horizon_steps,
            trigger=trigger,
            observation=observation,
        ),
        best_reward_seen=format_best_reward_seen(best_reward_seen),
        top_k=int(top_k),
        prompt_guidance=canonical_prompt_guidance(HURDLE_BETA_FAMILY),
    )


def format_seen_products_for_prompt(
    seen_products: Optional[Dict[str, dict]],
    best_product_asin: Optional[str] = None,
    top_n: int = 5,
) -> str:
    if not seen_products:
        return "No items seen yet."

    first_info = next(iter(seen_products.values())) if seen_products else {}
    is_paper_domain = isinstance(first_info, dict) and (
        "paper_id" in first_info or "Abstract" in first_info or "CitationCount" in first_info
    )
    if is_paper_domain:
        def paper_sort_key(item):
            pid, info = item
            reward = info.get(PAPER_RELEVANCE_REWARD_KEY, None)
            try:
                reward_val = float(reward) if reward is not None else -1.0
            except Exception:
                reward_val = -1.0
            try:
                rank_val = int(info.get("FirstSeenRank", info.get("LastVisibleRank", 10**9)) or 10**9)
            except Exception:
                rank_val = 10**9
            return (-reward_val, rank_val, str(pid))

        sorted_items = sorted(seen_products.items(), key=paper_sort_key)
        top_items = sorted_items[:top_n]
        lines = [f"Papers seen so far ({len(sorted_items)} total):"]
        for idx, (pid, info) in enumerate(top_items, start=1):
            marker = " (best)" if pid == best_product_asin else ""
            lines.append(f"\n  {idx}. PaperID: {pid}{marker}")
            lines.append(f"     Title: {info.get('Title', 'N/A')}")
            reward = info.get(PAPER_RELEVANCE_REWARD_KEY, None)
            if reward is not None:
                try:
                    lines.append(f"     RelevanceReward: {float(reward):.3f}")
                except Exception:
                    lines.append(f"     RelevanceReward: {reward}")
            rank = info.get("FirstSeenRank", info.get("LastVisibleRank", None))
            if rank is not None:
                lines.append(f"     ResultRank: {rank}")
            abstract = str(info.get("Abstract", "") or "").strip()
            if abstract:
                if len(abstract) > 220:
                    abstract = abstract[:200].rstrip() + " ..."
                lines.append(f"     Abstract: {abstract}")
        if len(sorted_items) > len(top_items):
            remaining = sorted_items[len(top_items):]
            reward_vals = []
            for _, info in remaining:
                try:
                    reward = info.get(PAPER_RELEVANCE_REWARD_KEY, None)
                    if reward is not None:
                        reward_vals.append(float(reward))
                except Exception:
                    pass
            if reward_vals:
                lines.append(
                    f"\n  Remaining {len(remaining)} papers: mean opened-paper reward {sum(reward_vals)/len(reward_vals):.3f}, "
                    f"max {max(reward_vals):.3f}"
                )
        return "\n".join(lines)

    def webshop_sort_key(item):
        asin, info = item
        reward = info.get(COUNTERFACTUAL_REWARD_KEY, info.get("reward", -1.0))
        try:
            reward_val = float(reward)
        except Exception:
            reward_val = -1.0
        price = info.get("Price", None)
        try:
            price_val = float(str(price).replace("$", "").strip()) if price is not None else 0.0
        except Exception:
            price_val = 0.0
        return (-reward_val, price_val, str(asin))

    sorted_items = sorted(seen_products.items(), key=webshop_sort_key)
    top_items = sorted_items[:top_n]
    lines = [f"Items seen so far ({len(sorted_items)} total):"]
    for idx, (asin, info) in enumerate(top_items, start=1):
        marker = " (best)" if asin == best_product_asin else ""
        lines.append(f"\n  {idx}. ASIN: {asin}{marker}")
        lines.append(f"     Title: {info.get('Title', 'N/A')}")
        reward = info.get(COUNTERFACTUAL_REWARD_KEY, None)
        if reward is not None:
            try:
                lines.append(f"     BuyNowReward: {float(reward):.3f}")
            except Exception:
                lines.append(f"     BuyNowReward: {reward}")
        price = info.get("Price", None)
        if price is not None:
            lines.append(f"     Price: {price}")
        attrs = info.get("Attributes", None)
        if attrs:
            attr_txt = ", ".join(str(a) for a in list(attrs)[:6])
            lines.append(f"     Attributes: {attr_txt}")
    if len(sorted_items) > len(top_items):
        remaining = sorted_items[len(top_items):]
        reward_vals = []
        for _, info in remaining:
            try:
                reward_vals.append(float(info.get(COUNTERFACTUAL_REWARD_KEY, info.get('reward', 0.0)) or 0.0))
            except Exception:
                pass
        if reward_vals:
            lines.append(
                f"\n  Remaining {len(remaining)} items: mean reward {sum(reward_vals)/len(reward_vals):.3f}, "
                f"max {max(reward_vals):.3f}"
            )
    return "\n".join(lines)
