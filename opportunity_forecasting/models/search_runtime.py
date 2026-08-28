"""Fixed search-model action and environment-state helpers."""

from __future__ import annotations

import hashlib
import html
import re
from typing import Any, Optional

import torch

from opportunity_forecasting.data.webshop_rewards import (
    COUNTERFACTUAL_REWARD_KEY,
    compute_current_session_buy_now_reward,
    is_product_page_observation,
)
from opportunity_forecasting.models.prompted_forecaster import (
    format_seen_products_for_prompt as _format_seen_products_for_prompt,
)


SELECTION_PROMPT_QWEN = """You are a search assistant. Select the single best action from the list to progress toward the goal.

Search Goal: {goal}

Recent Action History (last {hist_k}):
{action_history}

Current Page (first 1200 chars):
{observation}

Available Actions (copy one EXACTLY as written).

- Search actions help you explore or refine the query.
- Navigation actions help you move between pages or inspect candidate items.

Available Actions:
{actions}

STRICT OUTPUT FORMAT (no extra text):
<think>brief reasoning</think>
<answer>ONE_ACTION_FROM_LIST</answer>
"""


def parse_qwen_answer_only(text: str) -> tuple[str, str]:
    """Extract the action and optional rationale from Qwen's XML response."""
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    thoughts = re.findall(r"<think>\s*(.*?)\s*</think>", text, flags=re.S | re.I)
    return (
        answers[-1].strip() if answers else "",
        thoughts[-1].strip() if thoughts else "",
    )


def _categorise_action(action: str) -> str:
    normalized = action.strip().lower()
    if normalized.startswith("search["):
        return "search"
    if (
        normalized.startswith("buy[")
        or "buy now" in normalized
        or normalized.startswith("stop")
    ):
        return "terminal"
    return "navigation"


def _format_actions(valid_actions: list[str]) -> str:
    groups = {
        "broad": [],
        "refine": [],
        "navigation": [],
        "terminal": [],
    }
    for action in valid_actions:
        category = _categorise_action(action)
        if category == "search":
            match = re.match(r"search\[(.*)\]", action, flags=re.I)
            query = match.group(1) if match else action
            groups["broad" if len(query.strip().split()) <= 3 else "refine"].append(
                action
            )
        else:
            groups[category].append(action)

    lines: list[str] = []
    number = 1
    if groups["broad"] or groups["refine"]:
        lines.append("Search actions:")
        if groups["broad"]:
            lines.append("  Broad search actions (good for initial exploration):")
            for action in groups["broad"]:
                lines.append(f"    {number}. {action}")
                number += 1
        if groups["refine"]:
            lines.append("  Refine search actions (use when you know more about what you need):")
            for action in groups["refine"]:
                lines.append(f"    {number}. {action}")
                number += 1
    if groups["navigation"]:
        lines.append("Navigation and inspection actions:")
        for action in groups["navigation"]:
            lines.append(f"  {number}. {action}")
            number += 1
    if groups["terminal"]:
        lines.append("Terminal actions (commit to the current best item or stop):")
        for action in groups["terminal"]:
            lines.append(f"  {number}. {action}")
            number += 1
    return "\n".join(lines) if lines else "None"


def _match_generated_action(text: str, valid_actions: list[str]) -> str:
    normalized = html.unescape(text.strip())
    lowered = normalized.lower()
    for action in valid_actions:
        if action.lower() == lowered:
            return action

    without_number = re.sub(r"^\d+\.\s*", "", normalized).strip()
    for action in valid_actions:
        if action.lower() == without_number.lower():
            return action

    if not lowered.startswith(("click[", "search[")):
        wrapped = f"click[{normalized}]".lower()
        for action in valid_actions:
            if action.lower() == wrapped:
                return action

    item_id = re.search(r"b[0-9a-z]{9}", lowered)
    if item_id:
        for action in valid_actions:
            if item_id.group(0) in action.lower():
                return action

    if len(normalized) > 3:
        for action in valid_actions:
            action_lower = action.lower()
            if lowered in action_lower or action_lower in lowered:
                return action

    try:
        index = int(normalized) - 1
        if 0 <= index < len(valid_actions):
            return valid_actions[index]
    except ValueError:
        pass
    return valid_actions[0] if valid_actions else "stop"


def select_action_qwen(
    goal: str,
    obs: str,
    valid_actions: list[str],
    model: Any,
    tokenizer: Any,
    action_history: Optional[list[str]] = None,
    history_k: int = 60,
) -> tuple[str, str, str, str]:
    """Select one valid action with the fixed Qwen search model."""
    recent = (action_history or [])[-history_k:]
    prompt = SELECTION_PROMPT_QWEN.format(
        goal=goal,
        action_history="\n".join(f"- {action}" for action in recent) or "None",
        observation=(obs or "")[:1200],
        actions=_format_actions(valid_actions),
        hist_k=history_k,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    action_text, rationale = parse_qwen_answer_only(response)
    return (
        _match_generated_action(action_text, valid_actions),
        rationale,
        prompt,
        response,
    )


def observation_signature(observation: Optional[str], max_len: int = 600) -> Optional[str]:
    if not observation:
        return None
    snippet = observation[:max_len].encode("utf-8", errors="ignore")
    return hashlib.md5(snippet).hexdigest()


def extract_asins_from_observation(observation: str, env: Any = None) -> list[str]:
    if env is not None and hasattr(env, "extract_item_ids_from_observation"):
        try:
            ids = env.extract_item_ids_from_observation(observation)
            unique = list(dict.fromkeys(str(item).strip() for item in ids or []))
            if any(unique):
                return [item for item in unique if item]
        except Exception:
            pass
    matches = re.findall(r"\bB[0-9A-Z]{9}\b", (observation or "").upper())
    return list(dict.fromkeys(matches))


def _item_info(item_id: str, env: Any) -> Optional[dict[str, Any]]:
    if hasattr(env, "get_item_info"):
        try:
            info = env.get_item_info(item_id)
            if isinstance(info, dict):
                return info
        except Exception:
            pass
    try:
        product = env.browser.server.product_item_dict[item_id]
    except Exception:
        return None
    return {
        "asin": item_id,
        "Title": product.get("Title", "N/A"),
        "Price": product.get("Price", "N/A"),
        "Rating": product.get("Rating", "N/A"),
        "Description": (
            product.get("Description", "")[:200] + "..."
            if product.get("Description")
            else "N/A"
        ),
        "BulletPoints": (
            product.get("BulletPoints", [])[:3]
            if isinstance(product.get("BulletPoints"), list)
            else "N/A"
        ),
        COUNTERFACTUAL_REWARD_KEY: None,
    }


def update_seen_products(
    observation: str,
    seen_products: dict[str, dict],
    env: Any,
) -> None:
    if hasattr(env, "update_seen_items_dict"):
        try:
            env.update_seen_items_dict(observation, seen_products)
            return
        except Exception:
            pass

    for item_id in extract_asins_from_observation(observation, env=env):
        if item_id not in seen_products:
            info = _item_info(item_id, env)
            if info:
                seen_products[item_id] = info

    if not is_product_page_observation(observation, env=env):
        return
    try:
        reward, options, item_id = compute_current_session_buy_now_reward(
            env, obs=observation
        )
    except Exception:
        return
    if not item_id:
        return
    if item_id not in seen_products:
        info = _item_info(item_id, env)
        if info:
            seen_products[item_id] = info
    if item_id in seen_products:
        previous = seen_products[item_id].get(COUNTERFACTUAL_REWARD_KEY)
        try:
            previous_value = float(previous) if previous is not None else 0.0
        except (TypeError, ValueError):
            previous_value = 0.0
        seen_products[item_id][COUNTERFACTUAL_REWARD_KEY] = max(
            previous_value, float(reward)
        )
        if options:
            seen_products[item_id]["buy_now_options"] = dict(options)


def format_seen_products_for_prompt(
    seen_products: Optional[dict[str, dict]],
    best_product_asin: Optional[str] = None,
    top_n: int = 5,
) -> str:
    return _format_seen_products_for_prompt(
        seen_products=seen_products,
        best_product_asin=best_product_asin,
        top_n=top_n,
    )


def _search_texts_from_goal(env: Any) -> list[str]:
    session = env.browser.server.user_sessions.get(env.session, {})
    goal = session.get("goal", {}) if isinstance(session, dict) else {}
    attributes = goal.get("attributes", []) or []
    query = goal.get("query", "") or ""
    instruction = goal.get("instruction_text", "") or ""

    def strip_price(text: str) -> str:
        text = " ".join(text.split())
        patterns = (
            r"\bunder\s*\$?\s*\d+[.,]?\d*(?:\s*(?:dollars|bucks|usd))?\b",
            r"\bless\s+than\s*\$?\s*\d+[.,]?\d*(?:\s*(?:dollars|bucks|usd))?\b",
            r"\bbelow\s*\$?\s*\d+[.,]?\d*\b",
            r"\$\s*\d+[.,]?\d*\s*(?:or\s*less|max|maximum)",
            r"\b(?:max|maximum|budget)\s*\$?\s*\d+[.,]?\d*\b",
        )
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.I)
        return re.sub(r"\s{2,}", " ", text).strip(" ,;.-")

    texts = [query, *(f"{attribute} {query}" for attribute in attributes), strip_price(instruction)]
    normalized = (" ".join(text.lower().split()) for text in texts if text)
    return list(dict.fromkeys(text for text in normalized if text)) or ["products"]


def get_valid_actions_from_env(env: Any) -> list[str]:
    available = env.get_available_actions()
    if isinstance(available, dict) and isinstance(
        available.get("valid_actions"), list
    ):
        return [str(action) for action in available["valid_actions"]]
    if available["has_search_bar"]:
        return [f"search[{text}]" for text in _search_texts_from_goal(env)]

    actions: list[str] = []
    seen: set[str] = set()
    for raw in available["clickables"]:
        text = str(raw).strip()
        if text.lower().replace(" ", "") in {"buynow", "buy", "buynow!"}:
            text = "buy now"
        if text.lower() not in seen:
            seen.add(text.lower())
            actions.append(f"click[{text}]")
    return actions or ["stop"]
