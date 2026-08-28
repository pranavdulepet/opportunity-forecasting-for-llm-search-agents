"""
Shared reward helpers for WebShop checkpointing, labeling, and evaluation.

The active pipeline uses a single dense notion of "best reward seen so far":
the best reward the environment would return if we clicked "buy now" on a
visited product page right now, using the session's currently selected options.
Search-result impressions alone do not count.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

COUNTERFACTUAL_REWARD_KEY = "buy_now_reward"
CURRENT_PAGE_REWARD_MODE = "product_page_buy_now_current_options"

COUNTERFACTUAL_REWARD_MODE = CURRENT_PAGE_REWARD_MODE
PAPER_RELEVANCE_REWARD_KEY = "paper_relevance_reward"
PAPER_RELEVANCE_REWARD_MODE = "paper_page_litsearch_webshop_relevance_v4"
PAPER_RELEVANCE_REWARD_MODE_ALIASES = (PAPER_RELEVANCE_REWARD_MODE,)


def webshop_get_reward(**kwargs):
    from opportunity_forecasting.data.webshop_setup import ensure_public_webshop_imports

    ensure_public_webshop_imports()
    from web_agent_site.engine.goal import get_reward

    return get_reward(**kwargs)


def _get_server(env):
    if hasattr(env, "browser") and hasattr(env.browser, "server"):
        return env.browser.server
    return getattr(env, "server", None)


def _get_session_dict(env) -> Optional[dict]:
    server = _get_server(env)
    session = getattr(env, "session", None)
    if server is None or session is None:
        return None
    sess = server.user_sessions.get(str(session), None)
    return sess if isinstance(sess, dict) else None


def reward_keys_for_mode(reward_mode: str) -> Tuple[str, ...]:
    if str(reward_mode or "").strip().lower() in {
        alias.lower() for alias in PAPER_RELEVANCE_REWARD_MODE_ALIASES
    }:
        return (PAPER_RELEVANCE_REWARD_KEY, "reward")
    return (COUNTERFACTUAL_REWARD_KEY, "reward")


def is_product_page_observation(obs: Optional[str], env=None) -> bool:
    if env is not None and hasattr(env, "is_item_page_observation"):
        try:
            return bool(env.is_item_page_observation(obs))
        except Exception:
            pass
    return "buy now" in str(obs or "").lower()


def get_goal_dict_from_env(env) -> Optional[dict]:
    sess = _get_session_dict(env)
    if sess is None:
        return None
    goal = sess.get("goal", None)
    return goal if isinstance(goal, dict) else None


def get_product_and_price_from_env(asin: str, env) -> Tuple[Optional[dict], Optional[float]]:
    server = _get_server(env)
    if server is None:
        return None, None
    product_dict = getattr(server, "product_item_dict", {}) or {}
    product = product_dict.get(str(asin).upper(), None)
    price = (getattr(server, "product_prices", {}) or {}).get(str(asin).upper(), None)
    return product, price


def get_current_page_asin_from_env(env, obs: Optional[str] = None) -> Optional[str]:
    if obs is not None and not is_product_page_observation(obs, env=env):
        return None
    sess = _get_session_dict(env)
    if sess is None:
        return None
    asin = sess.get("asin", None)
    if asin is None:
        return None
    asin_txt = str(asin).strip().upper()
    return asin_txt or None


def get_current_session_options_from_env(env) -> Dict[str, str]:
    sess = _get_session_dict(env)
    if sess is None:
        return {}
    raw_options = sess.get("options", {}) or {}
    if not isinstance(raw_options, dict):
        return {}
    return {str(k): str(v) for k, v in raw_options.items()}


def compute_current_session_buy_now_reward(
    env,
    *,
    obs: Optional[str] = None,
) -> Tuple[float, Dict[str, str], Optional[str]]:
    if hasattr(env, "compute_current_page_reward"):
        try:
            reward, meta, item_id = env.compute_current_page_reward(obs=obs)
            meta_dict = dict(meta) if isinstance(meta, dict) else {}
            item_txt = str(item_id).strip() if item_id is not None else None
            return float(reward), meta_dict, (item_txt or None)
        except Exception:
            pass

    if obs is not None and not is_product_page_observation(obs, env=env):
        return 0.0, {}, None

    server = _get_server(env)
    if server is None:
        return 0.0, {}, None

    session = getattr(env, "session", None)
    try:
        session_key = int(session) if session is not None else -1
    except Exception:
        session_key = -1

    asin = get_current_page_asin_from_env(env, obs=obs)
    if asin is None:
        return 0.0, {}, None

    options = get_current_session_options_from_env(env)
    cache = getattr(server, "_ccs_current_page_reward_cache", None)
    if cache is None:
        cache = {}
        setattr(server, "_ccs_current_page_reward_cache", cache)
    key = (session_key, asin, tuple(sorted(options.items())))
    cached = cache.get(key, None)
    if cached is not None:
        return float(cached[0]), dict(cached[1]), asin

    goal = get_goal_dict_from_env(env)
    product, price = get_product_and_price_from_env(asin, env)
    if not isinstance(product, dict) or not isinstance(goal, dict) or price is None:
        return 0.0, {}, asin

    try:
        reward = webshop_get_reward(
            purchased_product=product,
            goal=goal,
            price=price,
            options=options,
        )
        reward_val = max(0.0, float(reward))
    except Exception:
        reward_val = 0.0
    cache[key] = (reward_val, dict(options))
    return reward_val, dict(options), asin


def best_reward_from_seen_products(seen_products: Dict[str, dict]) -> float:
    best = 0.0
    for _, info in (seen_products or {}).items():
        if not isinstance(info, dict):
            continue
        for key in (COUNTERFACTUAL_REWARD_KEY, PAPER_RELEVANCE_REWARD_KEY, "reward"):
            value = info.get(key, None)
            if value is None:
                continue
            try:
                best = max(best, float(value))
            except Exception:
                pass
    return float(best)
