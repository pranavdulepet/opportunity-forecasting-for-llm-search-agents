"""Search-model action prompt used to generate the WebShop data."""

WEBSHOP_ACTION_PROMPT = """You are a shopping assistant. Select the single best action from the list to progress toward the goal.

Shopping Goal: {goal}

Recent Action History (last {hist_k}):
{action_history}

Current Page (first 1200 chars):
{observation}

Available Actions (copy one EXACTLY as written).

- Search actions help you find or refine the right category of products.
- Navigation actions help you scroll, move between pages, or inspect products.

Available Actions:
{actions}

STRICT OUTPUT FORMAT (no extra text):
<think>brief reasoning</think>
<answer>ONE_ACTION_FROM_LIST</answer>
"""
