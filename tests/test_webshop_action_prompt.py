from opportunity_forecasting.data.webshop_action_prompt import WEBSHOP_ACTION_PROMPT


def test_webshop_action_prompt_is_shopping_specific() -> None:
    assert WEBSHOP_ACTION_PROMPT.startswith(
        "You are a shopping assistant. Select the single best action"
    )
    assert "Shopping Goal: {goal}" in WEBSHOP_ACTION_PROMPT
    assert "right category of products" in WEBSHOP_ACTION_PROMPT
    assert "scroll, move between pages, or inspect products" in WEBSHOP_ACTION_PROMPT
