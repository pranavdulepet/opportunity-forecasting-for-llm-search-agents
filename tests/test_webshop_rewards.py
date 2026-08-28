from __future__ import annotations

from types import SimpleNamespace

import opportunity_forecasting.data.webshop_rewards as rewards


def test_webshop_reward_uses_only_opened_page_current_options(monkeypatch) -> None:
    session = {
        "asin": "B000000001",
        "goal": {"instruction_text": "blue shirt"},
        "options": {"color": "blue"},
    }
    server = SimpleNamespace(
        user_sessions={"0": session},
        product_item_dict={"B000000001": {"Title": "shirt"}},
        product_prices={"B000000001": 20.0},
    )
    env = SimpleNamespace(
        session=0,
        browser=SimpleNamespace(server=server),
        is_item_page_observation=lambda observation: "buy now" in observation.lower(),
    )
    calls = []

    def get_reward(**kwargs):
        calls.append(kwargs)
        return 0.75 if kwargs["options"] == {"color": "blue"} else 0.25

    monkeypatch.setattr(rewards, "webshop_get_reward", get_reward)

    assert rewards.compute_current_session_buy_now_reward(
        env,
        obs="search results",
    ) == (0.0, {}, None)
    assert calls == []

    reward, options, asin = rewards.compute_current_session_buy_now_reward(
        env,
        obs="Buy Now",
    )
    assert (reward, options, asin) == (0.75, {"color": "blue"}, "B000000001")
    assert calls[-1]["options"] == {"color": "blue"}

    session["options"] = {"color": "red"}
    reward, options, asin = rewards.compute_current_session_buy_now_reward(
        env,
        obs="Buy Now",
    )
    assert (reward, options, asin) == (0.25, {"color": "red"}, "B000000001")
    assert calls[-1]["options"] == {"color": "red"}
