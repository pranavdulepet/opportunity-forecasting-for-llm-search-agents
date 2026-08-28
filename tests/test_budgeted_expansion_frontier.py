from opportunity_forecasting.evaluation.allocation import (
    _priority_for_stream,
    extract_threads,
    fixed_round_robin_curve,
    heuristic_greedy_curve,
    forecast_greedy_curve,
    oracle_frontier,
    shared_goal_ids,
    validate_shared_reward_streams,
)


def _hb(mean_delta):

    return {
        "forecast_family": "hurdle_beta",
        "delta_zero_prob": 0.0,
        "delta_one_prob": 0.0,
        "delta_pos_mean": float(mean_delta),
        "delta_pos_concentration": 4.0,
    }


def _cache():
    return {
        "goal_ids": [1, 2],
        "goal_predictions": {
            "1": {
                "checkpoints": [
                    {
                        "checkpoint_step": 2,
                        "visited_product_page": False,
                        "best_reward_seen": 0.0,
                        **_hb(0.9),
                    },
                    {
                        "checkpoint_step": 5,
                        "visited_product_page": True,
                        "best_reward_seen": 0.2,
                        **_hb(0.5),
                    },
                    {
                        "checkpoint_step": 7,
                        "visited_product_page": False,
                        "best_reward_seen": 0.6,
                        **_hb(0.1),
                    },
                ]
            },
            "2": {
                "checkpoints": [
                    {
                        "checkpoint_step": 4,
                        "visited_product_page": True,
                        "best_reward_seen": 0.1,
                        **_hb(0.2),
                    },
                    {
                        "checkpoint_step": 6,
                        "visited_product_page": True,
                        "best_reward_seen": 0.9,
                        **_hb(0.0),
                    },
                ]
            },
        },
    }


def test_extract_threads_starts_after_first_candidate_but_keeps_later_decision_points():
    streams = extract_threads(_cache(), [1, 2], offline_back_steps=1, offline_to_product_steps=2)

    assert [[c.goal_id for c in stream] for stream in streams] == [[1, 1], [2, 2]]
    assert [c.checkpoint_step for c in streams[0]] == [5, 7]
    assert [c.stop_step_cost for c in streams[0]] == [7, 9]
    assert streams[0][0].predicted_mean_delta == 0.5


def test_forecast_greedy_expands_highest_expected_remaining_first():
    streams = extract_threads(_cache(), [1, 2], offline_back_steps=1, offline_to_product_steps=2)

    curve = forecast_greedy_curve(
        domain="toy",
        split="test",
        label="model",
        streams=streams,
        priority_mode="predicted_mean_delta",
    )

    assert [round(p.mean_final_reward, 3) for p in curve] == [0.15, 0.35, 0.75]
    assert [round(p.mean_final_steps, 3) for p in curve] == [6.5, 7.5, 8.5]


def test_invalid_forecast_receives_zero_allocation_priority():
    cache = _cache()
    checkpoint = cache["goal_predictions"]["1"]["checkpoints"][1]
    checkpoint.pop("delta_zero_prob")
    streams = extract_threads(cache, [1, 2], offline_back_steps=1, offline_to_product_steps=2)

    assert not streams[0][0].forecast_valid
    assert streams[0][0].predicted_mean_delta is None
    assert _priority_for_stream(streams[0], 0, priority_mode="predicted_mean_delta") == 0.0


def test_explicit_expected_delta_takes_precedence_over_distribution_fields():
    cache = _cache()
    cache["goal_predictions"]["1"]["checkpoints"][1]["expected_delta"] = 0.01
    cache["goal_predictions"]["1"]["checkpoints"][1]["forecast_numeric_domain_ok"] = True

    streams = extract_threads(cache, [1, 2], offline_back_steps=1, offline_to_product_steps=2)

    assert streams[0][0].predicted_mean_delta == 0.01


def test_reward_stream_mismatch_is_rejected():
    cache_a = _cache()
    cache_b = _cache()
    cache_b["goal_predictions"]["1"]["checkpoints"][1]["best_reward_seen"] = 0.4

    streams_a = extract_threads(cache_a, [1, 2], offline_back_steps=1, offline_to_product_steps=2)
    streams_b = extract_threads(cache_b, [1, 2], offline_back_steps=1, offline_to_product_steps=2)
    import pytest

    with pytest.raises(ValueError, match="same fixed replay reward stream"):
        validate_shared_reward_streams({"a": streams_a, "b": streams_b})


def test_decision_point_mismatch_is_rejected():
    streams_a = extract_threads(_cache(), [1, 2], offline_back_steps=1, offline_to_product_steps=2)
    streams_b = extract_threads(_cache(), [1, 2], offline_back_steps=1, offline_to_product_steps=2)
    streams_b[0] = streams_b[0][1:]

    import pytest

    with pytest.raises(ValueError, match="same fixed replay reward stream"):
        validate_shared_reward_streams({"prompted": streams_a, "learned": streams_b})


def test_fixed_and_oracle_curves_are_available():
    cache = _cache()
    assert shared_goal_ids([cache, cache]) == [1, 2]
    streams = extract_threads(cache, [1, 2], offline_back_steps=1, offline_to_product_steps=2)

    fixed = fixed_round_robin_curve(domain="toy", split="test", streams=streams)
    oracle = oracle_frontier(domain="toy", split="test", streams=streams)

    assert fixed[0].series == "Fixed round-robin"
    assert oracle[-1].mean_final_reward == 0.75
    assert max(p.mean_final_reward for p in oracle) >= max(p.mean_final_reward for p in fixed)


def test_low_current_best_heuristic_uses_only_prefix_reward():
    streams = extract_threads(_cache(), [1, 2], offline_back_steps=1, offline_to_product_steps=2)

    curve = heuristic_greedy_curve(
        domain="toy",
        split="test",
        streams=streams,
        mode="low_current_best",
    )

    assert curve[0].series == "Heuristic: low current best"
    assert [round(p.mean_final_reward, 3) for p in curve] == [0.15, 0.55, 0.75]
    assert [round(p.mean_final_steps, 3) for p in curve] == [6.5, 7.5, 8.5]
