from opportunity_forecasting.evaluation.stopping import (
    _eval_cached_model,
    score_forecast,
)


def test_hurdle_beta_cache_does_not_bypass_distributional_scoring():
    ckpt = {
        "forecast_family": "hurdle_beta",
        "expected_delta": 0.4,
        "expected_std_delta": 0.0,
    }
    score = score_forecast(
        ckpt,
        stop_formulation="policy2",
        lambda_risk=1.0,
        c_step=0.01,
        checkpoint_step=2,
        total_horizon_steps=60,
    )
    assert not score["score_valid"]


def test_explicit_scalar_cache_uses_expected_delta_score():
    ckpt = {
        "forecast_family": "explicit_residual_gaussian",
        "expected_delta": 0.4,
        "expected_std_delta": 0.1,
    }
    score = score_forecast(
        ckpt,
        stop_formulation="policy2",
        lambda_risk=2.0,
        c_step=0.01,
        checkpoint_step=2,
        total_horizon_steps=60,
    )
    assert abs(score["score_val"] - (0.4 - 0.3 - 0.2)) < 1e-9


def test_explicit_residual_cache_uses_remaining_horizon_cost_proxy():
    ckpt = {
        "forecast_family": "explicit_residual_scalar",
        "expected_delta": 0.4,
        "expected_std_delta": 0.0,
    }
    score = score_forecast(
        ckpt,
        stop_formulation="policy1",
        lambda_risk=0.0,
        c_step=0.01,
        checkpoint_step=2,
        total_horizon_steps=60,
    )
    assert abs(score["cost_units"] - 30.0) < 1e-9
    assert abs(score["score_val"] - (0.4 - 0.3)) < 1e-9


def test_invalid_forecast_is_fail_safe_continue_in_stopping_replay():
    cache = {
        "model_path": "invalid-prompt",
        "model_type": "forecast",
        "goal_ids": [1],
        "goal_predictions": {
            "1": {
                "checkpoints": [
                    {
                        "checkpoint_step": 5,
                        "candidate_available": True,
                        "best_reward_seen": 0.2,
                        "final_reward": 0.8,
                        "forecast_family": "hurdle_beta",
                    }
                ]
            }
        },
    }

    results, _summary, diagnostics = _eval_cached_model(
        cache_payload=cache,
        stop_formulation="policy1",
        lambda_risk=0.0,
        c_step=1.0,
        total_horizon_steps=60,
        offline_back_steps=1,
        offline_to_product_steps=2,
    )

    assert not results[0].stopped
    assert results[0].final_reward == 0.8
    assert diagnostics["valid_score_rate"] == 0.0
