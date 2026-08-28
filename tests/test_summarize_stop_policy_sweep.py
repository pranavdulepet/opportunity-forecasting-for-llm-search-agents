from opportunity_forecasting.evaluation.summarize_stopping import common_support_frontier_rows, per_model_frontier_rows


def _row(label, c_step, steps, reward, valid):
    return {
        "policy": "policy1",
        "model_label": label,
        "model_path": label,
        "c_step": c_step,
        "lambda_risk": 1.0,
        "mean_final_reward": reward,
        "mean_final_steps": steps,
        "stop_rate": 0.0,
        "utility_proxy": reward,
        "valid_score_rate": valid,
        "valid_forecast_rate": valid,
        "numeric_parse_rate": valid,
    }


def test_frontiers_are_clipped_to_common_observed_support():
    rows = [
        _row("trained", -0.1, 60.0, 0.8, 1.0),
        _row("prompt", -0.1, 60.0, 0.8, 0.8),
        _row("trained", 1.0, 10.0, 0.5, 1.0),
        _row("prompt", 1.0, 12.0, 0.55, 0.8),
    ]

    common = common_support_frontier_rows(rows)
    trained = [row for row in common if row["model_label"] == "trained"]
    assert min(row["mean_final_steps"] for row in trained) == 12.0
    assert max(row["mean_final_steps"] for row in trained) == 60.0
    assert trained[0]["policy"] == "common_support_interpolation"

    frontier = per_model_frontier_rows(rows)
    for label in ("trained", "prompt"):
        steps = [row["mean_final_steps"] for row in frontier if row["model_label"] == label]
        assert min(steps) == 12.0
        assert max(steps) == 60.0
