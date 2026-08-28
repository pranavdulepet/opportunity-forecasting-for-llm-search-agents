import math

from opportunity_forecasting.evaluation.diagnostics import _forecast_metrics, summarize


def test_zoib_cache_diagnostics_score_distribution():
    ckpt = {
        "forecast_family": "hurdle_beta",
        "delta_zero_prob": 0.2,
        "delta_one_prob": 0.1,
        "delta_pos_mean": 0.25,
        "delta_pos_concentration": 8.0,
    }
    label = {
        "deltas": [0.0, 0.2, 0.4],
        "target_ev": 0.2,
        "target_event": 2.0 / 3.0,
    }

    row = _forecast_metrics(ckpt, label, thresholds=[0.1])

    assert row["valid"] == 1.0
    assert math.isfinite(row["nll"])
    assert math.isfinite(row["crps"])
    assert math.isfinite(row["threshold_brier"])


def test_residual_gaussian_cache_diagnostics_score_distribution():
    ckpt = {
        "forecast_family": "explicit_residual_gaussian",
        "expected_delta": 0.20,
        "expected_std_delta": 0.10,
    }
    label = {
        "deltas": [0.0, 0.1, 0.3],
        "target_ev": 0.13333333333333333,
        "target_event": 2.0 / 3.0,
    }

    row = _forecast_metrics(ckpt, label, thresholds=[0.1])

    assert math.isfinite(row["nll"])
    assert math.isfinite(row["crps"])
    assert math.isfinite(row["threshold_brier"])
    assert math.isfinite(row["event_brier"])


def test_summary_excludes_invalid_numeric_forecasts_from_error_metrics():
    rows = [
        {
            "valid": 1.0,
            "pred_ev": 0.2,
            "target_ev": 0.3,
            "pred_event": 0.4,
            "target_event": 0.5,
            "ev_abs_error": 0.1,
            "ev_sq_error": 0.01,
            "event_brier": 0.01,
        },
        {
            "valid": 0.0,
            "pred_ev": 0.0,
            "target_ev": 1.0,
            "pred_event": 0.0,
            "target_event": 1.0,
            "ev_abs_error": 1.0,
            "ev_sq_error": 1.0,
            "event_brier": 1.0,
        },
    ]

    summary = summarize(rows)

    assert summary["metric_support"] == "valid_numeric_forecasts"
    assert summary["num_valid"] == 1
    assert summary["valid_rate"] == 0.5
    assert abs(summary["ev_mae"] - 0.1) < 1e-12
