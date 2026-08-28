import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opportunity_forecasting.models.distributions import (
    HURDLE_BETA_FAMILY,
    compute_hurdle_beta_targets,
    forecast_brier_event_calibration,
    forecast_continuous_ranked_probability_score,
    forecast_from_fields,
    forecast_implied_moments,
    forecast_mean_log_likelihood,
    forecast_numeric_domain_ok,
    forecast_tail_probability,
    parse_forecast_response,
)


def test_parse_hurdle_beta_response_extracts_all_fields():
    txt = (
        "<think>Residual upside looks small but non-zero.</think>\n"
        "<delta_zero_prob>0.625</delta_zero_prob>\n"
        "<delta_one_prob>0.125</delta_one_prob>\n"
        "<delta_pos_mean>0.28</delta_pos_mean>\n"
        "<delta_pos_concentration>6.5</delta_pos_concentration>\n"
        "<analysis>Most continuations stall, but a small positive tail remains.</analysis>"
    )
    parsed = parse_forecast_response(txt)
    assert parsed.family == HURDLE_BETA_FAMILY
    assert parsed.delta_zero_prob == 0.625
    assert parsed.delta_one_prob == 0.125
    assert parsed.delta_pos_mean == 0.28
    assert parsed.delta_pos_concentration == 6.5
    assert forecast_numeric_domain_ok(parsed) is True


def test_hurdle_beta_implied_moments_match_closed_form():
    forecast = forecast_from_fields(
        {
            "forecast_family": HURDLE_BETA_FAMILY,
            "delta_zero_prob": 0.25,
            "delta_one_prob": 0.0,
            "delta_pos_mean": 0.4,
            "delta_pos_concentration": 4.0,
        }
    )
    mean_delta, std_delta = forecast_implied_moments(forecast)
    assert mean_delta is not None
    assert std_delta is not None
    assert abs(mean_delta - 0.3) < 1e-9
    assert 0.0 < std_delta < 0.3


def test_zero_one_inflated_beta_handles_endpoint_one():
    forecast = forecast_from_fields(
        {
            "forecast_family": HURDLE_BETA_FAMILY,
            "delta_zero_prob": 0.25,
            "delta_one_prob": 0.20,
            "delta_pos_mean": 0.4,
            "delta_pos_concentration": 4.0,
        }
    )
    assert forecast_numeric_domain_ok(forecast) is True
    mean_delta, std_delta = forecast_implied_moments(forecast)
    assert mean_delta is not None
    assert std_delta is not None
    assert abs(mean_delta - (0.20 + 0.55 * 0.4)) < 1e-9
    tail_099 = forecast_tail_probability(forecast, 0.99)
    assert tail_099 is not None and 0.20 <= tail_099 <= 0.21
    assert forecast_mean_log_likelihood([1.0, 1.0, 0.0, 0.5], forecast) > -20.0


def test_hurdle_beta_targets_capture_zero_mass_and_positive_tail():
    targets = compute_hurdle_beta_targets([0.0, 0.0, 0.2, 0.4, 0.8, 0.0, 1.0])
    assert abs(targets["delta_zero_prob_target"] - (3.0 / 7.0)) < 1e-9
    assert abs(targets["delta_one_prob_target"] - (1.0 / 7.0)) < 1e-9
    assert 0.0 < targets["delta_pos_mean_target"] < 1.0
    assert 2.0 <= targets["delta_pos_concentration_target"] <= 32.0


def test_hurdle_beta_tail_prob_and_scores_are_finite():
    forecast = forecast_from_fields(
        {
            "forecast_family": HURDLE_BETA_FAMILY,
            "delta_zero_prob": 0.5,
            "delta_one_prob": 0.0,
            "delta_pos_mean": 0.35,
            "delta_pos_concentration": 5.0,
        }
    )
    deltas = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4]
    p_tail = forecast_tail_probability(forecast, 0.25)
    ll = forecast_mean_log_likelihood(deltas, forecast)
    cal = forecast_brier_event_calibration(deltas, forecast, [0.05, 0.25, 0.5])
    assert p_tail is not None and 0.0 <= p_tail <= 1.0
    assert ll < 10.0
    assert 0.0 <= cal <= 1.0


def test_hurdle_beta_crps_prefers_better_matched_forecast():
    deltas = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4]
    good = forecast_from_fields(
        {
            "forecast_family": HURDLE_BETA_FAMILY,
            "delta_zero_prob": 0.35,
            "delta_one_prob": 0.0,
            "delta_pos_mean": 0.32,
            "delta_pos_concentration": 6.0,
        }
    )
    bad = forecast_from_fields(
        {
            "forecast_family": HURDLE_BETA_FAMILY,
            "delta_zero_prob": 0.0,
            "delta_one_prob": 0.0,
            "delta_pos_mean": 0.95,
            "delta_pos_concentration": 32.0,
        }
    )
    crps_good = forecast_continuous_ranked_probability_score(deltas, good, num_grid_points=81)
    crps_bad = forecast_continuous_ranked_probability_score(deltas, bad, num_grid_points=81)
    assert 0.0 <= crps_good <= 1.0
    assert 0.0 <= crps_bad <= 1.0
    assert crps_good < crps_bad
