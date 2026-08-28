"""Zero-one-inflated Beta forecasts for bounded remaining upside."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

try:
    from scipy.special import betainc as _scipy_betainc
    from scipy.special import betaln as _scipy_betaln
except ImportError:
    _scipy_betainc = None
    _scipy_betaln = None


HURDLE_BETA_FAMILY = "hurdle_beta"
MAX_REWARD_DELTA = 1.0
HURDLE_BETA_CONCENTRATION_MAX = 32.0
HURDLE_BETA_BETA_EPS = 1e-6
HURDLE_BETA_ZERO_TOL = 1e-12

_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_FIELD_PATTERNS = {
    field: re.compile(fr"<{field}>\s*{_NUMBER}\s*</{field}>", re.I | re.S)
    for field in (
        "delta_zero_prob",
        "delta_one_prob",
        "delta_pos_mean",
        "delta_pos_concentration",
    )
}
_ANALYSIS_PATTERN = re.compile(r"<analysis>\s*(.*?)\s*</analysis>", re.I | re.S)


@dataclass
class ForecastParseResult:
    family: Optional[str]
    analysis_text: str = ""
    raw_text: str = ""
    delta_zero_prob: Optional[float] = None
    delta_one_prob: Optional[float] = None
    delta_pos_mean: Optional[float] = None
    delta_pos_concentration: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "forecast_family": self.family,
            "delta_zero_prob": self.delta_zero_prob,
            "delta_one_prob": self.delta_one_prob,
            "delta_pos_mean": self.delta_pos_mean,
            "delta_pos_concentration": self.delta_pos_concentration,
        }


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _clip_open_unit_interval(
    value: float, *, eps: float = HURDLE_BETA_BETA_EPS
) -> float:
    return min(1.0 - eps, max(eps, float(value)))


def forecast_from_fields(fields: dict[str, Any]) -> ForecastParseResult:
    has_zoib_fields = any(
        fields.get(name) is not None
        for name in (
            "delta_zero_prob",
            "delta_one_prob",
            "delta_pos_mean",
            "delta_pos_concentration",
        )
    )
    family = str(fields.get("forecast_family") or "").strip().lower()
    if family != HURDLE_BETA_FAMILY and not has_zoib_fields:
        family = None
    else:
        family = HURDLE_BETA_FAMILY
    return ForecastParseResult(
        family=family,
        analysis_text=str(fields.get("analysis_text") or ""),
        raw_text=str(fields.get("raw_text") or ""),
        delta_zero_prob=_finite_float(fields.get("delta_zero_prob")),
        delta_one_prob=_finite_float(fields.get("delta_one_prob")),
        delta_pos_mean=_finite_float(fields.get("delta_pos_mean")),
        delta_pos_concentration=_finite_float(
            fields.get("delta_pos_concentration")
        ),
    )


def canonical_prompt_guidance(family: str = HURDLE_BETA_FAMILY) -> str:
    if family != HURDLE_BETA_FAMILY:
        raise ValueError(f"Unsupported forecast family: {family}")
    return """Return exactly one XML forecast block.
- Start immediately with <think> and end immediately with </analysis>.
- Do not repeat the prompt, schema, or instruction text.
- Do not include markdown or prose outside the tags.
- Use this exact tag order: <think>, <delta_zero_prob>, <delta_one_prob>, <delta_pos_mean>, <delta_pos_concentration>, <analysis>.
- delta_zero_prob is P(Delta = 0) and must be in [0, 1].
- delta_one_prob is P(Delta = 1) and must be in [0, 1].
- delta_zero_prob + delta_one_prob must be at most 1.
- delta_pos_mean is E[Delta | 0 < Delta < 1] and must be in (0, 1).
- delta_pos_concentration is the positive Beta concentration and must be in (0, 32].
- <think> should explain the forecast from the current state.
- <analysis> should briefly explain the remaining uncertainty.
"""


def _last_tagged_float(text: str, field: str) -> Optional[float]:
    matches = list(_FIELD_PATTERNS[field].finditer(text))
    if matches:
        return _finite_float(matches[-1].group(1))
    partial = re.findall(fr"<{field}>\s*{_NUMBER}", text, flags=re.I | re.S)
    return _finite_float(partial[-1]) if partial else None


def parse_forecast_response(text: str) -> ForecastParseResult:
    raw = str(text or "")
    analysis = _ANALYSIS_PATTERN.findall(raw)
    parsed = ForecastParseResult(
        family=None,
        analysis_text=analysis[-1].strip() if analysis else "",
        raw_text=raw,
        delta_zero_prob=_last_tagged_float(raw, "delta_zero_prob"),
        delta_one_prob=_last_tagged_float(raw, "delta_one_prob"),
        delta_pos_mean=_last_tagged_float(raw, "delta_pos_mean"),
        delta_pos_concentration=_last_tagged_float(
            raw, "delta_pos_concentration"
        ),
    )
    if any(
        value is not None
        for value in (
            parsed.delta_zero_prob,
            parsed.delta_one_prob,
            parsed.delta_pos_mean,
            parsed.delta_pos_concentration,
        )
    ):
        parsed.family = HURDLE_BETA_FAMILY
    return parsed


def forecast_numeric_domain_ok(forecast: ForecastParseResult) -> bool:
    if forecast.family != HURDLE_BETA_FAMILY:
        return False
    values = (
        forecast.delta_zero_prob,
        forecast.delta_one_prob,
        forecast.delta_pos_mean,
        forecast.delta_pos_concentration,
    )
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return False
    p_zero = float(forecast.delta_zero_prob)
    p_one = float(forecast.delta_one_prob)
    interior_mean = float(forecast.delta_pos_mean)
    concentration = float(forecast.delta_pos_concentration)
    return (
        0.0 <= p_zero <= 1.0
        and 0.0 <= p_one <= 1.0
        and p_zero + p_one <= 1.0 + HURDLE_BETA_BETA_EPS
        and 0.0 < interior_mean < 1.0
        and 0.0 < concentration <= HURDLE_BETA_CONCENTRATION_MAX
    )


def _beta_parameters(mean: float, concentration: float) -> tuple[float, float]:
    mean = _clip_open_unit_interval(mean)
    concentration = min(
        HURDLE_BETA_CONCENTRATION_MAX,
        max(HURDLE_BETA_BETA_EPS, concentration),
    )
    return (
        max(HURDLE_BETA_BETA_EPS, mean * concentration),
        max(HURDLE_BETA_BETA_EPS, (1.0 - mean) * concentration),
    )


def forecast_implied_moments(
    forecast: ForecastParseResult,
) -> tuple[Optional[float], Optional[float]]:
    if not forecast_numeric_domain_ok(forecast):
        return None, None
    p_zero = float(forecast.delta_zero_prob)
    p_one = float(forecast.delta_one_prob)
    interior_prob = max(0.0, 1.0 - p_zero - p_one)
    interior_mean = float(forecast.delta_pos_mean)
    concentration = float(forecast.delta_pos_concentration)
    interior_variance = (
        interior_mean * (1.0 - interior_mean) / (concentration + 1.0)
    )
    interior_second_moment = interior_variance + interior_mean**2
    mean = p_one + interior_prob * interior_mean
    second_moment = p_one + interior_prob * interior_second_moment
    variance = max(0.0, second_moment - mean**2)
    return mean, math.sqrt(variance)


def _beta_log_density(value: float, alpha: float, beta: float) -> float:
    value = _clip_open_unit_interval(value)
    if _scipy_betaln is not None:
        log_beta = float(_scipy_betaln(alpha, beta))
    else:
        log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (
        (alpha - 1.0) * math.log(value)
        + (beta - 1.0) * math.log(1.0 - value)
        - log_beta
    )


def _regularized_incomplete_beta(
    value: float, alpha: float, beta: float
) -> float:
    value = min(1.0, max(0.0, value))
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0

    def continued_fraction(a: float, b: float, x: float) -> float:
        fp_min = 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) >= fp_min else fp_min)
        result = d
        for iteration in range(1, 201):
            twice = 2 * iteration
            numerator = iteration * (b - iteration) * x
            denominator = (qam + twice) * (a + twice)
            d = 1.0 + numerator * d / denominator
            d = d if abs(d) >= fp_min else fp_min
            c = 1.0 + numerator / denominator / c
            c = c if abs(c) >= fp_min else fp_min
            d = 1.0 / d
            result *= d * c

            numerator = -(a + iteration) * (qab + iteration) * x
            denominator = (a + twice) * (qap + twice)
            d = 1.0 + numerator * d / denominator
            d = d if abs(d) >= fp_min else fp_min
            c = 1.0 + numerator / denominator / c
            c = c if abs(c) >= fp_min else fp_min
            d = 1.0 / d
            change = d * c
            result *= change
            if abs(change - 1.0) < 3e-14:
                break
        return result

    log_term = (
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + alpha * math.log(value)
        + beta * math.log1p(-value)
    )
    term = math.exp(log_term)
    if value < (alpha + 1.0) / (alpha + beta + 2.0):
        result = term * continued_fraction(alpha, beta, value) / alpha
    else:
        result = 1.0 - term * continued_fraction(beta, alpha, 1.0 - value) / beta
    return min(1.0, max(0.0, result))


def _beta_tail_probability(value: float, alpha: float, beta: float) -> float:
    if value <= 0.0:
        return 1.0
    if value >= 1.0:
        return 0.0
    value = _clip_open_unit_interval(value)
    cdf = (
        float(_scipy_betainc(alpha, beta, value))
        if _scipy_betainc is not None
        else _regularized_incomplete_beta(value, alpha, beta)
    )
    return min(1.0, max(0.0, 1.0 - cdf))


def forecast_tail_probability(
    forecast: ForecastParseResult, threshold: float
) -> Optional[float]:
    if not forecast_numeric_domain_ok(forecast):
        return None
    p_zero = float(forecast.delta_zero_prob)
    p_one = float(forecast.delta_one_prob)
    interior_prob = max(0.0, 1.0 - p_zero - p_one)
    if threshold <= 0.0:
        return 1.0 - p_zero
    if threshold >= 1.0:
        return 0.0
    alpha, beta = _beta_parameters(
        float(forecast.delta_pos_mean),
        float(forecast.delta_pos_concentration),
    )
    return p_one + interior_prob * _beta_tail_probability(threshold, alpha, beta)


def forecast_mean_log_likelihood(
    deltas: Sequence[float], forecast: ForecastParseResult
) -> float:
    if not deltas:
        return 0.0
    if not forecast_numeric_domain_ok(forecast):
        return -1e6
    p_zero = _clip_probability(float(forecast.delta_zero_prob))
    p_one = _clip_probability(float(forecast.delta_one_prob))
    interior_prob = max(HURDLE_BETA_BETA_EPS, 1.0 - p_zero - p_one)
    alpha, beta = _beta_parameters(
        float(forecast.delta_pos_mean),
        float(forecast.delta_pos_concentration),
    )
    values: list[float] = []
    for delta in deltas:
        delta = min(MAX_REWARD_DELTA, max(0.0, float(delta)))
        if delta <= HURDLE_BETA_ZERO_TOL:
            values.append(math.log(max(HURDLE_BETA_BETA_EPS, p_zero)))
        elif delta >= 1.0 - HURDLE_BETA_BETA_EPS:
            values.append(math.log(max(HURDLE_BETA_BETA_EPS, p_one)))
        else:
            values.append(
                math.log(interior_prob) + _beta_log_density(delta, alpha, beta)
            )
    return sum(values) / len(values)


def forecast_brier_event_calibration(
    deltas: Sequence[float],
    forecast: ForecastParseResult,
    thresholds: Sequence[float],
) -> float:
    if not deltas or not thresholds:
        return 0.0
    scores: list[float] = []
    for threshold in thresholds:
        probability = forecast_tail_probability(forecast, threshold)
        if probability is None:
            return 1.0
        observed = sum(delta > threshold for delta in deltas) / len(deltas)
        scores.append((probability - observed) ** 2)
    return sum(scores) / len(scores)


def forecast_continuous_ranked_probability_score(
    deltas: Sequence[float],
    forecast: ForecastParseResult,
    *,
    num_grid_points: int = 41,
) -> float:
    """Approximate CRPS by trapezoidal integration over [0, 1]."""
    if not deltas:
        return 0.0
    grid_size = max(3, num_grid_points)
    grid = [index / (grid_size - 1) for index in range(grid_size)]
    cdf: list[float] = []
    for point in grid:
        tail = forecast_tail_probability(forecast, point)
        if tail is None:
            return 1.0
        cdf.append(min(1.0, max(0.0, 1.0 - tail)))

    scores: list[float] = []
    for delta in deltas:
        observed_value = min(1.0, max(0.0, float(delta)))
        area = 0.0
        for index in range(len(grid) - 1):
            indicator_left = 1.0 if observed_value <= grid[index] else 0.0
            indicator_right = 1.0 if observed_value <= grid[index + 1] else 0.0
            left = (cdf[index] - indicator_left) ** 2
            right = (cdf[index + 1] - indicator_right) ** 2
            area += 0.5 * (left + right) * (grid[index + 1] - grid[index])
        scores.append(area)
    return sum(scores) / len(scores)


def compute_hurdle_beta_targets(
    continuation_deltas: Sequence[float],
) -> dict[str, float]:
    deltas = [
        min(MAX_REWARD_DELTA, max(0.0, float(delta)))
        for delta in continuation_deltas
        if delta is not None
    ]
    if not deltas:
        return {
            "delta_zero_prob_target": 1.0,
            "delta_one_prob_target": 0.0,
            "delta_pos_mean_target": 0.5,
            "delta_pos_concentration_target": 2.0,
        }
    zeros = sum(delta <= HURDLE_BETA_ZERO_TOL for delta in deltas)
    ones = sum(delta >= 1.0 - HURDLE_BETA_BETA_EPS for delta in deltas)
    interior = [
        delta
        for delta in deltas
        if HURDLE_BETA_ZERO_TOL < delta < 1.0 - HURDLE_BETA_BETA_EPS
    ]
    p_zero = zeros / len(deltas)
    p_one = ones / len(deltas)
    if not interior:
        interior_mean = 0.5
        concentration = 2.0
    else:
        interior_mean = sum(interior) / len(interior)
        if len(interior) == 1:
            concentration = 4.0
        else:
            variance = sum(
                (delta - interior_mean) ** 2 for delta in interior
            ) / len(interior)
            max_variance = max(
                HURDLE_BETA_BETA_EPS,
                interior_mean * (1.0 - interior_mean),
            )
            variance = max(
                max_variance / (HURDLE_BETA_CONCENTRATION_MAX + 1.0),
                variance,
            )
            concentration = min(
                HURDLE_BETA_CONCENTRATION_MAX,
                max(2.0, max_variance / variance - 1.0),
            )
    return {
        "delta_zero_prob_target": _clip_probability(p_zero),
        "delta_one_prob_target": _clip_probability(p_one),
        "delta_pos_mean_target": _clip_open_unit_interval(interior_mean),
        "delta_pos_concentration_target": concentration,
    }
