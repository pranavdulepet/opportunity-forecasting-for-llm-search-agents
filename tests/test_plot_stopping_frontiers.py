import pytest

from opportunity_forecasting.figures.allocation import portable_source as budget_source
from opportunity_forecasting.figures.stopping import (
    normalized_auc,
    portable_source as stopping_source,
    shared_endpoints,
)


def test_frontier_helpers_require_common_x_support_and_compute_auc():
    grouped = {
        "a": [
            {"mean_final_steps": 1.0, "mean_final_reward": 0.2},
            {"mean_final_steps": 3.0, "mean_final_reward": 0.6},
        ],
        "b": [
            {"mean_final_steps": 1.0, "mean_final_reward": 0.3},
            {"mean_final_steps": 3.0, "mean_final_reward": 0.5},
        ],
    }
    assert shared_endpoints(grouped, ("a", "b")) == (1.0, 3.0)
    assert normalized_auc(grouped["a"]) == pytest.approx(0.4)

    grouped["b"][0]["mean_final_steps"] = 2.0
    with pytest.raises(ValueError, match="do not share x endpoints"):
        shared_endpoints(grouped, ("a", "b"))


def test_verification_metadata_uses_paths_relative_to_runtime_source(tmp_path):
    source_root = tmp_path / "relocated"
    source = source_root / "budget" / "webshop.csv"
    assert budget_source(source, source_root) == "budget/webshop.csv"
    assert stopping_source(source, source_root) == "budget/webshop.csv"
