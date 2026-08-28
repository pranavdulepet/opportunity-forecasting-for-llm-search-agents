import math

import torch

from opportunity_forecasting.models.train_regression import (
    DEFAULT_GAUSSIAN_STD_FLOOR,
    ResidualHeadModel,
    _is_better_trained_checkpoint,
    _safe_float,
    build_cache,
    residual_loss,
)


class _Backbone(torch.nn.Module):
    class _Out:
        def __init__(self, h):
            self.last_hidden_state = h

    def forward(self, input_ids, attention_mask, output_hidden_states=False, use_cache=False):
        batch, seq = input_ids.shape
        h = torch.ones((batch, seq, 4), dtype=torch.float32)
        return self._Out(h)

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)


def test_scalar_checkpoint_null_gaussian_floor_uses_default():
    assert _safe_float(None, DEFAULT_GAUSSIAN_STD_FLOOR) == DEFAULT_GAUSSIAN_STD_FLOOR


def test_checkpoint_selection_excludes_untrained_initialization():
    assert _is_better_trained_checkpoint(epoch=1, metric=3.0, best_metric=float("inf"))
    try:
        _is_better_trained_checkpoint(epoch=0, metric=0.1, best_metric=float("inf"))
    except ValueError as exc:
        assert "trained epoch" in str(exc)
    else:
        raise AssertionError("epoch 0 must not be selectable as a trained checkpoint")


def test_residual_scalar_head_is_pure_ev_loss():
    preds = {
        "expected_gain": torch.tensor([[0.25], [0.10]], dtype=torch.float32),
    }
    batch = {
        "target_gain": torch.tensor([[0.20], [0.00]], dtype=torch.float32),
        "target_event": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
    }

    loss, metrics = residual_loss(
        preds, batch, target_family="residual_scalar"
    )
    expected = torch.nn.functional.smooth_l1_loss(
        preds["expected_gain"], batch["target_gain"]
    )

    assert torch.isclose(loss, expected)
    assert metrics["ev_loss"] == metrics["loss"]


def test_residual_gaussian_head_outputs_mean_std_and_loss():
    model = ResidualHeadModel(
        _Backbone(),
        4,
        output_dim=2,
        target_family="residual_gaussian",
        gaussian_std_floor=0.01,
    )
    preds = model(torch.ones((2, 5), dtype=torch.long), torch.ones((2, 5), dtype=torch.long))
    batch = {
        "target_gain": torch.tensor([[0.20, 0.05], [0.00, 0.00]], dtype=torch.float32),
        "target_event": torch.tensor([[0.75], [0.00]], dtype=torch.float32),
    }

    loss, metrics = residual_loss(
        preds, batch, target_family="residual_gaussian"
    )

    assert preds["expected_gain"].shape == (2, 1)
    assert preds["std_gain"].shape == (2, 1)
    assert torch.all(preds["std_gain"] > 0)
    assert math.isfinite(loss.item())
    assert torch.isclose(loss, torch.tensor(metrics["nll"]), atol=1e-6)
    assert metrics["conditional_loss"] >= 0
    assert metrics["mean_pred_std"] > 0


def test_residual_gaussian_head_enforces_std_floor():
    model = ResidualHeadModel(
        _Backbone(),
        4,
        output_dim=2,
        target_family="residual_gaussian",
        gaussian_std_floor=0.01,
    )
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias[1] = -100.0
    preds = model(torch.ones((2, 5), dtype=torch.long), torch.ones((2, 5), dtype=torch.long))

    assert torch.all(preds["std_gain"] >= 0.01)


def test_residual_gaussian_nll_includes_continuation_variance():
    preds = {
        "expected_gain": torch.tensor([[0.20]], dtype=torch.float32),
        "std_gain": torch.tensor([[0.10]], dtype=torch.float32),
    }
    batch = {
        "target_gain": torch.tensor([[0.20, 0.10]], dtype=torch.float32),
        "target_event": torch.tensor([[0.50]], dtype=torch.float32),
    }

    loss, metrics = residual_loss(
        preds, batch, target_family="residual_gaussian"
    )
    expected = 0.5 * math.log(2.0 * math.pi) + math.log(0.10) + 0.5

    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)
    assert math.isclose(metrics["nll"], expected, abs_tol=1e-6)


def test_residual_head_cache_contains_explicit_expected_delta(tmp_path):
    predictions = [
        (
            {
                "goal_idx": 5,
                "goal_text": "find a thing",
                "checkpoint_step": 2,
                "visited_product_page": True,
                "best_reward_seen": 0.25,
            },
            {
                "forecast_family": "explicit_residual_scalar",
                "expected_delta": 0.12,
                "expected_std_delta": 0.03,
                "event_probability": 1.0,
            },
        )
    ]

    payload = build_cache(
        examples=[],
        predictions=predictions,
        model_dir=tmp_path / "model",
        output_path=tmp_path / "cache.json",
        split_name="test",
        label_source=tmp_path / "labels.jsonl",
    )

    row = payload["goal_predictions"]["5"]["checkpoints"][0]
    assert row["expected_delta"] == 0.12
    assert row["expected_std_delta"] == 0.03
    assert row["forecast_numeric_domain_ok"] is True
