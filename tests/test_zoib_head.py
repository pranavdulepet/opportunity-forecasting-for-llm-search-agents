import torch

from opportunity_forecasting.evaluation.allocation import extract_threads
from opportunity_forecasting.models.distributions import HURDLE_BETA_FAMILY
from opportunity_forecasting.models.train_zoib import (
    build_cache_from_labeled_examples,
    hurdle_beta_distribution_loss,
    should_select_checkpoint,
)


def _batch(deltas):
    x = torch.tensor([deltas], dtype=torch.float32)
    return {
        "deltas": x,
        "delta_mask": torch.ones_like(x),
        "mean_delta": x.mean(dim=1),
        "pos_rate": (x > 1e-12).float().mean(dim=1),
        "p0_target": torch.tensor([float((x <= 1e-12).float().mean())]),
        "p1_target": torch.tensor([float((x >= 1.0 - 1e-6).float().mean())]),
        "m_pos_target": torch.tensor([0.4]),
        "k_pos_target": torch.tensor([4.0]),
    }


def _pred(p0, m, k, p1=0.0):
    return {
        "delta_zero_prob": torch.tensor([p0], dtype=torch.float32),
        "delta_one_prob": torch.tensor([p1], dtype=torch.float32),
        "delta_pos_mean": torch.tensor([m], dtype=torch.float32),
        "delta_pos_concentration": torch.tensor([k], dtype=torch.float32),
        "expected_delta": torch.tensor([p1 + max(0.0, 1.0 - p0 - p1) * m], dtype=torch.float32),
    }


def test_hurdle_head_loss_prefers_matching_zero_mass():
    batch = _batch([0.0, 0.0, 0.0, 0.4, 0.4, 0.4])
    good, _ = hurdle_beta_distribution_loss(_pred(0.5, 0.4, 8.0), batch)
    bad, _ = hurdle_beta_distribution_loss(_pred(0.05, 0.9, 2.0), batch)
    assert torch.isfinite(good)
    assert good < bad


def test_hurdle_head_loss_prefers_matching_one_mass():
    batch = _batch([1.0, 1.0, 0.0, 0.4, 0.4, 0.4])
    good, _ = hurdle_beta_distribution_loss(_pred(1.0 / 6.0, 0.4, 8.0, p1=2.0 / 6.0), batch)
    bad, _ = hurdle_beta_distribution_loss(_pred(1.0 / 6.0, 0.4, 8.0, p1=0.01), batch)
    assert torch.isfinite(good)
    assert good < bad


def test_checkpoint_selection_excludes_untrained_epoch_zero():
    assert not should_select_checkpoint(epoch=0, metric=0.1, best_metric=None)
    assert should_select_checkpoint(epoch=1, metric=0.2, best_metric=None)
    assert should_select_checkpoint(epoch=2, metric=0.1, best_metric=0.2)
    assert not should_select_checkpoint(epoch=2, metric=0.3, best_metric=0.2)


def test_hurdle_head_cache_matches_budgeted_frontier_schema(tmp_path):
    examples = [
        {
            "goal_idx": 7,
            "goal_text": "find a paper",
            "checkpoint_step": 3,
            "input": {"goal": "find a paper", "best_reward_seen": 0.2, "observation": "Paper page"},
            "continuation_deltas": [0.2, 0.0, 0.0, 0.2, 0.0, 0.0],
            "metadata": {"trigger": "paper_page"},
        },
        {
            "goal_idx": 7,
            "goal_text": "find a paper",
            "checkpoint_step": 6,
            "input": {"goal": "find a paper", "best_reward_seen": 0.5, "observation": "Results"},
            "continuation_deltas": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "metadata": {"trigger": "new_product"},
        },
    ]
    predictions = [
        (
            {
                "row_idx": 0,
                "goal_idx": 7,
                "goal_text": "find a paper",
                "checkpoint_step": 3,
                "visited_product_page": True,
                "best_reward_seen": 0.2,
            },
            {
                "forecast_family": HURDLE_BETA_FAMILY,
                "delta_zero_prob": 0.5,
                "delta_one_prob": 0.0,
                "delta_pos_mean": 0.2,
                "delta_pos_concentration": 4.0,
            },
        ),
        (
            {
                "row_idx": 1,
                "goal_idx": 7,
                "goal_text": "find a paper",
                "checkpoint_step": 6,
                "visited_product_page": True,
                "best_reward_seen": 0.5,
            },
            {
                "forecast_family": HURDLE_BETA_FAMILY,
                "delta_zero_prob": 1.0 - 1e-6,
                "delta_one_prob": 0.0,
                "delta_pos_mean": 0.5,
                "delta_pos_concentration": 2.0,
            },
        ),
    ]
    out = tmp_path / "head_predictions.json"
    payload = build_cache_from_labeled_examples(
        examples=examples,
        predictions=predictions,
        model_dir=tmp_path / "model",
        output_path=out,
        split_name="test",
        label_source=tmp_path / "labels.jsonl",
    )
    streams = extract_threads(payload, [7], offline_back_steps=0, offline_to_product_steps=0)
    assert out.exists()
    assert payload["prediction_health"]["valid_forecast_rate"] == 1.0
    assert len(streams) == 1
    assert [c.checkpoint_step for c in streams[0]] == [3, 6]
    assert round(streams[0][0].predicted_mean_delta, 3) == 0.1
