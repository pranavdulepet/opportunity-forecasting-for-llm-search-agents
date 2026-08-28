import json

from opportunity_forecasting.evaluation.merge_predictions import merge


def _payload(goal_id, checkpoint_count):
    return {
        "goal_ids": [goal_id],
        "goal_predictions": {
            str(goal_id): {
                "goal_idx": goal_id,
                "checkpoints": [
                    {"checkpoint_idx": index, "expected_delta": 0.1}
                    for index in range(checkpoint_count)
                ],
            }
        },
        "health": {
            "num_goals": 1,
            "num_checkpoints_total": checkpoint_count,
            "num_numeric_parsed": checkpoint_count,
            "num_valid_forecasts": checkpoint_count,
        },
    }


def test_merge_aggregates_learned_cache_health(tmp_path):
    shard0 = tmp_path / "shard0.json"
    shard1 = tmp_path / "shard1.json"
    output = tmp_path / "merged.json"
    shard0.write_text(json.dumps(_payload(1, 2)), encoding="utf-8")
    shard1.write_text(json.dumps(_payload(2, 3)), encoding="utf-8")

    merge([shard0, shard1], output)

    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["goal_ids"] == [1, 2]
    assert merged["health"]["num_goals"] == 2
    assert merged["health"]["num_checkpoints_total"] == 5
    assert merged["health"]["num_numeric_parsed"] == 5
    assert merged["health"]["num_valid_forecasts"] == 5
    assert "prediction_health" not in merged
