from opportunity_forecasting.evaluation.regression_controls import fit_ridge, predict_row


def test_feature_ridge_predicts_bounded_remaining_upside():
    rows = [
        {
            "goal_idx": 1,
            "checkpoint_step": 2,
            "input": {"best_reward_seen": 0.1, "visited_product_page": True},
            "continuation_deltas": [0.3, 0.5],
        },
        {
            "goal_idx": 2,
            "checkpoint_step": 8,
            "input": {"best_reward_seen": 0.9, "visited_product_page": True},
            "continuation_deltas": [0.0, 0.0],
        },
    ]

    for index, row in enumerate(rows):
        row["_decision_index"] = index
    weights = fit_ridge(rows, feature_set="all", ridge=1e-3)
    pred = predict_row(rows[0], weights)

    assert 0.0 <= pred <= 1.0
