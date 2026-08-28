from opportunity_forecasting.evaluation.select_stopping import select_rows


def test_selects_on_dev_and_reports_matching_test_row():
    dev = [
        {
            "model_label": "m",
            "policy": "policy1",
            "c_step": "0.01",
            "lambda_risk": "1",
            "utility_proxy": "0.5",
            "mean_final_reward": "0.8",
            "mean_final_steps": "20",
            "stop_rate": "0.2",
        },
        {
            "model_label": "m",
            "policy": "policy2",
            "c_step": "0.02",
            "lambda_risk": "0.5",
            "utility_proxy": "0.6",
            "mean_final_reward": "0.7",
            "mean_final_steps": "10",
            "stop_rate": "0.4",
        },
    ]
    test = [
        {
            "model_label": "m",
            "policy": "policy2",
            "c_step": "0.02",
            "lambda_risk": "0.5",
            "utility_proxy": "0.55",
            "mean_final_reward": "0.75",
            "mean_final_steps": "11",
            "stop_rate": "0.35",
        }
    ]

    rows = select_rows(dev, test, metric="utility_proxy")

    assert len(rows) == 1
    assert rows[0]["selected_policy"] == "policy2"
    assert rows[0]["matched_test_row"] == "1"
    assert rows[0]["test_mean_final_reward"] == "0.750000"


def test_shared_midpoint_budget_cannot_be_gamed_by_negative_policy_cost():
    dev = [
        {
            "model_label": label,
            "policy": "policy1",
            "c_step": c_step,
            "lambda_risk": "1",
            "utility_proxy": utility,
            "mean_final_reward": reward,
            "mean_final_steps": steps,
            "valid_score_rate": "1",
            "stop_rate": "0",
        }
        for label in ("a", "b")
        for c_step, utility, reward, steps in (
            ("-0.1", "100", "0.9", "20"),
            ("0.01", "0.2", "0.7", "10"),
            ("0.02", "0.1", "0.6", "5"),
        )
    ]
    test = [dict(row) for row in dev]

    rows = select_rows(dev, test, metric="budget_feasible_reward", target_step_fraction=0.5)

    assert {row["selected_c_step"] for row in rows} == {"0.01"}
    assert {row["dev_target_steps"] for row in rows} == {"12.500"}
    assert {row["selection_rule"] for row in rows} == {"dev_common_support_midpoint_budget"}
