import json
from pathlib import Path

import opportunity_forecasting.data.label_paper_search as labeler
from opportunity_forecasting.data.trajectories import run_episode_with_checkpoints
from opportunity_forecasting.data.label_paper_search import replay_to_checkpoint
from opportunity_forecasting.data.paper_search import (
    PAPER_REWARD_KEY,
    PAPER_REWARD_MODE,
    PaperSearchTextEnv,
    load_paper_search_data,
)
from opportunity_forecasting.models.search_runtime import format_seen_products_for_prompt, update_seen_products


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_toy_env(tmp_path: Path) -> PaperSearchTextEnv:
    query_path = tmp_path / "queries.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        query_path,
        [
            {
                "queryid": 0,
                "query": "graph neural network oversmoothing prior work",
                "gold_paper_ids": ["1001"],
            },
            {
                "queryid": 1,
                "query": "calibration for language model confidence",
                "gold_paper_ids": ["2001"],
            },
        ],
    )
    _write_jsonl(
        corpus_path,
        [
            {
                "corpusid": "1001",
                "title": "Revisiting Oversmoothing in Graph Neural Networks",
                "abstract": "We study oversmoothing in graph neural networks and characterize when it appears.",
                "citations": ["1002"],
            },
            {
                "corpusid": "1002",
                "title": "Understanding Oversmoothing in Graph Neural Networks",
                "abstract": "This survey covers graph neural networks, oversmoothing, message passing, and benchmarks.",
                "citations": [],
            },
            {
                "corpusid": "2001",
                "title": "Confidence Calibration for Large Language Models",
                "abstract": "We study calibration metrics and methods for language model confidence.",
                "citations": [],
            },
        ],
    )
    return PaperSearchTextEnv(
        query_path=str(query_path),
        corpus_path=str(corpus_path),
        page_size=1,
        max_results=5,
    )


def test_webshop_relevance_reward_keeps_gold_dominant_and_mid_signal(tmp_path: Path) -> None:
    env = _make_toy_env(tmp_path)
    env.reset(session=0)

    gold_reward = env._paper_utility("1001")
    related_reward = env._paper_utility("1002")

    assert env.paper_reward_mode == PAPER_REWARD_MODE
    assert gold_reward == 1.0
    assert 0.0 < related_reward <= 0.75
    assert related_reward < 0.90

def test_repeated_search_variants_are_masked(tmp_path: Path) -> None:
    env = _make_toy_env(tmp_path)
    env.reset(session=0)

    initial_searches = [a for a in env.get_available_actions()["valid_actions"] if a.startswith("search[")]
    assert len(initial_searches) == 1
    first_search = initial_searches[0]
    env.step(first_search)
    actions_after_first = env.get_available_actions()["valid_actions"]
    assert first_search not in actions_after_first

    remaining_searches = [a for a in actions_after_first if a.startswith("search[")]
    if remaining_searches:
        second_search = remaining_searches[0]
        env.step(second_search)
        actions_after_second = env.get_available_actions()["valid_actions"]
        assert first_search not in actions_after_second
        assert second_search not in actions_after_second


def test_search_variants_are_exposed_as_broad_to_specific_ladder(tmp_path: Path) -> None:
    query = (
        "Can you direct me to research that explores methods for transforming "
        "multi-hop questions into single-hop sub-questions to leverage existing "
        "single-hop answer models?"
    )
    query_path = tmp_path / "queries.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(query_path, [{"queryid": 0, "query": query, "gold_paper_ids": ["1001"]}])
    _write_jsonl(
        corpus_path,
        [
            {
                "corpusid": "1001",
                "title": "Question Decomposition for Multi-Hop QA",
                "abstract": "We decompose multi-hop questions into single-hop questions.",
            }
        ],
    )
    env = PaperSearchTextEnv(query_path=str(query_path), corpus_path=str(corpus_path))
    env.reset(session=0)

    search_actions = [a for a in env.get_available_actions()["valid_actions"] if a.startswith("search[")]
    assert len(search_actions) == 1
    assert search_actions[0] != f"search[{query}]"

    env.step(search_actions[0])
    search_actions = [a for a in env.get_available_actions()["valid_actions"] if a.startswith("search[")]
    assert search_actions == []

    click_action = next(a for a in env.get_available_actions()["valid_actions"] if a.startswith("click[paper "))
    env.step(click_action)
    search_actions = [a for a in env.get_available_actions()["valid_actions"] if a.startswith("search[")]
    assert len(search_actions) == 1
    assert search_actions[0] != f"search[{query}]"

    env.step(search_actions[0])
    search_actions = [a for a in env.get_available_actions()["valid_actions"] if a.startswith("search[")]
    assert search_actions == [f"search[{query}]"]


def test_paper_search_never_forces_commit_after_three_papers(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        query_path,
        [{"queryid": 0, "query": "graph neural network oversmoothing prior work", "gold_paper_ids": ["1001"]}],
    )
    _write_jsonl(
        corpus_path,
        [
            {
                "corpusid": "1001",
                "title": "Revisiting Oversmoothing in Graph Neural Networks",
                "abstract": "We study oversmoothing in graph neural networks.",
            },
            {
                "corpusid": "1002",
                "title": "Understanding Oversmoothing in Graph Neural Networks",
                "abstract": "A survey about graph neural networks and oversmoothing.",
            },
            {
                "corpusid": "1003",
                "title": "Graph Neural Network Benchmarks",
                "abstract": "Benchmarks for graph neural network representation learning.",
            },
        ],
    )
    env = PaperSearchTextEnv(query_path=str(query_path), corpus_path=str(corpus_path), page_size=1, max_results=5)
    env.reset(session=0)

    action = next(a for a in env.get_available_actions()["valid_actions"] if a.startswith("search["))
    env.step(action)
    while len(env._opened_paper_ids) < 3:
        actions = env.get_available_actions()["valid_actions"]
        paper_actions = [a for a in actions if a.startswith("click[paper ")]
        if paper_actions:
            env.step(paper_actions[0])
            continue
        if "click[next >]" in actions:
            env.step("click[next >]")
            continue
        if "click[back to results]" in actions:
            env.step("click[back to results]")
            continue
        search_actions = [a for a in actions if a.startswith("search[")]
        if search_actions:
            env.step(search_actions[0])
            continue
        raise AssertionError(f"Could not open comparison-set paper from actions: {actions}")

    actions = env.get_available_actions()["valid_actions"]
    assert "stop[select best paper]" in actions
    assert actions != ["stop[select best paper]"]


def test_paper_env_seen_items_and_reward(tmp_path: Path) -> None:
    env = _make_toy_env(tmp_path)
    obs, info = env.reset(session=0)
    assert "Search Literature" in obs
    actions = env.get_available_actions()["valid_actions"]
    assert any(a.startswith("search[") for a in actions)

    search_action = actions[0]
    obs, rew, done, _ = env.step(search_action)
    assert rew == 0.0
    assert not done
    assert "Results Page" in obs

    seen = {}
    update_seen_products(obs, seen, env)
    assert len(seen) == 1
    first_pid = next(iter(seen))
    assert first_pid in {"1001", "1002"}
    assert seen[first_pid].get(PAPER_REWARD_KEY) is None
    assert "Abstract" not in seen[first_pid]

    result_actions = env.get_available_actions()["valid_actions"]
    first_click = next(a for a in result_actions if a.startswith("click[paper "))
    assert "click[next >]" in result_actions
    assert not any(a.startswith("stop") for a in result_actions)

    obs, rew, done, _ = env.step(first_click)
    first_utility, _, first_pid = env.compute_current_page_reward()
    assert first_pid is not None
    assert rew == 0.0
    assert 0.0 < first_utility <= 1.0
    assert not done
    assert "CURRENT_PAPER_ID:" in obs

    update_seen_products(obs, seen, env)
    clicked_pid = str(first_pid)
    if clicked_pid == "1001":
        assert float(seen["1001"][PAPER_REWARD_KEY]) == 1.0
    else:
        assert 0.0 < float(seen["1002"][PAPER_REWARD_KEY]) <= 0.86
    assert "Abstract" in seen[clicked_pid]
    assert "GoldSimilarity" not in seen[clicked_pid]
    assert "IsGold" not in seen[clicked_pid]

    prompt_txt = format_seen_products_for_prompt(seen, top_n=3)
    assert "Papers seen so far" in prompt_txt
    assert "Oversmoothing" in prompt_txt
    assert "RelevanceReward:" in prompt_txt
    assert "QuerySimilarity:" not in prompt_txt

    paper_actions = env.get_available_actions()["valid_actions"]
    assert any(a.startswith("search[") for a in paper_actions)
    assert "stop[select best paper]" in paper_actions

    next_search = next(a for a in paper_actions if a.startswith("search["))
    obs, rew, done, _ = env.step(next_search)
    assert rew == 0.0
    assert not done
    assert "Results Page" in obs
    update_seen_products(obs, seen, env)
    assert seen[next(iter(seen))].get("FirstSeenRank") is not None

    for _ in range(3):
        available = env.get_available_actions()["valid_actions"]
        paper_clicks = [a for a in available if a.startswith("click[paper ")]
        if paper_clicks:
            second_click = paper_clicks[0]
            break
        if "click[next >]" in available:
            env.step("click[next >]")
            continue
        search_refinements = [a for a in available if a.startswith("search[")]
        assert search_refinements, available
        env.step(search_refinements[0])
    else:
        raise AssertionError("No second paper available after query refinements")
    assert second_click != first_click

    obs, rew, done, _ = env.step(second_click)
    second_utility, _, second_pid = env.compute_current_page_reward()
    assert second_pid is not None
    assert rew == 0.0
    assert 0.0 < float(second_utility) <= 1.0
    assert not done
    assert "CURRENT_PAPER_ID:" in obs

    update_seen_products(obs, seen, env)
    assert float(seen["1001"][PAPER_REWARD_KEY]) == 1.0
    assert env.get_final_reward() == 1.0

    obs, rew, done, _ = env.step("click[back to results]")
    assert rew == 0.0
    assert not done
    assert "Results Page 2" in obs

    obs, rew, done, _ = env.step("click[< prev]")
    assert rew == 0.0
    assert not done
    assert "Results Page 1" in obs


def test_checkpoint_and_replay_smoke(tmp_path: Path) -> None:
    env = _make_toy_env(tmp_path)

    def action_selector(goal, obs, valid_actions, action_history, step_seed):
        del goal, action_history, step_seed
        if "click[paper 1001]" in valid_actions:
            return "click[paper 1001]", "", "", ""
        paper_actions = [a for a in valid_actions if a.startswith("click[paper ")]
        if paper_actions:
            return paper_actions[0], "", "", ""
        if "click[next >]" in valid_actions:
            return "click[next >]", "", "", ""
        if "search[" in "".join(valid_actions):
            return next(a for a in valid_actions if a.startswith("search[")), "", "", ""
        return "stop", "", "", ""

    checkpoints = run_episode_with_checkpoints(
        env=env,
        goal_idx=0,
        max_steps=4,
        seed=123,
        action_selector=action_selector,
    )
    assert checkpoints
    assert any(bool(ckpt.get("visited_product_page", False)) for ckpt in checkpoints)
    assert checkpoints[-1]["trigger"] == "final"
    assert checkpoints[-1]["prefix_actions"]

    target_ckpt = checkpoints[-1]
    replay_obs, best_reward_seen, replay_seen = replay_to_checkpoint(
        env=_make_toy_env(tmp_path),
        goal_idx=int(target_ckpt["goal_idx"]),
        prefix_actions=list(target_ckpt["prefix_actions"]),
        reward_mode=PAPER_REWARD_MODE,
    )
    assert replay_obs
    assert best_reward_seen >= 0.0
    assert isinstance(replay_seen, dict)


def test_revisiting_same_paper_has_zero_incremental_reward(tmp_path: Path) -> None:
    env = _make_toy_env(tmp_path)
    env.reset(session=0)
    search_action = next(a for a in env.get_available_actions()["valid_actions"] if a.startswith("search["))
    env.step(search_action)
    click_action = next(a for a in env.get_available_actions()["valid_actions"] if a.startswith("click[paper "))

    _, first_rew, first_done, _ = env.step(click_action)
    assert first_rew == 0.0
    assert not first_done
    first_utility = float(env.get_final_reward())
    assert first_utility > 0.0

    _, rew, done, _ = env.step("click[back to results]")
    assert rew == 0.0
    assert not done
    assert click_action not in env.get_available_actions()["valid_actions"]

    _, second_rew, second_done, _ = env.step(click_action)
    assert second_rew == 0.0
    assert not second_done
    assert env.get_final_reward() == first_utility


def test_checkpointing_skips_repeated_paper_pages(tmp_path: Path) -> None:
    env = _make_toy_env(tmp_path)

    def action_selector(goal, obs, valid_actions, action_history, step_seed):
        del goal, obs, step_seed
        if not action_history:
            return next(a for a in valid_actions if a.startswith("search[")), "", "", ""
        if len(action_history) == 1:
            return next(a for a in valid_actions if a.startswith("click[paper ")), "", "", ""
        if len(action_history) == 2:
            return "click[back to results]", "", "", ""
        if len(action_history) == 3:
            return "click[paper 1001]", "", "", ""
        return "stop", "", "", ""

    checkpoints = run_episode_with_checkpoints(
        env=env,
        goal_idx=0,
        max_steps=5,
        seed=123,
        action_selector=action_selector,
    )
    paper_page_checkpoints = [ckpt for ckpt in checkpoints if ckpt.get("trigger") == "paper_page"]
    assert len(paper_page_checkpoints) == 1
    assert checkpoints[-1]["trigger"] == "final"


def test_official_litsearch_schema_preserves_gold_ids(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        query_path,
        [
            {
                "query_set": "inline_acl",
                "query": "knowledge distillation for compressing large language models",
                "specificity": 0,
                "quality": 2,
                "corpusids": [202719327],
            }
        ],
    )
    _write_jsonl(
        corpus_path,
        [
            {
                "corpusid": 202719327,
                "title": "Task-Agnostic Knowledge Distillation for Large Language Models",
                "abstract": "We study distillation methods for compressing large-scale language models.",
                "citations": [],
            }
        ],
    )

    queries, corpus = load_paper_search_data(query_path=str(query_path), corpus_path=str(corpus_path))
    assert len(queries) == 1
    assert queries[0]["query_id"] == "0"
    assert queries[0]["gold_paper_ids"] == ["202719327"]
    assert queries[0]["metadata"]["query_set"] == "inline_acl"
    assert "202719327" in corpus

    env = PaperSearchTextEnv(query_path=str(query_path), corpus_path=str(corpus_path), page_size=5, max_results=5)
    obs, _ = env.reset(session=0)
    assert "Search Literature" in obs
    search_action = next(a for a in env.get_available_actions()["valid_actions"] if a.startswith("search["))
    obs, rew, done, _ = env.step(search_action)
    assert rew == 0.0
    assert not done
    click_action = next(a for a in env.get_available_actions()["valid_actions"] if a.startswith("click[paper "))
    obs, rew, done, info = env.step(click_action)
    assert rew == 0.0
    assert not done
    assert info["gold_hit_at_stop"] is False
    assert env.get_final_reward() == 1.0


def test_unterminated_litsearch_continuation_keeps_best_opened_paper(tmp_path: Path, monkeypatch) -> None:
    env = _make_toy_env(tmp_path)

    def choose_first_paper(_llm, _goals, _obses, valid_actions_list, _histories, **_kwargs):
        actions = []
        for valid_actions in valid_actions_list:
            actions.append(next(a for a in valid_actions if a.startswith("click[paper ")))
        return actions

    monkeypatch.setattr(labeler, "_choose_actions_batched", choose_first_paper)
    rewards = labeler.run_continuations_lockstep(
        [env],
        llm=object(),
        goal_idx=0,
        goal_text="graph neural network oversmoothing prior work",
        prefix_actions=["search[graph neural network oversmoothing prior work]"],
        max_steps=1,
        temperature=0.0,
        top_p=1.0,
        action_max_new_tokens=16,
        per_cont_seeds=[123],
        reward_mode=PAPER_REWARD_MODE,
        tokenizer=None,
        max_model_len=None,
    )

    assert len(rewards) == 1
    assert rewards[0] > 0.0
