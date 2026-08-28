"""
Shared environment factory for WebShop and paper-search domains.
"""

from __future__ import annotations

from typing import Any, Dict

from opportunity_forecasting.data.paper_search import PAPER_REWARD_MODE, PAPER_REWARD_MODE_ALIASES, PaperSearchTextEnv
from opportunity_forecasting.data.webshop_rewards import COUNTERFACTUAL_REWARD_MODE


ENV_DOMAIN_CHOICES = ("webshop", "paper_search")


def infer_domain_from_reward_mode(reward_mode: str) -> str:
    raw = str(reward_mode or "").strip().lower()
    if raw in {alias.lower() for alias in PAPER_REWARD_MODE_ALIASES}:
        return "paper_search"
    return "webshop"


def add_shared_env_args(parser) -> None:
    parser.add_argument(
        "--domain",
        type=str,
        default="webshop",
        choices=ENV_DOMAIN_CHOICES,
        help="Environment/domain to use.",
    )
    parser.add_argument("--webshop_file_path", type=str, default=None)
    parser.add_argument("--human_goals", action="store_true")
    parser.add_argument("--limit_goals", type=int, default=None)
    parser.add_argument("--num_products", type=int, default=None)

    parser.add_argument(
        "--paper_query_path",
        type=str,
        default="",
        help="Optional local JSON/JSONL file for paper-search queries.",
    )
    parser.add_argument(
        "--paper_corpus_path",
        type=str,
        default="",
        help="Optional local JSON/JSONL file for the paper-search corpus.",
    )
    parser.add_argument(
        "--paper_qrels_path",
        type=str,
        default="",
        help="Optional local qrels file for paper-search gold paper ids.",
    )
    parser.add_argument(
        "--paper_dataset_name",
        type=str,
        default="princeton-nlp/LitSearch",
        help="Hugging Face dataset name for paper search when local files are not provided.",
    )
    parser.add_argument("--paper_query_config", type=str, default="query")
    parser.add_argument("--paper_corpus_config", type=str, default="corpus_clean")
    parser.add_argument("--paper_split", type=str, default="full")
    parser.add_argument("--paper_page_size", type=int, default=10)
    parser.add_argument("--paper_max_results", type=int, default=50)


def build_env_from_args(args) -> Any:
    domain = str(getattr(args, "domain", "webshop"))
    if domain == "paper_search":
        return PaperSearchTextEnv(
            query_path=str(getattr(args, "paper_query_path", "") or ""),
            corpus_path=str(getattr(args, "paper_corpus_path", "") or ""),
            qrels_path=str(getattr(args, "paper_qrels_path", "") or ""),
            dataset_name=str(getattr(args, "paper_dataset_name", "princeton-nlp/LitSearch")),
            query_config=str(getattr(args, "paper_query_config", "query")),
            corpus_config=str(getattr(args, "paper_corpus_config", "corpus_clean")),
            split=str(getattr(args, "paper_split", "full")),
            limit_queries=(
                int(getattr(args, "limit_goals"))
                if getattr(args, "limit_goals", None) is not None
                else None
            ),
            page_size=int(getattr(args, "paper_page_size", 10)),
            max_results=int(getattr(args, "paper_max_results", 50)),
            reward_mode=str(getattr(args, "reward_mode", PAPER_REWARD_MODE)),
        )

    from opportunity_forecasting.data.webshop_setup import ensure_public_webshop_imports

    ensure_public_webshop_imports()
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    env_kwargs: Dict[str, Any] = {
        "observation_mode": "text",
        "num_products": getattr(args, "num_products", None),
    }
    if getattr(args, "webshop_file_path", None):
        env_kwargs["file_path"] = getattr(args, "webshop_file_path")
    if bool(getattr(args, "human_goals", False)):
        env_kwargs["human_goals"] = 1
    if getattr(args, "limit_goals", None) is not None:
        env_kwargs["limit_goals"] = int(getattr(args, "limit_goals"))
    return WebAgentTextEnv(**env_kwargs)


def reward_mode_choices() -> list[str]:
    return [COUNTERFACTUAL_REWARD_MODE, *list(PAPER_REWARD_MODE_ALIASES)]
