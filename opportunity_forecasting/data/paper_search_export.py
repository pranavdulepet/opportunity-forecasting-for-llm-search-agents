"""
Export a WebShop-like scientific paper-search corpus.

The canonical paper-search domain starts from LitSearch and can optionally add
BEIR-style scientific IR datasets. All sources are normalized into one
query/corpus/qrels triple with source-prefixed ids to avoid collisions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from opportunity_forecasting.data.paper_search import load_paper_search_data


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _load_dataset_rows(
    dataset_name: str,
    config_name: str | None = None,
    split: str | None = None,
    revision: str | None = None,
) -> List[dict]:
    try:
        from datasets import Dataset, DatasetDict, load_dataset
    except Exception as e:
        raise RuntimeError("The `datasets` package is required for Hugging Face exports.") from e

    if split:
        try:
            ds = (
                load_dataset(
                    dataset_name,
                    config_name,
                    split=split,
                    revision=revision,
                )
                if config_name
                else load_dataset(dataset_name, split=split, revision=revision)
            )
            return [dict(row) for row in ds]
        except Exception:
            pass

    ds = (
        load_dataset(dataset_name, config_name, revision=revision)
        if config_name
        else load_dataset(dataset_name, revision=revision)
    )
    if isinstance(ds, DatasetDict):
        rows: List[dict] = []
        preferred = [split] if split else []
        preferred.extend(["train", "dev", "validation", "test", "corpus", "queries", "full"])
        seen = set()
        for split_name in preferred + list(ds.keys()):
            if not split_name or split_name in seen or split_name not in ds:
                continue
            seen.add(split_name)
            rows.extend(dict(row) for row in ds[split_name])
        return rows
    if isinstance(ds, Dataset):
        return [dict(row) for row in ds]
    return [dict(row) for row in ds]


def _combine_title_text(row: dict) -> Tuple[str, str]:
    title = _safe_text(row.get("title", row.get("Title", "")))
    text = _safe_text(row.get("text", row.get("abstract", row.get("Abstract", ""))))
    if title and text and title.lower() not in text.lower()[: max(120, len(title) + 32)]:
        combined = f"{title}. {text}"
    else:
        combined = text or title
    return title, combined


def _add_litsearch(
    *,
    query_rows: List[dict],
    corpus_rows: List[dict],
    qrel_rows: List[Tuple[str, str, float]],
    dataset_name: str,
    query_config: str,
    corpus_config: str,
    split: str,
    limit_queries: int | None,
    revision: str | None,
) -> dict:
    queries, corpus = load_paper_search_data(
        dataset_name=dataset_name,
        query_config=query_config,
        corpus_config=corpus_config,
        split=split,
        limit_queries=limit_queries,
        dataset_revision=revision,
    )
    doc_prefix = "litsearch:"
    query_prefix = "litsearch:"
    for pid, rec in corpus.items():
        corpus_rows.append(
            {
                "corpusid": f"{doc_prefix}{pid}",
                "title": rec.get("Title", "N/A"),
                "abstract": rec.get("Abstract", ""),
                "citations": [f"{doc_prefix}{cid}" for cid in rec.get("Citations", [])],
                "source_dataset": "LitSearch",
                "source_doc_id": str(pid),
            }
        )
    for q in queries:
        qid = f"{query_prefix}{q['query_id']}"
        gold_ids = [f"{doc_prefix}{pid}" for pid in q.get("gold_paper_ids", [])]
        query_rows.append(
            {
                "queryid": qid,
                "query": q["query_text"],
                "gold_paper_ids": gold_ids,
                "source_dataset": "LitSearch",
                "source_query_id": str(q["query_id"]),
            }
        )
        for pid in gold_ids:
            qrel_rows.append((qid, pid, 1.0))
    return {
        "source_dataset": "LitSearch",
        "num_queries": len(queries),
        "num_corpus_papers": len(corpus),
        "num_qrels": sum(len(q.get("gold_paper_ids", [])) for q in queries),
    }


def _qrel_value(row: dict, *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _add_beir_dataset(
    *,
    beir_name: str,
    query_rows: List[dict],
    corpus_rows: List[dict],
    qrel_rows: List[Tuple[str, str, float]],
    max_queries: int | None,
    revisions: dict[str, str],
) -> dict:
    dataset_name = f"BeIR/{beir_name}"
    qrels_name = f"BeIR/{beir_name}-qrels"
    source_label = f"BEIR/{beir_name}"
    doc_prefix = f"beir:{beir_name}:"
    query_prefix = f"beir:{beir_name}:"

    corpus_raw = _load_dataset_rows(
        dataset_name,
        "corpus",
        "corpus",
        revisions.get(dataset_name),
    )
    queries_raw = _load_dataset_rows(
        dataset_name,
        "queries",
        "queries",
        revisions.get(dataset_name),
    )
    qrels_raw = _load_dataset_rows(
        qrels_name,
        revision=revisions.get(qrels_name),
    )

    available_docs = set()
    for row in corpus_raw:
        raw_id = _safe_text(row.get("_id", row.get("id", row.get("corpus-id", ""))))
        if not raw_id:
            continue
        title, abstract = _combine_title_text(row)
        available_docs.add(raw_id)
        corpus_rows.append(
            {
                "corpusid": f"{doc_prefix}{raw_id}",
                "title": title or "N/A",
                "abstract": abstract,
                "citations": [],
                "source_dataset": source_label,
                "source_doc_id": raw_id,
            }
        )

    qrels_by_query: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in qrels_raw:
        raw_qid = _safe_text(_qrel_value(row, "query-id", "query_id", "queryid", "_id"))
        raw_did = _safe_text(_qrel_value(row, "corpus-id", "corpus_id", "doc_id", "paper_id"))
        if not raw_qid or not raw_did or raw_did not in available_docs:
            continue
        try:
            score = float(_qrel_value(row, "score", "relevance", default=1.0))
        except Exception:
            score = 1.0
        if score <= 0.0:
            continue
        qrels_by_query[raw_qid][raw_did] = max(score, qrels_by_query[raw_qid].get(raw_did, 0.0))

    added_queries = 0
    added_qrels = 0
    for row in queries_raw:
        raw_qid = _safe_text(row.get("_id", row.get("id", row.get("query-id", ""))))
        if not raw_qid or raw_qid not in qrels_by_query:
            continue
        title, query_text = _combine_title_text(row)
        query_text = query_text or title
        if not query_text:
            continue
        qid = f"{query_prefix}{raw_qid}"
        gold_ids = [f"{doc_prefix}{did}" for did in sorted(qrels_by_query[raw_qid])]
        query_rows.append(
            {
                "queryid": qid,
                "query": query_text,
                "gold_paper_ids": gold_ids,
                "source_dataset": source_label,
                "source_query_id": raw_qid,
            }
        )
        for did, score in sorted(qrels_by_query[raw_qid].items()):
            qrel_rows.append((qid, f"{doc_prefix}{did}", float(score)))
            added_qrels += 1
        added_queries += 1
        if max_queries is not None and added_queries >= int(max_queries):
            break

    return {
        "source_dataset": source_label,
        "num_queries": added_queries,
        "num_corpus_papers": len(available_docs),
        "num_qrels": added_qrels,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export paper-search queries/corpus/qrels to local files.")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="princeton-nlp/LitSearch")
    ap.add_argument("--query_config", type=str, default="query")
    ap.add_argument("--corpus_config", type=str, default="corpus_clean")
    ap.add_argument("--split", type=str, default="full")
    ap.add_argument("--limit_queries", type=int, default=None)
    ap.add_argument("--include_litsearch", type=int, default=1)
    ap.add_argument(
        "--beir_datasets",
        type=str,
        default="",
        help="Comma-separated BEIR dataset names to append, e.g. scidocs,scifact,nfcorpus.",
    )
    ap.add_argument("--max_beir_queries_per_dataset", type=int, default=None)
    ap.add_argument("--source_revisions", type=Path)
    args = ap.parse_args()
    revisions = (
        json.loads(args.source_revisions.read_text(encoding="utf-8"))["revisions"]
        if args.source_revisions
        else {}
    )

    query_rows: List[dict] = []
    corpus_rows: List[dict] = []
    qrel_rows: List[Tuple[str, str, float]] = []
    source_metadata: List[dict] = []

    if int(args.include_litsearch):
        source_metadata.append(
            _add_litsearch(
                query_rows=query_rows,
                corpus_rows=corpus_rows,
                qrel_rows=qrel_rows,
                dataset_name=str(args.dataset_name),
                query_config=str(args.query_config),
                corpus_config=str(args.corpus_config),
                split=str(args.split),
                limit_queries=(int(args.limit_queries) if args.limit_queries is not None else None),
                revision=revisions.get(str(args.dataset_name)),
            )
        )

    for beir_name in _parse_csv_list(args.beir_datasets):
        source_metadata.append(
            _add_beir_dataset(
                beir_name=beir_name,
                query_rows=query_rows,
                corpus_rows=corpus_rows,
                qrel_rows=qrel_rows,
                max_queries=(
                    int(args.max_beir_queries_per_dataset)
                    if args.max_beir_queries_per_dataset is not None
                    else None
                ),
                revisions=revisions,
            )
        )

    if not query_rows:
        raise ValueError("No paper-search queries were exported.")
    if not corpus_rows:
        raise ValueError("No paper-search corpus documents were exported.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_path = out_dir / "queries.jsonl"
    corpus_path = out_dir / "corpus.jsonl"
    qrels_path = out_dir / "qrels.tsv"
    metadata_path = out_dir / "metadata.json"

    _write_jsonl(query_path, query_rows)
    _write_jsonl(corpus_path, corpus_rows)
    with qrels_path.open("w") as f:
        f.write("query_id\tpaper_id\trelevance\n")
        for qid, pid, score in qrel_rows:
            f.write(f"{qid}\t{pid}\t{float(score):.6g}\n")

    metadata = {
        "dataset_name": str(args.dataset_name),
        "query_config": str(args.query_config),
        "corpus_config": str(args.corpus_config),
        "split": str(args.split),
        "include_litsearch": int(args.include_litsearch),
        "beir_datasets": _parse_csv_list(args.beir_datasets),
        "num_queries": int(len(query_rows)),
        "num_corpus_papers": int(len(corpus_rows)),
        "num_qrels": int(len(qrel_rows)),
        "query_path": str(query_path),
        "corpus_path": str(corpus_path),
        "qrels_path": str(qrels_path),
        "sources": source_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
