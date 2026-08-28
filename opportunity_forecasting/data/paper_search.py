"""
Thin text environment for LitSearch-derived paper discovery tasks.

This environment deliberately mirrors the WebShop interaction pattern:
- explicit ``search[...]`` actions
- results pages that reveal candidate papers
- paper pages that yield continuous reward
- final reward defined by the best opened paper seen so far
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PAPER_REWARD_KEY = "paper_relevance_reward"
PAPER_REWARD_MODE = "paper_page_litsearch_webshop_relevance_v4"
PAPER_REWARD_MODE_ALIASES = (PAPER_REWARD_MODE,)
_QREL_RELEVANCE_FLOOR = 0.90
_NON_QREL_RELEVANCE_CAP = 0.75
_NON_QREL_RELEVANCE_GAMMA = 2.25
_QUERY_RELEVANCE_WEIGHT = 0.40
_GOLD_RELEVANCE_WEIGHT = 0.60

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PAPER_RESULT_ID_RE = re.compile(r"PAPER_ID:\s*([0-9A-Za-z._:-]+)")
_CURRENT_PAPER_ID_RE = re.compile(r"CURRENT_PAPER_ID:\s*([0-9A-Za-z._:-]+)")
_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "by",
    "can",
    "done",
    "find",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "idea",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "literature",
    "me",
    "need",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "prior",
    "research",
    "review",
    "see",
    "should",
    "show",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "work",
}
_TITLE_TOKEN_BOOST = 2
_BM25_K1 = 1.5
_BM25_B = 0.75
_SEARCH_RANK_BM25_WEIGHT = 0.35
_SEARCH_RANK_NOISE_WEIGHT = 0.65
_SEARCH_RERANK_POOL_MULTIPLIER = 20
_SEARCH_RERANK_MIN_POOL = 1000
_MIN_OPENED_PAPERS_BEFORE_STOP = 1
_STOP_ACTION = "stop[select best paper]"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_paper_id(value: Any) -> str:
    txt = _safe_text(value)
    return txt or ""


def _normalize_reward_mode(value: str) -> str:
    mode = str(value or PAPER_REWARD_MODE).strip()
    if mode.lower() != PAPER_REWARD_MODE.lower():
        raise ValueError(f"Unsupported Paper Search reward mode: {mode}")
    return PAPER_REWARD_MODE


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _stable_unit_hash(text: str) -> float:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return float(value) / float(2**64 - 1)


def _shape_non_qrel_relevance(score: float) -> float:
    """Map semantic relevance to bounded partial credit for unjudged papers."""
    clipped = max(0.0, min(1.0, float(score)))
    shaped = _NON_QREL_RELEVANCE_CAP * (clipped ** _NON_QREL_RELEVANCE_GAMMA)
    return max(0.0, min(_NON_QREL_RELEVANCE_CAP, round(float(shaped), 6)))


def _snip(text: str, limit: int = 260) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: max(32, limit - 16)].rstrip() + " ..."


def _load_json_records(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[dict] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("rows", "data", "queries", "corpus", "records"):
            vals = obj.get(key)
            if isinstance(vals, list):
                return [x for x in vals if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON record layout: {path}")


def _load_tsv_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line_no == 1 and ("query-id" in line.lower() or "query_id" in line.lower()):
                continue
            parts = re.split(r"[\t,]", line)
            if len(parts) < 2:
                continue
            if len(parts) >= 4 and parts[1].strip().lower() in {"0", "q0"}:
                qid = _normalize_paper_id(parts[0])
                did = _normalize_paper_id(parts[2])
                score_part = parts[3]
            else:
                qid = _normalize_paper_id(parts[0])
                did = _normalize_paper_id(parts[1])
                score_part = parts[2] if len(parts) >= 3 else None
            if not qid or not did:
                continue
            score = 1.0
            if score_part is not None:
                try:
                    score = float(score_part)
                except Exception:
                    score = 1.0
                if score <= 0:
                    continue
            out[qid][did] = max(float(score), float(out[qid].get(did, 0.0)))
    return out


def _load_qrels(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Qrels path does not exist: {p}")
    if p.suffix.lower() in {".tsv", ".csv"}:
        return _load_tsv_qrels(p)
    rows = _load_json_records(p)
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in rows:
        qid = _safe_text(
            row.get("query_id", row.get("queryid", row.get("id", row.get("_id", ""))))
        )
        if not qid:
            continue
        docs_raw = (
            row.get("doc_ids")
            or row.get("paper_ids")
            or row.get("gold_paper_ids")
            or row.get("gold_ids")
            or row.get("corpus_ids")
        )
        if isinstance(docs_raw, (list, tuple, set)):
            doc_ids = [_normalize_paper_id(x) for x in docs_raw if _normalize_paper_id(x)]
        else:
            doc_id = _normalize_paper_id(
                row.get("doc_id", row.get("paper_id", row.get("corpus_id", "")))
            )
            doc_ids = [doc_id] if doc_id else []
        try:
            score = float(row.get("score", row.get("relevance", 1.0)))
        except Exception:
            score = 1.0
        if score <= 0.0:
            continue
        for did in doc_ids:
            out[qid][did] = max(float(score), float(out[qid].get(did, 0.0)))
    return out


def _load_hf_records(
    dataset_name: str,
    config_name: str,
    split: str,
    revision: Optional[str] = None,
) -> List[dict]:
    if revision is None:
        cached_rows = _load_cached_hf_records(dataset_name, config_name, split)
        if cached_rows is not None:
            return cached_rows
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "The `datasets` package is required to load LitSearch directly from Hugging Face. "
            "Install it or pass local JSON/JSONL paths instead."
        ) from e

    ds = load_dataset(
        dataset_name,
        config_name,
        split=split,
        revision=revision,
    )
    return [dict(row) for row in ds]


def _dataset_cache_slug(dataset_name: str) -> str:
    if "/" not in dataset_name:
        return re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    namespace, repo = dataset_name.split("/", 1)
    repo_slug = re.sub(r"[^a-z0-9]+", "_", repo.lower()).strip("_")
    return f"{namespace}___{repo_slug}"


def _candidate_hf_dataset_roots() -> List[Path]:
    roots: List[Path] = []
    datasets_cache = os.environ.get("HF_DATASETS_CACHE", "").strip()
    if datasets_cache:
        roots.append(Path(datasets_cache))
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home) / "datasets")
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
    user = os.environ.get("USER", "").strip()
    if user:
        roots.append(Path("/srv/local1") / user / "cache" / "huggingface" / "datasets")

    out: List[Path] = []
    seen: Set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _read_arrow_records(path: Path) -> List[dict]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception as e:
        raise RuntimeError(
            "Reading cached LitSearch Arrow shards requires `pyarrow` in the active environment."
        ) from e

    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.open_stream(source)
        except Exception:
            source.close()
            with pa.memory_map(str(path), "r") as file_source:
                reader = ipc.open_file(file_source)
                return reader.read_all().to_pylist()
        return reader.read_all().to_pylist()


def _normalized_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _find_dataset_cache_dir(root: Path, dataset_name: str) -> Optional[Path]:
    if "/" not in dataset_name:
        direct = root / _dataset_cache_slug(dataset_name)
        return direct if direct.exists() else None
    namespace, repo = dataset_name.split("/", 1)
    repo_norm = _normalized_alnum(repo)
    for candidate in root.glob(f"{namespace}___*"):
        tail = candidate.name.split("___", 1)[-1]
        if _normalized_alnum(tail) == repo_norm:
            return candidate
    return None


def _load_cached_hf_records(dataset_name: str, config_name: str, split: str) -> Optional[List[dict]]:
    if str(split or "full") != "full":
        return None
    for root in _candidate_hf_dataset_roots():
        dataset_root = _find_dataset_cache_dir(root, dataset_name)
        if dataset_root is None:
            continue
        cfg_root = dataset_root / str(config_name)
        if not cfg_root.exists():
            continue
        arrow_files = sorted(cfg_root.rglob("*.arrow"))
        if not arrow_files:
            continue
        rows: List[dict] = []
        for arrow_path in arrow_files:
            rows.extend(_read_arrow_records(arrow_path))
        if rows:
            return rows
    return None


def _pick_first_present(row: dict, keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _normalize_query_rows(
    rows: Sequence[dict],
    *,
    qrels: Optional[Dict[str, Dict[str, float]]] = None,
    limit_queries: Optional[int] = None,
) -> List[dict]:
    out: List[dict] = []
    for idx, row in enumerate(rows):
        query_text = _safe_text(
            _pick_first_present(
                row,
                ("query", "text", "question", "query_text", "goal", "instruction"),
            )
        )
        if not query_text:
            continue
        raw_qid = _pick_first_present(row, ("queryid", "query_id", "id", "_id"), default=idx)
        qid = _safe_text(raw_qid) or str(idx)
        docs_raw = _pick_first_present(
            row,
            (
                "gold_paper_ids",
                "gold_ids",
                "relevant_paper_ids",
                "paper_ids",
                "corpus_ids",
                "corpusids",
            ),
            default=[],
        )
        gold_ids: List[str] = []
        if isinstance(docs_raw, (list, tuple, set)):
            gold_ids = [_normalize_paper_id(x) for x in docs_raw if _normalize_paper_id(x)]
        elif docs_raw not in ("", None):
            gold_id = _normalize_paper_id(docs_raw)
            if gold_id:
                gold_ids = [gold_id]
        qrel_scores: Dict[str, float] = {}
        if qrels and qid in qrels:
            qrel_scores = {
                did: float(score)
                for did, score in qrels[qid].items()
                if _normalize_paper_id(did) and float(score) > 0.0
            }
            gold_ids.extend(sorted(qrel_scores))
        gold_ids = sorted({gid for gid in gold_ids if gid})
        for gid in gold_ids:
            qrel_scores.setdefault(gid, 1.0)
        out.append(
            {
                "goal_idx": len(out),
                "query_id": qid,
                "query_text": query_text,
                "gold_paper_ids": gold_ids,
                "qrel_scores": dict(sorted(qrel_scores.items())),
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key not in {"query", "text", "question", "query_text", "goal", "instruction"}
                },
            }
        )
        if limit_queries is not None and len(out) >= int(limit_queries):
            break
    if not out:
        raise ValueError("No usable paper-search queries were loaded.")
    return out


def _normalize_corpus_rows(rows: Sequence[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in rows:
        pid = _normalize_paper_id(
            _pick_first_present(row, ("corpusid", "paper_id", "id", "_id"))
        )
        if not pid:
            continue
        title = _safe_text(_pick_first_present(row, ("title", "Title"), default="N/A"))
        abstract = _safe_text(_pick_first_present(row, ("abstract", "text", "Abstract"), default=""))
        citations_raw = _pick_first_present(row, ("citations", "citation_ids", "references"), default=[])
        citations: List[str] = []
        if isinstance(citations_raw, (list, tuple, set)):
            citations = [_normalize_paper_id(x) for x in citations_raw if _normalize_paper_id(x)]
        out[pid] = {
            "paper_id": pid,
            "Title": title or "N/A",
            "Abstract": abstract,
            "Citations": citations,
            PAPER_REWARD_KEY: None,
        }
    if not out:
        raise ValueError("No usable paper-search corpus documents were loaded.")
    return out


def load_paper_search_data(
    *,
    query_path: str = "",
    corpus_path: str = "",
    qrels_path: str = "",
    dataset_name: str = "princeton-nlp/LitSearch",
    query_config: str = "query",
    corpus_config: str = "corpus_clean",
    split: str = "full",
    limit_queries: Optional[int] = None,
    dataset_revision: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, dict]]:
    qrels = _load_qrels(qrels_path)
    if query_path:
        query_rows = _load_json_records(Path(query_path))
    else:
        query_rows = _load_hf_records(
            dataset_name,
            query_config,
            split,
            revision=dataset_revision,
        )
    if corpus_path:
        corpus_rows = _load_json_records(Path(corpus_path))
    else:
        corpus_rows = _load_hf_records(
            dataset_name,
            corpus_config,
            split,
            revision=dataset_revision,
        )
    queries = _normalize_query_rows(query_rows, qrels=qrels, limit_queries=limit_queries)
    corpus = _normalize_corpus_rows(corpus_rows)
    return queries, corpus


def _weighted_doc_counts(title: str, abstract: str) -> Counter[str]:
    counts: Counter[str] = Counter(_tokenize(abstract))
    for tok in _tokenize(title):
        counts[tok] += _TITLE_TOKEN_BOOST
    return counts


def _tfidf_norm(term_counts: Counter[str], idf: Dict[str, float]) -> float:
    norm_sq = 0.0
    for tok, tf in term_counts.items():
        weight = float(tf) * float(idf.get(tok, 0.0))
        norm_sq += weight * weight
    return math.sqrt(max(0.0, norm_sq))


@dataclass
class _PaperBackend:
    queries: List[dict]
    corpus: Dict[str, dict]
    postings: Dict[str, Set[str]]
    idf: Dict[str, float]
    doc_term_counts: Dict[str, Counter[str]]
    title_term_counts: Dict[str, Counter[str]]
    doc_lengths: Dict[str, int]
    avg_doc_length: float
    doc_vector_norms: Dict[str, float]
    query_max_bm25_scores: Dict[str, float]
    doc_pair_cosine_cache: Dict[Tuple[str, str], float]

    def _bm25_score_terms(self, query_counts: Counter[str], paper_id: str) -> float:
        pid = _normalize_paper_id(paper_id)
        if not pid:
            return 0.0
        avgdl = max(1.0, float(self.avg_doc_length))
        score = 0.0
        for tok, qtf in query_counts.items():
            idf = float(self.idf.get(tok, 0.0))
            if idf <= 0.0:
                continue
            tf = float(self.doc_term_counts.get(pid, Counter()).get(tok, 0))
            if tf <= 0.0:
                continue
            doc_len = float(self.doc_lengths.get(pid, 1))
            denom = tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (doc_len / avgdl))
            if denom <= 0.0:
                continue
            score += float(qtf) * idf * ((tf * (_BM25_K1 + 1.0)) / denom)
        return max(0.0, float(score))

    def _bm25_scores_for_query(self, query_text: str) -> Dict[str, float]:
        query_counts = Counter(_tokenize(query_text))
        if not query_counts:
            return {}
        scores: Dict[str, float] = defaultdict(float)
        for tok, qtf in query_counts.items():
            posting = self.postings.get(tok, ())
            if not posting:
                continue
            idf = float(self.idf.get(tok, 0.0))
            if idf <= 0.0:
                continue
            for pid in posting:
                tf = float(self.doc_term_counts.get(pid, Counter()).get(tok, 0))
                if tf <= 0.0:
                    continue
                doc_len = float(self.doc_lengths.get(pid, 1))
                avgdl = max(1.0, float(self.avg_doc_length))
                denom = tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (doc_len / avgdl))
                if denom <= 0.0:
                    continue
                scores[pid] += float(qtf) * idf * ((tf * (_BM25_K1 + 1.0)) / denom)
        return dict(scores)

    def search_with_scores(
        self,
        query_text: str,
        *,
        max_results: int,
        rank_mode: str = "webshop_like",
    ) -> List[Tuple[str, float]]:
        scores = self._bm25_scores_for_query(query_text)
        if not scores:
            return []
        limit = max(1, int(max_results))
        pure_ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if rank_mode == "bm25":
            return [(pid, float(score)) for pid, score in pure_ranked[:limit]]

        pool_size = min(
            len(pure_ranked),
            max(limit, min(max(limit * _SEARCH_RERANK_POOL_MULTIPLIER, _SEARCH_RERANK_MIN_POOL), len(pure_ranked))),
        )
        pool = pure_ranked[:pool_size]
        max_score = max(1e-12, float(pool[0][1]))
        query_key = " ".join(_tokenize(query_text))

        def rank_key(item: Tuple[str, float]) -> Tuple[float, float, str]:
            pid, score = item
            bm25_norm = max(0.0, min(1.0, float(score) / max_score))
            noise = _stable_unit_hash(f"{query_key}\t{pid}")
            rank_score = (
                _SEARCH_RANK_BM25_WEIGHT * bm25_norm
                + _SEARCH_RANK_NOISE_WEIGHT * noise
            )
            return (-rank_score, -bm25_norm, str(pid))

        ranked = sorted(pool, key=rank_key)
        return [(pid, float(scores[pid])) for pid, _score in ranked[:limit]]

    def search(self, query_text: str, *, max_results: int) -> List[str]:
        return [pid for pid, _ in self.search_with_scores(query_text, max_results=max_results)]

    def query_bm25_score(self, query_text: str, paper_id: str) -> float:
        return self._bm25_score_terms(Counter(_tokenize(query_text)), paper_id)

    def normalized_query_bm25_score(self, query_text: str, paper_id: str) -> float:
        query_key = " ".join(_tokenize(query_text))
        if not query_key:
            return 0.0
        max_score = float(self.query_max_bm25_scores.get(query_key, 0.0))
        if max_score <= 0.0:
            top = self.search_with_scores(query_text, max_results=1, rank_mode="bm25")
            max_score = float(top[0][1]) if top else 0.0
            self.query_max_bm25_scores[query_key] = max_score
        if max_score <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self.query_bm25_score(query_text, paper_id) / max_score))

    def query_similarity(self, query_text: str, paper_id: str) -> float:
        pid = _normalize_paper_id(paper_id)
        doc_counts = self.doc_term_counts.get(pid)
        doc_norm = float(self.doc_vector_norms.get(pid, 0.0))
        if not doc_counts or doc_norm <= 0.0:
            return 0.0

        query_counts = Counter(_tokenize(query_text))
        if not query_counts:
            return 0.0
        query_norm = _tfidf_norm(query_counts, self.idf)
        if query_norm <= 0.0:
            return 0.0

        numer = 0.0
        for tok, qtf in query_counts.items():
            dtf = int(doc_counts.get(tok, 0))
            if dtf <= 0:
                continue
            idf = float(self.idf.get(tok, 0.0))
            numer += (float(qtf) * idf) * (float(dtf) * idf)
        if numer <= 0.0:
            return 0.0
        return max(0.0, min(1.0, numer / (query_norm * doc_norm)))

    def doc_tfidf_cosine(self, paper_a: str, paper_b: str) -> float:
        pid_a = _normalize_paper_id(paper_a)
        pid_b = _normalize_paper_id(paper_b)
        if not pid_a or not pid_b:
            return 0.0
        cache_key = (pid_a, pid_b) if pid_a <= pid_b else (pid_b, pid_a)
        cached = self.doc_pair_cosine_cache.get(cache_key)
        if cached is not None:
            return float(cached)
        counts_a = self.doc_term_counts.get(pid_a)
        counts_b = self.doc_term_counts.get(pid_b)
        norm_a = float(self.doc_vector_norms.get(pid_a, 0.0))
        norm_b = float(self.doc_vector_norms.get(pid_b, 0.0))
        if not counts_a or not counts_b or norm_a <= 0.0 or norm_b <= 0.0:
            self.doc_pair_cosine_cache[cache_key] = 0.0
            return 0.0

        if len(counts_a) > len(counts_b):
            counts_a, counts_b = counts_b, counts_a
        numer = 0.0
        for tok, tf_a in counts_a.items():
            tf_b = int(counts_b.get(tok, 0))
            if tf_b <= 0:
                continue
            idf = float(self.idf.get(tok, 0.0))
            numer += (float(tf_a) * idf) * (float(tf_b) * idf)
        if numer <= 0.0:
            self.doc_pair_cosine_cache[cache_key] = 0.0
            return 0.0
        value = max(0.0, min(1.0, numer / (norm_a * norm_b)))
        self.doc_pair_cosine_cache[cache_key] = float(value)
        return value


_BACKEND_CACHE: Dict[Tuple[Any, ...], _PaperBackend] = {}


def _backend_cache_key(
    *,
    query_path: str,
    corpus_path: str,
    qrels_path: str,
    dataset_name: str,
    query_config: str,
    corpus_config: str,
    split: str,
    limit_queries: Optional[int],
) -> Tuple[Any, ...]:
    return (
        str(query_path or ""),
        str(corpus_path or ""),
        str(qrels_path or ""),
        str(dataset_name or ""),
        str(query_config or ""),
        str(corpus_config or ""),
        str(split or ""),
        int(limit_queries) if limit_queries is not None else None,
    )


def _build_backend(queries: List[dict], corpus: Dict[str, dict]) -> _PaperBackend:
    postings: Dict[str, Set[str]] = defaultdict(set)
    doc_term_counts: Dict[str, Counter[str]] = {}
    title_term_counts: Dict[str, Counter[str]] = {}
    doc_lengths: Dict[str, int] = {}
    df: Dict[str, int] = defaultdict(int)

    num_docs = 0
    total_doc_len = 0
    for pid, rec in corpus.items():
        title_counts = Counter(_tokenize(rec.get("Title", "")))
        weighted_counts = _weighted_doc_counts(rec.get("Title", ""), rec.get("Abstract", ""))
        if not weighted_counts:
            continue
        num_docs += 1
        doc_term_counts[pid] = weighted_counts
        title_term_counts[pid] = title_counts
        doc_len = int(sum(weighted_counts.values()))
        doc_lengths[pid] = doc_len
        total_doc_len += doc_len
        for tok in weighted_counts:
            postings[tok].add(pid)
            df[tok] += 1

    n = max(1, num_docs)
    idf: Dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log(1.0 + ((n - float(freq) + 0.5) / (float(freq) + 0.5)))

    avg_doc_length = float(total_doc_len) / float(max(1, num_docs))
    doc_vector_norms = {
        pid: _tfidf_norm(term_counts, idf) for pid, term_counts in doc_term_counts.items()
    }

    return _PaperBackend(
        queries=queries,
        corpus=corpus,
        postings=postings,
        idf=idf,
        doc_term_counts=doc_term_counts,
        title_term_counts=title_term_counts,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        doc_vector_norms=doc_vector_norms,
        query_max_bm25_scores={},
        doc_pair_cosine_cache={},
    )


class PaperSearchTextEnv:
    ccs_domain = "paper_search"
    paper_reward_key = PAPER_REWARD_KEY
    paper_reward_mode = PAPER_REWARD_MODE

    def __init__(
        self,
        *,
        query_path: str = "",
        corpus_path: str = "",
        qrels_path: str = "",
        dataset_name: str = "princeton-nlp/LitSearch",
        query_config: str = "query",
        corpus_config: str = "corpus_clean",
        split: str = "full",
        limit_queries: Optional[int] = None,
        page_size: int = 10,
        max_results: int = 50,
        reward_mode: str = PAPER_REWARD_MODE,
        seed: int = 123,
    ):
        self.query_path = str(query_path or "")
        self.corpus_path = str(corpus_path or "")
        self.qrels_path = str(qrels_path or "")
        self.dataset_name = str(dataset_name)
        self.query_config = str(query_config)
        self.corpus_config = str(corpus_config)
        self.split = str(split)
        self.limit_queries = int(limit_queries) if limit_queries is not None else None
        self.page_size = max(1, int(page_size))
        self.max_results = max(self.page_size, int(max_results))
        self.paper_reward_mode = _normalize_reward_mode(reward_mode)
        self._rng = random.Random(int(seed))

        key = _backend_cache_key(
            query_path=self.query_path,
            corpus_path=self.corpus_path,
            qrels_path=self.qrels_path,
            dataset_name=self.dataset_name,
            query_config=self.query_config,
            corpus_config=self.corpus_config,
            split=self.split,
            limit_queries=self.limit_queries,
        )
        backend = _BACKEND_CACHE.get(key)
        if backend is None:
            queries, corpus = load_paper_search_data(
                query_path=self.query_path,
                corpus_path=self.corpus_path,
                qrels_path=self.qrels_path,
                dataset_name=self.dataset_name,
                query_config=self.query_config,
                corpus_config=self.corpus_config,
                split=self.split,
                limit_queries=self.limit_queries,
            )
            backend = _build_backend(queries=queries, corpus=corpus)
            _BACKEND_CACHE[key] = backend
        self.backend = backend

        self.session: Optional[int] = None
        self.instruction_text = ""
        self._page_type = "search_home"
        self._search_query = ""
        self._search_variants: List[str] = []
        self._issued_search_queries: Set[str] = set()
        self._opened_count_at_last_search = 0
        self._results: List[str] = []
        self._results_page = 0
        self._current_paper_id: Optional[str] = None
        self._best_reward_seen = 0.0
        self._best_paper_id: Optional[str] = None
        self._has_opened_paper = False
        self._opened_paper_ids: Set[str] = set()
        self._query_gold_ids_cache: Dict[str, Set[str]] = {}
        self._query_qrel_scores_cache: Dict[str, Dict[str, float]] = {}
        self._gold_similarity_cache: Dict[Tuple[str, str], float] = {}
        self._paper_utility_cache: Dict[Tuple[str, str, str], float] = {}

    def close(self) -> None:
        return None

    @property
    def num_queries(self) -> int:
        return len(self.backend.queries)

    def _current_query(self) -> dict:
        if self.session is None:
            raise RuntimeError("Environment not reset.")
        return self.backend.queries[int(self.session)]

    def _query_goal(self) -> str:
        return self._current_query()["query_text"]

    def _query_gold_ids(self) -> Set[str]:
        query = self._current_query()
        qid = str(query.get("query_id", self.session))
        cached = self._query_gold_ids_cache.get(qid)
        if cached is not None:
            return set(cached)
        gold_ids = {gid for gid in query.get("gold_paper_ids", []) if gid}
        self._query_gold_ids_cache[qid] = set(gold_ids)
        return gold_ids

    def _query_qrel_scores(self) -> Dict[str, float]:
        query = self._current_query()
        qid = str(query.get("query_id", self.session))
        cached = self._query_qrel_scores_cache.get(qid)
        if cached is not None:
            return dict(cached)
        raw = query.get("qrel_scores", {}) or {}
        out: Dict[str, float] = {}
        if isinstance(raw, dict):
            for pid, score in raw.items():
                norm_pid = _normalize_paper_id(pid)
                if not norm_pid:
                    continue
                try:
                    score_val = float(score)
                except Exception:
                    score_val = 1.0
                if score_val > 0.0:
                    out[norm_pid] = max(score_val, out.get(norm_pid, 0.0))
        self._query_qrel_scores_cache[qid] = dict(out)
        return out

    def _default_search_queries(self) -> List[str]:
        goal = " ".join(self._query_goal().split())
        toks = _tokenize(goal)
        keywords = [tok for tok in toks if tok not in _DEFAULT_STOPWORDS]
        deduped_keywords = list(dict.fromkeys(keywords))
        compressed = deduped_keywords or toks

        variants = [
            " ".join(compressed[:6]).strip(),
            " ".join(compressed[:12]).strip(),
            goal,
        ]
        uniq: List[str] = []
        seen: Set[str] = set()
        for txt in variants:
            norm = " ".join(str(txt or "").split())
            if norm and norm not in seen:
                seen.add(norm)
                uniq.append(norm)
        return uniq[:3] or ["research paper"]

    def _search_actions(self) -> List[str]:
        for txt in self._search_variants:
            norm = " ".join(str(txt or "").split())
            if norm in self._issued_search_queries:
                continue
            return [f"search[{norm}]"]
        return []

    def _gold_similarity(self, paper_id: str) -> float:
        pid = _normalize_paper_id(paper_id)
        if not pid or pid not in self.backend.corpus:
            return 0.0
        query_id = str(self._current_query().get("query_id", self.session))
        cache_key = (query_id, pid)
        cached = self._gold_similarity_cache.get(cache_key)
        if cached is not None:
            return float(cached)
        gold_ids = {gid for gid in self._query_gold_ids() if gid in self.backend.corpus}
        if not gold_ids:
            self._gold_similarity_cache[cache_key] = 0.0
            return 0.0
        if pid in gold_ids:
            self._gold_similarity_cache[cache_key] = 1.0
            return 1.0
        best = 0.0
        for gid in gold_ids:
            best = max(best, float(self.backend.doc_tfidf_cosine(pid, gid)))
        value = max(0.0, min(1.0, best))
        self._gold_similarity_cache[cache_key] = float(value)
        return value

    def _paper_utility(self, paper_id: str) -> float:
        pid = _normalize_paper_id(paper_id)
        if not pid or pid not in self.backend.corpus:
            return 0.0
        query_id = str(self._current_query().get("query_id", self.session))
        cache_key = (self.paper_reward_mode, query_id, pid)
        cached = self._paper_utility_cache.get(cache_key)
        if cached is not None:
            return float(cached)
        qrel_scores = {
            qid: score
            for qid, score in self._query_qrel_scores().items()
            if qid in self.backend.corpus
        }
        gold_ids = {gid for gid in self._query_gold_ids() if gid in self.backend.corpus}
        if pid in gold_ids and pid not in qrel_scores:
            qrel_scores[pid] = 1.0
        if pid in qrel_scores:
            max_qrel = max(float(v) for v in qrel_scores.values()) if qrel_scores else 1.0
            if max_qrel <= 0.0:
                self._paper_utility_cache[cache_key] = 1.0
                return 1.0
            norm = max(0.0, min(1.0, float(qrel_scores[pid]) / float(max_qrel)))
            if norm >= 1.0 - 1e-12:
                self._paper_utility_cache[cache_key] = 1.0
                return 1.0
            value = round(
                _QREL_RELEVANCE_FLOOR
                + (1.0 - _QREL_RELEVANCE_FLOOR) * norm,
                6,
            )
            self._paper_utility_cache[cache_key] = value
            return value

        query_bm25 = self.backend.normalized_query_bm25_score(self._query_goal(), pid)
        query_cos = self.backend.query_similarity(self._query_goal(), pid)
        clipped_bm25 = max(0.0, min(1.0, query_bm25))
        clipped_cos = max(0.0, min(1.0, query_cos))
        query_relevance = 0.5 * clipped_bm25 + 0.5 * clipped_cos
        if gold_ids:
            gold_relevance = max(0.0, min(1.0, float(self._gold_similarity(pid))))
            semantic_relevance = (
                _QUERY_RELEVANCE_WEIGHT * query_relevance
                + _GOLD_RELEVANCE_WEIGHT * gold_relevance
            )
        else:
            semantic_relevance = query_relevance
        value = _shape_non_qrel_relevance(semantic_relevance)
        self._paper_utility_cache[cache_key] = value
        return value

    def _update_best_seen(self, paper_id: str, reward: float) -> None:
        reward_val = float(reward)
        if reward_val > float(self._best_reward_seen) + 1e-12:
            self._best_reward_seen = reward_val
            self._best_paper_id = _normalize_paper_id(paper_id) or self._best_paper_id

    def reset(self, session: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if session is None:
            session = self._rng.randrange(len(self.backend.queries))
        sess = int(session)
        if sess < 0 or sess >= len(self.backend.queries):
            raise IndexError(f"PaperSearchTextEnv session out of range: {sess}")
        self.session = sess
        self.instruction_text = self._query_goal()
        self._page_type = "search_home"
        self._search_query = ""
        self._search_variants = self._default_search_queries()
        self._issued_search_queries.clear()
        self._opened_count_at_last_search = 0
        self._results = []
        self._results_page = 0
        self._current_paper_id = None
        self._best_reward_seen = 0.0
        self._best_paper_id = None
        self._has_opened_paper = False
        self._opened_paper_ids.clear()
        self._gold_similarity_cache.clear()
        self._paper_utility_cache.clear()
        obs = self._render_observation()
        return obs, {"goal": self.instruction_text, "query_id": self._current_query()["query_id"]}

    def _render_observation(self) -> str:
        goal = self._query_goal()
        if self._page_type == "search_home":
            return (
                "Instruction: [SEP] "
                f"{goal} [SEP] Search Literature [SEP] "
                "Use a search action to retrieve relevant prior work papers."
            )

        if self._page_type == "results":
            total = len(self._results)
            page_no = self._results_page + 1
            start = self._results_page * self.page_size
            end = min(total, start + self.page_size)
            lines: List[str] = [
                "Instruction:",
                goal,
                f"Results Page {page_no} (Total results: {total})",
                f"Current search query: {self._search_query or goal}",
                f"Papers opened so far: {len(self._opened_paper_ids)}",
            ]
            if start >= end:
                lines.append("No results found.")
            else:
                for pid in self._results[start:end]:
                    rec = self.backend.corpus.get(pid, {})
                    lines.extend(
                        [
                            "PAPER_RESULT",
                            f"PAPER_ID: {pid}",
                            f"Title: {rec.get('Title', 'N/A')}",
                            "Open this paper to inspect the abstract and judge relevance.",
                        ]
                    )
            return " [SEP] ".join(lines)

        if self._page_type == "paper":
            pid = str(self._current_paper_id or "")
            rec = self.backend.corpus.get(pid, {})
            start = self._results_page * self.page_size
            end = min(len(self._results), start + self.page_size)
            lines = [
                "Instruction:",
                goal,
                "Paper Page",
                f"CURRENT_PAPER_ID: {pid}",
                f"Current search query: {self._search_query or goal}",
                f"Papers opened so far: {len(self._opened_paper_ids)}",
                f"Use {_STOP_ACTION} only when the best paper seen so far is strong enough.",
                f"Title: {rec.get('Title', 'N/A')}",
                f"Abstract: {_snip(rec.get('Abstract', ''), limit=600)}",
            ]
            other_ids = [
                other_pid
                for other_pid in self._results[start:end]
                if str(other_pid) != pid and other_pid not in self._opened_paper_ids
            ]
            if other_ids:
                lines.append("Other visible papers from current results:")
                for other_pid in other_ids:
                    other = self.backend.corpus.get(other_pid, {})
                    lines.extend(
                        [
                            "PAPER_RESULT",
                            f"PAPER_ID: {other_pid}",
                            f"Title: {other.get('Title', 'N/A')}",
                        ]
                    )
            return " [SEP] ".join(lines)

        return f"Instruction: [SEP] {goal}"

    def get_available_actions(self) -> Dict[str, Any]:
        if self._page_type == "search_home":
            search_actions = self._search_actions()
            return {"valid_actions": search_actions}

        if self._page_type == "results":
            actions: List[str] = []
            start = self._results_page * self.page_size
            end = min(len(self._results), start + self.page_size)
            for pid in self._results[start:end]:
                if pid in self._opened_paper_ids:
                    continue
                actions.append(f"click[paper {pid}]")
            if self._results_page > 0:
                actions.append("click[< prev]")
            if end < len(self._results):
                actions.append("click[next >]")
            opened_current_query = len(self._opened_paper_ids) > self._opened_count_at_last_search
            if opened_current_query or not actions:
                actions = self._search_actions() + actions
            return {"valid_actions": actions}

        if self._page_type == "paper":
            actions: List[str] = []
            actions.extend(self._search_actions())
            actions.append("click[back to results]")
            start = self._results_page * self.page_size
            end = min(len(self._results), start + self.page_size)
            if self._results_page > 0:
                actions.append("click[< prev]")
            if end < len(self._results):
                actions.append("click[next >]")
            if len(self._opened_paper_ids) >= _MIN_OPENED_PAPERS_BEFORE_STOP:
                actions.append(_STOP_ACTION)
            return {"valid_actions": actions}

        return {"valid_actions": self._search_actions()}

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        act = _safe_text(action)
        reward = 0.0
        done = False

        if act.lower().startswith("search[") and act.endswith("]"):
            query_text = act[len("search[") : -1].strip()
            self._search_query = query_text or self._query_goal()
            self._issued_search_queries.add(" ".join(str(self._search_query or "").split()))
            self._opened_count_at_last_search = len(self._opened_paper_ids)
            self._results = self.backend.search(self._search_query, max_results=self.max_results)
            self._results_page = 0
            self._current_paper_id = None
            self._page_type = "results"
        elif act.lower() == "click[next >]" and self._page_type in {"results", "paper"}:
            max_page = max(0, (len(self._results) - 1) // self.page_size)
            self._results_page = min(max_page, self._results_page + 1)
            self._page_type = "results"
            self._current_paper_id = None
        elif act.lower() == "click[< prev]" and self._page_type in {"results", "paper"}:
            self._results_page = max(0, self._results_page - 1)
            self._page_type = "results"
            self._current_paper_id = None
        elif act.lower() == "click[back to results]" and self._page_type == "paper":
            self._page_type = "results"
            self._current_paper_id = None
        elif act.lower().startswith("click[paper ") and act.endswith("]"):
            pid = act[len("click[paper ") : -1].strip()
            if pid in self.backend.corpus:
                self._page_type = "paper"
                self._current_paper_id = pid
                self._has_opened_paper = True
                self._opened_paper_ids.add(pid)
                utility = self._paper_utility(pid)
                self._update_best_seen(pid, utility)
        elif (
            act.lower().startswith("stop")
            and self._page_type == "paper"
            and len(self._opened_paper_ids) >= _MIN_OPENED_PAPERS_BEFORE_STOP
        ):
            done = True

        obs = self._render_observation()
        best_pid = str(self._best_paper_id or "")
        best_is_gold = bool(best_pid and best_pid in self._query_gold_ids())
        info = {
            "goal": self._query_goal(),
            "query_id": self._current_query()["query_id"],
            "best_reward_seen": float(self._best_reward_seen),
            "best_paper_id": best_pid or None,
            "gold_hit_at_stop": bool(done and best_is_gold),
        }
        return obs, float(reward), bool(done), info

    def extract_item_ids_from_observation(self, obs: str) -> List[str]:
        if self._page_type == "paper":
            current = _CURRENT_PAPER_ID_RE.findall(str(obs or ""))
            return [pid for pid in current if pid]
        return [pid for pid in _PAPER_RESULT_ID_RE.findall(str(obs or "")) if pid]

    def get_item_info(self, item_id: str, *, include_abstract: bool = True) -> Optional[dict]:
        pid = _normalize_paper_id(item_id)
        if not pid or pid not in self.backend.corpus:
            return None
        rec = self.backend.corpus[pid]
        info = {
            "paper_id": pid,
            "Title": rec.get("Title", "N/A"),
            PAPER_REWARD_KEY: None,
        }
        if include_abstract:
            info["Abstract"] = _snip(rec.get("Abstract", ""), limit=280)
        return info

    def update_seen_items_dict(self, obs: str, seen_products: Dict[str, dict]) -> None:
        if self._page_type == "results":
            start = self._results_page * self.page_size
            end = min(len(self._results), start + self.page_size)
            for offset, pid in enumerate(self._results[start:end], start=1):
                rank = int(start + offset)
                if pid not in seen_products:
                    info = self.get_item_info(pid, include_abstract=False)
                    if info is not None:
                        seen_products[pid] = info
                if pid in seen_products:
                    seen_products[pid].setdefault("FirstSeenRank", rank)
                    seen_products[pid]["LastVisibleRank"] = rank
            return

        if self._page_type == "paper" and self._current_paper_id:
            pid = str(self._current_paper_id)
            if pid not in seen_products:
                info = self.get_item_info(pid, include_abstract=True)
                if info is not None:
                    seen_products[pid] = info
            if pid in seen_products:
                prev = seen_products[pid].get(PAPER_REWARD_KEY)
                prev_val = float(prev) if prev is not None else 0.0
                reward = self._paper_utility(pid)
                seen_products[pid][PAPER_REWARD_KEY] = max(prev_val, float(reward))
                info = self.get_item_info(pid, include_abstract=True) or {}
                for key, value in info.items():
                    if key == PAPER_REWARD_KEY and seen_products[pid].get(PAPER_REWARD_KEY) is not None:
                        continue
                    seen_products[pid][key] = value

    def is_item_page_observation(self, obs: Optional[str] = None) -> bool:
        if obs is None:
            return self._page_type == "paper"
        return bool(_CURRENT_PAPER_ID_RE.search(str(obs)))

    def compute_current_page_reward(
        self,
        *,
        obs: Optional[str] = None,
    ) -> Tuple[float, Dict[str, str], Optional[str]]:
        if not self.is_item_page_observation(obs):
            return 0.0, {}, None
        pid = str(self._current_paper_id or "")
        if not pid:
            ids = _CURRENT_PAPER_ID_RE.findall(str(obs or ""))
            pid = ids[-1] if ids else ""
        if not pid:
            return 0.0, {}, None
        reward = self._paper_utility(pid)
        return float(reward), {}, pid

    def format_seen_items_for_prompt(
        self,
        seen_products: Optional[Dict[str, dict]],
        best_item_id: Optional[str] = None,
        top_n: int = 5,
    ) -> str:
        if not seen_products:
            return "No papers seen yet."

        def _sort_key(item: Tuple[str, dict]) -> Tuple[float, int, str]:
            pid, info = item
            reward = info.get(PAPER_REWARD_KEY, None)
            try:
                reward_val = float(reward) if reward is not None else -1.0
            except Exception:
                reward_val = -1.0
            try:
                rank_val = int(info.get("FirstSeenRank", info.get("LastVisibleRank", 10**9)) or 10**9)
            except Exception:
                rank_val = 10**9
            return (-reward_val, rank_val, str(pid))

        items = sorted(seen_products.items(), key=_sort_key)
        top_items = items[: max(1, int(top_n))]
        lines = [f"Papers seen so far ({len(items)} total):"]
        for idx, (pid, info) in enumerate(top_items, start=1):
            marker = " (best)" if best_item_id and str(pid) == str(best_item_id) else ""
            lines.append(f"\n  {idx}. PaperID: {pid}{marker}")
            lines.append(f"     Title: {info.get('Title', 'N/A')}")
            reward = info.get(PAPER_REWARD_KEY, None)
            if reward is not None:
                lines.append(f"     RelevanceReward: {float(reward):.3f}")
            rank = info.get("FirstSeenRank", info.get("LastVisibleRank", None))
            if rank is not None:
                lines.append(f"     ResultRank: {rank}")
            abstract = _safe_text(info.get("Abstract", ""))
            if abstract:
                lines.append(f"     Abstract: {_snip(abstract, limit=240)}")
        return "\n".join(lines)

    def get_final_reward(self) -> float:
        return float(self._best_reward_seen)

    def get_final_gold_hit(self) -> bool:
        best_pid = str(self._best_paper_id or "")
        return bool(best_pid and best_pid in self._query_gold_ids())
