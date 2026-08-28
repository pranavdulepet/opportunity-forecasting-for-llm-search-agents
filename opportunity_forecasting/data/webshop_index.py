"""Build the full WebShop Lucene index from pinned source files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


from opportunity_forecasting import REPO_ROOT


def contents(product: dict) -> str:
    option_texts = []
    for option_name, option_values in product.get("options", {}).items():
        option_texts.append(f"{option_name}: {', '.join(option_values)}")
    bullet_points = product.get("BulletPoints") or [""]
    return " ".join(
        [
            str(product.get("Title", "")),
            str(product.get("Description", "")),
            str(bullet_points[0]),
            ", and ".join(option_texts),
        ]
    ).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webshop-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = args.webshop_root.resolve()
    data_dir = root / "data"
    search_dir = root / "search_engine"
    resources = search_dir / "resources"
    index = search_dir / "indexes"
    os.environ["WEBSHOP_DATA_DIR"] = str(data_dir)
    os.environ["WEBSHOP_SEARCH_ENGINE_DIR"] = str(search_dir)

    from opportunity_forecasting.data.webshop_setup import ensure_public_webshop_imports

    ensure_public_webshop_imports()
    from web_agent_site.engine.engine import load_products

    products, _, _, _ = load_products(
        filepath=str(data_dir / "items_shuffle.json"),
        human_goals=True,
    )
    resources.mkdir(parents=True, exist_ok=True)
    documents = resources / "documents.jsonl"
    indexable_documents = 0
    with documents.open("w", encoding="utf-8") as handle:
        for product in products:
            document_contents = contents(product)
            if document_contents.strip():
                indexable_documents += 1
            handle.write(
                json.dumps(
                    {
                        "id": product["asin"],
                        "contents": document_contents,
                        "product": product,
                    }
                )
                + "\n"
            )

    if index.exists():
        shutil.rmtree(index)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(resources),
            "--index",
            str(index),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(args.threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ],
        check=True,
    )
    from pyserini.index.lucene import IndexReader

    stats = IndexReader(str(index)).stats()
    if int(stats["documents"]) != indexable_documents:
        raise ValueError(
            f"Index has {stats['documents']} documents for {indexable_documents} nonempty products"
        )
    print(
        json.dumps(
            {
                "documents": indexable_documents,
                "empty_products": len(products) - indexable_documents,
                "index": str(index),
                "source_products": len(products),
                "status": "built",
            }
        )
    )


if __name__ == "__main__":
    main()
