"""
Evaluation script to measure retrieval performance (Recall@k and MRR)
against the synthetic ground-truth test dataset for FIA Regulations.

Supports two evaluation modes:
  1. HTTP API mode (Default - works from host without local dependencies):
     python scripts/evaluate_retrieval.py --api-url http://localhost:8000
  2. Direct Python mode (for running inside Docker or with local packages):
     python scripts/evaluate_retrieval.py --direct
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_COLLECTION = "fia_regulations"
logging.basicConfig(level=logging.WARNING)


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Loads the evaluation questions dataset from JSON."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_via_http(
    api_url: str,
    query: str,
    k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Fetches top-k chunks from the running FastAPI /search endpoint."""
    params = urllib.parse.urlencode({"q": query, "k": k, "collection_name": collection_name})
    url = f"{api_url.rstrip('/')}/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "EvaluationHarness/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", [])


def query_via_direct(
    query: str,
    k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Queries directly via Python imports (inside Docker)."""
    from app.services.vectorstore import search_similar

    return search_similar(query=query, k=k, collection_name=collection_name)


def check_api_health(api_url: str) -> tuple[bool, str]:
    """Checks if the FastAPI server is reachable."""
    try:
        url = f"{api_url.rstrip('/')}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("api") == "ok":
                return True, "API healthy"
            return False, f"Unexpected health response: {data}"
    except urllib.error.URLError as exc:
        return False, f"Cannot connect to {api_url} ({exc.reason})"
    except Exception as exc:
        return False, str(exc)


def evaluate_query(
    item: dict[str, Any],
    max_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    api_url: str | None = None,
    use_direct: bool = False,
) -> dict[str, Any]:
    """
    Evaluates a single query against ChromaDB and determines the rank of the first hit.
    """
    query = item["query"]
    expected_art = item.get("expected_article", "").strip().upper()
    expected_sub = item.get("expected_sub_section", "").strip().upper()

    try:
        if use_direct or not api_url:
            retrieved = query_via_direct(query=query, k=max_k, collection_name=collection_name)
        else:
            retrieved = query_via_http(api_url=api_url, query=query, k=max_k, collection_name=collection_name)
    except Exception as exc:
        return {
            "query": query,
            "error": str(exc),
            "first_match_rank": None,
            "retrieved": [],
        }

    first_match_rank = None
    for rank, chunk in enumerate(retrieved, start=1):
        chunk_art = str(chunk.get("article_number", "")).strip().upper()
        chunk_sub = str(chunk.get("sub_section", "")).strip().upper()
        citation = str(chunk.get("citation", "")).upper()

        article_matches = (expected_art == chunk_art) or (expected_art in citation)
        sub_matches = True
        if expected_sub:
            sub_matches = (expected_sub in chunk_sub) or (expected_sub in citation)

        if article_matches and sub_matches:
            first_match_rank = rank
            break

    return {
        "id": item.get("id"),
        "query": query,
        "category": item.get("category", "General"),
        "expected_article": expected_art,
        "expected_sub": expected_sub,
        "first_match_rank": first_match_rank,
        "retrieved_count": len(retrieved),
        "top_retrieved": [c.get("citation", "") for c in retrieved[:3]],
    }


def run_evaluation(
    dataset_path: Path,
    k_values: list[int] = [1, 3, 5],
    collection_name: str = DEFAULT_COLLECTION,
    api_url: str = "http://localhost:8000",
    use_direct: bool = False,
) -> dict[str, Any]:
    """Runs retrieval evaluation across all questions in the dataset."""
    dataset = load_dataset(dataset_path)
    total_queries = len(dataset)
    max_k = max(k_values)

    mode_desc = "Direct Python Import" if use_direct else f"HTTP API ({api_url})"
    print(f"\n=======================================================")
    print(f" FIA Regulation Retrieval Evaluation ({total_queries} queries)")
    print(f" Mode: {mode_desc}")
    print(f" Target Collection: {collection_name}")
    print(f" Metrics evaluated: Recall@{k_values} & MRR")
    print(f"=======================================================\n")

    # If running in HTTP mode, perform initial connection check
    if not use_direct and api_url:
        healthy, reason = check_api_health(api_url)
        if not healthy:
            print(f"[ERROR] API is unreachable at {api_url}: {reason}")
            print("\nTo fix this:")
            print("  1. Start the Docker services: docker compose up -d")
            print("  2. Ingest your cleaned markdown file:")
            print("     curl -X POST http://localhost:8000/ingest -H \"Content-Type: application/json\" -d \"{}\"")
            print("  3. Run this evaluation again: python scripts/evaluate_retrieval.py\n")
            return {"error": reason}

    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    failures = []
    empty_results = 0
    category_stats: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(dataset, start=1):
        cat = item.get("category", "General")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "hits_at_3": 0, "hits_at_5": 0}
        category_stats[cat]["total"] += 1

        eval_result = evaluate_query(
            item,
            max_k=max_k,
            collection_name=collection_name,
            api_url=api_url,
            use_direct=use_direct,
        )

        if "error" in eval_result:
            print(f"[{idx}/{total_queries}] ERROR on Q{item.get('id')}: {eval_result['error']}")
            failures.append(eval_result)
            continue

        if eval_result["retrieved_count"] == 0:
            empty_results += 1

        rank = eval_result["first_match_rank"]
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            for k in k_values:
                if rank <= k:
                    hits_at_k[k] += 1
            if rank <= 3:
                category_stats[cat]["hits_at_3"] += 1
            if rank <= 5:
                category_stats[cat]["hits_at_5"] += 1
            print(f"[{idx:02d}/{total_queries:02d}] HIT  @ rank {rank} | Q: {item['query'][:55]}...")
        else:
            reciprocal_ranks.append(0.0)
            failures.append(eval_result)
            print(f"[{idx:02d}/{total_queries:02d}] MISS @ max_k {max_k} | Q: {item['query'][:55]}...")

    if empty_results == total_queries:
        print(f"\n[WARNING] All queries returned 0 results!")
        print(f"It appears the collection '{collection_name}' is empty.")
        print(f"Please run ingestion first by calling:")
        print(f'  curl -X POST {api_url}/ingest -H "Content-Type: application/json" -d "{{}}"\n')

    # Calculate overall metrics
    recall_scores = {
        f"Recall@{k}": round((hits / total_queries) * 100, 1) if total_queries else 0.0
        for k, hits in hits_at_k.items()
    }
    mrr = round(sum(reciprocal_ranks) / total_queries, 3) if total_queries else 0.0

    print(f"\n-------------------------------------------------------")
    print(f"                 OVERALL SCORECARD                     ")
    print(f"-------------------------------------------------------")
    for metric, score in recall_scores.items():
        print(f"  {metric:<15}: {score:>6.1f}%")
    print(f"  {'MRR':<15}: {mrr:>6.3f}")
    print(f"-------------------------------------------------------")

    # Category breakdown
    print(f"\n  Breakdown by Category:")
    for cat, stats in sorted(category_stats.items()):
        tot = stats["total"]
        r3 = round((stats["hits_at_3"] / tot) * 100, 1) if tot else 0.0
        r5 = round((stats["hits_at_5"] / tot) * 100, 1) if tot else 0.0
        print(f"   * {cat:<26}: Recall@3={r3:>5.1f}% | Recall@5={r5:>5.1f}% ({tot} questions)")

    # Print failures
    if failures and empty_results < total_queries:
        print(f"\n-------------------------------------------------------")
        print(f"               FAILURE ANALYSIS ({len(failures)} misses)")
        print(f"-------------------------------------------------------")
        for f in failures:
            print(f"\nQ: {f.get('query')}")
            print(f"  Expected: Article {f.get('expected_article')} ({f.get('expected_sub', '')})")
            print(f"  Top retrieved instead:")
            for r_idx, cit in enumerate(f.get("top_retrieved", []), start=1):
                print(f"    [{r_idx}] {cit}")

    return {
        "total_queries": total_queries,
        "recall_scores": recall_scores,
        "mrr": mrr,
        "failure_count": len(failures),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval Recall@k")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "tests" / "data" / "synthetic_eval_set.json",
        help="Path to evaluation questions JSON",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,3,5",
        help="Comma-separated list of k values (e.g. '1,3,5')",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of running FastAPI server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run via direct Python imports inside Docker instead of HTTP API",
    )
    args = parser.parse_args()

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    run_evaluation(
        dataset_path=args.dataset,
        k_values=k_values,
        collection_name=args.collection,
        api_url=args.api_url,
        use_direct=args.direct,
    )


if __name__ == "__main__":
    main()
