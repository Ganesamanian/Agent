from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agent_system import AppConfig, ModelProvider, RagStore
from data.rag_eval import DEFAULT_RETRIEVAL_EVAL_SET

load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/reuse the Milvus Lite RAG database and run retrieval evaluation.")
    parser.add_argument("--embedding-provider", choices=["openai", "gemini"], default=None)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if unchanged.")
    parser.add_argument("--status", action="store_true", help="Print collection status and exit.")
    parser.add_argument("--query", default="", help="Optional retrieval query to run after build.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--no-eval", action="store_true", help="Skip retrieval evaluation.")
    return parser.parse_args()


def evaluate_retrieval(rag_store: RagStore, *, embedding_provider: Optional[str], top_k: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    hit_count = 0
    reciprocal_rank_total = 0.0

    for case in DEFAULT_RETRIEVAL_EVAL_SET:
        retrieved = rag_store.retrieve(case["query"], top_k=top_k, embedding_provider=embedding_provider)
        retrieved_names = [Path(item.source_path or "").name for item in retrieved]
        expected = set(case["expected_docs"])
        hit = any(name in expected for name in retrieved_names)
        rank = 0
        for index, name in enumerate(retrieved_names, start=1):
            if name in expected:
                rank = index
                break
        reciprocal_rank = 1.0 / rank if rank else 0.0
        hit_count += int(hit)
        reciprocal_rank_total += reciprocal_rank
        rows.append(
            {
                "query": case["query"],
                "expected_docs": case["expected_docs"],
                "retrieved_docs": retrieved_names,
                "hit": hit,
                "rank": rank or None,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    total = len(DEFAULT_RETRIEVAL_EVAL_SET)
    return {
        "summary": {
            "total_cases": total,
            "hit_rate": hit_count / total if total else 0.0,
            "mrr": reciprocal_rank_total / total if total else 0.0,
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env()
    provider = ModelProvider(config)
    rag_store = RagStore(config, provider)

    embedding_provider = config.get_embedding_provider(args.embedding_provider)
    embedding_model = config.get_embedding_model(embedding_provider)

    if args.status:
        print(json.dumps(rag_store.status(embedding_provider=embedding_provider, embedding_model=embedding_model), indent=2))
        return

    build = rag_store.build_or_reuse(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        force_rebuild=args.rebuild,
    )

    payload: Dict[str, Any] = {"build": build}
    if not args.no_eval:
        payload["evaluation"] = evaluate_retrieval(rag_store, embedding_provider=embedding_provider, top_k=args.top_k)
    if args.query:
        payload["query_result"] = [
            item.to_dict()
            for item in rag_store.retrieve(args.query, top_k=args.top_k, embedding_provider=embedding_provider)
        ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
