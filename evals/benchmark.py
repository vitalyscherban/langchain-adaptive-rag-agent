"""Cost vs quality benchmark: optimized adaptive pipeline vs a naive baseline.

Compares the token-optimized pipeline (dynamic top-k + MMR re-ranking +
contextual compression + hard token budget) against a naive baseline
(fixed top-k=10 similarity search, no MMR, no compression) across a small set
of representative simple and complex queries against the sample documents.

Runs fully offline using a deterministic hashing-based embedder
(:class:`evals.fake_embeddings.DeterministicHashEmbeddings`) so it requires
no API key or network access, making it safe to run in CI or sandboxed
environments. Set ``EVAL_USE_REAL_EMBEDDINGS=true`` to use real OpenAI
embeddings instead (requires ``OPENAI_API_KEY``).

Usage::

    python -m evals.benchmark
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from evals.fake_embeddings import DeterministicHashEmbeddings
from src.config import Settings, get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loaders import load_documents_from_dir
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.compressor import ContextCompressor, count_tokens
from src.retrieval.prompt_assembler import STATIC_SYSTEM_PROMPT
from src.routing.classifier import classify_query
from src.routing.model_router import ModelRouter

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

BENCHMARK_QUERIES = [
    "What payment methods are supported?",
    "What port does the API use?",
    "How often are encryption keys rotated?",
    "What is the support SLA for Enterprise plans?",
    "Compare the roles Owner, Admin, Developer, and Viewer and explain what each one can and cannot do.",
    "Explain how billing works and also describe what happens if a payment fails, "
    "including how retries and account suspension are handled.",
    "Walk me through the full onboarding process from creating an organization "
    "to setting up monitoring, and explain why separating environments into "
    "different projects matters.",
    "What is the relationship between our compliance certifications and how "
    "incident response and data residency are handled for Enterprise customers?",
]

BASELINE_TOP_K = 10
ASSUMED_COMPLETION_TOKENS = 200


@dataclass
class QueryComparison:
    query: str
    complexity: str
    model: str
    optimized_top_k: int
    optimized_context_tokens: int
    optimized_prompt_tokens: int
    optimized_estimated_cost_usd: float
    baseline_context_tokens: int
    baseline_prompt_tokens: int
    baseline_estimated_cost_usd: float
    token_savings_pct: float
    cost_savings_pct: float


def _build_embeddings(settings: Settings) -> Embeddings:
    use_real = os.getenv("EVAL_USE_REAL_EMBEDDINGS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if use_real:
        from src.retrieval.vectorstore import get_embeddings

        return get_embeddings(settings)
    return DeterministicHashEmbeddings()


def _estimate_cost(settings: Settings, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    is_cheap = model == settings.cheap_model
    prompt_price = (
        settings.cheap_model_prompt_price_per_1k if is_cheap else settings.strong_model_prompt_price_per_1k
    )
    completion_price = (
        settings.cheap_model_completion_price_per_1k
        if is_cheap
        else settings.strong_model_completion_price_per_1k
    )
    return (prompt_tokens / 1000) * prompt_price + (completion_tokens / 1000) * completion_price


def run_benchmark(settings: Settings | None = None) -> list[QueryComparison]:
    """Ingest the sample docs and compare optimized vs baseline retrieval for each query."""

    settings = settings or get_settings()
    embeddings = _build_embeddings(settings)

    tmp_dir = tempfile.mkdtemp(prefix="adaptive_rag_eval_")
    try:
        vectorstore = Chroma(
            collection_name="eval_benchmark",
            embedding_function=embeddings,
            persist_directory=tmp_dir,
        )
        documents = load_documents_from_dir(SAMPLE_DOCS_DIR)
        chunks = chunk_documents(documents, settings, embeddings)
        vectorstore.add_documents(chunks)

        retriever = AdaptiveRetriever(vectorstore, settings)
        compressor = ContextCompressor(embeddings=embeddings, kind="embeddings_filter", settings=settings)
        router = ModelRouter(settings)

        system_prompt_tokens = count_tokens(STATIC_SYSTEM_PROMPT)

        results: list[QueryComparison] = []
        for query in BENCHMARK_QUERIES:
            complexity = classify_query(query)
            decision = router.route(query)
            question_tokens = count_tokens(query)

            # --- Optimized: adaptive top-k + MMR + compression + budget ---
            retrieval = retriever.retrieve(query, complexity=complexity)
            compression = compressor.compress(query, retrieval.documents)
            optimized_context_tokens = compression.token_count
            optimized_prompt_tokens = system_prompt_tokens + optimized_context_tokens + question_tokens
            optimized_cost = _estimate_cost(
                settings, decision.model_name, optimized_prompt_tokens, ASSUMED_COMPLETION_TOKENS
            )

            # --- Baseline: naive fixed top-k=10 similarity search, no compression ---
            baseline_docs = vectorstore.similarity_search(query, k=BASELINE_TOP_K)
            baseline_context = "\n\n".join(doc.page_content for doc in baseline_docs)
            baseline_context_tokens = count_tokens(baseline_context)
            baseline_prompt_tokens = system_prompt_tokens + baseline_context_tokens + question_tokens
            baseline_cost = _estimate_cost(
                settings, decision.model_name, baseline_prompt_tokens, ASSUMED_COMPLETION_TOKENS
            )

            token_savings_pct = _pct_reduction(baseline_prompt_tokens, optimized_prompt_tokens)
            cost_savings_pct = _pct_reduction(baseline_cost, optimized_cost)

            results.append(
                QueryComparison(
                    query=query,
                    complexity=complexity.value,
                    model=decision.model_name,
                    optimized_top_k=retrieval.top_k,
                    optimized_context_tokens=optimized_context_tokens,
                    optimized_prompt_tokens=optimized_prompt_tokens,
                    optimized_estimated_cost_usd=optimized_cost,
                    baseline_context_tokens=baseline_context_tokens,
                    baseline_prompt_tokens=baseline_prompt_tokens,
                    baseline_estimated_cost_usd=baseline_cost,
                    token_savings_pct=token_savings_pct,
                    cost_savings_pct=cost_savings_pct,
                )
            )

        return results
    finally:
        # Chroma keeps file handles open on Windows; ignore cleanup errors
        # rather than letting them mask a successful benchmark run.
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _pct_reduction(baseline: float, optimized: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - optimized) / baseline * 100


def print_report(results: list[QueryComparison]) -> None:
    header = (
        f"{'Query':<60} {'Complexity':<10} {'Model':<14} {'Opt tok':>8} "
        f"{'Base tok':>9} {'Tok save%':>10} {'Cost save%':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        truncated_query = (r.query[:57] + "...") if len(r.query) > 60 else r.query
        print(
            f"{truncated_query:<60} {r.complexity:<10} {r.model:<14} "
            f"{r.optimized_prompt_tokens:>8} {r.baseline_prompt_tokens:>9} "
            f"{r.token_savings_pct:>9.1f}% {r.cost_savings_pct:>10.1f}%"
        )

    avg_token_savings = statistics.mean(r.token_savings_pct for r in results)
    avg_cost_savings = statistics.mean(r.cost_savings_pct for r in results)
    total_optimized_cost = sum(r.optimized_estimated_cost_usd for r in results)
    total_baseline_cost = sum(r.baseline_estimated_cost_usd for r in results)

    print("-" * len(header))
    print(f"Average token savings: {avg_token_savings:.1f}%")
    print(f"Average cost savings:  {avg_cost_savings:.1f}%")
    print(
        f"Total estimated cost:  optimized=${total_optimized_cost:.5f} "
        f"vs baseline=${total_baseline_cost:.5f}"
    )


def main() -> int:
    results = run_benchmark()
    print_report(results)

    report_path = Path(__file__).resolve().parent / "report.json"
    report_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print(f"\nDetailed report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
