"""CLI entry point.

Usage::

    python -m src.main ingest <dir>
    python -m src.main query "<question>"
"""

from __future__ import annotations

import argparse
import sys

from src.pipeline import AdaptiveRagPipeline


def _cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = AdaptiveRagPipeline()
    num_chunks = pipeline.ingest(args.directory)
    print(f"Indexed {num_chunks} chunks from '{args.directory}'.")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    pipeline = AdaptiveRagPipeline()
    outcome = pipeline.query(args.question)
    print(outcome.answer)
    print(
        f"\n[complexity={outcome.usage.complexity} model={outcome.usage.model} "
        f"tokens={outcome.usage.total_tokens} cost=${outcome.usage.estimated_cost_usd:.5f} "
        f"latency={outcome.usage.latency_seconds:.2f}s]",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.main", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from a directory.")
    ingest_parser.add_argument("directory", help="Path to a directory of .txt/.md documents.")
    ingest_parser.set_defaults(func=_cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Ask a question against the indexed docs.")
    query_parser.add_argument("question", help="The question to ask.")
    query_parser.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
