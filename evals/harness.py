"""
evals/harness.py — pytest-based eval runner for all squads.

Run with: uv run python evals/harness.py
Or: task eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_evals(output_json: Path | None = None) -> dict:
    """Run all eval datasets against their respective squad judges."""
    results: dict = {
        "router": [],
        "spiritual": [],
        "career": [],
        "intelligence": [],
        "creative": [],
    }
    # TODO: populate with dataset-driven evals per squad
    # Each dataset in evals/datasets/<squad>/*.jsonl
    # Each judge in evals/judges/<squad>.py
    logger.info("Eval harness placeholder — add datasets to evals/datasets/")
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", output_json)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    run_evals(args.output_json)
