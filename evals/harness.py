"""Eval harness for Sovereign Edge — router accuracy + squad quality checks.

Runs two eval types:
  1. Router eval — keyword classifier accuracy on labeled JSONL dataset (fast, no LLM)
  2. Squad eval — heuristic quality checks on LLM responses (requires API keys)

Usage:
    # Router only (fast, no API keys)
    uv run python evals/harness.py --router-only

    # Full eval (requires GROQ_API_KEY or other LLM key)
    uv run python evals/harness.py

    # Save results to JSON
    uv run python evals/harness.py --output-json evals/results/latest.json

    # CI mode — exit 1 if any eval falls below threshold
    uv run python evals/harness.py --router-only --fail-threshold 0.7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_EVALS_DIR = Path(__file__).parent
_DATASETS_DIR = _EVALS_DIR / "datasets"
_RESULTS_DIR = _EVALS_DIR / "results"

# Accuracy thresholds for CI pass/fail
_ROUTER_PASS_THRESHOLD = 0.75  # ≥75% intent accuracy required
_SQUAD_PASS_THRESHOLD = 0.60  # ≥60% heuristic pass rate required


# ── Router eval ───────────────────────────────────────────────────────────────


def run_router_eval() -> dict:
    """Evaluate router keyword classifier on labeled examples."""
    sys.path.insert(0, str(_EVALS_DIR))
    from judges.router_judge import evaluate, score

    dataset = _DATASETS_DIR / "router" / "router_eval.jsonl"
    if not dataset.exists():
        logger.warning("Router eval dataset not found: %s", dataset)
        return {"skipped": True, "reason": "dataset not found"}

    t0 = time.perf_counter()
    results = evaluate(dataset)
    elapsed = time.perf_counter() - t0
    scored = score(results)
    scored["elapsed_s"] = round(elapsed, 3)

    accuracy = scored["accuracy"]
    logger.info(
        "Router eval: %d/%d correct (%.0f%%) in %.2fs",
        scored["correct"],
        scored["total"],
        accuracy * 100,
        elapsed,
    )
    if scored["failures"]:
        logger.info("Failures:")
        for f in scored["failures"]:
            logger.info(
                "  expected=%-15s got=%-15s conf=%.2f | %s",
                f["expected"],
                f["got"],
                f["conf"],
                f["text"],
            )

    return scored


# ── Squad evals ───────────────────────────────────────────────────────────────


async def run_squad_eval(squad_name: str, dataset_path: Path) -> dict:
    """Run one squad against its eval dataset using the live LLM."""
    sys.path.insert(0, str(_EVALS_DIR))
    from judges.squad_judge import SquadEvalResult, check, load_dataset, score

    if not dataset_path.exists():
        logger.warning("Squad eval dataset not found: %s", dataset_path)
        return {"squad": squad_name, "skipped": True, "reason": "dataset not found"}

    rows = load_dataset(dataset_path)
    results: list[SquadEvalResult] = []
    t0 = time.perf_counter()

    squad = _load_squad(squad_name)
    if squad is None:
        return {"squad": squad_name, "skipped": True, "reason": "squad import failed"}

    for row in rows:
        query = row["query"]
        try:
            state = _build_state(query)
            response = await squad.run(state)  # type: ignore[arg-type]
        except Exception:
            logger.error("Squad eval error for '%s'", query, exc_info=True)
            response = ""

        result = check(query, response, row)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        logger.info("[%s] %s | %s", status, squad_name, query[:60])
        if result.failures:
            for fail in result.failures:
                logger.info("       ↳ %s", fail)

    elapsed = time.perf_counter() - t0
    scored = score(results)
    scored["squad"] = squad_name
    scored["elapsed_s"] = round(elapsed, 3)
    logger.info(
        "%s eval: %d/%d passed (%.0f%%) in %.2fs",
        squad_name,
        scored["passed"],
        scored["total"],
        scored["accuracy"] * 100,
        elapsed,
    )
    return scored


def _load_squad(squad_name: str) -> object | None:
    import importlib

    squad_map = {
        "spiritual": ("spiritual.squad", "SpiritualSquad"),
        "career": ("career.squad", "CareerSquad"),
        "intelligence": ("intelligence.squad", "IntelligenceSquad"),
        "creative": ("creative.squad", "CreativeSquad"),
    }
    if squad_name not in squad_map:
        logger.error("Unknown squad: %s", squad_name)
        return None
    module_name, class_name = squad_map[squad_name]
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        return cls()
    except Exception:
        logger.error("Failed to import squad %s.%s", module_name, class_name, exc_info=True)
        return None


def _build_state(query: str) -> dict:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=query)],
        "intent": "eval",
        "intent_confidence": 1.0,
        "memory_context": "",
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


async def _run_all(
    router_only: bool = False,
    squads: list[str] | None = None,
) -> dict:
    results: dict = {}

    logger.info("=" * 60)
    logger.info("ROUTER EVAL")
    logger.info("=" * 60)
    results["router"] = run_router_eval()

    if not router_only:
        eval_squads = squads or ["spiritual", "career", "intelligence", "creative"]
        for squad_name in eval_squads:
            logger.info("=" * 60)
            logger.info("SQUAD EVAL: %s", squad_name.upper())
            logger.info("=" * 60)
            dataset = _DATASETS_DIR / squad_name / f"{squad_name}_eval.jsonl"
            results[squad_name] = await run_squad_eval(squad_name, dataset)

    return results


def run_evals(
    output_json: Path | None = None,
    router_only: bool = False,
    squads: list[str] | None = None,
) -> dict:
    """Public API — run all evals and optionally write results to JSON."""
    results = asyncio.run(_run_all(router_only=router_only, squads=squads))

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", output_json)

    return results


def _check_thresholds(results: dict, fail_threshold: float) -> bool:
    all_pass = True
    for name, result in results.items():
        if result.get("skipped"):
            continue
        accuracy = result.get("accuracy", 0.0)
        threshold = _ROUTER_PASS_THRESHOLD if name == "router" else _SQUAD_PASS_THRESHOLD
        effective = max(threshold, fail_threshold)
        if accuracy < effective:
            logger.error("FAIL: %s accuracy=%.2f < threshold=%.2f", name, accuracy, effective)
            all_pass = False
        else:
            logger.info("PASS: %s accuracy=%.2f", name, accuracy)
    return all_pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Sovereign Edge eval harness")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--router-only",
        action="store_true",
        help="Run only the router eval (no LLM calls, fast)",
    )
    parser.add_argument(
        "--squads",
        nargs="*",
        default=None,
        help="Specific squads to eval: spiritual career intelligence creative",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.0,
        help="Exit code 1 if any eval accuracy falls below this threshold",
    )
    args = parser.parse_args()

    output = args.output_json
    if output is None:
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = _RESULTS_DIR / f"eval_{ts}.json"

    results = run_evals(
        output_json=output,
        router_only=args.router_only,
        squads=args.squads,
    )

    logger.info("=" * 60)
    logger.info("EVAL SUMMARY")
    logger.info("=" * 60)
    for name, r in results.items():
        if r.get("skipped"):
            logger.info("  %-15s SKIPPED (%s)", name, r.get("reason", ""))
        else:
            pct = r.get("accuracy", 0.0) * 100
            n = r.get("total", 0)
            logger.info("  %-15s %.0f%% (%d samples)", name, pct, n)

    if args.fail_threshold > 0:
        passed = _check_thresholds(results, args.fail_threshold)
        sys.exit(0 if passed else 1)
