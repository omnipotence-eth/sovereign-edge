"""Jetson Orin Nano inference benchmark — compare Ollama models and quantizations.

Measures tokens/sec, latency percentiles (p50/p95/p99), and memory usage
across different model configurations. Designed to run ON the Jetson.

Usage:
    python scripts/jetson_benchmark.py                          # default: qwen3:0.6b
    python scripts/jetson_benchmark.py --models qwen3:0.6b qwen3:4b
    python scripts/jetson_benchmark.py --runs 20 --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_PROMPT = (
    "Explain the key differences between transformer and RNN architectures "
    "for sequence modeling in three concise paragraphs."
)
DEFAULT_WARMUP_PROMPT = "Hello, how are you?"


@dataclass
class RunResult:
    """Single inference run metrics."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ms: float
    prompt_eval_duration_ms: float
    eval_duration_ms: float
    tokens_per_second: float


@dataclass
class BenchmarkResult:
    """Aggregated benchmark for a single model."""

    model: str
    runs: int
    prompt: str
    tokens_per_second_mean: float
    tokens_per_second_std: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    tegrastats_snapshot: dict[str, str] | None = None
    individual_runs: list[RunResult] = field(default_factory=list)


def get_tegrastats_snapshot() -> dict[str, str] | None:
    """Capture a single tegrastats reading (Jetson-specific)."""
    try:
        result = subprocess.run(
            ["/usr/bin/tegrastats", "--interval", "100", "--count", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return {"raw": result.stdout.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_gpu_memory() -> dict[str, str] | None:
    """Get GPU memory usage via nvidia-smi (works on both Jetson and desktop)."""
    try:
        result = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) == 3:
                return {
                    "gpu_memory_used_mb": parts[0].strip(),
                    "gpu_memory_total_mb": parts[1].strip(),
                    "gpu_memory_free_mb": parts[2].strip(),
                }
    except FileNotFoundError:
        pass
    return None


def ollama_generate(model: str, prompt: str, timeout: float = 120.0) -> RunResult:
    """Run a single Ollama generation and extract timing metrics."""
    with httpx.Client(base_url=OLLAMA_BASE, timeout=timeout) as client:
        resp = client.post(
            "/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()

    total_ns = data.get("total_duration", 0)
    prompt_eval_ns = data.get("prompt_eval_duration", 0)
    eval_ns = data.get("eval_duration", 0)
    eval_count = data.get("eval_count", 1)

    tokens_per_sec = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0

    return RunResult(
        model=model,
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=eval_count,
        total_duration_ms=total_ns / 1e6,
        prompt_eval_duration_ms=prompt_eval_ns / 1e6,
        eval_duration_ms=eval_ns / 1e6,
        tokens_per_second=tokens_per_sec,
    )


def warmup(model: str) -> None:
    """Warmup run to load model into GPU memory."""
    logger.info("warming up model=%s", model)
    try:
        ollama_generate(model, DEFAULT_WARMUP_PROMPT, timeout=180.0)
    except httpx.HTTPError as e:
        logger.warning("warmup failed model=%s err=%s", model, e)


def benchmark_model(
    model: str,
    prompt: str,
    num_runs: int,
) -> BenchmarkResult:
    """Run N inference passes and compute aggregate statistics."""
    warmup(model)
    time.sleep(1)  # let GPU settle

    results: list[RunResult] = []
    for i in range(num_runs):
        logger.info("run %d/%d model=%s", i + 1, num_runs, model)
        try:
            result = ollama_generate(model, prompt)
            results.append(result)
            logger.info(
                "  tok/s=%.1f completion_tokens=%d latency=%.0fms",
                result.tokens_per_second,
                result.completion_tokens,
                result.total_duration_ms,
            )
        except httpx.HTTPError as e:
            logger.error("run %d failed model=%s err=%s", i + 1, model, e)

    if not results:
        logger.error("all runs failed model=%s", model)
        return BenchmarkResult(
            model=model,
            runs=0,
            prompt=prompt,
            tokens_per_second_mean=0,
            tokens_per_second_std=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            avg_prompt_tokens=0,
            avg_completion_tokens=0,
        )

    tps_values = [r.tokens_per_second for r in results]
    latencies = [r.total_duration_ms for r in results]
    sorted_latencies = sorted(latencies)

    def percentile(data: list[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    tegra = get_tegrastats_snapshot()
    gpu_mem = get_gpu_memory()
    snapshot = {}
    if tegra:
        snapshot.update(tegra)
    if gpu_mem:
        snapshot.update(gpu_mem)

    return BenchmarkResult(
        model=model,
        runs=len(results),
        prompt=prompt,
        tokens_per_second_mean=statistics.mean(tps_values),
        tokens_per_second_std=statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0,
        latency_p50_ms=percentile(sorted_latencies, 50),
        latency_p95_ms=percentile(sorted_latencies, 95),
        latency_p99_ms=percentile(sorted_latencies, 99),
        avg_prompt_tokens=statistics.mean(r.prompt_tokens for r in results),
        avg_completion_tokens=statistics.mean(r.completion_tokens for r in results),
        tegrastats_snapshot=snapshot or None,
        individual_runs=results,
    )


def print_results(benchmarks: list[BenchmarkResult]) -> None:
    """Print a comparison table to stdout."""
    print("\n" + "=" * 80)
    print("JETSON INFERENCE BENCHMARK RESULTS")
    print("=" * 80)

    header = f"{'Model':<25} {'tok/s':>8} {'±std':>8} {'p50ms':>8} {'p95ms':>8} {'p99ms':>8}"
    print(header)
    print("-" * 80)

    for b in benchmarks:
        print(
            f"{b.model:<25} {b.tokens_per_second_mean:>8.1f} "
            f"{b.tokens_per_second_std:>8.1f} {b.latency_p50_ms:>8.0f} "
            f"{b.latency_p95_ms:>8.0f} {b.latency_p99_ms:>8.0f}"
        )

    print("-" * 80)
    for b in benchmarks:
        print(f"\n{b.model}: {b.runs} runs, avg {b.avg_completion_tokens:.0f} completion tokens")
        if b.tegrastats_snapshot:
            for k, v in b.tegrastats_snapshot.items():
                print(f"  {k}: {v}")
    print()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Jetson Ollama inference benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3:0.6b"],
        help="Ollama model tags to benchmark (default: qwen3:0.6b)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of inference runs per model (default: 10)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to use for benchmarking",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON results to file",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_BASE,
        help="Ollama base URL (default: http://localhost:11434)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    global OLLAMA_BASE
    OLLAMA_BASE = args.ollama_url

    # Verify Ollama is reachable
    try:
        httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0).raise_for_status()
    except httpx.HTTPError:
        logger.error("cannot reach Ollama at %s — is it running?", OLLAMA_BASE)
        sys.exit(1)

    benchmarks: list[BenchmarkResult] = []
    for model in args.models:
        logger.info("benchmarking model=%s runs=%d", model, args.runs)
        result = benchmark_model(model, args.prompt, args.runs)
        benchmarks.append(result)

    print_results(benchmarks)

    if args.output:
        output_data = [
            {k: v for k, v in asdict(b).items() if k != "individual_runs"} for b in benchmarks
        ]
        args.output.write_text(json.dumps(output_data, indent=2))
        logger.info("results written to %s", args.output)


if __name__ == "__main__":
    main()
