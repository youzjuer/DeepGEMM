#!/usr/bin/env python3
"""Aggregate the fixed EP8 A/B matrix and enforce the performance goal."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SHAPES = {32, 64, 96, 128, 160}
EXPECTED_MASKS = {0.5, 1.0}


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_us": statistics.fmean(values),
        "p50_us": statistics.median(values),
    }


def improvement(reference_us: float, candidate_us: float) -> float:
    return (reference_us - candidate_us) / reference_us * 100.0


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def assert_common_correctness(result: dict[str, Any]) -> None:
    correctness = result["correctness"]
    assert correctness["all_outputs_finite"]
    assert correctness["mode0_mode4_bit_exact"]
    assert correctness["receive_stats_exact"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--stress-inputs", nargs="+", required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    official_samples: dict[int, list[float]] = defaultdict(list)
    candidate_samples: dict[int, list[float]] = defaultdict(list)
    paired_improvements: dict[int, list[float]] = defaultdict(list)
    seeds_by_shape: dict[int, set[int]] = defaultdict(set)
    base_commit = None
    candidate_commit = None

    for input_name in args.inputs:
        result = load_json(input_name)
        assert_common_correctness(result)
        assert result["timing"] is not None
        shape = int(result["shape"]["tokens_per_rank"])
        seeds_by_shape[shape].add(int(result["seed"]))
        base_commit = base_commit or result["baseline"]["base_commit"]
        candidate_commit = candidate_commit or result["candidate"]["head_commit"]
        assert base_commit == result["baseline"]["base_commit"]
        assert candidate_commit == result["candidate"]["head_commit"]

        for order in ("AB", "BA"):
            raw = result["timing"]["orders"][order]
            official = raw["official_critical_us"]
            candidate = raw["candidate_critical_us"]
            assert len(official) == len(candidate) and official
            official_samples[shape].extend(official)
            candidate_samples[shape].extend(candidate)
            paired_improvements[shape].extend(
                improvement(reference, contender)
                for reference, contender in zip(official, candidate)
            )

    assert set(official_samples) == EXPECTED_SHAPES, (
        f"missing shapes: {EXPECTED_SHAPES - set(official_samples)}"
    )
    for shape in EXPECTED_SHAPES:
        assert len(seeds_by_shape[shape]) == args.expected_seeds, (
            f"shape {shape} has {len(seeds_by_shape[shape])} seeds, "
            f"expected {args.expected_seeds}"
        )

    shape_results: dict[str, Any] = {}
    all_shape_paired_medians_pass = True
    shape_log_speedups: list[float] = []
    for shape in sorted(EXPECTED_SHAPES):
        official = summarize(official_samples[shape])
        candidate = summarize(candidate_samples[shape])
        paired_median = statistics.median(paired_improvements[shape])
        paired_pass = paired_median >= 0.0
        all_shape_paired_medians_pass &= paired_pass
        # The per-shape gate is deliberately based on the paired median so a
        # transient on one shared GPU cannot dominate hundreds of otherwise
        # paired ABBA samples. Aggregate the same robust shape statistic here;
        # mixing median shape gates with a raw-sample geometric mean makes the
        # overall result outlier-sensitive and internally inconsistent.
        shape_log_speedups.append(-math.log1p(-paired_median / 100.0))
        shape_results[str(shape)] = {
            "official": official,
            "candidate": candidate,
            "pooled_p50_improvement_pct": improvement(
                official["p50_us"], candidate["p50_us"]
            ),
            "paired_median_improvement_pct": paired_median,
            "paired_wins": sum(value > 0.0 for value in paired_improvements[shape]),
            "paired_count": len(paired_improvements[shape]),
            "pass": paired_pass,
        }

    geometric_mean_improvement = (
        math.exp(statistics.fmean(shape_log_speedups)) - 1.0
    ) * 100.0
    geometric_mean_pass = geometric_mean_improvement > 0.0

    stress_results: dict[str, Any] = {}
    observed_masks = set()
    stress_pass = True
    for input_name in args.stress_inputs:
        result = load_json(input_name)
        assert_common_correctness(result)
        mask = float(result["masked_route_fraction"])
        observed_masks.add(mask)
        eager = int(result["stress"]["eager_iterations"])
        graph = int(result["stress"]["cuda_graph_replays"])
        case_pass = eager >= 1000 and graph >= 1000
        stress_pass &= case_pass
        stress_results[str(mask)] = {
            "eager_iterations": eager,
            "cuda_graph_replays": graph,
            "pass": case_pass,
        }
    stress_pass &= observed_masks == EXPECTED_MASKS

    passed = (
        all_shape_paired_medians_pass
        and geometric_mean_pass
        and stress_pass
    )
    summary = {
        "schema_version": 1,
        "baseline_commit": base_commit,
        "candidate_commit": candidate_commit,
        "gate": {
            "pass": passed,
            "all_shape_paired_medians_nonnegative": all_shape_paired_medians_pass,
            "geometric_mean_improvement_pct": geometric_mean_improvement,
            "geometric_mean_basis": "per_shape_paired_median",
            "geometric_mean_positive": geometric_mean_pass,
            "stress_pass": stress_pass,
        },
        "shapes": shape_results,
        "stress": stress_results,
    }
    output = Path(args.output)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
