#!/usr/bin/env python3
"""Same-process EP8 A/B for 32- versus 64-expert NVFP4 waves."""

import argparse
import os
from pathlib import Path

import torch

import bench_nvfp4_mode4_vs_official_ep8 as benchmark


WAVE_ENV = "DG_NVFP4_MEGAMOE_EXPERTS_PER_WAVE"
STORE_ENV = "DG_NVFP4_MEGAMOE_VECTORIZED_STORE"
SF_RING_ENV = "DG_NVFP4_MEGAMOE_MIN_SF_BLOCK_M"


def _set_wave(mode: int) -> None:
    assert mode in (0, 4)
    os.environ[benchmark.EPOCH_ENV] = "1"
    os.environ[benchmark.MODE_ENV] = "4"
    os.environ[STORE_ENV] = "0"
    os.environ[SF_RING_ENV] = "0"
    os.environ[WAVE_ENV] = "32" if mode == 0 else "64"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    benchmark._set_nvfp4_mode = _set_wave
    run_args = argparse.Namespace(
        num_processes=8,
        official_repo=args.official_repo,
        num_tokens=args.num_tokens,
        hidden=4096,
        intermediate_hidden=1024,
        num_experts=512,
        num_topk=10,
        activation_clamp=10.0,
        fast_math=1,
        seed=args.seed,
        masked_route_fraction=0.0,
        warmup=0,
        repeat=0,
        stress_iterations=0,
        cuda_graph_replays=0,
        mode0_warmup=args.warmup,
        mode0_repeat=args.repeat,
        json_output=args.json_output,
    )
    torch.multiprocessing.start_processes(
        benchmark._run_rank,
        args=(run_args.num_processes, run_args),
        nprocs=run_args.num_processes,
        join=True,
        start_method="fork",
    )


if __name__ == "__main__":
    main()
