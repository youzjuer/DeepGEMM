#!/usr/bin/env python3
"""Isolate scheduler/integration overhead with identical FP8xFP4 data paths."""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist

import bench_nvfp4_mode4_vs_official_ep8 as common


def _summary(values):
    return {
        "mean_us": statistics.fmean(values),
        "p50_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def _run_rank(local_rank: int, num_local_ranks: int, args) -> None:
    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    official = common._load_clean_official(args.official_repo)
    assert world_size == 8
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    local_experts = args.num_experts // world_size
    x_bf16 = torch.randn(
        (args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda"
    )
    l1_bf16 = torch.randn(
        (local_experts, args.intermediate_hidden * 2, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2_bf16 = torch.randn(
        (local_experts, args.hidden, args.intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    scores = torch.randn(
        (args.num_tokens, args.num_experts), dtype=torch.float, device="cuda"
    )
    topk_weights, topk_idx = torch.topk(
        scores, args.num_topk, dim=-1, largest=True, sorted=False
    )
    x_fp8 = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    l1_fp4 = common._cast_weights_to_fp4(l1_bf16, official)
    l2_fp4 = common._cast_weights_to_fp4(l2_bf16, official)
    official_weights = official.transform_weights_for_mega_moe(l1_fp4, l2_fp4)
    candidate_weights = deep_gemm.transform_weights_for_mega_moe(l1_fp4, l2_fp4)

    buffer_args = dict(
        group=group,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.num_tokens,
        num_topk=args.num_topk,
        hidden=args.hidden,
        intermediate_hidden=args.intermediate_hidden,
        mma_type="fp8xfp4",
    )
    official_buffer = official.get_symm_buffer_for_mega_moe(**buffer_args)
    candidate_buffer = deep_gemm.get_symm_buffer_for_mega_moe(**buffer_args)
    common._copy_routes_and_inputs(official_buffer, x_fp8, topk_idx, topk_weights)
    common._copy_routes_and_inputs(candidate_buffer, x_fp8, topk_idx, topk_weights)
    official_y = torch.empty(
        (args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda"
    )
    candidate_y = torch.empty_like(official_y)

    def run_official():
        official.fp8_fp4_mega_moe(
            official_y, official_weights[0], official_weights[1], official_buffer,
            activation_clamp=args.activation_clamp, fast_math=True,
        )

    def run_candidate():
        deep_gemm.fp8_fp4_mega_moe(
            candidate_y, candidate_weights[0], candidate_weights[1], candidate_buffer,
            activation_clamp=args.activation_clamp, fast_math=True,
        )

    run_official()
    run_candidate()
    torch.cuda.synchronize()
    common._assert_equal_on_all_ranks(
        torch.equal(official_y, candidate_y),
        "candidate FP8 output differs from official",
        group,
    )
    for i in range(args.warmup):
        first, second = (run_official, run_candidate) if i % 2 == 0 else (run_candidate, run_official)
        common._time_one(first, group)
        common._time_one(second, group)

    samples = {
        "AB": {"official_us": [], "candidate_us": []},
        "BA": {"official_us": [], "candidate_us": []},
    }
    for i in range(args.repeat):
        order = "AB" if i % 2 == 0 else "BA"
        if order == "AB":
            official_us, _ = common._time_one(run_official, group)
            candidate_us, _ = common._time_one(run_candidate, group)
        else:
            candidate_us, _ = common._time_one(run_candidate, group)
            official_us, _ = common._time_one(run_official, group)
        samples[order]["official_us"].append(official_us)
        samples[order]["candidate_us"].append(candidate_us)

    if rank == 0:
        result = {
            "shape": {"ep": 8, "tokens": args.num_tokens},
            "bit_exact": True,
            "orders": samples,
        }
        for order, values in samples.items():
            baseline = _summary(values["official_us"])
            candidate = _summary(values["candidate_us"])
            result[order] = {
                "official": baseline,
                "candidate": candidate,
                "p50_improvement_pct":
                    (baseline["p50_us"] - candidate["p50_us"]) / baseline["p50_us"] * 100,
            }
        Path(args.json_output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"AB": result["AB"], "BA": result["BA"]}, indent=2))

    dist.barrier(group=group)
    official_buffer.destroy()
    candidate_buffer.destroy()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument("--num-topk", type=int, default=10)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        _run_rank, args=(8, args), nprocs=8
    )


if __name__ == "__main__":
    main()
