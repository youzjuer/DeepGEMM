#!/usr/bin/env python3
"""EP8 A/B benchmark for official MegaMoE versus packed-FP4 Mode4.

The baseline is the FP8xFP4 MegaMoE path inherited from official DeepGEMM
559d79f.  The candidate uses the same scheduler, routes, source tensors and
timing method, changing only the candidate quantization contract and enabling
epoch workspace plus Mode4 readiness.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import (
    per_token_cast_to_fp4,
    per_token_cast_to_fp8,
    per_token_cast_to_mxfp4,
    per_token_cast_to_nvfp4,
)
from deep_gemm.utils.dist import init_dist


OFFICIAL_BASE_COMMIT = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
MODE_ENV = "DG_NVFP4_MEGAMOE_DISPATCH_READY_MODE"
EPOCH_ENV = "DG_NVFP4_MEGAMOE_EPOCH_WORKSPACE"
GROUPED_SF_ENV = "DG_NVFP4_MEGAMOE_WARP_RECIPROCAL_TABLE"

PROFILE_STAGES = (
    ("gemm_l1", "stage_gemm_l1"),
    ("gemm_l2", "stage_gemm_l2"),
    ("epilogue_l1", "stage_epilogue_l1"),
    ("epilogue_l2", "stage_epilogue_l2"),
    ("combine", "stage_combine"),
    ("route_metadata", "stage_route_metadata"),
    ("dispatch_publish_barrier", "stage_dispatch_publish_barrier"),
    ("remote_pull", "stage_remote_pull"),
    ("cleanup_barrier", "stage_cleanup_barrier"),
    ("remote_output_push", "stage_remote_output_push"),
    ("combine_barrier", "stage_combine_barrier"),
)


def _union_ns(intervals: List[Tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    begin, end = ordered[0]
    for next_begin, next_end in ordered[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def _parse_profile(profile: torch.Tensor, layout: Dict[str, int]) -> Dict[str, object]:
    profile = profile.cpu().reshape(-1, layout["words_per_cta"])
    stage_shift = layout["stage_shift"]
    timestamp_mask = layout["timestamp_mask"]
    stage_names = {layout[key]: name for name, key in PROFILE_STAGES}
    stages: Dict[str, List[Tuple[int, int]]] = {name: [] for name, _ in PROFILE_STAGES}
    kernel: List[Tuple[int, int]] = []

    def add(encoded_start: int, end: int) -> None:
        encoded_start &= (1 << 64) - 1
        start = encoded_start & timestamp_mask
        end &= timestamp_mask
        stage = encoded_start >> stage_shift
        if start and end >= start and stage in stage_names:
            stages[stage_names[stage]].append((start, end))

    for cta in profile:
        for warp in range(layout["max_warps"]):
            offset = layout["kernel_offset"] + warp * 2
            start, end = int(cta[offset]), int(cta[offset + 1])
            if start and end >= start:
                kernel.append((start, end))
        block_count = min(
            int(cta[layout["block_start_count_offset"]]),
            int(cta[layout["block_end_count_offset"]]),
            layout["max_blocks_per_cta"],
        )
        for block in range(block_count):
            offset = layout["block_offset"] + block * 2
            add(int(cta[offset]), int(cta[offset + 1]))
        for kind, base in enumerate((layout["compute_offset"], layout["communication_offset"])):
            for warp in range(layout["max_warps"]):
                count = min(
                    int(cta[layout["counter_offset"] + kind * layout["max_warps"] + warp]),
                    layout["max_intervals_per_warp"],
                )
                for interval in range(count):
                    offset = base + (warp * layout["max_intervals_per_warp"] + interval) * 2
                    add(int(cta[offset]), int(cta[offset + 1]))
    assert kernel
    span = max(end for _, end in kernel) - min(start for start, _ in kernel)
    return {
        "kernel_span_us": span / 1000.0,
        "stage_active_us": {name: _union_ns(values) / 1000.0 for name, values in stages.items()},
    }


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()


def _load_clean_official(repo: Path):
    """Load clean official DeepGEMM beside the candidate package.

    The alternate package name keeps its pybind extension, JIT compiler and
    Python MegaMoE wrappers independent from the candidate while both share
    the same process group and CUDA timing stream.
    """
    package_dir = repo / "deep_gemm"
    module_name = "deep_gemm_official_559d79f"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _summary(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    assert ordered
    p90_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.9) - 1)
    return {
        "count": len(ordered),
        "mean_us": statistics.fmean(ordered),
        "p50_us": statistics.median(ordered),
        "p90_us": ordered[p90_index],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _relative_improvement(reference_us: float, candidate_us: float) -> float:
    return (reference_us - candidate_us) / reference_us * 100.0


def _cast_weights_to_fp4(
    weights: torch.Tensor, official_deep_gemm,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = weights.shape
    packed = torch.empty((num_groups, n, k // 2), dtype=torch.int8, device="cuda")
    scales = torch.empty((num_groups, n, k // 32), dtype=torch.float, device="cuda")
    for group_idx in range(num_groups):
        packed[group_idx], scales[group_idx] = per_token_cast_to_fp4(
            weights[group_idx], use_ue8m0=True, gran_k=32
        )
    scales = official_deep_gemm.transform_sf_into_required_layout(
        scales, n, k, (1, 32), num_groups
    )
    return packed, scales


def _cast_weights_to_packed_fp4(
    weights: torch.Tensor, contract: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = weights.shape
    packed = torch.empty((num_groups, n, k // 2), dtype=torch.int8, device="cuda")
    packed_sf_k = k // (64 if contract == "nvfp4" else 128)
    scales = torch.empty((num_groups, n, packed_sf_k), dtype=torch.int32, device="cuda")
    quantize = per_token_cast_to_nvfp4 if contract == "nvfp4" else per_token_cast_to_mxfp4
    for group_idx in range(num_groups):
        packed[group_idx], scales[group_idx] = quantize(weights[group_idx])
    return packed, scales


def _copy_routes_and_inputs(
    buffer,
    x: Tuple[torch.Tensor, torch.Tensor],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    num_tokens = topk_idx.size(0)
    buffer.x[:num_tokens].copy_(x[0])
    buffer.x_sf[:num_tokens].copy_(x[1])
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)


def _all_rank_values(value: float, group: dist.ProcessGroup) -> List[float]:
    local = torch.tensor([value], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    return [item.item() for item in gathered]


def _time_one(
    fn: Callable[[], None], group: dist.ProcessGroup
) -> Tuple[float, List[float]]:
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    per_rank_us = _all_rank_values(start.elapsed_time(end) * 1e3, group)
    return max(per_rank_us), per_rank_us


def _assert_equal_on_all_ranks(
    condition: bool, message: str, group: dist.ProcessGroup
) -> None:
    local = torch.tensor([int(condition)], dtype=torch.int32, device="cuda")
    dist.all_reduce(local, op=dist.ReduceOp.MIN, group=group)
    assert local.item() == 1, message


def _cross_precision_metrics(
    official: torch.Tensor, candidate: torch.Tensor, group: dist.ProcessGroup
) -> Dict[str, float]:
    official_f = official.float()
    candidate_f = candidate.float()
    diff_sq = torch.sum((official_f - candidate_f) ** 2)
    official_sq = torch.sum(official_f ** 2)
    candidate_sq = torch.sum(candidate_f ** 2)
    dot = torch.sum(official_f * candidate_f)
    packed = torch.stack([diff_sq, official_sq, candidate_sq, dot])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
    rel_l2 = torch.sqrt(packed[0] / packed[1].clamp_min(1e-24)).item()
    cosine = (packed[3] / torch.sqrt((packed[1] * packed[2]).clamp_min(1e-24))).item()
    return {"relative_l2": rel_l2, "cosine": cosine}


def _set_nvfp4_mode(mode: int) -> None:
    assert mode in (0, 4)
    os.environ[EPOCH_ENV] = "1"
    os.environ[MODE_ENV] = str(mode)


def _run_rank(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    official_deep_gemm = _load_clean_official(args.official_repo)
    assert world_size == args.num_processes == 8, "This evaluator requires EP8"
    assert args.num_experts % world_size == 0
    candidate_mma_type = (
        "nvfp4xnvfp4" if args.candidate_contract == "nvfp4" else "mxfp4xmxfp4"
    )
    candidate_api = (
        deep_gemm.nvfp4_mega_moe
        if args.candidate_contract == "nvfp4"
        else deep_gemm.mxfp4_mega_moe
    )
    candidate_quantize = (
        per_token_cast_to_nvfp4
        if args.candidate_contract == "nvfp4"
        else per_token_cast_to_mxfp4
    )
    assert not args.compare_grouped_sf_reuse or args.candidate_contract == "nvfp4"
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    num_local_experts = args.num_experts // world_size
    x_bf16 = torch.randn(
        (args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda"
    )
    l1_bf16 = torch.randn(
        (num_local_experts, args.intermediate_hidden * 2, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2_bf16 = torch.randn(
        (num_local_experts, args.hidden, args.intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    scores = torch.randn(
        (args.num_tokens, args.num_experts), dtype=torch.float, device="cuda"
    )
    topk_weights, topk_idx = torch.topk(
        scores, args.num_topk, dim=-1, largest=True, sorted=False
    )
    if args.masked_route_fraction:
        mask = torch.rand(topk_idx.shape, dtype=torch.float, device="cuda") < args.masked_route_fraction
        topk_idx.masked_fill_(mask, -1)
        topk_weights.masked_fill_(mask, 0.0)

    x_fp8 = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    x_candidate = candidate_quantize(x_bf16)
    official_weights = official_deep_gemm.transform_weights_for_mega_moe(
        _cast_weights_to_fp4(l1_bf16, official_deep_gemm),
        _cast_weights_to_fp4(l2_bf16, official_deep_gemm),
    )
    candidate_weights = deep_gemm.transform_weights_for_mega_moe(
        _cast_weights_to_packed_fp4(l1_bf16, args.candidate_contract),
        _cast_weights_to_packed_fp4(l2_bf16, args.candidate_contract),
    )
    del l1_bf16, l2_bf16, scores

    buffer_args = dict(
        group=group,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.num_tokens,
        num_topk=args.num_topk,
        hidden=args.hidden,
        intermediate_hidden=args.intermediate_hidden,
    )
    official_buffer = official_deep_gemm.get_symm_buffer_for_mega_moe(
        **buffer_args, mma_type="fp8xfp4"
    )
    mode0_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        **buffer_args, mma_type=candidate_mma_type
    )
    mode4_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        **buffer_args, mma_type=candidate_mma_type
    )
    _copy_routes_and_inputs(official_buffer, x_fp8, topk_idx, topk_weights)
    _copy_routes_and_inputs(mode0_buffer, x_candidate, topk_idx, topk_weights)
    _copy_routes_and_inputs(mode4_buffer, x_candidate, topk_idx, topk_weights)
    torch.cuda.synchronize()

    official_y = torch.empty(
        (args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda"
    )
    mode0_y = torch.empty_like(official_y)
    mode4_y = torch.empty_like(official_y)

    def run_official(stats=None) -> None:
        official_deep_gemm.fp8_fp4_mega_moe(
            y=official_y,
            l1_weights=official_weights[0],
            l2_weights=official_weights[1],
            sym_buffer=official_buffer,
            cumulative_local_expert_recv_stats=stats,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )

    def run_mode0(stats=None) -> None:
        _set_nvfp4_mode(4 if args.compare_grouped_sf_reuse else 0)
        if args.compare_grouped_sf_reuse:
            os.environ[GROUPED_SF_ENV] = "0"
        candidate_api(
            y=mode0_y,
            l1_weights=candidate_weights[0],
            l2_weights=candidate_weights[1],
            sym_buffer=mode0_buffer,
            cumulative_local_expert_recv_stats=stats,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )

    def run_mode4(stats=None, kernel_profile=None) -> None:
        _set_nvfp4_mode(4)
        if args.compare_grouped_sf_reuse:
            os.environ[GROUPED_SF_ENV] = "1"
        candidate_api(
            y=mode4_y,
            l1_weights=candidate_weights[0],
            l2_weights=candidate_weights[1],
            sym_buffer=mode4_buffer,
            cumulative_local_expert_recv_stats=stats,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            kernel_profile=kernel_profile,
        )

    if args.ncu_profile_only:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        if args.ncu_profile_only == "official":
            run_official()
        else:
            run_mode4()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        dist.barrier(group=group)
        official_buffer.destroy()
        mode0_buffer.destroy()
        mode4_buffer.destroy()
        dist.destroy_process_group()
        return

    # Compile all three variants before correctness or timing begins.
    run_official()
    run_mode0()
    run_mode4()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    initial_stats = torch.arange(
        num_local_experts, dtype=torch.int32, device="cuda"
    ) + (rank + 1) * 1000
    official_stats = initial_stats.clone()
    mode0_stats = initial_stats.clone()
    mode4_stats = initial_stats.clone()
    run_official(official_stats)
    run_mode0(mode0_stats)
    run_mode4(mode4_stats)
    torch.cuda.synchronize()

    local_finite = (
        torch.isfinite(official_y).all().item()
        and torch.isfinite(mode0_y).all().item()
        and torch.isfinite(mode4_y).all().item()
    )
    _assert_equal_on_all_ranks(local_finite, "non-finite output", group)
    mode_outputs_bit_exact = torch.equal(mode0_y, mode4_y)
    if not getattr(args, "allow_mode_difference", False):
        _assert_equal_on_all_ranks(
            mode_outputs_bit_exact,
            "packed-FP4 Mode4 output differs from Mode0",
            group,
        )
    _assert_equal_on_all_ranks(
        torch.equal(official_stats, mode0_stats)
        and torch.equal(mode0_stats, mode4_stats),
        "route receive statistics differ between paths",
        group,
    )
    precision_metrics = _cross_precision_metrics(official_y, mode4_y, group)
    mode_precision_metrics = _cross_precision_metrics(mode0_y, mode4_y, group)

    stress = {"eager_iterations": 0, "cuda_graph_replays": 0}
    if args.stress_iterations:
        run_mode0()
        torch.cuda.synchronize()
        reference = mode0_y.clone()
        for iteration in range(args.stress_iterations):
            run_mode4()
            torch.cuda.synchronize()
            assert torch.equal(mode4_y, reference), f"eager Mode4 mismatch at {iteration}"
        _assert_equal_on_all_ranks(True, "eager Mode4 stress failed", group)
        stress["eager_iterations"] = args.stress_iterations

    if args.cuda_graph_replays:
        run_mode4()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_mode4()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        for replay in range(args.cuda_graph_replays):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(mode4_y, mode0_y), f"CUDA Graph Mode4 mismatch at {replay}"
        _assert_equal_on_all_ranks(True, "CUDA Graph Mode4 stress failed", group)
        stress["cuda_graph_replays"] = args.cuda_graph_replays

    timing = None
    if args.repeat:
        for warmup_idx in range(args.warmup):
            if warmup_idx % 2 == 0:
                _time_one(run_official, group)
                _time_one(run_mode4, group)
            else:
                _time_one(run_mode4, group)
                _time_one(run_official, group)

        samples = {
            "AB": {"official_critical_us": [], "candidate_critical_us": [],
                   "official_per_rank_us": [], "candidate_per_rank_us": []},
            "BA": {"official_critical_us": [], "candidate_critical_us": [],
                   "official_per_rank_us": [], "candidate_per_rank_us": []},
        }
        for repeat_idx in range(args.repeat):
            order = "AB" if repeat_idx % 2 == 0 else "BA"
            if order == "AB":
                official_us, official_ranks = _time_one(run_official, group)
                candidate_us, candidate_ranks = _time_one(run_mode4, group)
            else:
                candidate_us, candidate_ranks = _time_one(run_mode4, group)
                official_us, official_ranks = _time_one(run_official, group)
            samples[order]["official_critical_us"].append(official_us)
            samples[order]["candidate_critical_us"].append(candidate_us)
            samples[order]["official_per_rank_us"].append(official_ranks)
            samples[order]["candidate_per_rank_us"].append(candidate_ranks)

        timing = {"warmup_per_path": args.warmup, "repeat_per_path": args.repeat,
                  "orders": samples}
        for order, order_samples in samples.items():
            official_summary = _summary(order_samples["official_critical_us"])
            candidate_summary = _summary(order_samples["candidate_critical_us"])
            timing[order] = {
                "official": official_summary,
                "candidate": candidate_summary,
                "p50_improvement_pct": _relative_improvement(
                    official_summary["p50_us"], candidate_summary["p50_us"]
                ),
                "mean_improvement_pct": _relative_improvement(
                    official_summary["mean_us"], candidate_summary["mean_us"]
                ),
            }

    mode0_mode4_timing = None
    if args.mode0_repeat:
        for warmup_idx in range(args.mode0_warmup):
            if warmup_idx % 2 == 0:
                _time_one(run_mode0, group)
                _time_one(run_mode4, group)
            else:
                _time_one(run_mode4, group)
                _time_one(run_mode0, group)

        mode_samples = {
            "AB": {"mode0_critical_us": [], "mode4_critical_us": []},
            "BA": {"mode0_critical_us": [], "mode4_critical_us": []},
        }
        for repeat_idx in range(args.mode0_repeat):
            order = "AB" if repeat_idx % 2 == 0 else "BA"
            if order == "AB":
                mode0_us, _ = _time_one(run_mode0, group)
                mode4_us, _ = _time_one(run_mode4, group)
            else:
                mode4_us, _ = _time_one(run_mode4, group)
                mode0_us, _ = _time_one(run_mode0, group)
            mode_samples[order]["mode0_critical_us"].append(mode0_us)
            mode_samples[order]["mode4_critical_us"].append(mode4_us)

        mode0_mode4_timing = {
            "warmup_per_path": args.mode0_warmup,
            "repeat_per_path": args.mode0_repeat,
            "orders": mode_samples,
        }
        for order, order_samples in mode_samples.items():
            mode0_summary = _summary(order_samples["mode0_critical_us"])
            mode4_summary = _summary(order_samples["mode4_critical_us"])
            mode0_mode4_timing[order] = {
                "mode0": mode0_summary,
                "mode4": mode4_summary,
                "p50_improvement_pct": _relative_improvement(
                    mode0_summary["p50_us"], mode4_summary["p50_us"]
                ),
                "mean_improvement_pct": _relative_improvement(
                    mode0_summary["mean_us"], mode4_summary["mean_us"]
                ),
            }

    kernel_profile_records = None
    if args.kernel_profile_repeats:
        layout = deep_gemm.get_mega_moe_kernel_profile_layout()
        profile = deep_gemm.allocate_mega_moe_kernel_profile()
        records = []
        for _ in range(args.kernel_profile_repeats):
            profile.zero_()
            dist.barrier(group=group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_mode4(kernel_profile=profile)
            end.record()
            end.synchronize()
            record = _parse_profile(profile, layout)
            record["cuda_event_us"] = start.elapsed_time(end) * 1000.0
            gathered = [None] * world_size
            dist.all_gather_object(gathered, record, group=group)
            records.append(gathered)
        kernel_profile_records = records

    if rank == 0:
        result = {
            "schema_version": 1,
            "baseline": {
                "name": "official_latest_megamoe_fp8xfp4",
                "repository": "https://github.com/deepseek-ai/DeepGEMM",
                "base_commit": OFFICIAL_BASE_COMMIT,
                "checkout": str(args.official_repo),
                "entrypoint": "deep_gemm.fp8_fp4_mega_moe",
                "loaded_as_independent_extension": True,
            },
            "candidate": {
                "name": f"youzjuer_megamoe_{args.candidate_contract}_mode4",
                "head_commit": _git_output("rev-parse", "HEAD"),
                "entrypoint": f"deep_gemm.{args.candidate_contract}_mega_moe",
                "quantization_contract": args.candidate_contract,
                "epoch_workspace": True,
                "dispatch_ready_mode": 4,
            },
            "shape": {
                "ep": world_size,
                "tokens_per_rank": args.num_tokens,
                "hidden": args.hidden,
                "intermediate_hidden": args.intermediate_hidden,
                "global_experts": args.num_experts,
                "local_experts": num_local_experts,
                "topk": args.num_topk,
                "shared_experts": 0,
            },
            "seed": args.seed,
            "masked_route_fraction": args.masked_route_fraction,
            "correctness": {
                "all_outputs_finite": True,
                "mode0_mode4_bit_exact": mode_outputs_bit_exact,
                "mode0_mode4_precision": mode_precision_metrics,
                "receive_stats_exact": True,
                "cross_precision": precision_metrics,
            },
            "stress": stress,
            "timing": timing,
            "mode0_mode4_timing": mode0_mode4_timing,
            "kernel_profile": kernel_profile_records,
        }
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({
            "output": str(output),
            "shape": result["shape"],
            "correctness": result["correctness"],
            "stress": stress,
            "AB": None if timing is None else timing["AB"],
            "BA": None if timing is None else timing["BA"],
            "mode0_mode4": mode0_mode4_timing,
        }, indent=2), flush=True)

    dist.barrier(group=group)
    official_buffer.destroy()
    mode0_buffer.destroy()
    mode4_buffer.destroy()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument("--num-topk", type=int, default=10)
    parser.add_argument(
        "--candidate-contract", choices=("nvfp4", "mxfp4"), default="mxfp4"
    )
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--masked-route-fraction", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--stress-iterations", type=int, default=0)
    parser.add_argument("--cuda-graph-replays", type=int, default=0)
    parser.add_argument("--mode0-warmup", type=int, default=0)
    parser.add_argument("--mode0-repeat", type=int, default=0)
    parser.add_argument("--compare-grouped-sf-reuse", action="store_true")
    parser.add_argument("--kernel-profile-repeats", type=int, default=0)
    parser.add_argument("--ncu-profile-only", choices=("official", "candidate"))
    parser.add_argument("--external-rank", type=int)
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()
    assert 0.0 <= args.masked_route_fraction <= 1.0
    assert args.warmup >= 0 and args.repeat >= 0
    assert args.stress_iterations >= 0 and args.cuda_graph_replays >= 0
    assert args.mode0_warmup >= 0 and args.mode0_repeat >= 0
    assert args.kernel_profile_repeats >= 0
    official_head = subprocess.check_output(
        ["git", "-C", str(args.official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert official_head == OFFICIAL_BASE_COMMIT, (
        f"official checkout is {official_head}, expected {OFFICIAL_BASE_COMMIT}"
    )
    if args.external_rank is None:
        torch.multiprocessing.spawn(
            _run_rank, args=(args.num_processes, args), nprocs=args.num_processes
        )
    else:
        assert 0 <= args.external_rank < args.num_processes
        _run_rank(args.external_rank, args.num_processes, args)


if __name__ == "__main__":
    main()
