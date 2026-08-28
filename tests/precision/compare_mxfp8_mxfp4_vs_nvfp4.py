"""Compare MXFP8xMXFP4 and NVFP4xNVFP4 quantization accuracy.

The experiment isolates the numerical formats from communication, routing,
kernel scheduling, and fast-math approximations.  Both candidates start from
the same BF16-rounded tensors.  Quantized operands are dequantized exactly as
their block-scale contracts specify, then multiplied with FP32 accumulation.

Three MegaMoE-shaped calculations are reported:

* GEMM1: [M, H] @ [2I, H].T
* GEMM2: [M, I] @ [H, I].T
* PIPELINE: GEMM1 -> SwiGLU -> requantize -> GEMM2

This is a format-accuracy oracle, not a fused-kernel correctness test.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

MX_SCHEME = "mxfp8_x_mxfp4"
NV_SCHEME = "nvfp4_x_nvfp4"
SCENARIOS = (
    "normal",
    "activation_outlier",
    "weight_outlier",
    "both_outlier",
    "heavy_tail",
    "block_skew",
)


@dataclass(frozen=True)
class ErrorMetrics:
    rel_l2: float
    cosine: float
    snr_db: float
    rmse: float
    mae: float
    bias: float
    p50_abs: float
    p90_abs: float
    p99_abs: float
    max_abs: float
    reference_rms: float
    candidate_rms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--m-values", type=int, nargs="+", default=(1, 8, 32, 160))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--outlier-rate", type=float, default=1.0 / 64.0)
    parser.add_argument("--outlier-factor", type=float, default=32.0)
    parser.add_argument(
        "--source-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Round generated tensors to this source dtype before quantization",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--validate-helpers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check the standalone format model against deep_gemm/utils/math.py",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def round_source(x: torch.Tensor, source_dtype: str) -> torch.Tensor:
    if source_dtype == "bf16":
        return x.to(torch.bfloat16).float()
    return x.float()


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def randn(shape: Sequence[int], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    generator_device = str(generator.device)
    target_device = device if generator_device.startswith(device.type) else torch.device("cpu")
    result = torch.randn(tuple(shape), device=target_device, dtype=torch.float32, generator=generator)
    return result if result.device == device else result.to(device)


def rand(shape: Sequence[int], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    generator_device = str(generator.device)
    target_device = device if generator_device.startswith(device.type) else torch.device("cpu")
    result = torch.rand(tuple(shape), device=target_device, dtype=torch.float32, generator=generator)
    return result if result.device == device else result.to(device)


def make_values(
    shape: Sequence[int],
    kind: str,
    device: torch.device,
    generator: torch.Generator,
    outlier_rate: float,
    outlier_factor: float,
) -> torch.Tensor:
    if kind == "normal":
        return randn(shape, device, generator)
    if kind == "heavy_tail":
        numerator = randn(shape, device, generator)
        chi_square = sum(randn(shape, device, generator).square() for _ in range(3))
        # Student-t(df=3) has variance 3; divide by sqrt(3) for unit variance.
        return (numerator / torch.sqrt(chi_square.clamp_min(1e-12) / 3.0)) / math.sqrt(3.0)
    if kind == "outlier":
        values = randn(shape, device, generator)
        mask = rand(shape, device, generator) < outlier_rate
        return torch.where(mask, values * outlier_factor, values)
    if kind == "block_skew":
        if shape[-1] % 32 != 0:
            raise ValueError(f"block_skew expects K divisible by 32, got {tuple(shape)}")
        values = randn(shape, device, generator)
        pattern = torch.ones((shape[-1] // 32, 32), device=device, dtype=torch.float32)
        pattern[0::2, :16] = outlier_factor
        pattern[1::2, 16:] = outlier_factor
        return (values.view(*shape[:-1], shape[-1] // 32, 32) * pattern).view(*shape)
    raise ValueError(f"Unsupported value kind: {kind}")


def scenario_kinds(scenario: str) -> tuple[str, str]:
    if scenario == "normal":
        return "normal", "normal"
    if scenario == "activation_outlier":
        return "outlier", "normal"
    if scenario == "weight_outlier":
        return "normal", "outlier"
    if scenario == "both_outlier":
        return "outlier", "outlier"
    if scenario == "heavy_tail":
        return "heavy_tail", "heavy_tail"
    if scenario == "block_skew":
        return "block_skew", "block_skew"
    raise ValueError(f"Unsupported scenario: {scenario}")


def normalize_rows(x: torch.Tensor, target_rms: float) -> torch.Tensor:
    row_rms = x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    return x * (target_rms / row_rms)


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values up to the UE8M0 power-of-two scale."""
    bits = x.abs().float().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exponent.clamp(1, 254) << 23).view(torch.float32)


def fp4_e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Round to {0, .5, 1, 1.5, 2, 3, 4, 6} with sign."""
    magnitude = x.abs()
    index = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        index += (magnitude > boundary).to(torch.uint8)
    levels = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=x.device,
        dtype=torch.float32,
    )
    rounded = levels[index.to(torch.long)]
    return torch.where(x < 0, -rounded, rounded)


def quantize_mxfp8_dequantize(x: torch.Tensor) -> torch.Tensor:
    """MXFP8 E4M3 payload with group-32 UE8M0 scales."""
    if x.ndim != 2 or x.shape[1] % 32 != 0:
        raise ValueError(f"MXFP8 expects 2D K divisible by 32, got {tuple(x.shape)}")
    rows, cols = x.shape
    grouped = x.view(rows, cols // 32, 32)
    scales = ceil_to_ue8m0(grouped.abs().amax(dim=2).clamp_min(1e-4) / 448.0)
    payload = (grouped * scales.reciprocal().unsqueeze(2)).to(torch.float8_e4m3fn).float()
    return (payload * scales.unsqueeze(2)).view_as(x)


def quantize_mxfp4_dequantize(x: torch.Tensor) -> torch.Tensor:
    """MXFP4 E2M1 payload with group-32 UE8M0 scales."""
    if x.ndim != 2 or x.shape[1] % 32 != 0:
        raise ValueError(f"MXFP4 expects 2D K divisible by 32, got {tuple(x.shape)}")
    rows, cols = x.shape
    grouped = x.view(rows, cols // 32, 32)
    scales = ceil_to_ue8m0(grouped.abs().amax(dim=2).clamp_min(1e-4) / 6.0)
    payload = fp4_e2m1_round(grouped * scales.reciprocal().unsqueeze(2))
    return (payload * scales.unsqueeze(2)).view_as(x)


def quantize_nvfp4_dequantize(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 E2M1 payload with group-16 E4M3 scales and global scale 1."""
    if x.ndim != 2 or x.shape[1] % 16 != 0:
        raise ValueError(f"NVFP4 expects 2D K divisible by 16, got {tuple(x.shape)}")
    rows, cols = x.shape
    grouped = x.view(rows, cols // 16, 16)
    scales = (grouped.abs().amax(dim=2) / 6.0).clamp_min(2.0**-9)
    scales = scales.to(torch.float8_e4m3fn).float()
    payload = fp4_e2m1_round(grouped * scales.reciprocal().unsqueeze(2))
    return (payload * scales.unsqueeze(2)).view_as(x)


def validate_against_deepgemm_helpers(device: torch.device) -> dict[str, float]:
    """Prove that the standalone dequantized values match DeepGEMM helpers."""
    math_path = Path(__file__).resolve().parents[2] / "deep_gemm" / "utils" / "math.py"
    spec = importlib.util.spec_from_file_location("deepgemm_precision_math", math_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {math_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = make_generator(device, 20260819)
    sample = round_source(
        normalize_rows(
            make_values((7, 128), "heavy_tail", device, generator, 1.0 / 64.0, 32.0),
            1.0,
        ),
        "bf16",
    )
    fp8, fp8_sf = module.per_token_cast_to_fp8(
        sample, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    fp8_reference = module.cast_back_from_fp8(
        fp8, fp8_sf, gran_k=32, use_packed_ue8m0=True
    )
    mxfp4, mxfp4_sf = module.per_token_cast_to_mxfp4(sample)
    mxfp4_reference = module.cast_back_from_mxfp4(mxfp4, mxfp4_sf)
    nvfp4, nvfp4_sf = module.per_token_cast_to_nvfp4(sample)
    nvfp4_reference = module.cast_back_from_nvfp4(nvfp4, nvfp4_sf)

    comparisons = {
        "mxfp8": (quantize_mxfp8_dequantize(sample), fp8_reference),
        "mxfp4": (quantize_mxfp4_dequantize(sample), mxfp4_reference),
        "nvfp4": (quantize_nvfp4_dequantize(sample), nvfp4_reference),
    }
    max_abs_differences = {
        name: (candidate - reference).abs().max().item()
        for name, (candidate, reference) in comparisons.items()
    }
    mismatches = {
        name: int((candidate != reference).sum().item())
        for name, (candidate, reference) in comparisons.items()
    }
    if any(mismatches.values()):
        raise AssertionError(
            "Standalone quantization model differs from DeepGEMM helpers: "
            f"mismatches={mismatches}, max_abs={max_abs_differences}"
        )
    return max_abs_differences


def error_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> ErrorMetrics:
    candidate = candidate.float().reshape(-1)
    reference = reference.float().reshape(-1)
    error = candidate - reference

    # Output tensors are small enough that FP64 reductions are affordable and
    # avoid a metric artifact being mistaken for a format difference.
    candidate64 = candidate.double()
    reference64 = reference.double()
    error64 = error.double()
    reference_norm = torch.linalg.vector_norm(reference64)
    candidate_norm = torch.linalg.vector_norm(candidate64)
    error_norm = torch.linalg.vector_norm(error64)
    denominator = max(reference_norm.item(), sys.float_info.min)
    rel_l2 = error_norm.item() / denominator
    cosine_denominator = max(reference_norm.item() * candidate_norm.item(), sys.float_info.min)
    cosine = torch.dot(reference64, candidate64).item() / cosine_denominator
    snr_db = 999.0 if error_norm.item() == 0 else 20.0 * math.log10(denominator / error_norm.item())

    absolute_error = error.abs()
    quantiles = torch.quantile(
        absolute_error,
        torch.tensor((0.5, 0.9, 0.99), device=absolute_error.device),
    ).cpu().tolist()
    count = max(reference.numel(), 1)
    return ErrorMetrics(
        rel_l2=rel_l2,
        cosine=cosine,
        snr_db=snr_db,
        rmse=math.sqrt(error64.square().sum().item() / count),
        mae=error64.abs().mean().item(),
        bias=error64.mean().item(),
        p50_abs=float(quantiles[0]),
        p90_abs=float(quantiles[1]),
        p99_abs=float(quantiles[2]),
        max_abs=absolute_error.max().item(),
        reference_rms=math.sqrt(reference64.square().sum().item() / count),
        candidate_rms=math.sqrt(candidate64.square().sum().item() / count),
    )


def metric_pair(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, dict[str, float]]:
    return {
        "fp32": asdict(error_metrics(candidate, reference)),
        "bf16_output": asdict(
            error_metrics(candidate.to(torch.bfloat16).float(), reference.to(torch.bfloat16).float())
        ),
    }


def swiglu(x: torch.Tensor, intermediate: int) -> torch.Tensor:
    gate, up = x[:, :intermediate], x[:, intermediate:]
    return torch.nn.functional.silu(gate) * up


def add_records(
    records: list[dict],
    mode: str,
    m_values: Iterable[int],
    n: int,
    k: int,
    scenario: str,
    seed: int,
    reference: torch.Tensor,
    mx_candidate: torch.Tensor,
    nv_candidate: torch.Tensor,
) -> None:
    for m in m_values:
        reference_slice = reference[:m]
        candidate_slices = {
            MX_SCHEME: mx_candidate[:m],
            NV_SCHEME: nv_candidate[:m],
        }
        per_scheme = {
            scheme: metric_pair(candidate, reference_slice)
            for scheme, candidate in candidate_slices.items()
        }
        mx_rel_l2 = per_scheme[MX_SCHEME]["bf16_output"]["rel_l2"]
        nv_rel_l2 = per_scheme[NV_SCHEME]["bf16_output"]["rel_l2"]
        records.append(
            {
                "mode": mode,
                "shape": {"m": m, "n": n, "k": k},
                "scenario": scenario,
                "seed": seed,
                "schemes": per_scheme,
                "winner_by_bf16_rel_l2": MX_SCHEME if mx_rel_l2 < nv_rel_l2 else NV_SCHEME,
                "nv_over_mx_bf16_rel_l2": nv_rel_l2 / max(mx_rel_l2, sys.float_info.min),
            }
        )


def run_case(
    device: torch.device,
    hidden: int,
    intermediate: int,
    m_values: Sequence[int],
    scenario: str,
    seed: int,
    source_dtype: str,
    outlier_rate: float,
    outlier_factor: float,
) -> list[dict]:
    activation_kind, weight_kind = scenario_kinds(scenario)
    max_m = max(m_values)
    generator = make_generator(device, seed)

    make: Callable[[Sequence[int], str, float], torch.Tensor] = (
        lambda shape, kind, target_rms: round_source(
            normalize_rows(
                make_values(shape, kind, device, generator, outlier_rate, outlier_factor),
                target_rms,
            ),
            source_dtype,
        )
    )
    x1 = make((max_m, hidden), activation_kind, 1.0)
    x2 = make((max_m, intermediate), activation_kind, 1.0)
    w1 = make((2 * intermediate, hidden), weight_kind, 1.0 / math.sqrt(hidden))
    w2 = make((hidden, intermediate), weight_kind, 1.0 / math.sqrt(intermediate))
    route_weight = torch.softmax(randn((max_m, 10), device, generator), dim=-1)[:, :1]

    # Quantize each operand once and reuse it across M slices.  Quantization is
    # row-local, so a prefix has the same result as quantizing that prefix alone.
    x1_mx = quantize_mxfp8_dequantize(x1)
    x2_mx = quantize_mxfp8_dequantize(x2)
    w1_mx = quantize_mxfp4_dequantize(w1)
    w2_mx = quantize_mxfp4_dequantize(w2)

    x1_nv = quantize_nvfp4_dequantize(x1)
    x2_nv = quantize_nvfp4_dequantize(x2)
    w1_nv = quantize_nvfp4_dequantize(w1)
    w2_nv = quantize_nvfp4_dequantize(w2)

    reference_l1 = x1 @ w1.t()
    mx_l1 = x1_mx @ w1_mx.t()
    nv_l1 = x1_nv @ w1_nv.t()

    reference_l2 = x2 @ w2.t()
    mx_l2 = x2_mx @ w2_mx.t()
    nv_l2 = x2_nv @ w2_nv.t()

    # MegaMoE stores the L1 accumulator through a BF16 boundary before its
    # SwiGLU epilogue, then applies the route weight before L2 requantization.
    reference_activation = swiglu(reference_l1.to(torch.bfloat16).float(), intermediate) * route_weight
    mx_activation = swiglu(mx_l1.to(torch.bfloat16).float(), intermediate) * route_weight
    nv_activation = swiglu(nv_l1.to(torch.bfloat16).float(), intermediate) * route_weight
    mx_activation = quantize_mxfp8_dequantize(mx_activation)
    nv_activation = quantize_nvfp4_dequantize(nv_activation)
    reference_pipeline = reference_activation.to(torch.bfloat16).float() @ w2.t()
    mx_pipeline = mx_activation @ w2_mx.t()
    nv_pipeline = nv_activation @ w2_nv.t()

    records: list[dict] = []
    add_records(
        records, "gemm1", m_values, 2 * intermediate, hidden, scenario, seed,
        reference_l1, mx_l1, nv_l1,
    )
    add_records(
        records, "gemm2", m_values, hidden, intermediate, scenario, seed,
        reference_l2, mx_l2, nv_l2,
    )
    add_records(
        records, "swiglu_pipeline", m_values, hidden, intermediate, scenario, seed,
        reference_pipeline, mx_pipeline, nv_pipeline,
    )
    return records


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def summarize(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        shape = record["shape"]
        key = (record["mode"], shape["m"], shape["n"], shape["k"], record["scenario"])
        groups.setdefault(key, []).append(record)

    summary = []
    for key, group in sorted(groups.items()):
        mode, m, n, k, scenario = key
        mx_rel = [r["schemes"][MX_SCHEME]["bf16_output"]["rel_l2"] for r in group]
        nv_rel = [r["schemes"][NV_SCHEME]["bf16_output"]["rel_l2"] for r in group]
        mx_cos = [r["schemes"][MX_SCHEME]["bf16_output"]["cosine"] for r in group]
        nv_cos = [r["schemes"][NV_SCHEME]["bf16_output"]["cosine"] for r in group]
        ratios = [r["nv_over_mx_bf16_rel_l2"] for r in group]
        summary.append(
            {
                "mode": mode,
                "shape": {"m": m, "n": n, "k": k},
                "scenario": scenario,
                "num_seeds": len(group),
                "median_bf16_rel_l2": {MX_SCHEME: median(mx_rel), NV_SCHEME: median(nv_rel)},
                "median_bf16_cosine": {MX_SCHEME: median(mx_cos), NV_SCHEME: median(nv_cos)},
                "median_nv_over_mx_rel_l2": median(ratios),
                "wins": {
                    MX_SCHEME: sum(r["winner_by_bf16_rel_l2"] == MX_SCHEME for r in group),
                    NV_SCHEME: sum(r["winner_by_bf16_rel_l2"] == NV_SCHEME for r in group),
                },
            }
        )
    return summary


def write_csv(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "mode", "m", "n", "k", "scenario", "seed", "scheme",
        "output_cast", "rel_l2", "cosine", "snr_db", "rmse", "mae",
        "bias", "p50_abs", "p90_abs", "p99_abs", "max_abs",
        "reference_rms", "candidate_rms",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for scheme, output_casts in record["schemes"].items():
                for output_cast, metrics in output_casts.items():
                    writer.writerow(
                        {
                            "mode": record["mode"],
                            **record["shape"],
                            "scenario": record["scenario"],
                            "seed": record["seed"],
                            "scheme": scheme,
                            "output_cast": output_cast,
                            **metrics,
                        }
                    )


def print_summary(summary: Sequence[dict]) -> None:
    print(
        "mode              M  scenario               MX rel-L2   NV rel-L2   "
        "NV/MX   winner"
    )
    for row in summary:
        mx = row["median_bf16_rel_l2"][MX_SCHEME]
        nv = row["median_bf16_rel_l2"][NV_SCHEME]
        wins = row["wins"]
        winner = MX_SCHEME if wins[MX_SCHEME] > wins[NV_SCHEME] else NV_SCHEME
        print(
            f"{row['mode']:<17} {row['shape']['m']:>3}  {row['scenario']:<21} "
            f"{mx:>10.6f}  {nv:>10.6f}  {row['median_nv_over_mx_rel_l2']:>6.3f}  {winner}"
        )


def main() -> None:
    args = parse_args()
    if args.hidden % 32 != 0 or args.intermediate % 32 != 0:
        raise ValueError("hidden and intermediate must both be divisible by 32")
    if min(args.m_values) <= 0:
        raise ValueError("all M values must be positive")
    if not 0.0 <= args.outlier_rate <= 1.0:
        raise ValueError("outlier-rate must be between 0 and 1")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    helper_validation = (
        validate_against_deepgemm_helpers(device) if args.validate_helpers else None
    )
    started = time.time()
    records: list[dict] = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            print(f"running scenario={scenario} seed={seed}", flush=True)
            records.extend(
                run_case(
                    device=device,
                    hidden=args.hidden,
                    intermediate=args.intermediate,
                    m_values=tuple(sorted(set(args.m_values))),
                    scenario=scenario,
                    seed=seed,
                    source_dtype=args.source_dtype,
                    outlier_rate=args.outlier_rate,
                    outlier_factor=args.outlier_factor,
                )
            )
            synchronize(device)

    summary = summarize(records)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "deepgemm_git_head": git_head(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        environment.update(
            {
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            }
        )
    payload = {
        "schema": "megamoe-format-accuracy-v1",
        "method": (
            "same BF16-rounded sources; exact block-scale quantize/dequantize; "
            "FP32 matmul; metrics before and after BF16 output cast"
        ),
        "format_contracts": {
            MX_SCHEME: {
                "activation": "E4M3 payload, UE8M0 scale, group 32",
                "weight": "E2M1 payload, UE8M0 scale, group 32",
            },
            NV_SCHEME: {
                "activation": "E2M1 payload, E4M3 scale, group 16, global scale 1",
                "weight": "E2M1 payload, E4M3 scale, group 16, global scale 1",
            },
        },
        "scope_limit": (
            "Format-level oracle only; excludes routing, EP communication, fast-math, "
            "tensor-core accumulation details, and model-level NVFP4 global scales."
        ),
        "config": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "m_values": sorted(set(args.m_values)),
            "seeds": args.seeds,
            "scenarios": args.scenarios,
            "source_dtype": args.source_dtype,
            "outlier_rate": args.outlier_rate,
            "outlier_factor": args.outlier_factor,
            "row_rms_normalization": {
                "activation": 1.0,
                "weight": "1/sqrt(K)",
            },
            "route_weight": "first component of a deterministic topk=10 softmax sample",
        },
        "environment": environment,
        "deepgemm_helper_validation_max_abs": helper_validation,
        "elapsed_seconds": time.time() - started,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.csv is not None:
        write_csv(args.csv, records)
    print_summary(summary)
    print(f"wrote {args.output}")
    if args.csv is not None:
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
