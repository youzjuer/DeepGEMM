"""Automatic persistent-replica scheduling for NVFP4 MegaMoE.

The scheduler deliberately makes decisions at window boundaries.  A fixed
number of steps in each window sample logical expert routes into device rank
load counters.  At the end of a window those counters are reduced across the
EP group, and that completed window selects the baseline or persistent-replica
path for the next window.  This keeps all ranks on the same collective path
and amortizes statistics and host synchronization over multiple decode steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.distributed as dist

from .. import _C


@dataclass(frozen=True)
class PersistentReplicaGuardMetrics:
    """Host snapshot produced once per completed route window."""

    window_routes: int
    rank_loads: Tuple[int, ...]
    predicted_replica_rank_loads: Tuple[float, ...]
    rank_max_mean: float
    predicted_relief: float
    replicated_route_fraction: float
    use_replica_next_window: bool


@dataclass(frozen=True)
class PersistentReplicaStepResult:
    """Path selected for one launch and an optional window transition."""

    used_replica: bool
    window_complete: bool
    use_replica_next_window: bool
    metrics: Optional[PersistentReplicaGuardMetrics]


class PersistentReplicaRouteGuard:
    """GPU route histogram plus a previous-window skew guard.

    ``expert_routing_map`` has the same ``(choices, counts)`` contract as
    :func:`nvfp4_mega_moe`.  Route collection is CUDA Graph capture-safe.  The
    window finalization performs one integer all-reduce across ``group`` and a
    synchronous host snapshot; callers should therefore choose a window size
    that amortizes this boundary operation.
    """

    def __init__(
        self,
        *,
        num_logical_experts: int,
        master_experts_per_rank: int,
        physical_experts_per_rank: int,
        expert_routing_map: Tuple[torch.Tensor, torch.Tensor],
        group: Optional[Any] = None,
        window_size: int = 32,
        samples_per_window: int = 4,
        enter_threshold: float = 1.25,
        exit_threshold: float = 1.15,
        min_predicted_relief: float = 0.01,
        min_tokens_per_rank: int = 0,
        initial_use_replica: bool = False,
    ) -> None:
        choices, counts = expert_routing_map
        if num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be positive")
        if master_experts_per_rank <= 0 or physical_experts_per_rank <= 0:
            raise ValueError("expert counts per rank must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 1 <= samples_per_window <= window_size:
            raise ValueError("samples_per_window must be in [1, window_size]")
        if not 1.0 <= exit_threshold <= enter_threshold:
            raise ValueError(
                "thresholds must satisfy 1 <= exit_threshold <= enter_threshold"
            )
        if not 0.0 <= min_predicted_relief < 1.0:
            raise ValueError("min_predicted_relief must be in [0, 1)")
        if min_tokens_per_rank < 0:
            raise ValueError("min_tokens_per_rank must be nonnegative")
        if choices.dtype != torch.int32 or counts.dtype != torch.int32:
            raise TypeError("routing choices and counts must have dtype int32")
        if not choices.is_cuda or not counts.is_cuda:
            raise ValueError("routing choices and counts must be CUDA tensors")
        if not choices.is_contiguous() or not counts.is_contiguous():
            raise ValueError("routing choices and counts must be contiguous")
        if choices.ndim != 2 or counts.ndim != 1:
            raise ValueError("routing choices must be 2D and counts must be 1D")
        if (
            choices.shape[0] != num_logical_experts
            or counts.numel() != num_logical_experts
        ):
            raise ValueError("routing map does not match num_logical_experts")
        if choices.device != counts.device:
            raise ValueError("routing choices and counts must share a CUDA device")

        inferred_ranks = num_logical_experts // master_experts_per_rank
        if inferred_ranks * master_experts_per_rank != num_logical_experts:
            raise ValueError(
                "num_logical_experts must be divisible by master_experts_per_rank"
            )
        if dist.is_available() and dist.is_initialized():
            num_ranks = dist.get_world_size(group)
            if num_ranks != inferred_ranks:
                raise ValueError(
                    f"EP group has {num_ranks} ranks, expected {inferred_ranks}"
                )
        else:
            if inferred_ranks != 1:
                raise RuntimeError("a distributed group is required for EP > 1")
            num_ranks = 1

        self.group = group
        self.num_logical_experts = num_logical_experts
        self.master_experts_per_rank = master_experts_per_rank
        self.physical_experts_per_rank = physical_experts_per_rank
        self.num_ranks = num_ranks
        self.window_size = window_size
        self.samples_per_window = samples_per_window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_predicted_relief = min_predicted_relief
        self.min_tokens_per_rank = min_tokens_per_rank
        self._use_replica = bool(initial_use_replica)
        self._tokens_per_rank: Optional[int] = None
        self._rank_idx = dist.get_rank(group) if num_ranks > 1 else 0
        self._routing_choices = choices
        self._routing_counts = counts
        # [baseline rank loads, predicted replica rank loads, replica routes].
        self._route_loads = torch.zeros(
            2 * num_ranks + 1, dtype=torch.int32, device=choices.device
        )

    @property
    def use_replica(self) -> bool:
        return self._use_replica

    @property
    def route_loads(self) -> torch.Tensor:
        return self._route_loads

    def observe(self, logical_topk_idx: torch.Tensor) -> None:
        """Accumulate one step of logical routes on the current CUDA stream."""
        if logical_topk_idx.dtype != torch.int64:
            raise TypeError("logical_topk_idx must have dtype int64")
        if not logical_topk_idx.is_cuda or not logical_topk_idx.is_contiguous():
            raise ValueError("logical_topk_idx must be a contiguous CUDA tensor")
        if logical_topk_idx.device != self._route_loads.device:
            raise ValueError("logical_topk_idx is on the wrong CUDA device")
        if logical_topk_idx.ndim != 2:
            raise ValueError("logical_topk_idx must have shape [tokens, top_k]")
        tokens_per_rank = logical_topk_idx.shape[0]
        if self._tokens_per_rank is None:
            self._tokens_per_rank = tokens_per_rank
        elif self._tokens_per_rank != tokens_per_rank:
            raise ValueError(
                "token count changed; use one scheduler per CUDA Graph shape"
            )
        _C.accumulate_route_loads(
            logical_topk_idx,
            self._routing_choices,
            self._routing_counts,
            self._route_loads,
            self._rank_idx,
            self.master_experts_per_rank,
            self.physical_experts_per_rank,
        )

    def clear_counts(self) -> None:
        """Clear the current device window without changing the selected path."""
        self._route_loads.zero_()

    def finalize_window(self) -> PersistentReplicaGuardMetrics:
        """Collect the completed window and choose the next window's path."""
        if self._tokens_per_rank is None:
            raise RuntimeError("observe must be called before finalize_window")
        if self.num_ranks > 1:
            dist.all_reduce(self._route_loads, op=dist.ReduceOp.SUM, group=self.group)
        snapshot = self._route_loads.cpu()
        self._route_loads.zero_()

        offset = self.num_ranks
        host_rank_loads = tuple(int(v) for v in snapshot[:offset].tolist())
        host_replica_loads_int = tuple(
            int(v) for v in snapshot[offset : 2 * offset].tolist()
        )
        host_replica_loads = tuple(float(v) for v in host_replica_loads_int)
        host_total_routes = sum(host_rank_loads)
        host_mean_load = host_total_routes / self.num_ranks
        host_baseline_max = max(host_rank_loads, default=0)
        host_replica_max = max(host_replica_loads_int, default=0)
        host_rank_max_mean = (
            host_baseline_max / max(host_mean_load, 1.0)
            if host_total_routes > 0
            else 1.0
        )
        host_predicted_relief = (
            max(host_baseline_max - host_replica_max, 0) / host_baseline_max
            if host_baseline_max > 0
            else 0.0
        )
        host_replicated_fraction = (
            int(snapshot[2 * offset].item()) / host_total_routes
            if host_total_routes > 0
            else 0.0
        )

        threshold = self.exit_threshold if self._use_replica else self.enter_threshold
        relief_threshold = self.min_predicted_relief * (
            0.5 if self._use_replica else 1.0
        )
        enough_tokens = self._tokens_per_rank >= self.min_tokens_per_rank
        self._use_replica = bool(
            host_total_routes > 0
            and enough_tokens
            and host_rank_max_mean >= threshold
            and host_predicted_relief >= relief_threshold
            and host_replicated_fraction > 0.0
        )

        return PersistentReplicaGuardMetrics(
            window_routes=host_total_routes,
            rank_loads=host_rank_loads,
            predicted_replica_rank_loads=host_replica_loads,
            rank_max_mean=host_rank_max_mean,
            predicted_relief=host_predicted_relief,
            replicated_route_fraction=host_replicated_fraction,
            use_replica_next_window=self._use_replica,
        )


class PersistentReplicaAutoScheduler:
    """Select between prewarmed baseline and replica MegaMoE paths.

    The two callables must use persistent tensors and write their result to a
    common output tensor.  They are invoked in the same order on every EP rank.
    ``logical_topk_idx`` must remain logical and must be refilled before every
    launch; the replica MegaMoE path may mutate only its copied symmetric-buffer
    scratch indices.
    """

    def __init__(
        self,
        *,
        guard: PersistentReplicaRouteGuard,
        logical_topk_idx: torch.Tensor,
        baseline_step: Callable[[], None],
        replica_step: Callable[[], None],
    ) -> None:
        self.guard = guard
        self.logical_topk_idx = logical_topk_idx
        self.baseline_step = baseline_step
        self.replica_step = replica_step
        self._baseline_graph: Optional[torch.cuda.CUDAGraph] = None
        self._replica_graph: Optional[torch.cuda.CUDAGraph] = None
        self._stats_graph: Optional[torch.cuda.CUDAGraph] = None
        self._steps_in_window = 0
        self._last_metrics: Optional[PersistentReplicaGuardMetrics] = None

    @property
    def captured(self) -> bool:
        return (
            self._baseline_graph is not None
            and self._replica_graph is not None
            and self._stats_graph is not None
        )

    @property
    def use_replica(self) -> bool:
        return self.guard.use_replica

    @property
    def last_metrics(self) -> Optional[PersistentReplicaGuardMetrics]:
        return self._last_metrics

    def _rank_sync(self) -> None:
        torch.cuda.synchronize(self.logical_topk_idx.device)
        if self.guard.num_ranks > 1:
            dist.barrier(group=self.guard.group)

    def warmup(self, iterations: int = 2) -> None:
        """Compile and warm both paths in a rank-consistent order."""
        if iterations <= 0:
            raise ValueError("warmup iterations must be positive")
        for _ in range(iterations):
            self.baseline_step()
            self.replica_step()
        self._rank_sync()
        # Warm the small NCCL reduction and host snapshot used at a window
        # boundary. Preserve the requested initial path: warmup observations
        # must not become a real previous-window decision.
        initial_use_replica = self.guard._use_replica
        self.guard.observe(self.logical_topk_idx)
        self.guard.finalize_window()
        self.guard._use_replica = initial_use_replica
        self.guard.clear_counts()
        self._rank_sync()

    def capture(self, *, warmup_iterations: int = 2, pool: Any = None) -> None:
        """Prewarm and capture one CUDA Graph for each path."""
        self.warmup(warmup_iterations)

        def capture_one(step: Callable[[], None]) -> torch.cuda.CUDAGraph:
            self.guard.clear_counts()
            self._rank_sync()
            graph = torch.cuda.CUDAGraph()
            if pool is None:
                context = torch.cuda.graph(graph)
            else:
                context = torch.cuda.graph(graph, pool=pool)
            with context:
                step()
            self._rank_sync()
            self.guard.clear_counts()
            return graph

        self._baseline_graph = capture_one(self.baseline_step)
        self._replica_graph = capture_one(self.replica_step)
        self._stats_graph = capture_one(
            lambda: self.guard.observe(self.logical_topk_idx)
        )
        self._steps_in_window = 0
        self._last_metrics = None

    def run(self) -> PersistentReplicaStepResult:
        """Launch the selected path and update the previous-window guard."""
        used_replica = self.guard.use_replica
        sample_before = (
            self._steps_in_window
            * self.guard.samples_per_window
            // self.guard.window_size
        )
        sample_after = (
            (self._steps_in_window + 1)
            * self.guard.samples_per_window
            // self.guard.window_size
        )
        should_sample = sample_after > sample_before
        if self.captured:
            if should_sample:
                assert self._stats_graph is not None
                self._stats_graph.replay()
            graph = self._replica_graph if used_replica else self._baseline_graph
            assert graph is not None
            graph.replay()
        else:
            if should_sample:
                self.guard.observe(self.logical_topk_idx)
            (self.replica_step if used_replica else self.baseline_step)()

        self._steps_in_window += 1
        metrics = None
        if self._steps_in_window == self.guard.window_size:
            metrics = self.guard.finalize_window()
            self._last_metrics = metrics
            self._steps_in_window = 0

        return PersistentReplicaStepResult(
            used_replica=used_replica,
            window_complete=metrics is not None,
            use_replica_next_window=self.guard.use_replica,
            metrics=metrics,
        )
