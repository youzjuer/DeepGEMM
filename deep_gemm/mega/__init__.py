import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu'):
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def get_mega_moe_kernel_profile_layout() -> dict[str, int]:
    """Return the stable device-timeline layout used by NVFP4 MegaMoE."""
    values = _C.get_mega_moe_kernel_profile_layout()
    keys = (
        'max_warps',
        'max_blocks_per_cta',
        'max_intervals_per_warp',
        'kernel_offset',
        'block_offset',
        'compute_offset',
        'communication_offset',
        'counter_offset',
        'block_start_count_offset',
        'block_end_count_offset',
        'overflow_offset',
        'words_per_cta',
        'stage_shift',
        'timestamp_mask',
        'stage_gemm_l1',
        'stage_gemm_l2',
        'stage_epilogue_l1',
        'stage_epilogue_l2',
        'stage_combine',
        'stage_route_metadata',
        'stage_dispatch_publish_barrier',
        'stage_remote_pull',
        'stage_cleanup_barrier',
        'stage_remote_output_push',
        'stage_combine_barrier',
    )
    return dict(zip(keys, map(int, values)))


def allocate_mega_moe_kernel_profile() -> torch.Tensor:
    """Allocate a zeroed timeline buffer for one profiled MegaMoE launch.

    The caller must clear the tensor before every reuse. Passing ``None`` to
    :func:`nvfp4_mega_moe` selects the normal JIT specialization with all
    timeline code compiled out.
    """
    layout = get_mega_moe_kernel_profile_layout()
    num_sms = _C.get_num_sms()
    return torch.zeros(
        (num_sms, layout['words_per_cta']),
        dtype=torch.int64,
        device='cuda',
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def _ensure_mn_major_packed_sf(sf: torch.Tensor) -> torch.Tensor:
    """Place packed 4-byte SF words in the MN-major TMA layout."""
    if sf.stride(-2) == 1:
        return sf
    *prefix, mn, packed_sf_k = sf.shape
    padded_mn = align(mn, 4)  # four int32 values are 16-byte TMA aligned
    backing = torch.empty((*prefix, packed_sf_k, padded_mn),
                          dtype=sf.dtype, device=sf.device)
    result = backing.transpose(-1, -2)[..., :mn, :]
    result.copy_(sf)
    return result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # Low-precision path: interleave gate/up for weight and packed SF,
        # then transpose L1 SF for UTCCP. This applies to both MXFP8/MXFP4
        # and NVFP4/NVFP4.
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf_input = _ensure_mn_major_packed_sf(l1_weights[1])
        l2_sf_input = _ensure_mn_major_packed_sf(l2_weights[1])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_sf_input))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_sf_input))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


def nvfp4_mega_moe(y: torch.Tensor,
                    l1_weights: Tuple[torch.Tensor, torch.Tensor],
                    l2_weights: Tuple[torch.Tensor, torch.Tensor],
                    sym_buffer: SymmBuffer,
                    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                    recipe: Tuple[int, int, int] = (1, 1, 16),
                    activation: str = 'swiglu',
                    activation_clamp: Optional[float] = None,
                    fast_math: bool = True,
                    expert_routing_map: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    kernel_profile: Optional[torch.Tensor] = None):
    """Run fused NVFP4 MegaMoE.

    ``expert_routing_map`` optionally contains ``(choices, counts)``.  Choices
    is a contiguous int32 ``[num_logical_experts, max_instances]`` tensor of
    physical expert IDs and counts is a contiguous int32 tensor with one valid
    instance count per logical expert.  When present, ``sym_buffer.topk_idx``
    contains logical IDs and dispatch deterministically selects one persistent
    physical instance for each route.  Dispatch rewrites those registered
    scratch indices to physical IDs; callers must refill them before the next
    launch, as in the normal symmetric-buffer contract.

    ``kernel_profile`` optionally enables the device-side timeline profiler.
    Allocate it with :func:`allocate_mega_moe_kernel_profile`, zero it before
    each reuse, and do not use profiled launches for production timing.
    """
    _C.nvfp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        expert_routing_map,
        kernel_profile
    )


def mxfp4_mega_moe(y: torch.Tensor,
                    l1_weights: Tuple[torch.Tensor, torch.Tensor],
                    l2_weights: Tuple[torch.Tensor, torch.Tensor],
                    sym_buffer: SymmBuffer,
                    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                    recipe: Tuple[int, int, int] = (1, 1, 32),
                    activation: str = 'swiglu',
                    activation_clamp: Optional[float] = None,
                    fast_math: bool = True,
                    expert_routing_map: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    kernel_profile: Optional[torch.Tensor] = None):
    """Run fused E2M1xE2M1 MegaMoE with group-32 UE8M0 scaling."""
    _C.mxfp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        expert_routing_map,
        kernel_profile
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math
    )


from .scheduler import (
    PersistentReplicaAutoScheduler,
    PersistentReplicaGuardMetrics,
    PersistentReplicaRouteGuard,
    PersistentReplicaStepResult,
)
