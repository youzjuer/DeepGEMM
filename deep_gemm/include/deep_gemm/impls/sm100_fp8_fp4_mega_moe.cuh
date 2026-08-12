#pragma once

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_conversion.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/profiling/mega_moe.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    uint32_t kFP4ScaleGranularity = 0,
    bool kUseEpochWorkspace = false,
    bool kUseExpertRoutingMap = false,
    uint32_t kDispatchReadyMode = 0,
    bool kEnableKernelProfile = false,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const int* expert_routing_choices,
                            const int* expert_routing_counts,
                            const uint32_t num_logical_experts,
                            const uint32_t max_expert_instances,
                            uint64_t* kernel_profile,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;
    constexpr bool kUsePackedFP4 = kFP4ScaleGranularity != 0;
    constexpr bool kUseNVFP4 = kFP4ScaleGranularity == 16;
    constexpr bool kUseMXFP4 = kFP4ScaleGranularity == 32;

    // Template checks
    DG_STATIC_ASSERT(kFP4ScaleGranularity == 0 or kUseNVFP4 or kUseMXFP4,
                     "Packed FP4 scales must use group 16 or group 32");
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(not kUseEpochWorkspace or kUsePackedFP4,
                     "Epoch workspace is currently supported only by packed-FP4 MegaMoE");
    DG_STATIC_ASSERT(not kUseExpertRoutingMap or kUsePackedFP4,
                     "Expert routing maps are currently supported only by packed-FP4 MegaMoE");
    DG_STATIC_ASSERT(kDispatchReadyMode <= 4,
                     "Unsupported dispatch readiness mode");
    DG_STATIC_ASSERT(kDispatchReadyMode == 0 or (kUsePackedFP4 and kUseEpochWorkspace),
                     "Barrierless dispatch readiness requires packed-FP4 epoch workspace");
    DG_STATIC_ASSERT(not kEnableKernelProfile or kUsePackedFP4,
                     "Kernel profiling is currently supported only by packed-FP4 MegaMoE");
    DG_STATIC_ASSERT(not kUsePackedFP4 or not kHasShared,
                     "Packed-FP4 MegaMoE does not yet support shared experts");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Optional device-side timeline.  The JIT specializes this entire block
    // away for normal launches, leaving the production kernel unchanged.
    using ProfileLayout = profile::MegaMoEKernelProfileLayout;
    DG_STATIC_ASSERT(kNumThreads / 32 <= ProfileLayout::kMaxWarps,
                     "MegaMoE profile layout has too few warp slots");
    uint32_t profile_compute_count = 0;
    uint32_t profile_communication_count = 0;
    uint64_t* cta_profile = nullptr;
    if constexpr (kEnableKernelProfile) {
        DG_DEVICE_ASSERT(kernel_profile != nullptr);
        cta_profile = kernel_profile +
            static_cast<uint64_t>(sm_idx) * ProfileLayout::kWordsPerCTA;
    }
    const auto profile_now = [&]() -> uint64_t {
        if constexpr (kEnableKernelProfile) {
            if (lane_idx == 0)
                return profile::read_globaltimer();
        }
        return 0;
    };
    const auto profile_mark_overflow = [&]() {
        if constexpr (kEnableKernelProfile) {
            if (lane_idx == 0)
                cta_profile[ProfileLayout::kOverflowOffset] = 1;
        }
    };
    const auto profile_append_interval = [&](const ProfileLayout::IntervalKind kind,
                                             const ProfileLayout::IntervalStage stage,
                                             const uint64_t start,
                                             const uint64_t end) {
        if constexpr (kEnableKernelProfile) {
            if (lane_idx == 0) {
                auto& count = kind == ProfileLayout::Compute ?
                    profile_compute_count : profile_communication_count;
                if (count < ProfileLayout::kMaxIntervalsPerWarp) {
                    cta_profile[ProfileLayout::get_interval_offset(
                        kind, warp_idx, count, false)] =
                            ProfileLayout::encode_stage(start, stage);
                    cta_profile[ProfileLayout::get_interval_offset(
                        kind, warp_idx, count, true)] = end;
                } else {
                    profile_mark_overflow();
                }
                ++ count;
            }
        }
    };
    if constexpr (kEnableKernelProfile) {
        if (lane_idx == 0)
            cta_profile[ProfileLayout::get_kernel_offset(warp_idx, false)] =
                profile::read_globaltimer();
    }

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights_sf);
    }

    // Workspaces and Buffer
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ true,
        /*use_packed_fp4=*/ kUsePackedFP4,
        /*sf_gran_k=*/ kUsePackedFP4 ? kFP4ScaleGranularity : 32,
        kNumSharedExperts
    );
    const auto workspace_layout = buffer.workspace;
    const auto input_token_buffer = buffer.input_token_buffer;
    const auto input_sf_buffer = buffer.input_sf_buffer;
    const auto input_topk_idx_buffer = buffer.input_topk_idx_buffer;
    const auto input_topk_weights_buffer = buffer.input_topk_weights_buffer;
    const auto l1_token_buffer = buffer.l1_token_buffer;
    const auto l1_sf_buffer = buffer.l1_sf_buffer;
    const auto l1_topk_weights_buffer = buffer.l1_topk_weights_buffer;
    const auto l2_token_buffer = buffer.l2_token_buffer;
    const auto l2_sf_buffer = buffer.l2_sf_buffer;
    const auto combine_token_buffer = buffer.combine_token_buffer;

    // SF and its buffer configs
    constexpr uint32_t kGranK = kUsePackedFP4 ? kFP4ScaleGranularity : 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types
    // TMA unpacks packed FP4 payloads to one byte per element in shared
    // memory. The legacy path keeps FP8 activations and FP4 weights.
    using a_dtype_t = std::conditional_t<kUsePackedFP4,
        cutlass::detail::float_e2m1_unpacksmem_t, cutlass::float_e4m3_t>;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;
    using a_smem_dtype_t = std::conditional_t<kUsePackedFP4, cutlass::float_e2m1_t, a_dtype_t>;
    using b_smem_dtype_t = std::conditional_t<kUsePackedFP4, cutlass::float_e2m1_t, b_dtype_t>;
    using shared_a_dtype_t = cutlass::float_e4m3_t;
    using shared_b_dtype_t = cutlass::float_e4m3_t;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_DESC_BLOCK_K = kUsePackedFP4 ? 256 : 128;
    constexpr uint32_t UMMA_K = kUsePackedFP4 ? 64 : 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M / 2; // float2

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) a_dtype_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
        } smem_d;
        alignas(kSharedMemoryAlignment) a_smem_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K / (kUsePackedFP4 ? 2 : 1)];
        alignas(kSharedMemoryAlignment) b_smem_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K / (kUsePackedFP4 ? 2 : 1)];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / (kGranK * 4))];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / (kGranK * 4))];
        float2 amax_reduction[kNumEpilogueWarps][AMAX_REDUCTION_WARP_BUFFER_SIZE];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        uint32_t tmem_ptr_in_smem;
        uint32_t metadata_bank_idx;
        uint32_t barrier_generation;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemColsPerSet = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemColsPerSet = SF_BLOCK_N / 32;
    constexpr uint32_t kNumSFTmemSets = kUsePackedFP4 ? UMMA_DESC_BLOCK_K / (kGranK * 4) : 1;
    constexpr uint32_t kNumSFATmemCols = kNumSFATmemColsPerSet * kNumSFTmemSets;
    constexpr uint32_t kNumSFBTmemCols = kNumSFBTmemColsPerSet * kNumSFTmemSets;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync()) {
            // The bytes must be 8 bytes aligned
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
            const auto barrier_generation = ptx::ld_acq(
                workspace_layout.get_nvl_barrier_counter_ptr());
            shared_storage.barrier_generation = barrier_generation;
            shared_storage.metadata_bank_idx = kUseEpochWorkspace ?
                ((barrier_generation >> 1) & 1u) : 0u;
        }
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at 2 CTAs, A + B
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrive only at the leader CTA
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    // NOTES: Using `.relaxed` is allowed here since `fence_barrier_init` is `.release.cluster`,
    // and `barrier.cluster.wait.aligned` is by default `.acquire`
    comm::cluster_sync_with_relaxed_arrive();

    // Two NVLink barriers are issued per epoch-enabled launch. Their device
    // generation therefore alternates the active metadata bank without a host
    // argument, including under CUDA Graph replay. Each active bank is cleaned
    // while its launch combines, then remains unused for one full launch before
    // reuse; that intervening dispatch barrier supplies cross-rank confirmation.
    const auto workspace = workspace_layout.with_metadata_bank(
        shared_storage.metadata_bank_idx);

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks,
        kNumSharedExperts,
        kDispatchReadyMode>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos,
            shared_storage.expert_token_count
    );

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;
    constexpr uint32_t kDispatchReadyCacheBarrierIdx =
        kEpilogueWGBarrierStartIdx + kNumEpilogueWarpgroups;
    DG_STATIC_ASSERT(kDispatchReadyCacheBarrierIdx < 16,
                     "Insufficient named barriers for readiness cache");

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // Adjust registers
    // NOTES: more experts per rank will cost more schedulers' registers
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 160;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& apply_expert_routing, const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    const uint32_t token_topk_idx = i * kNumTopk + lane_idx;
                    auto topk_idx_ptr =
                        input_topk_idx_buffer.get_base_ptr<int64_t>() + token_topk_idx;
                    expert_idx = static_cast<int>(
                        __ldg(topk_idx_ptr));
                    if (expert_idx >= 0) {
                        if constexpr (
                            kUseExpertRoutingMap and
                            std::decay_t<decltype(apply_expert_routing)>::value
                        ) {
                            DG_DEVICE_ASSERT(expert_routing_choices != nullptr);
                            DG_DEVICE_ASSERT(expert_routing_counts != nullptr);
                            DG_DEVICE_ASSERT(static_cast<uint32_t>(expert_idx) < num_logical_experts);
                            const int num_instances = __ldg(expert_routing_counts + expert_idx);
                            DG_DEVICE_ASSERT(num_instances > 0);
                            DG_DEVICE_ASSERT(static_cast<uint32_t>(num_instances) <= max_expert_instances);
                            uint32_t instance_idx = 0;
                            if (num_instances > 1) {
                                const uint32_t global_token_idx =
                                    sym_buffer.rank_idx * num_tokens + token_topk_idx / kNumTopk;
                                instance_idx = global_token_idx % static_cast<uint32_t>(num_instances);
                            }
                            expert_idx = __ldg(
                                expert_routing_choices +
                                expert_idx * max_expert_instances + instance_idx);
                            DG_DEVICE_ASSERT(expert_idx >= 0 and expert_idx < static_cast<int>(kNumExperts));
                            // The registered top-k buffer is launch scratch.
                            // Cache the physical ID so the metadata scan below
                            // does not repeat table loads and route selection.
                            *topk_idx_ptr = static_cast<int64_t>(expert_idx);
                        }
                        process(token_topk_idx, expert_idx);
                    }
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx(std::true_type{}, [&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        const auto profile_metadata_start = profile_now();
        read_topk_idx(std::false_type{}, [&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });
        profile_append_interval(
            ProfileLayout::Communication, ProfileLayout::RouteMetadata,
            profile_metadata_start, profile_now());

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Publish expert counts and wait until every rank can safely pull.
        const auto profile_publish_start = profile_now();

        // Write expert count
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_count = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_count & 0xffffffff;
                const auto recv_sum_ptr = sym_buffer.map(
                    workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx),
                    dst_rank_idx);
                if constexpr (kDispatchReadyMode == 1 or
                              kDispatchReadyMode == 2 or
                              kDispatchReadyMode == 3)
                    ptx::atomic_add_rel_sys(recv_sum_ptr, expert_count);
                else
                    ptx::atomic_add_sys(recv_sum_ptr, expert_count);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        if constexpr (kDispatchReadyMode == 1 or
                      kDispatchReadyMode == 2 or
                      kDispatchReadyMode == 4) {
            // Epoch mode normally advances its generation twice per launch,
            // once at each cross-rank barrier. Keep barrier #1's monotonic
            // signal/counter arrive so candidate and baseline launches can
            // share a workspace, but do not wait for remote arrivals or
            // broadcast completion to 148 CTAs. Modes 1/2 use per-expert
            // completion; mode 4 lets each CTA wait on tag-1 directly.
            if (sm_idx == 0) {
                auto* signal_ptr = reinterpret_cast<uint32_t*>(
                    workspace.get_nvl_barrier_signal_ptr(
                        kBeforeDispatchPullBarrierTag - 1));
                if (thread_idx < kNumRanks)
                    ptx::red_add_rel_sys(
                        reinterpret_cast<uint32_t*>(
                            sym_buffer.map(signal_ptr, thread_idx)),
                        1u);
                ptx::sync_aligned(
                    kNumDispatchThreads, kDispatchBarrierIdx);
                if (thread_idx == 0)
                    ptx::red_add(
                        workspace.get_nvl_barrier_counter_ptr(), 1u);
            }
        } else if constexpr (kUseEpochWorkspace) {
            comm::nvlink_epoch_barrier<
                kNumRanks, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
                /* After the NVLink barrier, there is a grid sync */ true
            );
        } else {
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
                /* After the NVLink barrier, there is a grid sync */ true
            );
        }
        if constexpr (kDispatchReadyMode == 4) {
            // Every rank releases tag-1 only after publishing all expert
            // counts and route metadata. One acquire per CTA therefore makes
            // the whole publication visible. Warp 0 then snapshots local
            // expert counts in retired dispatch SMEM; the CTA barrier carries
            // that visibility to every scheduler role without repeating a
            // system-scope acquire for every expert and every warp.
            if (warp_idx == 0) {
                auto* signal_ptr = reinterpret_cast<uint32_t*>(
                    workspace.get_nvl_barrier_signal_ptr(
                        kBeforeDispatchPullBarrierTag - 1));
                if (lane_idx == 0) {
                    const uint32_t target =
                        (shared_storage.barrier_generation / 2u + 1u) *
                        kNumRanks;
                    while (ptx::ld_acq_sys(signal_ptr) != target);
                }
                __syncwarp();
                for (uint32_t expert_idx = lane_idx;
                     expert_idx < kNumExpertsPerRank;
                     expert_idx += 32) {
                    const auto value = ptx::ld_volatile(
                        workspace.get_expert_recv_count_sum_ptr(expert_idx));
                    DG_DEVICE_ASSERT(
                        static_cast<uint32_t>(value >> 32) ==
                        kNumSMs * kNumRanks);
                    shared_storage.expert_token_count[expert_idx] =
                        static_cast<uint32_t>(value);
                }
            }
            ptx::sync_unaligned(
                kNumDispatchThreads + 32, kDispatchReadyCacheBarrierIdx);
        }
        profile_append_interval(
            ProfileLayout::Communication,
            ProfileLayout::DispatchPublishBarrier,
            profile_publish_start, profile_now());

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // Wait token data arrival
        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks
            constexpr uint32_t kNumInputTokenBytes = kUsePackedFP4 ? kHidden / 2 : kHidden;
            constexpr uint32_t kNumChunks = kNumInputTokenBytes / kNumBytesPerPull;
            DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kNumInputTokenBytes,
                             "kNumBytesPerPull must divide token bytes");

            // TMA load token from remote rank and store into local
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }

            const bool profile_is_remote_pull =
                current_rank_in_expert_idx != sym_buffer.rank_idx;
            const auto profile_pull_start = profile_now();

            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();

            // Load and store SF (overlaps with last chunk's TMA load from remote)
            constexpr uint32_t kNumSFUint32 = kHidden / (kGranK * 4);
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % (kGranK * 4) == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
            const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_block);
            if constexpr (kUseNVFP4) {
                // Group-16 scales double the number of words per token.  Pull
                // adjacent words together so H4096 still needs one warp-wide
                // remote-load round, then scatter to the existing MN-major
                // local layout.
                DG_STATIC_ASSERT(kNumSFUint32 % 2 == 0, "Invalid NVFP4 SF");
                constexpr uint32_t kNumSFUint2 = kNumSFUint32 / 2;
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint2, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFUint2) {
                        const auto sf_pair = reinterpret_cast<const uint2*>(remote_sf_ptr)[j];
                        const uint32_t sf_idx = j * 2;
                        local_sf_ptr[sf_idx * kNumSFRingTokens + sf_ring_token_idx] = sf_pair.x;
                        local_sf_ptr[(sf_idx + 1) * kNumSFRingTokens + sf_ring_token_idx] = sf_pair.y;
                    }
                }
            } else {
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFUint32)
                        local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
                }
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                // Load weights
                const auto weight = *sym_buffer.map(
                    buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Complete last chunk's store
                issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), 
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
            if (profile_is_remote_pull)
                profile_append_interval(
                    ProfileLayout::Communication,
                    ProfileLayout::RemotePull,
                    profile_pull_start, profile_now());
        }

        // The epilogue reaches this handoff after publishing all remote L2
        // outputs. Cleanup therefore overlaps combine in both lifecycle modes.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                *workspace.get_l1_task_count_ptr() = 0;
                *workspace.get_l2_task_count_ptr() = 0;
                *workspace.get_shared_l1_task_count_ptr() = 0;
                *workspace.get_shared_l2_task_count_ptr() = 0;
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Legacy mode clears per-rank counts in place. Epoch mode
                // leaves them intact because every source rank overwrites all
                // expert entries before the next dispatch barrier.
                if constexpr (not kUseEpochWorkspace) {
                    for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                        *workspace.get_expert_recv_count_ptr(j, i) = 0;
                }
                __syncwarp();

                // Clean L1 and L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        if constexpr (not kUseEpochWorkspace) {
            // Wait for all ranks to finish cleaning
            const auto profile_cleanup_barrier_start = profile_now();
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                /* Before the NVLink barrier, there is a grid sync */ true,
                /* At the end of kernel does not need to sync */ false
            );
            profile_append_interval(
                ProfileLayout::Communication,
                ProfileLayout::CleanupBarrier,
                profile_cleanup_barrier_start, profile_now());
        }
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        uint32_t profile_block_idx = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts_sf;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait the entire token arrival
            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
            } else if (task_info.block_phase == sched::BlockPhase::Linear2) {
                const auto ptr = workspace.get_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (L2_SHAPE_K / BLOCK_N) * 2 * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (SHARED_L2_SHAPE_K / BLOCK_N) * 2;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            if constexpr (kEnableKernelProfile) {
                if (lane_idx == 0) {
                    if (profile_block_idx < ProfileLayout::kMaxBlocksPerCTA) {
                        cta_profile[ProfileLayout::get_block_offset(
                                profile_block_idx, false)] = ProfileLayout::encode_stage(
                                profile::read_globaltimer(),
                                task_info.block_phase == sched::BlockPhase::Linear2 ?
                                    ProfileLayout::GemmL2 : ProfileLayout::GemmL1);
                    } else {
                        profile_mark_overflow();
                    }
                }
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute token offsets from block index
                uint32_t m_idx = block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / (kGranK * 4));

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    if constexpr (kUsePackedFP4) {
                        tma::copy_packed_fp4<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode>(
                            tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                            reinterpret_cast<uint8_t*>(shared_storage.smem_a[stage_idx]),
                            k_idx, m_idx, 2);
                    } else {
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                            shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    }
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        // Two CTAs contribute one local shared-memory tile each.
                        constexpr uint32_t kAClusterTransactionBytes =
                            sizeof(SharedStorage::smem_a[0]) * 2;
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            kAClusterTransactionBytes + sizeof(SharedStorage::smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
            ++ profile_block_idx;
        }
        if constexpr (kEnableKernelProfile) {
            if (lane_idx == 0)
                cta_profile[ProfileLayout::kBlockStartCountOffset] = profile_block_idx;
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
            const auto tensor_map_sfb_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights_sf;

            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);
            const auto n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = task_info.is_shared() ? n_block_idx * BLOCK_N : task_info.local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = task_info.is_shared() ?
                    k_block_idx * (BLOCK_K / 128) :
                    task_info.local_expert_idx * shape_sfb_k +
                        k_block_idx * (BLOCK_K / (kGranK * 4));

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[stage_idx]), k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * 2 + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    } else {
                        if constexpr (kUsePackedFP4) {
                            tma::copy_packed_fp4<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode>(
                                tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx],
                                reinterpret_cast<uint8_t*>(shared_storage.smem_b[stage_idx]),
                                k_idx, n_idx, 2);
                        } else {
                            tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx],
                                shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                        }
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            constexpr uint32_t kBClusterTransactionBytes =
                                sizeof(SharedStorage::smem_b[0]) * (kUsePackedFP4 ? 2 : 1);
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                                kBClusterTransactionBytes + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            using instr_a_dtype_t = std::conditional_t<kUsePackedFP4, cutlass::float_e2m1_t, b_dtype_t>;
            using instr_b_dtype_t = std::conditional_t<kUsePackedFP4, cutlass::float_e2m1_t, a_dtype_t>;
            using sf_dtype_t = std::conditional_t<kUseNVFP4, cutlass::float_ue4m3_t, cutlass::float_ue8m0_t>;
            auto routed_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                instr_a_dtype_t, instr_b_dtype_t, float, sf_dtype_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto shared_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                shared_b_dtype_t, shared_a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<
                cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_DESC_BLOCK_K, kSwizzleAMode>(
                    shared_storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<
                cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_DESC_BLOCK_K, kSwizzleBMode>(
                    shared_storage.smem_b[0], 0, 0);
            auto shared_b_desc = [&]() {
                if constexpr (kUsePackedFP4) {
                    return b_desc;
                } else {
                    return mma::sm100::make_umma_desc<
                        cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_DESC_BLOCK_K, kSwizzleBMode>(
                            reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[0]), 0, 0);
                }
            }();
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(SharedStorage::smem_a[0]) / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(SharedStorage::smem_b[0]) / 16 : 0u;
            uint32_t shared_b_desc_lo = lane_idx < kNumStages ? shared_b_desc.lo + lane_idx * sizeof(SharedStorage::smem_b[0]) / 16 : 0u;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            while (scheduler.get_next_task(task_info)) {
                const auto num_k_blocks = task_info.shape_k / BLOCK_K;

                // Dynamic update of UMMA N based on effective M
                auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(task_info.is_shared() ? shared_b_desc_lo : b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_DESC_BLOCK_K; ++ umma_k_block_idx) {
                            const auto copy_sf_to_tmem = [&](const uint32_t& sf_word_idx, const uint32_t& tmem_set_idx) {
                                using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                                #pragma unroll
                                for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                    auto smem_ptr = shared_storage.smem_sfa[stage_idx] + sf_word_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA +
                                        tmem_set_idx * kNumSFATmemColsPerSet + i * 4);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                    auto smem_ptr = shared_storage.smem_sfb[stage_idx] + sf_word_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB +
                                        tmem_set_idx * kNumSFBTmemColsPerSet + i * 4);
                                }
                            };

                            // MXFP8 shares one SF word across all four K32
                            // instructions. NVFP4 preserves its validated
                            // group-16 publication order. MXFP4 publishes each
                            // independent group-32 TMEM set immediately before
                            // its first consumer below; this lets the next
                            // set's UTCCP overlap the prior set's MMA instead of
                            // serializing every scale copy in the critical
                            // prefix.
                            if constexpr (kUseNVFP4) {
                                #pragma unroll
                                for (uint32_t sf_set_idx = 0; sf_set_idx < kNumSFTmemSets; ++ sf_set_idx)
                                    copy_sf_to_tmem(
                                        umma_k_block_idx * kNumSFTmemSets + sf_set_idx,
                                        sf_set_idx);
                            } else if constexpr (not kUsePackedFP4) {
                                copy_sf_to_tmem(umma_k_block_idx, 0);
                            }
                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_DESC_BLOCK_K / UMMA_K; ++ k) {
                                if constexpr (kUseMXFP4) {
                                    if (k % 2 == 0) {
                                        const uint32_t sf_set_idx = k / 2;
                                        copy_sf_to_tmem(
                                            umma_k_block_idx * kNumSFTmemSets + sf_set_idx,
                                            sf_set_idx);
                                    }
                                }
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(
                                        instr_desc,
                                        kUseNVFP4 ? 0u : kUseMXFP4 ? (k % 2) * 2 : k,
                                        kUseNVFP4 ? 0u : kUseMXFP4 ? (k % 2) * 2 : k);
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_smem_dtype_t>(
                                        a_desc_base_lo,
                                        umma_k_block_idx * UMMA_DESC_BLOCK_K * LOAD_BLOCK_M,
                                        k * UMMA_K);
                                if (task_info.is_shared()) {
                                    b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t>(
                                            b_desc_base_lo,
                                            umma_k_block_idx * UMMA_DESC_BLOCK_K * LOAD_BLOCK_N,
                                            k * UMMA_K);
                                } else {
                                    b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_smem_dtype_t>(
                                            b_desc_base_lo,
                                            umma_k_block_idx * UMMA_DESC_BLOCK_K * LOAD_BLOCK_N,
                                            k * UMMA_K);
                                }
                                if constexpr (kUseNVFP4) {
                                    ptx::SM100_MMA_MXF4NVF4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                        kTmemStartColOfSFB + k * kNumSFBTmemColsPerSet,
                                        kTmemStartColOfSFA + k * kNumSFATmemColsPerSet);
                                } else if constexpr (kUseMXFP4) {
                                    ptx::SM100_MMA_MXF4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                        kTmemStartColOfSFB + (k / 2) * kNumSFBTmemColsPerSet,
                                        kTmemStartColOfSFA + (k / 2) * kNumSFATmemColsPerSet);
                                } else {
                                    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                        kTmemStartColOfSFB, kTmemStartColOfSFA);
                                }
                            }
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            }

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        if constexpr (kDispatchReadyMode == 4)
            ptx::sync_unaligned(
                kNumDispatchThreads + 32, kDispatchReadyCacheBarrierIdx);
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        uint32_t profile_block_idx = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            if constexpr (kEnableKernelProfile) {
                if (epilogue_warp_idx == 0 and lane_idx == 0) {
                    if (profile_block_idx < ProfileLayout::kMaxBlocksPerCTA) {
                        cta_profile[ProfileLayout::get_block_offset(
                            profile_block_idx, true)] = profile::read_globaltimer();
                    } else {
                        profile_mark_overflow();
                    }
                }
            }
            const auto profile_epilogue_start = profile_now();

            // Now we can release the task
            scheduler.release_task_info();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    // Wait L2 block empty
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }

                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                float stored_cached_weight = 1.0f;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    // Iterate all atoms in the store block
                    float2 activation_values[kNumAtomsPerStore][2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if (not task_info.is_shared() and (j * ATOM_M) % 32 == 0 and
                            (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *buffer.l1_topk_weights_buffer
                                .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .template get_base_ptr<float>();
                        }

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint2 raw_values[4];
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               raw_values[0].x, raw_values[0].y, raw_values[1].x, raw_values[1].y);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               raw_values[2].x, raw_values[2].y, raw_values[3].x, raw_values[3].y);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Apply SwiGLU: silu(gate) * up
                        auto fp32_values = reinterpret_cast<float2*>(raw_values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
                            auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // Amax reduction (thread-level)
                        float2 thread_local_amax = {0.f, 0.f};
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            thread_local_amax.x = cute::max(thread_local_amax.x, cute::abs(activation_values[i][k].x));
                            thread_local_amax.y = cute::max(thread_local_amax.y, cute::abs(activation_values[i][k].y));
                        }

                        // Amax reduction (warp-level)
                        amax_values[i].x = math::warp_reduce<4, true>(
                            thread_local_amax.x, math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            thread_local_amax.y, math::ReduceMax<float>());

                        // Reduce amax (warp-pair-level)
                        if constexpr (not kUseNVFP4) {
                            if (lane_idx < 4)
                                shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        }
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `shared_storage.amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    if constexpr (not kUsePackedFP4)
                        ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Quantize the post-SwiGLU activation and store unpacked
                    // bytes into shared memory for the output stage.
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        if constexpr (not kUseNVFP4) {
                            // MXFP8 uses one group-32 scale shared by a warp pair.
                            const float2 wp_amax =
                                shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                            amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                            amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);
                        }

                        // Calculate SF
                        float2 sf, sf_inv;
                        uint8_t sf_code_x = 0, sf_code_y = 0;
                        if constexpr (kUseNVFP4) {
                            // `warp_reduce<4, true>` gives the same amax to the
                            // eight lanes with the same `lane_idx % 4`.  Only
                            // the first copy needs to perform the UE4M3
                            // conversion and exact reciprocal; broadcast the
                            // bit-identical inverse to the other seven lanes.
                            sf_inv = {0.0f, 0.0f};
                            if (lane_idx < 4) {
                                constexpr float kMinUE4M3 = 1.0f / 512.0f;
                                const cutlass::float_ue4m3_t sf_x(cute::max(amax_values[i].x / 6.0f, kMinUE4M3));
                                const cutlass::float_ue4m3_t sf_y(cute::max(amax_values[i].y / 6.0f, kMinUE4M3));
                                sf = {static_cast<float>(sf_x), static_cast<float>(sf_y)};
                                sf_inv = {1.0f / sf.x, 1.0f / sf.y};
                                sf_code_x = sf_x.raw();
                                sf_code_y = sf_y.raw();
                            }
                            sf_inv.x = __shfl_sync(0xffffffff, sf_inv.x, lane_idx % 4);
                            sf_inv.y = __shfl_sync(0xffffffff, sf_inv.y, lane_idx % 4);
                        } else if constexpr (kUseMXFP4) {
                            math::get_e2m1_sf_and_sf_inv(amax_values[i], sf, sf_inv);
                        } else {
                            math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);
                        }

                        // Cast
                        const float2 upper = __fmul2_rn(activation_values[i][0], sf_inv);
                        const float2 lower = __fmul2_rn(activation_values[i][1], sf_inv);

                        // STSM
                        uint32_t row = lane_idx;
                        uint32_t col = warp_idx_in_wg;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                            + i * ATOM_M * L1_OUT_BLOCK_N
                            + row * L1_OUT_BLOCK_N
                            // Use 64B swizzle for SwiGLU, so divided by 2
                            + (col ^ (row / 2)) * kNumBankGroupBytes;
                        if constexpr (kUsePackedFP4) {
                            const cutlass::Array<float, 4> fp32x4 = {upper.x, upper.y, lower.x, lower.y};
                            const auto fp4x4 = cutlass::NumericArrayConverter<
                                cutlass::float_e2m1_t, float, 4>()(fp32x4);
                            const uint16_t packed = *reinterpret_cast<const uint16_t*>(&fp4x4);
                            const uint32_t unpacked =
                                (packed & 0x000fu) |
                                ((packed & 0x00f0u) << 4u) |
                                ((packed & 0x0f00u) << 8u) |
                                ((packed & 0xf000u) << 12u);
                            ptx::SM100_U8x4_STSM_T<uint32_t>::copy(unpacked, smem_ptr);
                        } else {
                            const auto fp8x4_values = __nv_fp8x4_e4m3(
                                make_float4(upper.x, upper.y, lower.x, lower.y));
                            ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);
                        }

                        // Store packed SF to `l2_sf_buffer` in MN-major layout.
                        // Group-32 MXFP8/MXFP4 has one writer per warp pair;
                        // group-16 NVFP4 has one scale writer per warp.
                        // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
                        if ((kUseNVFP4 or warp_idx_in_wg % 2 == 0) and lane_idx < 4) {
                            const uint32_t k_idx = kUseNVFP4 ?
                                n_block_idx * 4 + warp_idx_in_wg :
                                n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = (task_info.is_shared() ? kNumSharedSFTokens : kNumSFRingTokens) * sizeof(uint32_t);
                            const auto sf_base_ptr = task_info.is_shared() ?
                                buffer.shared_l2_sf_buffer.get_base_ptr<uint8_t>() : buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            // NOTES: originally there was:
                            //   - `const uint32_t token_idx_in_expert = task_info.m_block_idx * BLOCK_M + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2
                            //   - `task_info.pool_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert)`
                            // We find out that
                            //   1. `task_info.m_block_idx * BLOCK_M` mod `BLOCK_M` is 0, and `epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2` is always < `BLOCK_M`, so we can put `task_info.m_block_idx * BLOCK_M` outside
                            //   2. `lane_idx * 2` controls the lowest 3 bit of `token_idx_in_expert`, and `transform_sf_token_idx` is a bitwise-independent transformation if the input is less than `BLOCK_M`, so we can put `lane_idx * 2` outside
                            // This reduce the number of computation instructions.
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_token_idx = block_idx * SF_BLOCK_M
                                + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] = kUseNVFP4 ? sf_code_x :
                                (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] = kUseNVFP4 ? sf_code_y :
                                (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    if constexpr (kUsePackedFP4) {
                        // The FP4 TMA unpacked format requires an inner box of
                        // at least 128 logical elements. Post-SwiGLU is only 64
                        // elements wide, so cooperatively pack adjacent nibbles
                        // and write the local ring buffer directly.
                        constexpr uint32_t kPackedBytesPerRow = L1_OUT_BLOCK_N / 2;
                        const uint32_t wg_thread_idx = warp_idx_in_wg * 32 + lane_idx;
                        const auto smem_base = reinterpret_cast<const uint8_t*>(
                            shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx]);
                        #pragma unroll
                        for (uint32_t packed_idx = wg_thread_idx;
                             packed_idx < STORE_BLOCK_M * kPackedBytesPerRow;
                             packed_idx += 128) {
                            const uint32_t row_in_store = packed_idx / kPackedBytesPerRow;
                            const uint32_t packed_col = packed_idx % kPackedBytesPerRow;
                            const uint32_t logical_col = packed_col * 2;
                            const uint32_t bank_group = logical_col / kNumBankGroupBytes;
                            const uint32_t col_in_bank_group = logical_col % kNumBankGroupBytes;
                            constexpr uint32_t kNumBankGroups =
                                L1_OUT_BLOCK_N / kNumBankGroupBytes;
                            const uint32_t swizzle_row =
                                (row_in_store / 2) % kNumBankGroups;
                            const uint32_t physical_col =
                                (bank_group ^ swizzle_row) * kNumBankGroupBytes + col_in_bank_group;
                            const auto lo = smem_base[row_in_store * L1_OUT_BLOCK_N + physical_col] & 0x0f;
                            const auto hi = smem_base[row_in_store * L1_OUT_BLOCK_N + physical_col + 1] & 0x0f;
                            const uint32_t ring_token_idx = ring_m_idx + epilogue_wg_idx * WG_BLOCK_M +
                                s * STORE_BLOCK_M + row_in_store;
                            auto dst = l2_token_buffer.get_data_buffer(ring_token_idx).template get_base_ptr<uint8_t>();
                            dst[n_block_idx * kPackedBytesPerRow + packed_col] = lo | (hi << 4);
                        }
                    } else if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        const auto tensor_map_l1_output_ptr = task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            tensor_map_l1_output_ptr,
                            shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2 and increment L1 empty count
                // TODO: less epilogue sync scope
                if constexpr (not kUsePackedFP4)
                    ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        ptx::red_add_rel(
                            workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    } else {
                        ptx::red_add_rel(
                            workspace.get_l2_full_count_ptr(ring_block_idx), 1u);

                        // Increment L1 empty count for this physical slot (one per N block)
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
                if (warp_idx_in_wg == 0)
                    profile_append_interval(
                        ProfileLayout::Compute,
                        ProfileLayout::EpilogueL1,
                        profile_epilogue_start, profile_now());
            } else {
                // Increment L2 empty count for this physical slot (one per N block)
                if (not task_info.is_shared()) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_add(
                            workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    }
                    __syncwarp();
                }

                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    // TODO: check performance
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    const auto profile_l2_compute_start =
                        s == 0 ? profile_epilogue_start : profile_now();

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements
                        // Start from lane index 0 and 16
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Store into shared memory
                        // NOTES: each lane provides its own address for stmatrix; 2 warps share a BF16 swizzle atom
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (warp_idx_in_wg == 0)
                        profile_append_interval(
                            ProfileLayout::Compute,
                            ProfileLayout::EpilogueL2,
                            profile_l2_compute_start, profile_now());

                    // Write into remote buffers
                    // Each warp writes 2 rows (lane_idx/16 splits the warp into two halves, one per row)
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;
                    const auto profile_output_push_start = profile_now();
                    bool profile_wrote_remote = false;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                        if (task_info.is_shared()) {
                            dst_rank_idx = sym_buffer.rank_idx;
                            dst_token_idx = pool_m_idx + m_idx_in_block;
                            dst_topk_idx = kNumTopk;
                        } else {
                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                            dst_rank_idx = src_metadata.rank_idx;
                            dst_token_idx = src_metadata.token_idx;
                            dst_topk_idx = src_metadata.topk_idx;
                        }
                        if constexpr (kEnableKernelProfile)
                            profile_wrote_remote |= dst_rank_idx != sym_buffer.rank_idx;

                        // Read from shared memory
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // Write into remote
                        const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                    if constexpr (kEnableKernelProfile) {
                        const bool profile_any_remote =
                            __any_sync(0xffffffff, profile_wrote_remote);
                        profile_append_interval(
                            profile_any_remote ? ProfileLayout::Communication :
                                                 ProfileLayout::Compute,
                            profile_any_remote ? ProfileLayout::RemoteOutputPush :
                                                 ProfileLayout::EpilogueL2,
                            profile_output_push_start, profile_now());
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
            ++ profile_block_idx;
        }
        if constexpr (kEnableKernelProfile) {
            if (epilogue_warp_idx == 0 and lane_idx == 0)
                cta_profile[ProfileLayout::kBlockEndCountOffset] = profile_block_idx;
        }

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // NVLink barrier (grid sync + cross-rank signal + grid sync): ~4 us
        const auto profile_combine_barrier_start = profile_now();
        if constexpr (kUseEpochWorkspace) {
            comm::nvlink_epoch_barrier<
                kNumRanks, kNumSMs, kNumEpilogueThreads,
                kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
            );
        } else {
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumEpilogueThreads,
                kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
            );
        }
        profile_append_interval(
            ProfileLayout::Communication,
            ProfileLayout::CombineBarrier,
            profile_combine_barrier_start, profile_now());

        // Start active-bank cleanup after output publication and overlap it
        // with combine. Epoch mode reuses this bank only two launches later;
        // the intervening launch's dispatch barrier confirms every rank has
        // completed the cleanup before that reuse.
        ptx::sync_unaligned(
            kNumDispatchThreads + kNumEpilogueThreads,
            kDispatchWithEpilogueBarrierIdx);

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from start up to the barriers
        // 1 token, 1 topk latency: ~3 us
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes);

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
        });

        // Iterate over all tokens
        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const auto profile_combine_start = profile_now();
            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Iterate all chunks
            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                // Move mask and load
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        // Move
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;

                        // Load
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                // Load the first selection
                bool do_reduce = move_mask_and_load(load_stage_idx);

                // Accumulate all top-k contributions for this chunk in float registers
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    // Prefetch next top-k into the buffer while current is being accumulated
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                    // Accumulate
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                // Cast
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    // Wait share memory release and write
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                // TMA store the token chunk
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
            profile_append_interval(
                ProfileLayout::Compute,
                ProfileLayout::Combine,
                profile_combine_start, profile_now());
        }
    }

    if constexpr (kEnableKernelProfile) {
        if (lane_idx == 0) {
            cta_profile[ProfileLayout::get_counter_offset(
                ProfileLayout::Compute, warp_idx)] = profile_compute_count;
            cta_profile[ProfileLayout::get_counter_offset(
                ProfileLayout::Communication, warp_idx)] =
                    profile_communication_count;
            cta_profile[ProfileLayout::get_kernel_offset(warp_idx, true)] =
                profile::read_globaltimer();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
