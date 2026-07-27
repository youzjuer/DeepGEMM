#pragma once

#include <cstdint>

#include <cutlass/cutlass.h>

namespace deep_gemm::profile {

// Device-side timeline layout for the optional MegaMoE profiler.  The buffer
// is private to each rank; every CTA owns a disjoint slice, so profiling never
// adds global atomics to the measured kernel.
struct MegaMoEKernelProfileLayout {
    static constexpr uint32_t kMaxWarps = 16;
    static constexpr uint32_t kMaxBlocksPerCTA = 64;
    static constexpr uint32_t kMaxIntervalsPerWarp = 64;
    static constexpr uint32_t kNumIntervalKinds = 2;

    enum IntervalKind : uint32_t {
        Compute = 0,
        Communication = 1,
    };

    // The high bits of an interval start timestamp carry a stage tag.  This
    // keeps the trace footprint and number of global stores unchanged while
    // allowing the host to split compute/communication into useful phases.
    // %globaltimer is nanoseconds since device boot, so 59 timestamp bits
    // leave more than 18 years before wraparound.  Keeping bit 63 clear also
    // lets PyTorch expose the uint64 payload through an int64 tensor safely.
    static constexpr uint32_t kStageShift = 59;
    static constexpr uint64_t kTimestampMask = (1ull << kStageShift) - 1;
    enum IntervalStage : uint32_t {
        Unspecified = 0,
        GemmL1 = 1,
        GemmL2 = 2,
        EpilogueL1 = 3,
        EpilogueL2 = 4,
        Combine = 5,
        RouteMetadata = 6,
        DispatchPublishBarrier = 7,
        RemotePull = 8,
        CleanupBarrier = 9,
        RemoteOutputPush = 10,
        CombineBarrier = 11,
    };

    CUTLASS_HOST_DEVICE
    static constexpr uint64_t encode_stage(
        const uint64_t timestamp, const IntervalStage stage) {
        return (timestamp & kTimestampMask) |
            (static_cast<uint64_t>(stage) << kStageShift);
    }

    // One [start, end] pair per logical warp records the kernel envelope.
    static constexpr uint32_t kKernelOffset = 0;
    static constexpr uint32_t kKernelWords = kMaxWarps * 2;

    // A producer writes the start and epilogue warp 0 writes the completion of
    // each expert block.  This spans local A/B TMA, MMA and accumulator-ready.
    static constexpr uint32_t kBlockOffset = kKernelOffset + kKernelWords;
    static constexpr uint32_t kBlockWords = kMaxBlocksPerCTA * 2;

    // Extra compute intervals cover L1/L2 epilogues and final top-k combine.
    static constexpr uint32_t kComputeOffset = kBlockOffset + kBlockWords;
    static constexpr uint32_t kIntervalWords =
        kMaxWarps * kMaxIntervalsPerWarp * 2;

    // Communication intervals cover remote route publication, remote token
    // pulls, remote L2 output pushes and cross-rank barriers.
    static constexpr uint32_t kCommunicationOffset =
        kComputeOffset + kIntervalWords;

    // Per-warp interval counts, followed by block counters and an overflow
    // flag.  The stride is aligned for simple host-side tensor allocation.
    static constexpr uint32_t kCounterOffset =
        kCommunicationOffset + kIntervalWords;
    static constexpr uint32_t kCounterWords =
        kNumIntervalKinds * kMaxWarps;
    static constexpr uint32_t kBlockStartCountOffset =
        kCounterOffset + kCounterWords;
    static constexpr uint32_t kBlockEndCountOffset =
        kBlockStartCountOffset + 1;
    static constexpr uint32_t kOverflowOffset =
        kBlockEndCountOffset + 1;
    static constexpr uint32_t kUnalignedWordsPerCTA = kOverflowOffset + 1;
    static constexpr uint32_t kWordsPerCTA =
        ((kUnalignedWordsPerCTA + 31) / 32) * 32;

    CUTLASS_HOST_DEVICE
    static constexpr uint64_t get_required_numel(const uint32_t num_ctas) {
        return static_cast<uint64_t>(num_ctas) * kWordsPerCTA;
    }

    CUTLASS_HOST_DEVICE
    static constexpr uint32_t get_kernel_offset(
        const uint32_t warp_idx, const bool is_end) {
        return kKernelOffset + warp_idx * 2 + static_cast<uint32_t>(is_end);
    }

    CUTLASS_HOST_DEVICE
    static constexpr uint32_t get_block_offset(
        const uint32_t block_idx, const bool is_end) {
        return kBlockOffset + block_idx * 2 + static_cast<uint32_t>(is_end);
    }

    CUTLASS_HOST_DEVICE
    static constexpr uint32_t get_interval_offset(
        const IntervalKind kind, const uint32_t warp_idx,
        const uint32_t interval_idx, const bool is_end) {
        const uint32_t kind_base = kind == Compute ?
            kComputeOffset : kCommunicationOffset;
        return kind_base +
            (warp_idx * kMaxIntervalsPerWarp + interval_idx) * 2 +
            static_cast<uint32_t>(is_end);
    }

    CUTLASS_HOST_DEVICE
    static constexpr uint32_t get_counter_offset(
        const IntervalKind kind, const uint32_t warp_idx) {
        return kCounterOffset +
            static_cast<uint32_t>(kind) * kMaxWarps + warp_idx;
    }
};

CUTLASS_DEVICE
uint64_t read_globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

} // namespace deep_gemm::profile
