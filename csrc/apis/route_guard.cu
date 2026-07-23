#include "route_guard.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>

namespace deep_gemm::route_guard {

namespace {

__global__ void accumulate_route_histogram_kernel(
    const int64_t* topk_idx,
    const int64_t num_routes,
    int32_t* expert_counts,
    const int32_t num_experts) {
    const int64_t route_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (route_idx >= num_routes)
        return;

    const int64_t expert_idx = topk_idx[route_idx];
    // Negative IDs are masked routes.  Positive out-of-range IDs are ignored
    // here and remain the MegaMoE launcher's responsibility to reject.
    if (expert_idx >= 0 && expert_idx < num_experts)
        atomicAdd(expert_counts + expert_idx, 1);
}

__global__ void accumulate_route_loads_kernel(
    const int64_t* topk_idx,
    const int64_t num_routes,
    const int32_t num_tokens,
    const int32_t num_topk,
    const int32_t* routing_choices,
    const int32_t* routing_counts,
    const int32_t num_experts,
    const int32_t max_instances,
    int32_t* route_loads,
    const int32_t num_ranks,
    const int32_t rank_idx,
    const int32_t master_experts_per_rank,
    const int32_t physical_experts_per_rank) {
    const int64_t route_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (route_idx >= num_routes)
        return;

    const int64_t expert_idx_64 = topk_idx[route_idx];
    if (expert_idx_64 < 0 || expert_idx_64 >= num_experts)
        return;
    const auto expert_idx = static_cast<int32_t>(expert_idx_64);
    const int32_t baseline_rank = expert_idx / master_experts_per_rank;
    if (baseline_rank < 0 || baseline_rank >= num_ranks)
        return;
    atomicAdd(route_loads + baseline_rank, 1);

    const int32_t num_instances = routing_counts[expert_idx];
    if (num_instances <= 0 || num_instances > max_instances)
        return;
    int32_t instance_idx = 0;
    if (num_instances > 1) {
        const uint32_t token_idx = static_cast<uint32_t>(route_idx / num_topk);
        const uint32_t global_token_idx =
            static_cast<uint32_t>(rank_idx * num_tokens) + token_idx;
        instance_idx = static_cast<int32_t>(
            global_token_idx % static_cast<uint32_t>(num_instances));
        atomicAdd(route_loads + 2 * num_ranks, 1);
    }
    const int32_t physical_expert =
        routing_choices[expert_idx * max_instances + instance_idx];
    const int32_t replica_rank = physical_expert / physical_experts_per_rank;
    if (physical_expert >= 0 && replica_rank >= 0 && replica_rank < num_ranks)
        atomicAdd(route_loads + num_ranks + replica_rank, 1);
}

}  // namespace

void accumulate_route_histogram(
    const torch::Tensor& topk_idx,
    const torch::Tensor& expert_counts) {
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(expert_counts.is_cuda(), "expert_counts must be a CUDA tensor");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64,
                "topk_idx must have dtype int64");
    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32,
                "expert_counts must have dtype int32");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
    TORCH_CHECK(expert_counts.is_contiguous(),
                "expert_counts must be contiguous");
    TORCH_CHECK(expert_counts.dim() == 1,
                "expert_counts must be one-dimensional");
    TORCH_CHECK(topk_idx.get_device() == expert_counts.get_device(),
                "topk_idx and expert_counts must be on the same CUDA device");
    TORCH_CHECK(expert_counts.numel() > 0,
                "expert_counts must contain at least one expert");
    TORCH_CHECK(expert_counts.numel() <= INT32_MAX,
                "expert_counts contains too many experts");

    const auto num_routes = topk_idx.numel();
    if (num_routes == 0)
        return;

    constexpr int32_t kNumThreads = 256;
    const auto num_blocks = static_cast<int32_t>(std::min<int64_t>(
        (num_routes + kNumThreads - 1) / kNumThreads, 4096));
    const c10::cuda::CUDAGuard device_guard(topk_idx.device());
    const auto stream = at::cuda::getCurrentCUDAStream(topk_idx.get_device());
    accumulate_route_histogram_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        num_routes,
        expert_counts.data_ptr<int32_t>(),
        static_cast<int32_t>(expert_counts.numel()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void accumulate_route_loads(
    const torch::Tensor& topk_idx,
    const torch::Tensor& routing_choices,
    const torch::Tensor& routing_counts,
    const torch::Tensor& route_loads,
    const int rank_idx,
    const int master_experts_per_rank,
    const int physical_experts_per_rank) {
    TORCH_CHECK(topk_idx.is_cuda() && routing_choices.is_cuda() &&
                    routing_counts.is_cuda() && route_loads.is_cuda(),
                "all route guard tensors must be CUDA tensors");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64,
                "topk_idx must have dtype int64");
    TORCH_CHECK(routing_choices.scalar_type() == torch::kInt32 &&
                    routing_counts.scalar_type() == torch::kInt32 &&
                    route_loads.scalar_type() == torch::kInt32,
                "routing map and route_loads must have dtype int32");
    TORCH_CHECK(topk_idx.is_contiguous() && routing_choices.is_contiguous() &&
                    routing_counts.is_contiguous() && route_loads.is_contiguous(),
                "all route guard tensors must be contiguous");
    TORCH_CHECK(topk_idx.dim() == 2,
                "topk_idx must have shape [tokens, top_k]");
    TORCH_CHECK(routing_choices.dim() == 2 && routing_counts.dim() == 1,
                "invalid routing map shape");
    TORCH_CHECK(routing_choices.size(0) == routing_counts.numel(),
                "routing choices and counts disagree");
    TORCH_CHECK(topk_idx.get_device() == routing_choices.get_device() &&
                    topk_idx.get_device() == routing_counts.get_device() &&
                    topk_idx.get_device() == route_loads.get_device(),
                "all route guard tensors must share a CUDA device");
    TORCH_CHECK(master_experts_per_rank > 0 && physical_experts_per_rank > 0,
                "expert counts per rank must be positive");
    const int64_t num_experts = routing_counts.numel();
    TORCH_CHECK(num_experts % master_experts_per_rank == 0,
                "logical expert count must be divisible by experts per rank");
    const int64_t num_ranks = num_experts / master_experts_per_rank;
    TORCH_CHECK(rank_idx >= 0 && rank_idx < num_ranks,
                "rank_idx is outside the EP group");
    TORCH_CHECK(route_loads.dim() == 1 &&
                    route_loads.numel() == 2 * num_ranks + 1,
                "route_loads must have 2 * num_ranks + 1 elements");
    TORCH_CHECK(topk_idx.size(0) <= INT32_MAX && topk_idx.size(1) <= INT32_MAX,
                "topk_idx shape exceeds int32 range");

    const auto num_routes = topk_idx.numel();
    if (num_routes == 0)
        return;
    constexpr int32_t kNumThreads = 256;
    const auto num_blocks = static_cast<int32_t>(std::min<int64_t>(
        (num_routes + kNumThreads - 1) / kNumThreads, 4096));
    const c10::cuda::CUDAGuard device_guard(topk_idx.device());
    const auto stream = at::cuda::getCurrentCUDAStream(topk_idx.get_device());
    accumulate_route_loads_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        num_routes,
        static_cast<int32_t>(topk_idx.size(0)),
        static_cast<int32_t>(topk_idx.size(1)),
        routing_choices.data_ptr<int32_t>(),
        routing_counts.data_ptr<int32_t>(),
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(routing_choices.size(1)),
        route_loads.data_ptr<int32_t>(),
        static_cast<int32_t>(num_ranks),
        rank_idx,
        master_experts_per_rank,
        physical_experts_per_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace deep_gemm::route_guard
