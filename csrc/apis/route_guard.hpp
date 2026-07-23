#pragma once

#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace deep_gemm::route_guard {

void accumulate_route_histogram(
    const torch::Tensor& topk_idx,
    const torch::Tensor& expert_counts);

void accumulate_route_loads(
    const torch::Tensor& topk_idx,
    const torch::Tensor& routing_choices,
    const torch::Tensor& routing_counts,
    const torch::Tensor& route_loads,
    int rank_idx,
    int master_experts_per_rank,
    int physical_experts_per_rank);

inline void register_apis(pybind11::module_& m) {
    m.def(
        "accumulate_route_histogram",
        &accumulate_route_histogram,
        "Accumulate CUDA int64 logical expert routes into CUDA int32 counters");
    m.def(
        "accumulate_route_loads",
        &accumulate_route_loads,
        "Accumulate baseline and persistent-replica rank loads on CUDA");
}

}  // namespace deep_gemm::route_guard
