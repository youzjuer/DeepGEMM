import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


def run_scheduler_test(local_rank: int, num_processes: int) -> None:
    rank, world_size, group = init_dist(local_rank, num_processes)
    assert world_size == 2

    num_logical_experts = 8
    master_experts_per_rank = 4
    physical_experts_per_rank = 5
    choices = torch.empty((num_logical_experts, 2), dtype=torch.int32, device="cuda")
    counts = torch.ones(num_logical_experts, dtype=torch.int32, device="cuda")
    logical_ids = torch.arange(num_logical_experts, device="cuda")
    master_ids = (
        torch.div(logical_ids, master_experts_per_rank, rounding_mode="floor")
        * physical_experts_per_rank
        + logical_ids % master_experts_per_rank
    )
    choices[:, 0].copy_(master_ids)
    choices[:, 1].copy_(master_ids)
    choices[0].copy_(torch.tensor([0, 9], dtype=torch.int32, device="cuda"))
    counts[0] = 2

    skew_topk = torch.zeros((4, 2), dtype=torch.int64, device="cuda")
    uniform_topk = torch.tensor(
        [[0, 4], [0, 4], [0, 4], [0, 4]], dtype=torch.int64, device="cuda"
    )
    logical_topk = skew_topk.clone()
    marker = torch.zeros(1, dtype=torch.int32, device="cuda")

    guard = deep_gemm.PersistentReplicaRouteGuard(
        num_logical_experts=num_logical_experts,
        master_experts_per_rank=master_experts_per_rank,
        physical_experts_per_rank=physical_experts_per_rank,
        expert_routing_map=(choices, counts),
        group=group,
        window_size=1,
        samples_per_window=1,
        enter_threshold=1.25,
        exit_threshold=1.15,
        min_predicted_relief=0.01,
    )
    scheduler = deep_gemm.PersistentReplicaAutoScheduler(
        guard=guard,
        logical_topk_idx=logical_topk,
        baseline_step=lambda: marker.fill_(1),
        replica_step=lambda: marker.fill_(2),
    )
    scheduler.capture(warmup_iterations=1)

    # Window 0 observes skew on the baseline path and enables replicas for the
    # next window. Window 1 then executes the captured replica path.
    first = scheduler.run()
    assert not first.used_replica
    assert first.use_replica_next_window
    assert marker.item() == 1
    assert first.metrics is not None
    assert first.metrics.rank_loads == (16, 0)
    assert first.metrics.predicted_replica_rank_loads == (8.0, 8.0)

    second = scheduler.run()
    assert second.used_replica
    assert second.use_replica_next_window
    assert marker.item() == 2

    # The balanced window still uses the previous replica decision, then
    # disables replicas for the following window.
    logical_topk.copy_(uniform_topk)
    third = scheduler.run()
    assert third.used_replica
    assert not third.use_replica_next_window
    assert marker.item() == 2
    assert third.metrics is not None
    assert third.metrics.rank_loads == (8, 8)

    fourth = scheduler.run()
    assert not fourth.used_replica
    assert not fourth.use_replica_next_window
    assert marker.item() == 1

    logical_topk.fill_(-1)
    fifth = scheduler.run()
    assert not fifth.used_replica
    assert not fifth.use_replica_next_window
    assert fifth.metrics is not None
    assert fifth.metrics.window_routes == 0
    assert fifth.metrics.rank_loads == (0, 0)

    decisions = torch.tensor(
        [
            int(first.use_replica_next_window),
            int(second.use_replica_next_window),
            int(third.use_replica_next_window),
            int(fourth.use_replica_next_window),
            int(fifth.use_replica_next_window),
        ],
        dtype=torch.int32,
        device="cuda",
    )
    gathered = [torch.empty_like(decisions) for _ in range(world_size)]
    dist.all_gather(gathered, decisions, group=group)
    assert all(torch.equal(value, decisions) for value in gathered)

    if rank == 0:
        print("scheduler_cuda_graph=true", flush=True)
        print(
            "previous_window_transition=baseline,replica,replica,baseline,baseline",
            flush=True,
        )
        print("all_masked_routes_safe=true", flush=True)
        print("all_rank_decisions_identical=true", flush=True)
    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2)
    args = parser.parse_args()
    if args.num_processes != 2:
        raise ValueError("this test currently requires exactly two processes")
    mp.spawn(run_scheduler_test, args=(args.num_processes,), nprocs=args.num_processes)
