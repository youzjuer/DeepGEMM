# 一 名词解释

## 1.1 get_symm_buffer_for_mega_moe

### 1.1.1 位置

* **多卡时** ：用 `torch.distributed._symmetric_memory.empty(num_bytes, device='cuda')` 分配，然后 `symm_mem.rendezvous(buffer, group)` 建立握手 —— 每个 rank 在**自己的显存**里分配一块**大小、布局完全相同**的 buffer，rendezvous 后拿到 `handle.buffer_ptrs`（所有 rank 的 buffer 指针数组），使得 kernel 里任一 CTA 可以通过 NVLink  **直接读写其他 rank 的这块显存** 。
* **单卡时** （`group.size()==1`）：退化为普通 `torch.empty`，`buffer_ptrs` 只含本地指针。

## 1.2 SymmBuffer

| 变量                  | 类型             | shape                                 | 作用                                                                                                                                                                               |
| --------------------- | ---------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| grid_sync_count       | uint32           | [4]                                   | 同步相关                                                                                                                                                                           |
| nvl_barrier_counter   | uint32           | [1]                                   | 同步相关                                                                                                                                                                           |
| nvl_barrier_signal    | int              | [2]                                   | 同步相关                                                                                                                                                                           |
| expert_send_count     | uint64           | [kNumExperts]                         | [ep_idx]表示当前rank会发送多少token到第ep_idx个专家                                                                                                                                |
| expert_recv_count_ptr | uint64           | [kNumRanks × E_per_rank]             | [src_rank][local_ep_idx]表示当前rank的local_ep_idx会从src_rank收到多少token                                                                                                        |
| expert_recv_count_sum | uint64           | [E_per_rank]                          | [local_ep_idx]表示当前rank的第local_ep_idx个专家共收到多少个token                                                                                                                  |
| l1_arrival_count      | uint32           | [num_max_pool_blocks]                 | 用于同步dispatch pull和L1，[BLOCK_M, hidden]大小对应一个flag                                                                                                                       |
| l2_arrival_mask       | uint64           | [num_max_pool_blocks]                 | 用于同步L1 epilogue和L2                                                                                                                                                            |
| src_token_topk_idx    | uint32           | [E_per_rank × kNumRanks × max_recv] | [local_ep_idx][src_rank][token_idx]，表示当前rank的local_ep_idx中src_rank发送过来的所有token中，第token_idx个token对应src_rank的topk矩阵中的哪个元素（num_token x num_topk个元素） |
| token_src_metadata    | TokenSrcMetadata | [kNumMaxPoolTokens]                   | Combine回写需要的源信息{rank_idx, token_idx, topk_idx}                                                                                                                             |
