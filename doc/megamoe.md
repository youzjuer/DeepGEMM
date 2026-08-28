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

* expert_send_count

`expert_send_count[i]`： **本 rank 打算发送多少个 token 到全局第 i 个 expert** 。

| 位段     | 含义                                                     |
| -------- | -------------------------------------------------------- |
| 低 32 位 | token 数（本 rank 路由到该 expert 的 token-topk 对数量） |
| 高 32 位 | 已上报的 SM 个数（每个 SM 贡献 +1）                      |

* `get_src_token_topk_idx_ptr`

src_token_topk_idx，对应三维数组 `[local_ep][src_rank][slot]`，也就是每个expert必须要知道每个token的原始地址，

* expert_recv_count_ptr

**本 rank 的第 `local_ep` 个 expert，将从源 rank `j` 收到多少 token** 。

## 1.3 实例

用一个小到能手算的配置： **2 rank、4 expert（每 rank 2 个）、topk=2、每 rank 4 token、BLOCK_M=2、每 rank 2 个 SM** 。

E0/E1 在 rank0，E2/E3 在 rank1。只追踪  **E0** （rank0 的 local_ep=0）。

### ① 路由结果

| rank0 token | topk0        | topk1        |  | rank1 token | topk0        | topk1        |
| ----------- | ------------ | ------------ | - | ----------- | ------------ | ------------ |
| t0          | **E0** | E2           |  | t0          | E2           | **E0** |
| t1          | E1           | E2           |  | t1          | **E0** | E1           |
| t2          | **E0** | E3           |  | t2          | E3           | E2           |
| t3          | E2           | **E0** |  | t3          | E2           | E1           |

发往 E0 的：rank0 有 3 个（t0k0、t2k0、t3k1），rank1 有 2 个（t0k1、t1k0）。

### ② SM 级 atomic：分配基址

rank0 内 SM_a 处理 t0/t1，SM_b 处理 t2/t3。各自先在 shared memory 数出本地 E0 计数：SM_a=1，SM_b=2。

然后各做一次 `atomicAdd(expert_send_count[E0], (1<<32)|local_count)`。 **假设 SM_b 先到** ：

| 执行顺序 | 加的值         | 返回的旧值     | 低32位 = 本 SM 基址        |
| -------- | -------------- | -------------- | -------------------------- |
| SM_b     | `(1<<32)｜2` | `0`          | **0** → 占 slot 0,1 |
| SM_a     | `(1<<32)｜1` | `(1<<32)｜2` | **2** → 占 slot 2   |

最终 `expert_send_count[E0] = (2<<32)｜3` —— 高位 2 表示"2 个 SM 已上报"，低位 3 是 token 数。

注意：**SM_a 明明处理的是编号更小的 t0，却拿到了更靠后的 slot 2** —— slot 顺序由 atomic 竞争决定，与 token 编号、SM 编号都无关。

### ③ 写元数据

存的值是 `token_topk_idx = token_idx * kNumTopk + topk_idx`。

rank0 写自己（`dst_rank = E0/2 = 0`），落在泳道 `[ep0][src_rank=0]`：

| slot | 来自         | 计算   | 存入        |
| ---- | ------------ | ------ | ----------- |
| 0    | SM_b 的 t2k0 | 2×2+0 | **4** |
| 1    | SM_b 的 t3k1 | 3×2+1 | **7** |
| 2    | SM_a 的 t0k0 | 0×2+0 | **0** |

rank1  **跨 NVLink 写进 rank0 的 buffer** ，落在泳道 `[ep0][src_rank=1]`：

| slot | 来自 | 计算   | 存入        |
| ---- | ---- | ------ | ----------- |
| 0    | t0k1 | 0×2+1 | **1** |
| 1    | t1k0 | 1×2+0 | **2** |

两条泳道物理隔离，所以 rank0 和 rank1 可以完全并发写，零竞争。rank0 上的最终状态：

```text
src_token_topk_idx[ep0][rank0] = [4, 7, 0]    expert_recv_count[rank0][ep0] = 3
src_token_topk_idx[ep0][rank1] = [1, 2]       expert_recv_count[rank1][ep0] = 2
                                              expert_recv_count_sum[ep0] = (4<<32)｜5
```

高位 4 = `kNumSMs × kNumRanks` = 2×2 ✓（校验通过）。低位 5 就是 **E0 的 GEMM M 值** → `ceil(5/2) = 3` 个 M block。

### ③.5

rank0 和 rank1 各自的 SM0 把自己的 `expert_send_count` 低 32 位， **跨卡投递到目标 rank 的 `expert_recv_count` 里自己那一行** ：

* rank0 的 SM0：读本地 `expert_send_count[E0] = (2<<32)|3` → 取低位 3 → 写到 rank0（自己）的 `expert_recv_count[rank0][ep0] = 3`
* rank1 的 SM0：读本地 `expert_send_count[E0] = (?<<32)|2` → 取低位 2 → 跨 NVLink 写到 **rank0** 的 `expert_recv_count[rank1][ep0] = 2`

这一步为什么必须在 `grid_sync`（L539）之后：`expert_send_count[E0]` 要等 SM_a 和 SM_b **都**完成 atomicAdd 才等于 3。SM0 若提前读，可能只看到 2。这就是那次 grid sync 存在的理由之一。

### ④ 反查：min-peeling 把 slot n 解回源坐标

`remaining = [3, 2]`，调度器依次分配 n = 0..4：

 **Round 1** ：`num_active_ranks=2`，`length=min(3,2)=2`，`num_round_tokens=4` → n=0..3 命中

| n | `n % 2` → rank   | `offset + n/2` → 泳道内 idx | 读出 | 解码   |
| - | ------------------- | ------------------------------ | ---- | ------ |
| 0 | 0 →**rank0** | 0+0 = 0                        | 4    | t2, k0 |
| 1 | 1 →**rank1** | 0+0 = 0                        | 1    | t0, k1 |
| 2 | 0 →**rank0** | 0+1 = 1                        | 7    | t3, k1 |
| 3 | 1 →**rank1** | 0+1 = 1                        | 2    | t1, k0 |

n=4 未命中 → `slot_idx -= 4` 变 0，`offset += 2` 变 2，`remaining = [1, 0]`

 **Round 2** ：`num_active_ranks=1`（只剩 rank0），`length=1`

| n | rank                | 泳道内 idx | 读出 | 解码   |
| - | ------------------- | ---------- | ---- | ------ |
| 4 | 0 →**rank0** | 2+0 = 2    | 0    | t0, k0 |

### ⑤ 效果

稠密 pool（`pool_token_idx = 0×2 + n`，即 `l1_token_buffer` 的第 0..4 行）：

```text
slot:      0        1        2        3        4
来源:   rank0    rank1    rank0    rank1    rank0
        ↑______________↑ ↑______________↑ ↑___
         M block 0        M block 1      block 2(半空)
```

* **交织生效** ：相邻 slot 轮转不同 rank，所以 block0 的两次 TMA pull 一条走本地 HBM、一条走 NVLink 到 rank1 —— 若按 rank 拼接（`[rank0,rank0,rank0,rank1,rank1]`），block0 的两次 pull 会全压在本地，block1 全压在 rank1 链路上。
* **纯算术反查** ：整个 ④ 只有除、模、`__reduce_min_sync`，没有查表、没有前缀和、没有额外 grid sync。
* **combine 回写** ：slot 0 解出 (rank0, t2, k0)，结果就写回 rank0 的 `combine_token_buffer[topk=0][token=2]`；slot 1 解出 (rank1, t0, k1) → 跨 NVLink 写 rank1 的 `[topk=1][token=0]`。不同 slot 的目标地址天然不重叠，所以回写无需 barrier。

### 对照：如果不用 slot 而按源 token 编号索引

`[ep0][rank0]` 泳道会变成长度 4（源 token 数）的稀疏数组：

```text
idx:   0      1      2      3
      t0k0    —     t2k0   t3k1     ← 中间有洞
```

于是：M 算不出来（需要 popcount 归约）；"第 n 个有效元素在哪"无法用除模表达（需要前缀和 + 二分）；GEMM tile 里混着无效行，MMA 空转。**这三件事正是 slot 这层抽象一次性解决的。**
