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

## 1.3 实例1

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

## 1.4 实例2

追踪 **rank0 的 t0** 这一个 token 走完全程。它的路由是 `[E0, E2]` —— 两个副本去两块**不同的** GPU，最后两个结果又要汇合回 rank0，正好把余数的作用暴露出来。

### 起点：rank0 上 t0 的两行元数据

```text
rank0.topk_idx[t0]     = [ E0 ,  E2 ]      展平下标 =  0 ,  1
rank0.topk_weights[t0] = [0.7 , 0.3 ]
```

需要注意topk_weights和weights是两码事

### 阶段 1：dispatch —— 两份记录分道扬镳

```text
token_topk_idx = 0  ─→ expert E0 ─→ dst_rank = 0/2 = 0 ─→ 写 rank0 的 src_token_topk_idx[ep0][rank0][slot=2] = 0
token_topk_idx = 1  ─→ expert E2 ─→ dst_rank = 2/2 = 1 ─→ 写 rank1 的 src_token_topk_idx[ep0][rank0][slot=0] = 1
                                                  ↑ 跨 NVLink
```

两块 GPU 各拿到一个 uint32：rank0 拿到 `0`，rank1 拿到 `1`。

### 阶段 2：pull —— 商相同，余数不同

|              | rank0 处理 E0                       | rank1 处理 E2                       |
| ------------ | ----------------------------------- | ----------------------------------- |
| 读到的值     | `0`                               | `1`                               |
| 商 =`/2`   | **0**                         | **0** ← 相同                 |
| 余数 =`%2` | **0**                         | **1** ← 不同                 |
| 拉的 hidden  | rank0 的 `x[t0]`（本地）          | rank0 的 `x[t0]`（跨 NVLink）     |
| 取的权重     | `topk_weights[0]` = **0.7** | `topk_weights[1]` = **0.3** |

**两边拉的是同一份 hidden 向量** —— 因为商一样。整个 pull 过程里，余数唯一的作用就是让权重取对了那一列。

### 阶段 3：存 metadata —— 只有一个字段不同

```cpp
// rank0 上
token_src_metadata[pool行 a] = { rank_idx: 0, token_idx: 0, topk_idx: 0 }
// rank1 上
token_src_metadata[pool行 b] = { rank_idx: 0, token_idx: 0, topk_idx: 1 }
                               └────── 完全相同 ──────┘   └── 唯一区别 ──┘
```

### 阶段 4：combine —— 余数选定写回的那一维

两个 GPU 各自算完自己的 GEMM，互不知晓、时间上完全不协调：

```text
rank0 (E0 的输出)                          rank1 (E2 的输出)
      │                                          │
      │ get_rank_buffer(topk_idx=0)              │ get_rank_buffer(topk_idx=1)
      │ get_data_buffer(token_idx=0)             │ get_data_buffer(token_idx=0)
      │                                          │ 跨 NVLink
      ▼                                          ▼
┌──────────────── rank0 的 combine_token_buffer ────────────────┐
│                     t0      t1     t2     t3                  │
│  topk_slot 0    [ E0输出 ][    ][    ][    ]  ← rank0 本地写   │
│  topk_slot 1    [ E2输出 ][    ][    ][    ]  ← rank1 远程写   │
└───────────────────────────────────────────────────────────────┘
              ↑ 两个地址不同 → 无冲突、无 barrier
```

下游按权重求和：

y[t0]=0.7×buf[0][t0]+0.3×buf[1][t0]

### 如果去掉余数，只存 token index

两条 metadata 变成 `{rank:0, token:0}` 和 `{rank:0, token:0}` ——  **一模一样** 。回写地址都是 `buf[?][0]`：

```text
rank0 写 buf[0][t0] = E0输出
rank1 写 buf[0][t0] = E2输出      ← 覆盖！
                ↑ 同一个地址
```

后果有三层：

1. **结果丢失** ：t0 只剩一个 expert 的贡献，另一个被覆盖，输出错误。
2. **非确定性** ：哪个 GPU 先写不确定，同样的输入每次跑出不同结果 —— 这类 bug 最难查。
3. **无法用同步救** ：写入是 `float4` 的 bf16 向量，没有对应的原子加指令；改成跨卡加锁的话，每个输出行都要一次远程 CAS，性能直接崩掉。

而且即便侥幸不覆盖，**权重也会配错** —— `topk_weights[t0][0] = 0.7` 是给 E0 的，若 E2 的结果落在 slot 0，下游就会用 0.7 去乘 E2 的输出。

## 1.5 scheduler

MoE计算过程分成多个wave，一个wave完整处理多个expert，一个wave包含两个phase，第一个是L1，第二个是L2，先处理完成一个expert，再处理下一个expert，同一个expert内部按照先N方向后M方向的顺序遍历，所有sm按照sm_id得到自己要处理的block。

get_num_experts_per_wave_for_mega_moe会计算出一个wave是多少个expert，基本原理就是在能打满所有sm的前提下，expert越少越好，这样可以让L2尽快开始。

# 二 整体流程

## 2.1 流程简述

整条流水线分成四段：**dispatch → L1 → L2 → combine**。段间衔接只有两种手段 —— 跨 rank 用 NVLink barrier，同 rank 内用 workspace 上的 full / empty 计数器做生产者-消费者握手。

```text
【kernel 外】pre_dispatch
     bf16 token ──量化──▶ input_token_buffer{ x, x_sf, topk_idx, topk_weights }
     · 数据留在本 rank 等人来拉，不主动推送

【dispatch】
  1. 源 rank 推 metadata 到远端
        src_token_topk_idx + expert_recv_count / expert_recv_count_sum
  2. NVLink barrier ── kBeforeDispatchPullBarrierTag
  3. 目标 rank 按 metadata pull token / SF / weight
        一个 token 只对应一个源 rank；round-robin 在 token 之间摊开以打满 NVLink
  4. 写入 L1 token ring pool（per-expert 按 BLOCK_M 对齐）
        ──l1_full_count──▶ 通知 L1 可以开算

【L1】
  5. L1 GEMM ── FP8×FP4 或 FP4×FP4
  6. L1 epilogue ── SwiGLU → × route weight → 量化(FP8 或 FP4)
  7. 写入 L2 token ring pool
        ──l2_full_count──▶ 通知 L2 可以开算
        ──l1_empty_count──▶ 回收 L1 槽位

【L2】
  8. L2 GEMM
        ──l2_empty_count──▶ 回收 L2 槽位
  9. L2 epilogue ── 累加器转 BF16，按 token_src_metadata 推回
        源 rank 的 combine_token_buffer[topk_slot][token]
 10. NVLink barrier ── kBeforeCombineReduceBarrierTag

【combine】
 11. 源 rank 本地 reduce ── 遍历 topk_slot，FP32 累加
 12. BF16 cast → TMA store → y[token, hidden]
```

分段对照：

| 阶段 | 谁在做 | 产物落在哪 | 与下一段的衔接 |
| --- | --- | --- | --- |
| pre_dispatch | 独立 kernel（在 MegaMoE 之外） | 本 rank `input_token_buffer` | —— |
| dispatch 推 metadata | 源 rank 的 dispatch warp | 远端 `src_token_topk_idx`、`expert_recv_count(_sum)` | NVLink barrier |
| dispatch pull | 目标 rank 的 dispatch warp | `l1_token_buffer` / `l1_sf_buffer` / `l1_topk_weights_buffer`，外加 `token_src_metadata` | `l1_full_count` |
| L1 GEMM + epilogue | MMA warp + epilogue warpgroup | `l2_token_buffer` / `l2_sf_buffer` | `l2_full_count` 前进、`l1_empty_count` 回收 |
| L2 GEMM + epilogue | MMA warp + epilogue warpgroup | 源 rank 的 `combine_token_buffer[topk_slot][token]` | NVLink barrier、`l2_empty_count` 回收 |
| combine reduce | epilogue warp | `y[token, hidden]` | —— |

三个容易看漏的点：

- **route weight 在第 6 步就乘掉了**，不在 combine 阶段。所以 `combine_token_buffer` 里存的已经是加权后的 partial，最后的 reduce 退化成纯求和。
- **combine 是「推」不是「拉」**：L2 epilogue 直接写远端地址，返程地址来自第 3 步顺手记下的 `token_src_metadata{rank_idx, token_idx, topk_idx}`。
- **L1 / L2 都是 ring buffer**，不是全量池；槽位靠 empty 计数器循环回收，所以 full 计数器的目标值会跨 generation 累积，而不是每轮清零。
