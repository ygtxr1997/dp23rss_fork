# 学习更稳定、任务相关的 Quadtree Topology

## 1. 当前状态与目标

当前 PushT 主配置使用：

```text
256 x 256 RGB
-> 64 x 64 input
-> from-scratch ResNet10
-> NodeAlignedFPNRouter
-> quadtree topology
-> FoveationObsEncoder
-> diffusion policy
```

当前几何：

- patch grids：`4 x 4`、`8 x 8`、`16 x 16`。
- internal nodes：`16 + 64 = 80`。
- leaf 范围：16～256。
- `target_num_patches=0.25` 对应 76 leaves，即恰好 20 次 reachable split。

EMA 对 trainable/frozen global encoder 的处理已经修复，不再列入本计划。

最终目标不是单纯提高 `topology_div`，而是：

1. 训练早期能探索多种合法 topology。
2. 训练后期不同 task-relevant 输入产生稳定、不同的 topology。
3. learned topology 在相同 hard budget 下优于 random/fixed topology 的
   diffusion、action 和 rollout 表现。

## 2. 剩余核心问题

### 2.1 当前 topology 指标不可直接比较

- `random:0.25` 在 eval 下仍然随机采样。
- learned router 在 eval 下使用确定性的 `logit > 0`。

所以 random 的 `topology_div` 主要反映 sampling exploration，learned 的指标主要反映
deterministic input conditioning。两者需要拆开记录。

### 2.2 Expected budget 与 hard decoder 错位

若所有节点输出相同 probability，当前 expected leaf count 为：

```text
E[num_leaves] = 16 + 48p + 192p^2
```

当 `p ~= 0.448` 时 expected count 正好为 76，但对应 logit 小于 0，eval 会全部 stop，
得到 16-leaf coarsest topology。即：

```text
budget loss = 0
eval topology_div = 0
```

因此应让 hard decoder 直接满足 budget，而不是依赖每个节点的 0 threshold。

### 2.3 Temperature 不控制 hard exploration

训练 hard decision：

```text
sigmoid((logit + noise) / temperature) > 0.5
```

等价于：

```text
logit + noise > 0
```

temperature 只改变 surrogate gradient。真正的 exploration 需要独立的 noise scale。

### 2.4 Task gradient 的 counterfactual credit 很弱

当前 hard gather 只把已选择的 parent 或 children 送到下游。未选择分支没有 action
gradient，因此 router 很难比较 split 与 stop 哪个更有利，容易 early lock-in。

此外，当前 router 看不到 `agent_pos`，两个 observation frames 也独立采样 topology。

## 3. 实施阶段

### 阶段 1：拆分诊断指标

先将 routing stochasticity 与 `module.training` 解耦，使 model 保持 `eval()` 时也能显式
请求 stochastic routing，避免同时改变 BN/dropout。

至少记录：

```text
sample/topology_div_input_live
sample/topology_div_input_ema
sample/topology_div_stochastic
sample/topology_div_repeat
```

含义：

- `input_live/input_ema`：deterministic routing 下不同输入之间的差异。
- `stochastic`：不同输入各采样一次后的差异。
- `repeat`：同一输入重复采样 `S=4` 次的差异，直接反映 exploration。

同时保留少量关键诊断：

```text
router/logit_inter_sample_std
router/hard_sign_agreement
router/split_entropy
router/clean_expected_leaf_count
router/train_hard_leaf_count_mean/std
router/eval_hard_leaf_count_mean/std
grad/router_diffusion_grad_norm
grad/router_budget_grad_norm
grad/router_diffusion_budget_cosine
```

成功标准：能够明确区分输入无关随机性、deterministic collapse 和 threshold calibration
问题；live/EMA 指标只存在正常 EMA lag。

### 阶段 2：Exact-budget deterministic decoder

根据目标 leaf count 计算固定 split 数：

```text
target_num_splits = round((target_num_patches - min_num_patches) / 3)
```

当前配置得到 20。

eval 使用 greedy reachable decoding：

1. eligible set 初始化为 level-0 internal nodes。
2. 选择 eligible nodes 中 logit 最大的节点。
3. split 后暴露其 internal children。
4. 重复直到完成 20 次 split。

这样每个 topology 都严格为 76 leaves，并且使用 logits 的相对排序，不受共同负 bias 或
0 threshold 影响。

先在修复 EMA 后的 checkpoint 上比较：

```text
zero-threshold decoder
exact-budget greedy decoder
```

如果 greedy decoder 已显著提高 deterministic input diversity，说明当前 router 可能已经
学到相对节点排序，主要问题只是 decoder calibration。

时间维主实验使用 `shared_topology`：对同一样本的两个 observation frames 的 logits
取 mean，生成一棵 topology 后扩展到两个 frames。这样与 `random:0.25` 的时间语义一致，
也减少无意义的 history jitter。

第一轮只改 decoder，继续保留现有 clean expected budget。若 exact hard budget 稳定后
clean budget 仍把 logits 推向共同 probability，再单独将 `budget_loss_weight` 设为 0。

### 阶段 3：Budget-constrained stochastic sampler

训练时使用与阶段 2 相同的 reachable growth，但节点分数为：

```text
score = split_logit + routing_noise_scale(step) * gumbel_noise
```

要求：

- 每棵训练 topology 合法且严格满足 20 splits/76 leaves。
- `routing_noise_scale` 与 `router_temperature` 分离。
- early noise 较大，late 逐步降至 0。
- 同一样本的两个 frames 共享 topology。

建议配置：

```yaml
training:
  router_noise_scale_start: 1.0
  router_noise_scale_end: 0.0
  router_noise_scale_steps: 100000
  router_noise_scale_schedule: cosine
```

第一版继续使用现有 soft leaf/ST backward，只替换 hard sampler。若 diffusion gradient
仍很弱，再升级 estimator，不在首版同时实现复杂 structured relaxation。

只有 logits 很早饱和、noise schedule 仍无法维持探索时，才增加 reach-weighted entropy
bonus，并在训练后期退火到 0。不要直接最大化 batch Jaccard distance，因为这会奖励
输入无关随机树。

成功标准：

- early `topology_div_repeat > 0`。
- hard leaf count 全程严格为 76。
- noise 下降时 repeat diversity 平滑下降。
- deterministic input diversity 随训练增加，而不是随 exploration 一起消失。

### 阶段 4：改善 Task Credit Assignment

只有 exact-budget sampler 工作、router logits 会随输入变化，但 learned topology 仍不能
优于 random topology 时，才进入本阶段。

优先尝试 random-topology curriculum：

```text
A. 用 random:0.25 训练 FoveationObsEncoder + diffusion policy
B. 加载下游权重，启用 trainable global encoder + router
C. joint fine-tune，并逐渐降低 routing noise
```

这可以让下游先适应多种 topology，减少某棵早期随机树因共同适应而被持续强化。

如果仍不稳定，对同一输入采样 `K=2` 或 `K=4` 棵 exact-budget topology：

- 使用相同 diffusion timestep 和 diffusion noise。
- 分别计算 per-sample diffusion loss。
- 用同一输入的平均 loss 作为 baseline。
- 使用 relative advantage 更新 router。

共同 diffusion noise 可以减少 topology 差异与 diffusion 随机性的混淆。

更昂贵的 soft parent-vs-children counterfactual mixture 仅作为后续方案，不作为第一版。

### 阶段 5：Conditioning 和结构消融

在前四阶段可信后，再依次做：

1. 将 normalized `agent_pos` 传给 router；不使用推理时不可获得的 future action。
2. 比较 per-frame 与 shared-topology temporal routing。
3. FPN 增加 coordinate/position feature。
4. 两个 internal levels 使用独立 `1 x 1` heads，但层内保持空间共享。
5. global encoder、router、downstream visual encoder 使用独立 LR multiplier；bias/norm
   使用 `weight_decay=0`。
6. 最后再比较更高 router 输入分辨率、预训练 CNN 和 node-aligned ViT。

`ROUTER_DROPOUT` 对当前 NodeAlignedFPNRouter 没有作用，不作为 exploration 方案。

## 4. 实验顺序

每次只修改一个主要机制：

```text
E0: EMA 修复 + 拆分后的诊断指标
E1: E0 + exact-budget greedy eval
E2: E1 + exact-budget stochastic train sampler，固定 noise scale
E3: E2 + noise-scale annealing
E4: E3 + random-topology curriculum 或 K-sample advantage
E5: E4 + agent_pos/temporal conditioning
E6: position/head/backbone 消融
```

每个正式实验至少 3 seeds，最终结论建议 5 seeds。

## 5. 统一验收标准

### Budget

- exact-budget 模式下每个 sample 都严格为 76 leaves，而不只是 batch mean 接近 76。
- random、learned 和 fixed topology 在相同 hard budget 下比较。

### Exploration 与 input conditioning

- early 同一输入重复采样有非零 diversity。
- late exploration 可以下降，但不同 task-relevant 输入仍产生不同 deterministic topology。
- 不要求 learned topology 达到 `random:0.25` 的 Jaccard 0.8。
- 相邻 frames 和 color jitter 不产生无意义 topology 抖动。

### Task utility

同时比较：

```text
diffusion loss
sample action MSE
rollout reward/success
learned-vs-random task loss gap
```

最终成功条件是：

> 在严格相同预算下，learned router 根据 task-relevant 输入产生稳定 topology，并在
> action/rollout 上优于随机或固定 topology。

## 6. 主要代码位置

- `diffusion_policy/model/roi/foveation_net.py`
  - routing mode、exact-budget decoder/sampler、noise scale。
- `diffusion_policy/policy/foveation_dp_transformer_policy.py`
  - agent condition、multi-sample task loss、optimizer groups。
- `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py`
  - live/EMA、deterministic/stochastic/repeat diagnostics 和 schedules。
- `configs/pusht256_foveation_dp.yaml`
  - routing mode、noise schedule 和 optimizer multipliers。

实现继续遵循根目录 `AGENTS.md`：保持科研代码简短、Tensor 使用 shape-aware 命名、
新增 public API 添加简洁 docstring、不新增独立测试脚本，只进行静态编译和 diff 检查；
实际 topology、gradient 和 rollout 验证在训练服务器完成。
