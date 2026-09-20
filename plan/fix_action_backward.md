# Quadtree Router：从 Action-Only Collapse 到预算下的下游适配

当前路线是先用多拓扑训练动作路径（A），再固定动作路径学习 routing（B0/B）。
paired + budget 和 target decay 已实现；当前成功配方包含后期下游适配：固定 routing，
以原有低学习率训练 adaptive resolution visual encoder + diffusion action policy。
该阶段配置默认关闭，但启用后的性能收益已有实验支持；DDP／恢复等专项工程检查
与性能结论分开记录。

## 1. 问题与关键实验事实

### 1.1 最新结果（2026-09-18 实验反馈）

- 使用 [Stage B 脚本](../scripts/roi/sh_train_foveation_pusht_stageB.sh)，先通过 paired
  loss 优化 routing，再固定 routing、finetune 视觉编码器与动作 policy；在
  `target_num_patches=0.1` 下，三 seed 平均报告成绩为 **0.679**
  （`test/mean_score`，最大覆盖性能）。
- Paired 阶段本身（不进行后期下游适配）已明显优于已有直接端到端训练树 router 的
  对照；后期适配进一步明显提升表现。**PushT 的 paired 信号贡献视为已完成验证，
  不再安排重复实验。** 直接端到端对照的更新范围按已有记录整理，不自动等同于 B0。
- 当前最少 16、最多 256 leaves，目标 0.1 对应 40 leaves，约为 all-fine tokens
  的 15.6%。这是目标而非实际推理统计；逐 seed 数值、离散程度、实际数量与 checkpoint
  选择规则仍需从已有记录汇总，不补写未知数据。
- [测速脚本](../exp/pusht_compare_infer_speed.py) 已使用 `--include-router`，ratio 0.1
  比 all-fine 显著加速。当前测的是合成输入和固定 ratio 拓扑，未加载训练 checkpoint；
  checkpoint + 真实观测 + learned routing 测速列入后续工具计划。

下面保留早期实验解释方法演进，不能用它们覆盖最新结论，也不能直接以旧 all-fine
成绩证明当前公平设置下的性能超越。下一步重点是 [LIBERO 迁移](libero_exp.md)。

### 1.2 历史实验与问题来源

无 budget 的早期实验曾长期接近 all-coarse，all-fine 初始化也未避免退化。
零 budget 权重下不同 target 的 run 得到不同 count，不构成 target 的因果证据：
该 target 不直接控制 action loss、router sampling 或初始化。

以下为历史报告值；前三项随机拓扑基线各有 3 seeds，分数不能视为同一共享网络下
逐节点 split 收益的直接比较。

| 拓扑 | Leaves | 报告分数 / 主要观察 |
|---|---:|---|
| `random:0.0`，all-coarse | 16 | 0.465；采样 action MSE 约 0.0065 |
| `random:0.25`，固定 count、随机布局 | 76 | 0.526；diffusion loss 曾高于 coarse，但闭环成绩更高 |
| `random:1.0`，all-fine | 256 | 0.587；后期采样 action MSE 约 0.0045 |
| `fixed:0.25`，固定 count 与布局 | 76 | 分数未报告；后期误差介于 coarse/fine 之间 |

B0/B 分别从同一 A 来源初始化，均冻结下游；以下历史实验关闭 budget 和随机拓扑替换：

| 实验 | Count 与空间现象 | 任务表现 |
|---|---|---|
| B0：ST，3 seeds | 约 80 → 30，后期缓慢回升到约 65；约 60–70% 可视化例子在交互附近更细 | epoch300–500 达峰，略好于所加载 A 的报告峰值，之后略降 |
| B：paired | 早期升至约 250，epoch500 后仍约 212；交互区域 fine、边缘偶尔合并 | 前 500 epoch 峰值约 0.63，与 B0 相当；已报告的后续观察未见明显下降 |

这些结果说明 fine 路径可学习，paired 可主动形成高密度偏好，不能证明 split 普遍
更优、ST 是 collapse 唯一成因，或 B 已优于 B0。约 212 leaves 仍属于高密度结果，
leaf 数下降比例不等于实际延迟下降比例。paired 历史 run 的完整终点、窗口均分和
多 seed 复现仍未报告。

另一组历史观察来自预算衰减实验：目标在 1200/2000 epoch 后固定，最后约 800 epoch
有轻微性能下降。具体 run、下降幅度、窗口统计与多 seed 结果尚未汇总；不要与上述
无预算实验混为同一结论。

## 2. 机制与 A / B0 / B 约定

| 阶段 | 训练对象与拓扑 | 目的 |
|---|---|---|
| A | visual encoder + policy；绕过 router，独立采样多 ratio 随机树 | 维持多尺度覆盖 |
| B0 | 冻结下游；diffusion loss 经 ST 更新 routing | 最小训练对照 |
| B | 冻结下游；paired + 可选 budget 更新 routing | 根据局部真实拓扑比较学习选择 |
| B 后期适配（配置可选，当前成功配方启用） | 固定确定性 routing；diffusion loss 更新下游 | 适应预算压缩后的输入分布 |

A 的 `random:0.0,0.25,0.5,0.75,1.0` 对应 16、76、136、196、256 leaves，
逐样本选 ratio 后生成合法树，同一样本各观测帧共享树；ratio 不是逐节点 split 概率。
B0/B 加载 A 的 EMA 权重及 normalizer，新建 optimizer 和调度；A 绕过的 router
不是已训练 router。B0 虽冻结下游参数，仍需通过其输入反传，不能使用 `no_grad()`。

ST 通过已选 token 的内容门控提供代理梯度，没有直接计算另一棵树的 token 数量、
内容和 attention 变化；可能与真实 split 收益不同，但有偏不意味着必然偏向 stop。

paired 每个样本选一个 view、一个观测帧，再均匀选一个可达内部节点，比较该节点
stop/split 两棵合法树。其他位置保持不变，split 后的后代沿用已采样决策。
两次下游评估在 `no_grad()` 下共享观测、diffusion noise、timestep 和 mask：

```text
A = L_stop - L_split
paired_loss = -mean(p_split * detach(A))
total_loss = paired_loss_weight * paired_loss + budget_loss
```

`p_split` 来自 canonical split logit 的 clean sigmoid。该目标是可达节点采样下的
局部辅助目标，不是原始整树目标的无偏梯度。positive fraction 约 0.47 和全局
advantage mean 接近零，均不能单独判断空间收益或更新方向；概率饱和也可能削弱
`A * p * (1-p)`。需要定位失败时再做按区域/层级的局部诊断。

Budget 是归一化期望 patch 数与目标的平方误差，过高和过低都会受到约束，
不是单向计算惩罚。当前 paired 已将可微 budget 加入总损失；target decay 已实现，
无需再接回 budget 梯度或解除旧的零权重限制。

## 3. 后期下游适配：原因、配置与行为

目标固定不等于 router 固定：paired + budget 仍持续更新 routing。与此同时，下游
保留 A 权重，无法适应 patch 的尺度、数量和空间分布变化。低 LR 只作用于实际训练
的参数，原先冻结的下游没有利用它。这支持下游分布不适应的假设，但不能排除
router 过拟合、预算与任务质量冲突，以及训练随机采样和测试确定性 routing 的差异。

最新实验已观察到固定 routing 后适配下游的明显收益，为分布不适应的解释提供支持，
但不证明这是唯一原因。下一步优先迁移这一成功配方；联合或交替训练仅在新任务出现
明确失败时再考虑。

```yaml
training:
  target_num_patches_decay_epochs: 1200
  num_epochs: 2000
  # null 默认关闭；使用 zero-based epoch。1600 表示最后 400 epoch 适配。
  downstream_finetune_start_epoch: 1600
```

建议并检查：

```text
预算衰减启用时：decay_epochs <= downstream_finetune_start_epoch < num_epochs
finetune_epochs = num_epochs - downstream_finetune_start_epoch
因此 decay_epochs + finetune_epochs <= num_epochs
```

新阶段仅支持 paired router-only Stage B，要求 learned routing 可训练且
`random_topology_prob=0`。沿用完整 cosine LR 和 visual encoder LR multiplier，
不重启 warmup。越晚切换越温和，但剩余学习率过小可能不足以适配。

切换时采用 rank 0 当前 EMA routing（无 EMA 则采用 live），将完整参数及 buffers
同步到所有 rank 的 live routing，并停止后续 routing EMA 更新。Routing 范围包含
整个 `foveation_net`（global encoder、router 等）和 `router_condition_projection`。

| 项目 | 切换前 | 切换后 |
|---|---|---|
| Routing | train，随机采样，paired + budget 梯度 | eval，确定性路由，`no_grad()` |
| Visual encoder + policy | eval，paired 两次评估不建图 | train，正常 diffusion 梯度 |
| `total_loss` | weighted paired + budget | diffusion loss |
| Budget | 原有衰减 | 最终目标，仅记录诊断 |
| EMA | 更新 routing，固定下游 | 固定 routing，继续平滑下游 |

同一输入的 routing 规则固定，不代表所有图片共用一棵树。Policy 的 `train()` 维护
上述模式，防止 validation/rollout 后误开启 routing 采样。后期不再计算 paired 两臂。

## 4. 指标与实验对照

- `test/mean_score` 是各 rollout 最大归一化 coverage reward 的均值，不是严格 IoU。
  比较相同 seeds 下的后期窗口、峰值和稳定性，不只比较最佳 checkpoint。
- 前期 `train/num_patches_mean` 来自 live 随机路由，validation 和可视化通常使用
  EMA 确定性路由；进入适配后训练 routing 也变为确定性，切换处 count 可能出现变化。
  比较预算时应看实际 count，同一 target 不保证同一 count。
- 前期 `train/diffusion_loss` 是 paired 基础树的无梯度评估，后期则是训练目标；
  `train/total_loss` 在阶段之间含义不同，不能直接比较数值高低。
- `train/downstream_finetune_active` 标记阶段。后期保留 budget 和 patch 指标，
  跳过 router 梯度诊断；前期非零 budget weight 可以产生 budget 梯度。
- `vis/train_foveation/image` 只展示缓存训练样本；`vis/test_foveation/video_<seed>`
  展示测试轨迹的逐帧四叉树。检查细分是否跟随交互对象，不能仅从固定中央细分推断
  任务相关性。平均 leaves 不等于 batch padding 下的实际计算量。

汇总已有实验时标注 A 初始化、预算、总 epoch、LR、评估设置、seeds 和适配切换点，
分别报告 paired 阶段与后期适配后的结果。固定／随机树和 ST 的已有对照按其实际设置
整理，不把“再次验证 paired 是否有效”列为待办。新任务需要定位失败时再补针对性诊断。

## 5. 实现状态与验证

涉及配置、Stage B 启动脚本、foveation policy 和 workspace。启动示例：

```bash
TARGET_NUM_PATCHES=0.1 DOWNSTREAM_FINETUNE_START_EPOCH=1200 \
  bash scripts/roi/sh_train_foveation_pusht_stageB.sh
```

需按脚本原有约定设置 GPU、checkpoint 和与 Stage A 一致的架构参数。
上述 1200 是脚本示例的切换点，不表示已核实三个 seed 均使用这一配置；以各 run
保存的配置为准。脚本默认 target 为 0.5，适配默认 null，复现低预算适配需显式开启。

启用功能时，两阶段参数从初始化起全部注册进 optimizer/DDP，训练期间不动态修改
其 `requires_grad`、不重建 optimizer。通过阶段对应的 `no_grad()` 隔离停用分支，
其梯度保持 None；AdamW 不更新它，也不施加 weight decay。因此参数统计中的
requires-grad 数量表示注册资格，active/no-grad 状态才表示当前更新范围。

Workspace 保存阶段切换标记和下一 epoch 游标。恢复时保留 optimizer/scheduler/EMA
进度，重建阶段模式和 EMA 排除前缀，不重复用 EMA 覆盖已固定的 routing。
旧 router-only optimizer 没有下游参数，不能直接恢复为新模式；可以使用
`pretrained_ckpt`、`resume=false` 新开实验，但这不延续旧 LR 和 optimizer 状态。
新旧 optimizer 布局之间的完整 resume 会明确报错。

此前实现的本地验证限于 Python 静态编译、YAML 解析、shell 语法和 diff 检查；
不新增独立测试脚本。性能收益现已有服务器实验反馈；以下专项工程检查未在本次反馈中
逐项确认，应与性能结论分开保留：

- 功能关闭时保持基线行为；切换前后实际更新范围正确、LR 连续。
- 各 rank 切换一致，下游梯度正确同步；routing 参数、buffers 和确定性输出保持固定。
- EMA/live routing 一致；切换前后 checkpoint 恢复不重复切换，不重置优化进度。

联合训练、交替训练、paired + ST、更强探索和结构改造仅作为按需研究方向，
不作为 LIBERO 或当前成功配方的前置条件。自主时间计算分配保留为核心研究目标；
现有 budget 不是时间分配已成立的证据，候选成本正则与后续任务见 [libero_exp.md](libero_exp.md)。

参考：[AdaFocus](https://arxiv.org/abs/2105.03245)、
[Local Expectation Gradients](https://arxiv.org/abs/1503.01494)、
[COMA](https://arxiv.org/abs/1705.08926)。它们提供分阶段选择或反事实比较的启发，
当前 paired 不是这些论文的直接复现，也不因此获得无偏或收敛保证。
