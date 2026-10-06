# Spatial adaptive 与 Temporal adaptive：跨时间计算分配

背景见[项目概述](quadtree_introduction.md)，现有 paired 与三阶段训练见[训练机制](fix_action_backward.md)。本文记录用户报告的实验结果，以及已接入代码的 episode 内跨时间分配学习。**当前使用完整子树的 reachable stop/split、随机锚点主动匹配与失败回退；single_level 已移除。已有用户报告的初步运行观察，受控收益验证及完整运行时验收仍待完成。**

## 1. 目标与命名

目标是在测试时随操作进程、接触状态和任务需求改变每帧的 `num_patches`，既能增加，也能减少。继续使用现有 stop/split router，不新增预测数量的决策模块。

| 名称 | 本文中的含义 | 状态 |
| --- | --- | --- |
| Spatial adaptive | 现有局部 paired 路由学习，以动作预测收益学习 stop/split，配合平均预算控制整体成本 | 已有实现，作为基线 |
| Temporal adaptive | 同一 episode 的不同决策时刻，显式比较“减少同样数量的 patches，优先在哪里合并”，训练现有 router 的相对分配偏好 | 已实现，有初步运行观察，待完整验收 |

这是对**学习机制**的命名，不是把模型能力完全分开。Spatial adaptive 已可能产生随状态变化的时间数量差异；不能将其描述为固定每帧数量或完全没有时间自适应。Temporal adaptive 增加的是显式跨时刻比较，不是首次让数量变得可变，也不等于新增时序网络。

现有 paired 提供随状态变化的局部细分收益；平均预算允许观测间补偿，但不直接偏好某种时间分配。例如，软期望数量 40/40、25/55、55/25 的均值相同，平均预算项无法区分。lambda 与 alpha 提供共同的数量成本，不识别哪个时刻更值得增加计算。

现有时间差异主要是局部决策积累的结果；新增机制要直接学习跨时刻的**边际资源价值**。不以数量方差增大、接触时必然更多或人为制造波动为目标。

用户观察到，Stage B 通常从满树开始，随 cosine target 收紧，画面边缘和操作无关部分先合并。这说明当前预算与 paired 已在配合选择低代价合并，但首先是空间选择的证据。Temporal 要进一步比较不同时刻的合并代价：预算控制整体数量压力，Spatial 判断局部细分价值，Temporal 决定相同节省量优先从哪里获得。满树起点和逐步收缩是当前实验观察，不是上限型预算保证的轨迹。

## 2. 当前结果：以 alpha=0 作为主基线

以下为用户报告的结果，alpha 对应 `BUDGET_BASE_COST_WEIGHT`，按 `upper_bound_lagrangian_elu` 模式解读。延迟来自 `exp/pusht_compare_infer_speed.py` 的完整推理 wall 计时，patches mean 为测试输入上的硬数量。

| alpha | predict_action_wall | patches mean | IoU |
| --- | --- | --- | --- |
| 0.1 | 118.080 ms | 32.63 | 0.6610 |
| 0.01 | 训练中，待补充 | — | — |
| 0.001 | 121.756 ms | 40.24 | 0.6536 |
| 0.0001 | 132.997 ms | 41.22 | 0.7014 |
| 0 | 122.786 ms | 40.91 | 0.6903 |

- alpha=0、0.0001、0.001 的平均数量都在 40–41 附近；明显压缩来自 alpha=0.1，不能概括为“只要非零就有主要差异”。相对 alpha=0，alpha=0.1 数量减少约 20.2%，延迟降低约 3.8%，IoU 下降 0.0293。
- alpha=0.0001 的 IoU 高 0.0111，但延迟高约 8.3%；尚未通过重复训练、统一评估和重复测速确认差异是否稳定。平均数量接近不能解释全部延迟差异，还涉及数量分布、padding、层级分布及运行开销。
- 用户观察 alpha=0 的视频，认为路由分配基本符合预期，任务表现也不错。这是定性证据，尚不是跨时间分配收益的受控验证。
- **后续以 alpha=0 的 Spatial adaptive 为主基线；非零 alpha 保留为额外节省奖励的消融。** 等 0.01 完成后补充结果，不将继续搜索 alpha 作为主要方向。

历史上，逐观测 MSE 与 batch mean MSE 的数量波动肉眼相近；线性基础数量成本曾导致过快减量。饱和奖励用于缓解持续压缩压力，但不是时间自适应的必要条件，也不能单独保证避免坍缩。

用户报告两极化树形从 **step 1200 左右**出现。单层候选限制与强制孩子停止可能造成层级监督偏置，这是待验证的解释，不能记录成已证实的原因。此次恢复完整 reachable 干预，并用相同 checkpoint、seed 和预算下的实验检验；不以中间层叶子占比或树形观感单独判定成败。

### 2.1 当前 reachable Temporal 版本的初步观察

用户报告，加入 Temporal adaptive 后，PushT 评测效果没有明显变好或变差；`sample/topology_div` 随训练上升更稳定，曲线抖动明显减小。`train/temporal_match_fraction` 通常超过 0.99；`train/temporal_loss` 有时为正、有时为负，例如 −0.0006。

同一阶段的 advantage 日志观察如下：

- `paired_advantage_mean` 在零附近波动，正值较多。
- `paired_advantage_abs_mean` 在 0.0004 附近波动。
- `paired_advantage_positive_fraction` 从约 0.47 缓慢升至约 0.54，过程存在波动。

按第 4.5 节的统计口径，这说明当前采样候选的单次两臂评估中，正负收益均常见，平均绝对差约为 `4e-4`。均值接近零可以包含正负抵消；正值比例上升描述的是该采样分布下单次收益符号的变化。信号强度需结合原始两臂 diffusion loss 的尺度判断，符号稳定性及期望收益需通过固定候选的重复评估确认。

以上为初步运行观察，任务性能等价性及稳定性变化的来源仍待验证。本轮尚未记录完整对照数值、各运行的配置和多 seed 统计；新增 B 窗口、主动匹配后的候选分布和 Temporal 目标都可能影响结果，需要第 6 节的三组对照区分。

已核实 workspace 中的指标定义：`sample/topology_div` 在同一 view、同一相对观测帧位置下，计算 batch 内不同样本有效叶子集合的两两 Jaccard 距离，再对各 view/frame 的有效距离取均值；忽略 token padding，无可比较样本对时返回 0。

```text
Jaccard_distance(leaves_i, leaves_j)
    = 1 - size(intersection(leaves_i, leaves_j)) / size(union(leaves_i, leaves_j))
```

因此它衡量的是**样本间拓扑差异，而不是同一轨迹相邻时刻的拓扑变化**。其训练曲线更平稳，不直接证明轨迹中的路由更平滑；多样性更高也不自动意味着任务表现更好。目前结果支持进一步验证复杂任务，但不能推出 LIBERO 成功率会提升。

## 3. 保留的预算机制：控制整体花多少

### 3.1 Target、alpha 与 lambda

当前基线显式选择 `upper_bound_lagrangian_elu`，并设 alpha=0。代码默认预算模式仍是 `per_observation_mse`；**仅将 alpha 设为 0 不会自动切换预算类型。** 原线性 `upper_bound_lagrangian` 在目标以下仍保留带符号的 lambda 项，与这里不同。

记全局平均软期望数量为 $\bar N$，合法数量范围宽度为 $R=N_{\max}-N_{\min}$，归一化数量为 $c=(\bar N-N_{\min})/R$，当前 cosine target 为 $b_t=(N_{\mathrm{target}}-N_{\min})/R$，差值为 $d=c-b_t$。软期望数量来自包含祖先可达性的叶子概率之和。预算开启且 $R>0$ 时：

| 平均数量 | ELU 分段预算项，外层权重为 $w_b$ | alpha=0 时 |
| --- | --- | --- |
| 高于 target | $L_{\mathrm{budget}}=w_b\lambda d$ | 保留超支惩罚 |
| 等于或低于 target | $L_{\mathrm{budget}}=w_b\alpha\tau(e^{d/\tau}-1)$ | 预算项及其梯度为零 |

alpha 只控制目标内继续节省的奖励，不控制是否开启时间适应。alpha>0 时，目标内的数量梯度为 $w_b\alpha e^{d/\tau}$，越低于目标越弱；相同数量和 target 下，梯度仍与 alpha 成正比。恰好等于 target 时选择奖励分支。`BUDGET_ELU_TAU=0.025` 在 16–256 范围下对应 6 patches 的指数衰减尺度；alpha=0 时 tau 不影响这项预算 loss。

**无论 alpha 是否为零，target 都参与分支选择和 lambda 更新。** alpha=0 不是无预算基线。Target 是软上限，既不保证硬数量不超支，也不把低于目标的数量拉回去；lambda 当前为零时，一次超支 forward 也可能没有预算梯度，更新后的价格才用于后续步骤。

预算分支使用单次 forward 内所有 GPU、batch、观测帧和相机的加权平均，不约束每帧分别达到 target，也不是整条轨迹的预算账户。梯度累积大于 1 时，分段 loss 按各次 forward 计算，不等价于对整个累积窗口计算一次非线性 loss。

### 3.2 Lambda 的必要说明

成功 optimizer step 后更新 $\lambda_{\mathrm{next}}=\max(0,\lambda+\eta_\lambda\Delta)$，其中 $\Delta=(\bar N_{\mathrm{step}}-N_{\mathrm{target}})/R$，数量统计覆盖整个 step 的多卡与梯度累积窗口，且停止梯度。

- lambda 累计变化，可以上升、下降，最低为零；不按 epoch 重置。恢复训练会恢复乘子，加载 pretrained checkpoint 开始新实验时使用配置初值。
- `train/budget_lambda` 记录本次更新使用的旧值，新值用于下一次；持续低于上限且初值为零时一直为零符合规则。持续明显超支却长期精确为零，才需要检查更新路径。
- 判断时使用已有 `train/expected_num_patches_mean`、`train/target_num_patches` 和梯度日志，不能用测试硬数量替代训练软期望数量。
- 更新形式来自标准投影对偶梯度上升，但当前互斥分段 loss 只借用这套价格更新，不是对该分段目标严格执行标准对偶上升，不能直接套用凸优化收敛保证。
- AMP 跳步、预算关闭、固定拓扑、零数量范围及下游适配阶段不更新乘子。测试时不计算预算 loss，不根据 target 强制截断数量。

### 3.3 下一阶段沿用的预算设置

在原 Stage B 启动设置中使用以下配置；本文不改变脚本或 YAML 的默认值：

```bash
BUDGET_LOSS_TYPE=upper_bound_lagrangian_elu \
BUDGET_LOSS_WEIGHT=0.1 \
BUDGET_BASE_COST_WEIGHT=0.0 \
BUDGET_LAMBDA_INIT=0.0 BUDGET_LAMBDA_LR=0.01 \
TARGET_NUM_PATCHES=0.1 \
DOWNSTREAM_FINETUNE_START_EPOCH=1200 \
bash scripts/roi/sh_train_foveation_pusht_stageB.sh
```

继续保留原启动所需的 GPU、checkpoint、架构及数据设置，脚本末尾 Hydra 覆盖仍可使用。当前配置从 `target_num_patches_start=0.75` 用 1200 epochs 衰减到目标；16–256 范围下约从 196 降至 40，并非第一轮就以 40 为上限。**这是训练进度上的预算调度，不是测试轨迹中的时间分配机制。**

## 4. 已实现：reachable 随机锚点匹配与失败回退

### 4.1 在相同资源增量下比较局部价值

核心问题是：**减少同样数量的 patches，在时刻 A 还是 B 执行，动作预测损失增加得更少？** 这与保留相同额外计算给谁是对应的选择。原始预测误差更大，不代表细分更有帮助；也不直接监督 A 的绝对总数量必须大于 B。

早期两侧候选的合并代价都可能接近零，此时 Temporal 提供很弱的偏好，由预算推动节省，不将微小噪声归一化成强排序信号。随着预算收紧，跨时刻比较是否更有价值仍需实验判断。

### 4.2 同 episode 的两个有效决策窗口

- A 保留原样本的顶层 `obs/action`；B 从同一源数据集、同一 episode 的其他有效决策时刻均匀采样，使用自己的 `obs/action`。不额外要求最小时间间隔，B 可以早于或晚于 A。
- 在 PushT 内复用 `SequenceSampler.indices` 与 episode 边界。实际窗口起点为 `buffer_start_idx - sample_start_idx`，决策时刻再加 `n_obs_steps - 1`。只有决策时刻在 episode 真实范围内的窗口参与配对，其余窗口的 padding 保持原实现。
- 若 A 的决策时刻本身落在 padding 区域，或没有其他有效 B，则复用 A 的已增强数据作为占位，标记 `temporal_pair_valid=false`。Temporal 贡献为零，Spatial 和预算仍统计两个输入窗口。
- 额外数据放在 `temporal_pair` 中；直接调用 sampler 与数据转换，不递归调用 `__getitem__`。配对使用 DataLoader worker 的 NumPy 随机状态，不重新设种子。
- Merged 数据集分别在各子数据集内完成配对，避免跨文件同编号 episode 混淆。候选仅来自当前 sampler，训练与验证划分不交叉。

数据配对由 workspace 调用 `set_temporal_pairing(enabled)` 开关控制，不写入测速脚本读取的 dataset 构造配置。共享内存布尔标记让 persistent workers 在 epoch 边界看到变化；索引在训练 workers 启动前准备。验证数据集有独立的关闭标记。

router 阶段启用双窗口；进入 downstream 阶段后关闭，policy 仅优化 A 的动作 loss。恢复训练时按已恢复阶段及当前 epoch 初始化。验证、rollout、视频和测速使用原单窗口路径，推理不需要配对或未来信息。

#### LIBERO 数据接入（2026-09-24）

LIBERO Stage B 已接入相同训练机制，使用 `LiberoFTDataset.set_temporal_pairing(enabled)`。
与 PushT 的 SequenceSampler 不同，LIBERO 的 MergeDataset 每个索引已经是真实决策帧。
利用 MergeDataset 的文件 prefix 和 reader 的 demo prefix/lengths 构建全局 episode
区间，并按 max_len 截断；B 均匀采自同一文件、同一 demo 的其他可访问决策索引。
历史和动作 padding 仍按原读取方式重复边界，不排除含 padding 的真实决策窗口。
单帧区间使用无效占位，数据结构、随机数使用、A/B 独立增强及阶段开关语义与 PushT 一致。

当前 two_by_two Stage B 仅在 B_train 中配对，A_val 独立关闭配对，并复用 Stage A
统计，不因 Temporal 修改划分。单/多视角沿用同一 policy 路径：每对选择相同 view 和
相对观测帧位置，不要求移动相机中同一空间位置对应同一个物体。
Temporal 比较的是等数量增量的动作收益，不是跨时刻图像或物体的对应关系。

脚本默认 `TEMPORAL_ADAPTIVE=false`，权重默认 1.0；开启且权重为 0 保留双窗口与匹配。
batch 不自动改变，默认 64 对对应 128 个窗口。验证、rollout、视频和 downstream
finetune 使用原单窗口；resume 和 persistent workers 复用既有阶段切换。
启动示例、三组对照及服务器验收项见 [LIBERO 实验记录](libero_exp.md#1011-libero-temporal-接入2026-09-24)。
当前仅进行静态检查，运行时与任务收益仍待服务器验证；Temporal 不自动修复指示模型的
分辨率收益排序问题，相关诊断见 [分辨率损失分析](fix_libero_resolution_loss.md)。

### 4.3 完整 reachable 候选与随机锚点主动匹配

`single_level` 的候选限制和强制孩子停止逻辑已删除。`sample_paired_splits()` 仅使用完整子树的 reachable stop/split：祖先均 split 的内部节点都可作为候选，包括后代仍在细分的父节点。Spatial only 使用单窗口，在全部 reachable 内部节点中独立均匀采样，恢复原 Spatial adaptive 的训练语义。

接口移除 `node_sampling` 参数，增加可选 `temporal_pair_valid_P=None`，仍返回所选节点、stop 拓扑、split 拓扑三项。不传配对标记时，不计算匹配所需的子树数量，也不消耗随机锚点采样。树至少需要一个可细分层，否则明确报错。

Temporal 的 P 对窗口按 `[全部 A, 全部 B]` 排成 2P 个输入，采样全过程在 `no_grad()` 下完成：

1. 两侧分别生成独立随机 scores，在 reachable 候选中取得默认均匀采样结果。
2. 从最深内部层向上计算每个节点的实际子树叶子数，以及强制该节点 split 相对 stop 的数量增量。最深层叶子数量为 1；必须计算被祖先遮蔽的后代，不能用节点层级或当前可见叶子数量推测 delta。

   ```text
   split_leaf_count(node) = sum(actual_leaf_count(child))
   actual_leaf_count(node) = split_leaf_count(node) if node_is_split else 1
   delta(node) = split_leaf_count(node) - 1
   ```

3. 每对以 50% 概率选择 A 或 B 为锚点，保留锚点默认候选。另一侧在 `reachable & (delta == anchor_delta)` 集合内均匀采样，复用该侧原随机 scores。
4. 匹配集合为空时，保留另一侧默认候选。不重采锚点、不更换 B 时间窗口、不修改基础树凑数量。`temporal_pair_valid=false` 的占位对保留两侧普通独立采样，不应用匹配。
5. 从基础树复制两臂，只将所选父节点分别设为 stop/split，完整保留后代决策。基础树和 logits 不变，且一臂与基础树的实际叶子集合一致。

子树数量复用 `internal_children_M_4` 和层级 offsets，按层循环，batch/node 维度张量化，不逐节点执行 Python 循环。三层树的数量增量支持 `3/6/9/12/15`，实现不硬编码为三层。

**主动匹配会改变共享候选的 Spatial 采样分布。随机锚点消除固定 A/B 角色的不对称，但不会恢复两侧原本的独立采样分布。** 相同数量增量仅对齐资源尺度，不保证相同推理延迟。

#### 采样历史核查与当前行为

此前对 Git 历史、中间版本快照和当前本地代码的核查，区分了基础 router 采样与 paired 反事实干预：

| 版本 | paired 候选与干预 |
| --- | --- |
| 原始 Spatial paired（`b1c18e3`） | 在全部 reachable 内部节点中均匀采样，仅修改所选父节点，保留后代决策 |
| 中间未提交的 `single_level` 版本 | 限制候选；在 split 评估臂强制直接内部孩子 stop，使数量增量固定为 3。更深决策未清除，但被 stop 孩子遮蔽 |
| 当前 reachable 版本 | 已删除上述限制和孩子强制 stop；Temporal 仅增加随机锚点匹配，后代决策完整保留 |

所核查版本的基础节点决策采样、叶子可达性计算和打包逻辑未发生上述修改；变化发生在 paired 候选选择与反事实干预。基础 router 为所有内部节点采样决策，包括被 stop 祖先遮蔽的后代。中间版本虽然没有直接改写基础树或 logits，却改变了评估得到的 advantage，因而会改变训练信号。

例如，父节点原为 stop，隐藏的四个孩子决策为 `[stop, split, stop, stop]`，其中 split 孩子展开为四个叶子。当前强制父节点 split 后，得到 `1 + 4 + 1 + 1 = 7` 个叶子；stop 臂为 1，delta=6。**当前版本不存在“父节点改成 split 后，强制直接孩子 stop”的行为。**

中间 Stage B 启动示例曾同时设置 `TEMPORAL_ADAPTIVE=false` 和 `PAIRED_NODE_SAMPLING=single_level`，虽然脚本默认值仍为 reachable，复制该示例会覆盖默认值。因此关闭 Temporal 不足以证明旧实验恢复了原始 Spatial 语义。这里核实的是本地代码历史；服务器上的某次实验必须结合当时保存的配置和实际运行代码判断。

### 4.4 两次批量评估同时供给两个目标

把原始 P 对窗口按 `[全部 A, 全部 B]` 拼成 2P 个窗口，统一自由路由一次。每对共享随机 view、相对观测帧位置及 diffusion timestep；两侧候选按上一节主动匹配。

随后只进行 stop、split 两次批量动作评估，得到 A-stop、A-split、B-stop、B-split。A/B 使用各自的动作目标和独立 diffusion noise；同一窗口的两臂严格复用 noise、mask、输入增强及 timestep。其他 view 和观测帧不变，原动作路径处于 eval，评估在 `no_grad()` 中完成。

令 `advantage = loss_stop - loss_split`。Spatial 沿用 `-p_split * detach(advantage)`，在全部 2P 个窗口上平均，包括匹配失败和占位窗口，仍使用 `paired_loss` 日志键。

在已有两次 `pack_topology()` 评估中，读取所选 view/frame 的 `num_patches_B_T`，计算 split 减 stop 的实际数量差。只使用有效叶子数，不用 padding 后的 token 长度。Temporal 使用 canonical split logits（`split_action_value=0` 时先反转符号）：

```text
matched = temporal_pair_valid AND (actual_delta_A == actual_delta_B)
gain = advantage_A - advantage_B
prefer_A = sigmoid(canonical_logit_A - canonical_logit_B)
L_temporal = mean_over_all_P_pairs(-matched * prefer_A * detach(gain))
L_total = paired_loss_weight * L_spatial
        + temporal_loss_weight * L_temporal
        + L_budget
```

最终 `matched` 同时要求时间窗口配对有效和两臂实际数量差相等。Temporal 按全部原始 P 对平均，匹配失败或无效配对贡献零；不按成功数放大，也不将收益除以 delta。

交换 A/B 后，角色对应的梯度应一致；标量 loss 本身可以相差一个与 logits 无关的常数。`prefer_A` 沿用现有 router logits，实际路由仍允许两侧都增加或都不增加，推理时没有两侧独占资源约束。梯度方向及其与 Spatial 的关系见第 4.6 节。

默认 `temporal_loss_weight=1.0`，允许设为零。零权重仍保留双窗口和主动匹配，是相同采样流程下关闭 Temporal loss 的消融，不等同于原 Spatial only。不加入收益标准化、温度、额外调度或数量方差目标。这是待实验验证的局部代理目标，不是整树策略的无偏梯度或全局最优分配保证。

### 4.5 预算及日志

预算使用全部 2P 个自由路由窗口，包括匹配失败和占位窗口，每个窗口统计一次，不统计四个反事实候选。沿用 cosine target、全局软期望数量聚合、梯度累积处理和每次成功 optimizer step 一次 lambda 更新。

当前每对的观测帧数与 view 数固定，原始 batch 大小仍可作为 loss 的权重。双窗口让预算 `observation_count` 和梯度归一化的 `reference_count` 同时翻倍，两者的比例保持一致；Spatial 的窗口均值与 Temporal 的配对均值均保持原有尺度。分段预算仍按每次 forward 的多卡均值计算，lambda 使用完整 optimizer step 的数量统计；AMP 跳步和 downstream 阶段不更新 lambda。

保留 `train/temporal_loss`，记录未乘权重的 Temporal 辅助 loss，并按全部原始配对数汇总成当前 forward 的多卡均值。新增一个标量：

```text
train/temporal_match_fraction = global_matched_pair_count / global_valid_pair_count
```

policy 返回私有匹配成功数和有效时间窗口对数，workspace 分别跨卡求和后求比值，不平均各卡比例。匹配率分母不包含没有有效第二窗口的占位对；全局有效对数为零时记为 0。此分母与 Temporal loss 的全部 P 对分母不同。

三个现有 advantage 指标直接统计 `advantage = loss_stop - loss_split`，日志键带 `train/` 前缀：

| 指标 | 当前 forward 的计算定义 |
| --- | --- |
| `paired_advantage_mean` | `mean(advantage)` |
| `paired_advantage_abs_mean` | `mean(abs(advantage))` |
| `paired_advantage_positive_fraction` | `mean(advantage > 0)` |

positive fraction 中零值计入分母，只有严格大于零的值计入分子；无 split 概率、收益大小或 delta 加权，也没有近零阈值。Temporal 模式下，三项均覆盖本卡全部 `2P` 个窗口，包括匹配失败和占位窗口；Spatial only 则覆盖本卡原单窗口 batch。它们经标量转换后由主进程写入，记录的是**主进程所在卡当前 forward 的统计**，未进行跨卡或梯度累积窗口聚合。上文 Temporal loss 和匹配率各自单独跨卡聚合，统计范围需分别解读。

这些指标描述实际采样到的候选。reachable 集合随基础树变化；每个窗口只选择一个 view/frame 中的一个候选，层级节点数量影响各层被采到的比例，Temporal 主动匹配还会改变两侧候选分布。因此它们代表该采样分布下的局部收益，不能外推为所有节点、像素或整棵树的收益比例。同一窗口的两臂复用随机条件以减少比较噪声，单次 advantage 仍依赖 diffusion noise/timestep；单次正值比例和重复评估下稳定的期望收益应分别解释。

沿用原数量、target、lambda、paired/budget 梯度与视频，不新增 Temporal 梯度曲线。Temporal loss 接近零不独立证明收敛或任务收益，也可能反映匹配少或收益差弱。匹配率超过 0.99 仅说明有效时间窗口对中等增量候选容易找到，不代表收益排序准确或任务策略有效；还需检查匹配 delta 的分布，不能由高匹配率推断较大子树调整也得到充分监督。

### 4.6 三项目标的共同作用

**计算对象。** 当前 `upper_bound_lagrangian_elu`、alpha=0 设置下，Budget 使用当前 forward 内所有 GPU、batch、观测帧和相机的全局平均软期望数量。预算开启且数量范围非零时，平均数量高于 target 且 lambda>0 才产生减量压力；达到或低于 target 时预算梯度为零。当前实验 lambda 初始为零，首次超支 forward 也可能没有预算梯度，成功 optimizer step 更新后的 lambda 用于后续步骤。硬满树描述实际采样出的叶子集合；软期望数量由含祖先可达性的叶子概率求和，两者应分开判断。

Spatial 使用所选节点的 `a = loss_stop - loss_split` 评价局部细分价值。动作路径被冻结，两臂在 `no_grad()` 下评估；两臂 loss 的差作为停止梯度的监督信号，仅通过所选 logit 的 split 概率学习路由。Temporal 复用两侧收益，在时间窗口有效、实际数量增量相等的配对上比较 `a_A - a_B`，学习哪一侧更值得保留细分。

**对所选 logits 的直接梯度。** 以下 z 均为 canonical split logit，增大 z 表示提高 split 偏好；公式省略外层 loss 权重及窗口／配对平均系数，将 a 和 gain 视为停止梯度的常量：

```text
p = sigmoid(z)
a = loss_stop - loss_split
L_spatial_window = -p * detach(a)
dL_spatial_window/dz = -a * p * (1 - p)

q = sigmoid(z_A - z_B)
gain = a_A - a_B
L_temporal_pair = -matched * q * detach(gain)
dL_temporal_pair/dz_A = -matched * gain * q * (1 - q)
dL_temporal_pair/dz_B = +matched * gain * q * (1 - q)
```

梯度下降下，Spatial 在 a>0 时支持 split，在 a<0 时支持 stop，a=0 时直接梯度为零。对有效等增量配对，Temporal 在 gain>0 时提高 z_A、降低 z_B，gain<0 时方向相反；gain=0 或 matched=false 时直接梯度为零。两者都是有符号的收益代理目标：单窗口 Spatial 贡献的符号与 a 相反，有效配对的 Temporal 贡献符号与 gain 相反。

为比较两项目标，以下将收益较大一侧记为 A，即 `a_A > a_B`；若 B 较大，交换两侧即可：

| 两侧收益 | Spatial 方向 | Temporal 方向 | 同向或冲突 |
| --- | --- | --- | --- |
| 异号：`a_A > 0 > a_B` | A 支持 split，B 支持 stop | 提高 A、降低 B 的 split 偏好 | 两侧均同向 |
| 均正：`a_A > a_B > 0` | 两侧均支持 split | 提高 A、降低 B 的 split 偏好 | A 同向，B 冲突 |
| 均负：`0 > a_A > a_B` | 两侧均支持 stop | 提高 A、降低 B 的 split 偏好 | A 冲突，B 同向 |

某侧收益恰好为零时，该侧 Spatial 直接梯度为零，Temporal 仍可由收益差提供相对监督。这些方向针对所选 logits；实际参数更新由共享网络上所有样本的 Budget、Spatial、Temporal 梯度及各自权重共同决定，不能据此保证每个 logit 的更新方向。

**从满树起点到资源取舍。** 基础硬树为满树时，所选节点的 split 臂保持基础子树，stop 臂将该子树合并为一个叶子，测量合并带来的动作 loss 变化。随着 target 收紧且 lambda 建立正价格，预算产生数量压力；Spatial 学习保留细分的收益，Temporal 强化等增量的跨时刻取舍。合并后，只要祖先仍 split，当前为 stop 的内部节点仍在 reachable 集合中，可再次被选中并展开；后代决策保留，预算和收益变化时也可重新分配。

固定其余条件，将一个局部候选的数量成本近似为 split 概率 p 与数量增量 delta 的乘积，可写为：

```text
L_local ≈ -spatial_weight * p * a + c * p * delta
```

这里 `spatial_weight = paired_loss_weight`，delta 为该候选 split 相对 stop 的实际数量增量，c 表示当前有效的单位数量价格，吸收预算权重、数量归一化及平均系数；它与第 3.1 节表示归一化数量的 c 分属不同语境。该式仅用于解释收益与数量成本的局部竞争，省略了 Temporal 项；实际实现通过含祖先可达性的软数量和采样代理目标优化，并未直接最小化此局部近似。

细分收益普遍为正、预算存在正数量压力且局部近似适用时，Budget 与 Spatial 倾向于先压缩**单位节省量对应动作代价较低**的子树，即 a/delta 较小的候选。同 delta 时，Temporal 进一步强化保留高收益一侧的偏好。这个直觉依赖收益估计质量、目标权重和共享参数的共同作用，描述的是优化倾向；当前实现没有显式按 a/delta 排序或求解全局资源分配。

**结合实验解释。** loss 符号本身不衡量当前偏好是否正确。例如 `gain=0.002, prefer_A=0.3` 得到 −0.0006，此时偏好仍偏向 B；`gain=-0.003, prefer_A=0.2` 得到 +0.0006，此时偏好已偏向收益更高的 B。batch 和多卡均值还会正负抵消，所以第 2.1 节的 −0.0006 本身符合该目标的定义。当前单次 advantage 的正负都常见、绝对均值约 `4e-4`；应结合原始 loss 尺度、固定候选的重复评估和任务对照，判断监督是否稳定且有用。

## 5. 配置与实现范围

| policy 配置 | 默认值 | Stage B 环境变量 | 行为 |
| --- | --- | --- | --- |
| `temporal_adaptive` | `false` | `TEMPORAL_ADAPTIVE` | 开启双窗口与主动匹配 |
| `temporal_loss_weight` | `1.0` | `TEMPORAL_LOSS_WEIGHT` | Temporal 权重，允许为零 |
| `paired_node_sampling` | `reachable` | `PAIRED_NODE_SAMPLING` | 保留配置入口，仅接受 reachable |

Temporal 只支持 PushT 的 paired router-only 训练路径，要求 `obs_as_cond=true`、dataset 与 policy 的 `n_obs_steps` 一致，并沿用 learned router、无 random topology 的 Stage B 约束。不新增采样策略开关。传入 `single_level` 明确报错并提示已移除，避免旧命令静默改变训练含义。

模型参数、持久化 buffer 和推理接口不变，旧 checkpoint 权重仍可加载。旧配置缺少新字段时保持原训练路径；**使用含 `single_level` 的旧 checkpoint 配置实例化时，需要先将 `policy.paired_node_sampling` 覆盖为 `reachable`。**

三组实验共用原 GPU、Stage A checkpoint、架构、seed 和预算设置。启动示例保留 `tmp="..."` 形式，变量直接写在命令前；以下为回到原 Spatial adaptive only 的完整示例：

```bash
tmp="
CUDA_VISIBLE_DEVICES=1,3,5,7 \
 PRETRAINED_CKPT=/home/geyuan/code/dp23rss_fork/outputs/2026-09-05/23-54-36/checkpoints/epoch=1800-test_mean_score=0.613.ckpt \
 VIS_ENC_LAYERS=4 POOLING_TYPE=cls_token \
 PATCH_EMBEDDING_TYPE=conv PATCH_LEVEL_ADAPTER_CHANNELS=16 \
 PAIRED_NODE_SAMPLING=reachable PAIRED_LOSS_WEIGHT=1.0 \
 TEMPORAL_ADAPTIVE=false \
 BUDGET_LOSS_WEIGHT=0.1 TARGET_NUM_PATCHES=0.1 \
 BUDGET_LOSS_TYPE=upper_bound_lagrangian_elu BUDGET_BASE_COST_WEIGHT=0.0 \
 BUDGET_LAMBDA_INIT=0.0 BUDGET_LAMBDA_LR=0.01 \
 DOWNSTREAM_FINETUNE_START_EPOCH=1200 SEED=42 \
 bash scripts/roi/sh_train_foveation_pusht_stageB.sh
"
```

只有 A 窗口，独立均匀采样 reachable 节点；Temporal 权重不参与计算。

启用 **Spatial＋Temporal adaptive** 时，在上述完整命令中将 `TEMPORAL_ADAPTIVE=false` 替换为 `TEMPORAL_ADAPTIVE=true TEMPORAL_LOSS_WEIGHT=1.0`，其余设置保持一致。脚本开头同时给出这套完整示例，启用双窗口、随机锚点主动匹配和 Temporal loss。

运行 **双窗口主动匹配、关闭 Temporal loss 的消融** 时，使用 `TEMPORAL_ADAPTIVE=true TEMPORAL_LOSS_WEIGHT=0.0`。该消融包含额外窗口和匹配后的候选分布，不等同于原 Spatial only。脚本末尾的 Hydra 覆盖继续生效；预算模式和 alpha 的旧默认值不变，实验通过上述变量显式选择。

此前 reachable 主动匹配实现涉及以下六个文件。已有 dataset 配对、persistent workers 开关、恢复训练和 downstream 切换沿用现有实现，数据集文件未在该次修改中新增改动。本次实验讨论整理仅更新本文。

| 文件 | 实现内容 |
| --- | --- |
| `diffusion_policy/model/roi/foveation_net.py` | 删除单层分支，接入子树 delta、随机锚点、匹配和回退 |
| `diffusion_policy/policy/foveation_dp_transformer_policy.py` | 调用匹配采样、读取实际数量差、生成 Temporal mask 和私有计数 |
| `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py` | 跨卡汇总匹配率，保留多卡 Temporal loss 均值 |
| `configs/pusht256_foveation_dp.yaml` | 更新采样配置说明和兼容性要求 |
| `scripts/roi/sh_train_foveation_pusht_stageB.sh` | Spatial only、Temporal 和零权重消融示例，保留末尾 Hydra 覆盖 |
| `plan/study_temporal_adaptive.md` | 记录单层方案移除、主动匹配规则、实验观察、参数与验收边界 |

## 6. 验收与实验对照

此前实现的本地验证仅做 Python 静态编译、YAML 解析、`bash -n` 和 `git diff --check`，未新增独立测试脚本。本次仅更新文档，核对公式、日志口径与 Markdown 结构，并运行 `git diff --check`；无接口变化，不运行训练验证或新增测试脚本。下列运行时行为仍需服务器验收：

- 满树、最粗树、混合树及被遮蔽的后代 split：预计算 delta 与实际打包数量差一致；三层支持 `3/6/9/12/15`，更深树不依赖固定层数。
- 匹配成功、目标 delta 不存在、无有效第二窗口：锚点不被重采，回退候选合法，失败对 Temporal 为零而 Spatial 正常。
- 锚点两侧对称、匹配集合内均匀采样；基础树不被修改，完整后代决策保留，一臂保持基础树的实际叶子集合。
- 同 episode、不同有效时刻、各自动作目标；短 episode 和 padding 决策占位；train/val 及跨文件 episode 隔离。
- 已知收益差的梯度方向、零收益差、无有效配对、交换 A/B 后角色对应的梯度一致性；同一窗口粗细评估复用随机条件。
- Temporal loss 使用全部原始对数，匹配率仅使用有效窗口对数；跨卡分别汇总，全无有效对时无除零。
- 双窗口预算不重复计数；单卡/多卡均值与梯度尺度一致；AMP 跳步、梯度累积和恢复训练保持既有 lambda 行为。
- downstream 切换后 persistent workers 不再为新 epoch 加载额外窗口，Temporal 不参与优化；验证、rollout、视频和测速仍使用单窗口。

实验比较以下三种设置，使用相同 checkpoint、seed、预算、数据及训练安排：

| 对照 | `temporal_adaptive` | `paired_node_sampling` | `temporal_loss_weight` |
| --- | --- | --- | --- |
| 原 Spatial only | `false` | `reachable` | `1.0`（不参与） |
| Spatial＋Temporal | `true` | `reachable` | `1.0` |
| 双窗口主动匹配、关闭 Temporal loss | `true` | `reachable` | `0.0` |

Spatial＋Temporal 与零权重消融对比跨时刻 loss 的贡献；零权重消融与原 Spatial only 的差异同时包含额外窗口和主动匹配后的采样分布，不能将其单独归因于 Temporal loss。结合实际平均数量比较 IoU、完整推理延迟，以及数量变化是否与可重复的状态需求一致。不以中间层叶子占比、树形观感或数量方差增大单独判定成败；视频中每帧重新路由的曲线也不代表每个环境步都执行了一次策略。已有数字及定性观察来自用户服务器实验，本地未复现；受控收益验证、速度对照及完整运行时验收仍待完成。

### 6.1 优先补充的实验与指标

以下是后续研究建议，本次不新增日志、测试脚本或 LIBERO 接入。

| 优先关注点 | 建议记录 | 解释边界 |
| --- | --- | --- |
| 三组多 seed 对照 | 每个 seed 内使用相同 checkpoint、预算和训练安排，报告任务指标均值与标准差；同时记录 optimizer steps、处理窗口数和训练耗时 | 同步数不代表相同训练成本；零权重双窗口组用于分离 Temporal 目标的贡献 |
| 表现与计算量 | 不同预算下的任务表现—实际平均 patch 数曲线，并记录完整推理延迟；PushT 使用 IoU，LIBERO 使用相应成功率 | 相同预算配置不保证实际数量相同；相同平均数量也不保证延迟相同 |
| Temporal 信号质量 | 固定验证配对上的 gain 分布、排序一致率，以及 matched delta 直方图 | gain 接近零时排序易受噪声影响；匹配集中于 delta=3 不证明大子树监督充分 |
| 单候选收益稳定性 | 固定窗口、候选和两臂拓扑，重复采样 diffusion noise/timestep，记录 advantage 的均值、波动和正负符号稳定性，同时参考原始两臂 loss 尺度 | 每次重复内两臂仍共享 noise/timestep；区分单次符号比例与这些随机条件下的期望收益 |
| 轨迹内路由与响应 | 相邻观测的叶子划分变化、patch 数变化及变化量高分位数，区分普通时段与接触、遮挡、阶段切换 | 需考虑物体运动；固定树也能获得低变化量，不能只追求平滑 |
| 控制与困难阶段 | 动作突变、接触前后失败率、恢复成功率、长任务分阶段成功率 | 判断路由变化是否传递为控制收益，避免将必要动作变化视为不稳定 |

排序一致率只在匹配成功的验证对上评估，基础判据为：

```text
logit_diff = canonical_logit_A - canonical_logit_B
ranking_correct = (logit_diff * gain > 0)
```

同时报告按 `abs(gain)` 加权的一致率或按收益差大小分组的结果；全零收益差不提供排序证据。固定验证配对及评估随机条件有助于减少候选与评估噪声变化带来的混淆。上述诊断尚未实现，也不替代任务评测。

### 6.2 潜在收益与 LIBERO 验证动机

**相同资源增量下的分配效率。** Spatial 分别学习每个窗口 split 是否有帮助，Temporal 补充相同数量增量下谁更有帮助的比较，可能在预算紧张、不同操作阶段视觉需求差异较大时更有价值。但它不直接把 A 的 token 转移给 B，实际资源分配由 Spatial、Temporal 和预算共同形成。

**额外的相对收益监督。** 即使两侧 split 都有帮助，Temporal 仍能提供收益排序；这可能帮助 router 学习随任务状态变化的偏好。反面是它也会相对压低收益较小一侧的偏好，权重过大可能与 Spatial 信号产生拉扯，不能假定越强越好。

**共享条件下可能减少共同波动。** 同 episode 配对以及共享 view、相对观测位置、diffusion timestep，可能使两侧 advantage 存在共同波动，差分有机会消去其中一部分：

```text
Var(advantage_A - advantage_B)
    = Var(advantage_A) + Var(advantage_B)
      - 2 * Cov(advantage_A, advantage_B)
```

这一解释依赖协方差，差分并不保证比单侧信号方差更低。独立 diffusion noise、不同动作目标和窗口内容仍会引入噪声；当前 topology_div 曲线更平稳与降噪假设相容，但不能证明该机制成立。

Temporal 目标没有直接要求相邻时刻拓扑相同，A/B 也不一定相邻。理想行为是需求不变时减少无意义变化，关键事件发生时及时重新分配，而不是始终维持固定树。

LIBERO 是合理的后续验证方向，尤其可以观察接触、遮挡和多阶段操作。但任务更难不保证 Temporal 更有效：瓶颈与视觉分辨率分配有关时才更可能受益；若主要受限于动作建模、数据覆盖或长程规划，router 改善可能仍不提高成功率。优先用三组多 seed 对照、表现—实际 patch 数曲线和关键阶段失败分析，检验当前稳定性观察能否转化为实际收益。
