# Spatial adaptive 与 Temporal adaptive：跨时间计算分配

背景见[项目概述](quadtree_introduction.md)，现有 paired 与三阶段训练见[训练机制](fix_action_backward.md)。本文记录最新实验结论，以及下一步准备实现的 episode 内跨时间分配学习。**Temporal adaptive 尚未实现；现有分段预算已接入代码。**

## 1. 目标与命名

目标是在测试时随操作进程、接触状态和任务需求改变每帧的 `num_patches`，既能增加，也能减少。继续使用现有 stop/split router，不新增预测数量的决策模块。

| 名称 | 本文中的含义 | 状态 |
| --- | --- | --- |
| Spatial adaptive | 现有局部 paired 路由学习，以动作预测收益学习 stop/split，配合平均预算控制整体成本 | 已有实现，作为基线 |
| Temporal adaptive | 同一 episode 的不同决策时刻，显式比较“同样一份额外 patches 给谁更有价值”，训练现有 router 的相对分配偏好 | 下一步实现 |

这是对**学习机制**的命名，不是把模型能力完全分开。Spatial adaptive 已可能产生随状态变化的时间数量差异；不能将其描述为固定每帧数量或完全没有时间自适应。Temporal adaptive 增加的是显式跨时刻比较，不是首次让数量变得可变，也不等于新增时序网络。

现有 paired 提供随状态变化的局部细分收益；平均预算允许观测间补偿，但不直接偏好某种时间分配。例如，软期望数量 40/40、25/55、55/25 的均值相同，平均预算项无法区分。lambda 与 alpha 提供共同的数量成本，不识别哪个时刻更值得增加计算。

现有时间差异主要是局部决策积累的结果；新增机制要直接学习跨时刻的**边际资源价值**。不以数量方差增大、接触时必然更多或人为制造波动为目标。

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

## 4. 下一步：episode 内的 Temporal adaptive

### 4.1 比较边际资源价值

视频压缩的帧间码率分配、混合精度量化的位宽分配，都提供一个思路：在相同资源增量下，优先选择能减少更多任务误差的分配。本方案借鉴这一原则，并非某篇工作的直接复现。

对应问题是：**在当前拓扑附近，同样增加一些 patches，放在时刻 A 的收益，是否大于放在时刻 B 的收益？** 原始预测误差更大，不代表增加 patches 更有帮助；也不直接监督 A 的绝对总数量必须大于 B。

### 4.2 对 batch 中每个样本，在同一 episode 再取一个窗口

1. A 保留当前样本的 obs 窗口及其动作目标；B 从同一 episode 的另一个决策时刻随机采样，使用 B 自己的 obs 窗口和动作目标。B 可以早于或晚于 A，这只是离线训练配对。
2. 不跨 episode、不跨训练与验证划分。Merged 数据集还要保留源数据集身份，不能只靠可能重复的 episode 编号。窗口 padding 和 obs/action 对齐沿用现有 sampler。
3. 排除相同决策时刻，避免把 padding 形成的重复状态误当成有效时间对比。可设置最小时间间隔以减少近重复窗口；具体间隔与短 episode 的回退规则在实现时确定，不预设接触标签。
4. 当前 `n_obs_steps=2`。首版建议干预 A、B 窗口中各自的最新观测帧，其他帧在该窗口的粗细对照中保持不变，避免把历史帧与最新帧的相对重要性混入跨决策时刻比较。

这条路线研究一条轨迹上不同操作阶段的资源分配，不是只比较同一次策略输入中的两个相邻观测帧。**测试时仍输入正常 obs 窗口，不需要第二个窗口或未来信息。**

### 4.3 构造等数量增量的粗细候选

A、B 分别构造合法的粗、细两种拓扑，细相对粗增加相同的 $\Delta N$。例如：

| 比较方案 | A 最新帧 | B 最新帧 | 两帧合计 |
| --- | --- | --- | --- |
| 额外计算给 A | 34 | 40 | 74 |
| 额外计算给 B | 31 | 43 | 74 |

两侧基础数量可以不同。这里 A 的基础数量为 31，B 为 40，增量均为 3；不交换整帧拓扑，不要求两侧使用相同空间位置，也不要求获得增量的一侧总数更大。窗口内其余帧在两个方案中相同，因此整对窗口的合计数量也相同。

**不能直接假设现有 paired 干预都只差 3。** 当前 `sample_paired_splits()` 保留后代决策，打开一个节点可能释放更深子树。新候选需要保证等增量并按实际叶子数核对；可以从一次叶节点四分的 +3 候选起步，但可达性、后代停止和最大细分边界必须一起处理。没有合法匹配候选时不能伪造偏好。

等数量只是首版成本尺度，不保证等延迟。“paired 收益除以实际增量”可作为较轻的替代思路，但单独归一化原 paired 并不会自动形成跨时间竞争，首选仍是等增量对照。

### 4.4 评估收益，训练相对分配偏好

分别得到四个损失：$\ell_A^{coarse}$、$\ell_A^{fine}$、$\ell_B^{coarse}$、$\ell_B^{fine}$。每个窗口使用自己的动作目标；同一窗口的粗细评估复用完全相同的输入增强、diffusion noise、timestep 和 mask。A/B 之间也需使用可比较的 timestep 采样与损失归一化，避免噪声尺度主导排序。

定义两种分配的合计损失：$J_A=\ell_A^{fine}+\ell_B^{coarse}$，$J_B=\ell_A^{coarse}+\ell_B^{fine}$，相对收益为 $g=J_B-J_A$。

- $g>0$：同样的增量更值得给 A。
- $g<0$：更值得给 B。
- 差异接近零：学习信号应弱，不强迫数量差异。

这也等价于比较两侧各自的 paired 收益：$g=(\ell_A^{coarse}-\ell_A^{fine})-(\ell_B^{coarse}-\ell_B^{fine})$。新信息不是额外的任务标签，而是把两侧局部收益在相同成本下组织成显式相对学习信号。一个采样候选只代表局部边际价值，不能当成整帧的完整价值。

**候选 loss，尚未实现：** 对两个候选节点的 canonical split logits $z_A,z_B$，可令 $p_A=\operatorname{sigmoid}(z_A-z_B)$，表示在“给 A”与“给 B”两个候选之间的相对偏好，再使用 $L_{temporal}=-\operatorname{mean}[p_A\operatorname{stopgrad}(g)]$。正收益提高 A 相对 B 的细分偏好，负收益反向；共享 router 参数由两侧梯度共同更新。这里的 $p_A$ 不是新增网络，也不是推理时的独占资源开关。

这是待验证的局部代理目标，不是整树策略的无偏梯度或全局最优分配保证。最终 loss 形式、权重及合法候选采样范围在代码实现前确定，不将该候选写成已经通过实验的方案。

### 4.5 保留 Spatial adaptive 与平均预算

拟使用 $L_{total}=w_sL_{spatial}+w_tL_{temporal}+L_{budget}$，其中 $L_{spatial}$ 是现有 paired，$w_s$ 沿用 `PAIRED_LOSS_WEIGHT`，Temporal 项使用独立权重。下游动作路径仍在 router 阶段冻结，收益停止梯度；后期 downstream finetune 保持原规则。

Temporal 项回答“这份资源优先给谁”，不能替代判断“是否值得使用这份资源”：两侧可能都值得增加，也可能都不值得。保留 Spatial paired 与预算，使最终 router 仍可两边都增加或都不增加，不能在实际推理中强制每对只允许一方细分。

配对候选等总成本也不能保证自由路由后的全局数量满足上限，因此保留 cosine target、lambda 和 alpha=0 的预算项。建议开启 Temporal 时，将 A/B 的原始自由路由观测各统计一次并按观测数求均值；不能把四个反事实候选作为四份真实预算样本，也不能因窗口翻倍把 loss 或乘子更新次数翻倍。

## 5. 代码实现范围与待定细节

本次只更新文档。后续按 AGENTS.md 保持实现简短，新增 public 函数添加简洁 docstring，tensor/NumPy 变量保留 shape 后缀。

| 后续文件 | 预期修改 |
| --- | --- |
| `diffusion_policy/dataset/pusht_image_dataset.py` | 为 PushT 与 Merged 数据集接入 episode 内窗口配对，保留各自 obs/action 和身份信息；关闭时沿用旧样本路径 |
| `diffusion_policy/model/roi/foveation_net.py` | 构造并筛选等数量增量的合法拓扑候选，复用现有打包与路由输出 |
| `diffusion_policy/policy/foveation_dp_transformer_policy.py` | 评估 A/B 粗细收益，计算 Temporal 辅助 loss，保留 Spatial paired、预算及推理接口 |
| `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py` | 核对双窗口预算统计、全局均值、一次 step 一次 lambda 更新，以及训练阶段切换与最少日志 |
| `configs/pusht256_foveation_dp.yaml` | 增加显式开关、Temporal 权重及必要采样配置；默认关闭，方便复现现有基线 |
| `scripts/roi/sh_train_foveation_pusht_stageB.sh` | 增加对应环境变量和启动示例，保留末尾 Hydra 覆盖 |

优先复用 `SequenceSampler.indices` 与 replay buffer 的 episode 边界，在 PushT 数据集内完成索引映射；只有确有需要时再修改 `diffusion_policy/common/sampler.py`，不先改造通用采样体系。

实现前需明确：最小时间间隔及无有效配对时的处理、等增量候选的节点范围、Temporal loss 权重与收益噪声控制、Spatial paired 是否在 A/B 上对称计算及如何复用评估。保留这些为待定项，不引入未经确认的复杂调度。

每个原始样本的 Temporal 比较需要两个窗口各自的粗细评估，概念上为四次窗口评估，原 Spatial paired 为两次。若继续单独计算原 Spatial 候选，还可能有额外开销；实际耗时取决于候选与特征复用、批处理和内存。不能承诺训练仅增加固定比例，推理则无需增加配对计算。

## 6. 验证重点与日志

- **实现正确性：** 同 episode、不同有效时刻，各自动作目标正确；比较分支等增量且合法；粗细评估随机条件一致。交换 A/B 应反转偏好，收益相等时不产生强制偏好。关闭 Temporal 应回到原训练路径，预算不能重复计数，阶段切换不应继续优化冻结的 router。
- **区分机制收益与额外训练量：** 除 Spatial 基线与 Spatial+Temporal 外，必要时加入“同样双窗口训练、Temporal 权重为零”的对照，避免把更多数据或评估带来的变化归功于新 loss。
- **研究验收：** 在平均数量相近时比较 IoU 与完整推理延迟，并看数量变化是否与可重复的状态需求一致。数量方差增大不是成功标准；也不能将视频中每帧重新路由的曲线视作每个环境步都实际执行了一次策略。
- **日志保持少量：** 复用现有数量、target、lambda、paired/budget 梯度与视频。新目标接入时只优先记录其 loss；仅当合法配对经常被过滤时，再考虑有效配对比例，不预先扩展大量 W&B 指标。
- **验证边界：** 本次为文档更新；已有数字来自用户服务器实验，未在本地复现。后续代码按 AGENTS.md 只做本地静态编译及配置、脚本和 diff 检查，不新增独立测试脚本；训练稳定性、多 GPU 行为与实际收益由服务器实验验证。
