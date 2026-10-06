# 四叉树方法快速迁移与复现指南

面向负责迁移实现的大模型和研究人员：以当前 LIBERO 实验为参考，保留 Spatial、Temporal、预算和三阶段训练的关键语义，再适配目标模型。本文中的参数是参考配置，不代表其他模型上的最优值；π0.5／RoboTwin 的具体版本、路径和 API 需接入时核实。

目录：[参考入口](#reference) · [训练流程](#training) · [核心算法](#algorithm) · [迁移接口与示例](#transfer) · [必要观测](#logging) · [实施与验收](#acceptance)

<a id="reference"></a>
## 1. 方法与参考入口

方法在昂贵视觉编码前选择四叉树叶子，用不同尺度分配空间计算，并通过 Temporal 目标学习不同时刻的计算分配。叶子完整、无重叠地覆盖输入；粗细区域都转换成相同局部尺寸的 patch。它不同于编码后 pruning 或固定 ROI；预算退火本身也不等于学到了时间自适应。

优先阅读并复现以下两份实际实验入口，再沿调用关系查看基础脚本和模型：

- [Stage A remaining6](../scripts/roi/sh_train_foveation_libero_stageA_remaining6.sh)：按任务启动多拓扑适配，调用 [Stage A 基础脚本](../scripts/roi/sh_train_foveation_libero_stageA.sh)。
- [Stage B remaining6](../scripts/roi/sh_train_foveation_libero_stageB_remaining6.sh)：从 Stage A 运行目录选择 checkpoint，调用 [Stage B 基础脚本](../scripts/roi/sh_train_foveation_libero_stageB.sh)，一次运行完成路由学习与下游适配。

运行前设置 Stage A 的 `HDF5_FNS`、Stage B 的 `LOG_TIMES`，检查 GPU、数据／环境路径及环境变量覆盖。实际执行任务由未注释的列表项决定，不能从 `remaining6` 文件名推断任务数。最终有效配置还受基础脚本、[LIBERO 配置](../configs/liberoft_foveation_dp.yaml)及命令行覆盖影响。

<a id="training"></a>
## 2. 三阶段、两次运行

| 运行与阶段 | 更新对象 | 当前参考配置与目标 |
|---|---|---|
| Stage A：第一阶段，多拓扑适配 | 下游视觉编码器与动作模型（按配置冻结例外模块） | 80% `level:2` 全细树，20% `random:0.05-0.99` 随机混合树；使用原生动作目标 |
| Stage B 的 epoch 0–599：第二阶段，路由学习 | 路由分支；下游冻结且处于 eval 模式 | Spatial＋Temporal＋预算；默认加载 Stage A best checkpoint，继承 dataset 配置及数据划分种子 |
| 同次 Stage B 的 epoch 600–1200：第三阶段，下游适配 | 下游视觉／动作参数；路由固定 | 固定选定的 EMA 路由，关闭 Temporal 配对，使用原生动作目标 |

**第二、第三阶段共享同一次 LR 调度。** Stage B 共 1201 epochs，默认初始 warmup 为 1000 steps，此后 cosine 衰减；epoch 600 切换更新对象时不重启 scheduler、不重新 warmup。第一阶段是独立运行，有自己的调度。

预算目标另有一条 cosine 调度：归一化目标从 0.75 在前 600 epochs 退火至 0.1，此后保持不变。它与学习率 cosine 是两个不同调度；第三阶段不再优化预算目标。

当前 Stage B 默认启用 Temporal；Spatial／Temporal 权重均为 1，预算权重 0.1，预算模式为 `upper_bound_lagrangian_elu`，基础成本系数为 0，lambda 初值为 0、更新率为 0.01。迁移先保持训练关系，再按目标模型的动作损失尺度调整权重和训练时长。

阶段切换同时处理参数、训练模式、buffer、optimizer 残留状态和 EMA；关闭配对的开关应传到 persistent workers。恢复训练保存并还原阶段、live/EMA、optimizer、scheduler、步数、预算调度和 lambda，避免重复切换或重置调度。具体实现见 [workspace](../diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py) 的 `_set_stage_b_phase` 和 `load_payload`。

<a id="algorithm"></a>
## 3. 核心算法：迁移时保持这些语义

### Spatial

每窗口随机选相机和相对观测帧，在基础树的 reachable 内部节点中采样。reachable 只取决于祖先是否 split，本节点 stop/split 都可入选。两臂只改变该父节点，保留所有后代决策，所以其中一臂的叶子集合与基础树一致；两臂实际叶子数差不一定为 3。

同窗口 stop/split 复用图像增强、动作目标、noise、时间和 mask；冻结下游并关闭 dropout 等随机训练行为。用统一为“split 为正”的 logit z 计算：

```text
q_i = sigmoid(z_i)
A_i = loss_stop_i - loss_split_i
L_spatial = -mean_i(q_i * stop_gradient(A_i))
```

动作模型需返回逐窗口损失 `loss_B`。paired 阶段的基础树动作损失只是 detached 诊断，router 通过上述目标更新。当前 DP 的 masked MSE 仍除以完整展平维度；新模型沿用自身原生动作损失和 mask 分母，明确量纲差异。

### Temporal

每个原始样本增加同 episode、不同真实决策时刻的窗口，按 `[全部 A, 全部 B]` 组成 2P 个窗口。无法配对时复用 A 作为占位并标 valid=false；占位继续参加 Spatial 与预算。候选索引应处理 episode 隔离和 padding，验证集不共享训练配对开关。

两窗口选同相机与同相对帧，先各自采样 reachable 节点，再随机选 A/B 为锚点，在另一侧匹配**相同实际叶子增量**的节点；增量由保留后代决策后的完整子树决定。无匹配时保留默认节点。动作评估后以实际 packed 数量差确认 matched。

A/B 使用各自动作与独立 noise，共享时间变量；每个窗口内部的两臂继续共享随机条件。令 A_a、A_b 为两窗口的 Spatial advantage：

```text
m_p = valid_p AND (Delta_a == Delta_b)
g_p = A_a - A_b
L_temporal = -sum_p(m_p * sigmoid(z_a-z_b) * stop_gradient(g_p)) / P
match_fraction = sum_p(m_p) / sum_p(valid_p)
L_router = w_spatial * L_spatial + w_temporal * L_temporal + L_budget
```

Temporal loss 分母是全部原始 P，包含失败和无效占位；Spatial 分母为全部 2P。Temporal 不除以 matched 数或 Delta。有效数为零时标注无有效配对，不能把返回的零匹配率当成真实失败率。`temporal_adaptive=true` 且权重为零仍保留配对与匹配，不等同于关闭 Temporal。

### 预算

软期望 E 是叶子概率之和，硬数量 N 是实际叶子数，padding 长度是模型实际分配的序列长度，三者分别记录。每视图每帧的归一化预算 rho 换算为 `N_target=N_min+rho*(N_max-N_min)`；当前 [0,1] 内 float 被解释为 rho，绝对数与归一化数不要混用。

当前使用的 ELU 上界模式如下，其他模式直接参考 [FoveationNet](../diffusion_policy/model/roi/foveation_net.py) 的 `budget_loss`：

```text
u = (mean(E)-N_min)/(N_max-N_min); gap = u-rho
L_budget = budget_weight * (lambda*gap                         if gap>0
                           else alpha*tau*ELU(gap/tau))
lambda_next = max(0, lambda_used + lambda_lr*gap_update)
```

当前 alpha=0，因此目标以下不额外奖励继续省 token。`gap_update` 使用整个 optimizer 累积窗口的全局软期望；lambda 在成功更新后推进，日志记录本次使用的旧值。软目标不是逐帧硬上限。

跨卡先汇总 sum/count；ELU 目标使用**本次 forward 的全局均值**，不能平均各卡非线性 loss，也不等于整个梯度累积窗口只计算一次 ELU。保留实际观测数归一化及尾部短 batch 处理。预算概率的 `sampled_relaxed`／`clean_expected` 模式应写入配置。

<a id="transfer"></a>
## 4. 迁移接口与 π0.5＋RoboTwin 示例

先跑通目标原生 baseline，再接入以下接口；目标仓库的具体 API 通过阅读确认。

| 接入点 | 保留的契约／目标适配 |
|---|---|
| Dataset | RGB 值域、归一化、相机顺序、语言、状态／动作定义、观测与动作时间对齐；按 episode 隔离 train/val；提供 Temporal 双窗口及有效标记 |
| Router／patch bank | 图像金字塔与内部节点对齐特征；输出 logits、基础树及软预算统计；可复用同一 bank 构造两臂 |
| 叶子／视觉模型 | 输出 patches、bank 索引、原图坐标／尺度、有效 mask 和数量；正确处理多相机及可变长度，不用打包序号冒充空间坐标 |
| 动作损失 | 显式接收 time、noise、action target、mask，返回逐窗口 loss；支持两臂复用随机条件 |
| Evaluator | 沿用官方动作转换、chunk/replan、reset、并行环境和成功判据；在线预处理与离线一致 |

**SigLIP 与 prefix：** 保留兼容的预训练 patch projection 和 Transformer，将叶子 patch 接入视觉序列，适配位置／尺度编码、视觉 attention mask 和多模态 prefix。DP 的 pooling 不强制迁入 VLA，优先保留叶子 token 序列。先检查全细树与原生 dense 路径的差异，再处理粗树与混合树。

合法几何要求输入边长可被 `patch_size * 2^(L-1)` 整除。以 patch14、L=3、224×224 为例，根网格 4×4，N_min=16、N_max=256，rho=0.1 对应软目标 40。若目标尺寸不合法，明确 resize/crop/pad 方案；不要照搬 DP 的 patch16。坐标属于原图区域，padding mask 要贯通视觉编码器和 prefix；静态 bucket 需同时报告实际与 padded 长度。

**Flow matching：** 使用 π0.5 原生连续时间、插值、velocity target 和 mask 分母；Spatial／Temporal 的随机条件关系保持不变。把源 `diffusion_loss` 映射为 `action_loss` 或 `flow_matching_loss`，不要称为 epsilon MSE。目标损失尺度变化后重新校准路由与预算权重。

**RoboTwin：** 核实三相机实际字段及顺序、语言输入、双臂动作表示／单位／左右顺序／夹爪范围，不预设固定动作维度。分别对齐 clean/randomized、single-task/co-train 的数据与评测配置；若用 XPolicyLab adapter，确认它保持所选 RoboTwin 版本的官方协议。记录版本与 checkpoint，不宣称未验证的收益。

<a id="logging"></a>
## 5. 必要日志与可视化

沿用目标 trainer 的轻量日志周期，图片和额外诊断单独低频执行。下表保留源键语义，目标可改名但说明映射；新目标框架需要补齐有效计数及长度记录。

| 观测组 | 核心键／内容 | 关键解释 |
|---|---|---|
| 训练与表现 | `train/total_loss`、原生动作 loss、适用 `val/*`、`test/mean_score`、每任务分数及 episode 数 | 区分优化目标、detached 诊断和闭环表现；验证不伪造 paired 指标 |
| 阶段与 LR | `train/lr`、`train/downstream_finetune_active`、epoch、step、损失权重 | 区分 batch step 与 optimizer step；阶段切换处 LR 连续 |
| Spatial | `train/paired_loss`、`train/paired_stop_diffusion_loss`、`train/paired_split_diffusion_loss`、`train/paired_advantage_*` | paired 未乘权重；advantage 同看均值、绝对值和正值比例，不据此宣称闭环收益 |
| Temporal | `train/temporal_loss`、`train/temporal_match_fraction`；补原始／有效／matched 配对数 | loss 分母 P，匹配率分母 valid；零有效有明确标记 |
| 预算与数量 | `train/budget_loss`、`train/budget_lambda`、`train/expected_num_patches_mean`、`train/target_num_patches`、`train/num_patches_mean` | budget_loss 已含权重；lambda 为旧值；区分软期望、硬数量与目标 |
| 推理长度 | `test/num_patches_mean`、逐相机数量、`test/total_num_patches_mean`；补实际／padded 视觉长度 | 来自真实 policy 调用；说明每视图均值、跨视图总量、历史帧和调用次数口径 |
| 图片／视频 | `vis/train_foveation/{camera}`、代表性原始 rollout 与各相机四叉树视频 | 展示叶子布局及数量，附 task、episode、camera、时间和 checkpoint |

普通训练标量当前部分为主 rank 本地值，不能统称全局均值；比率跨卡先合计分子／分母。任务等权与 episode／调用加权分开命名。明确使用 live 或 EMA、train 或 eval 模式；固定诊断保存样本与种子，不推进训练 RNG。W&B 由主进程提交，并保留本地标量和媒体。

执行路由与视频重算路由要分清：当前 LIBERO 视频可能通过 `get_foveation_outputs` 事后重算，须标注 `recomputed`，不能冒充动作执行时的布局。真实调用数量可绘制逐相机曲线；一个 action chunk 中没有新 policy 调用的环境帧，不应被算作新路由。

拓扑差异、统一分辨率收益、采样动作 MSE 和梯度分项作为可选排查项，查看 workspace 实现即可。注意 `sample/topology_div` 是同相机／相对帧的样本间差异，非轨迹内相邻变化；paired 阶段 detached 动作损失的 router 梯度为零是正常行为。

首次正确复现不要求完整 invocation 数据表、分项计时或上传恢复系统。**若报告计算效率或加速收益，应测量包含预处理、router、视觉编码和动作生成的端到端延迟分布**，注明硬件、batch、warmup 和同步方式；诊断／媒体开销与模型测速分开，通信／模拟器耗时另记。

<a id="acceptance"></a>
## 6. 实施顺序与最小验收

实施顺序：原生 baseline → 数据与多拓扑视觉接入 → 第一阶段适配 → Spatial＋Temporal＋预算 → 同次运行切换下游适配 → 官方闭环。日志随模块接入，不等训练结束后补采。

核心源码入口：

- [FoveationNet](../diffusion_policy/model/roi/foveation_net.py)：几何、bank、`pack_topology`、`sample_paired_splits`、预算和叶子可视化。
- [Policy](../diffusion_policy/policy/foveation_dp_transformer_policy.py)：视觉接入、`_diffusion_loss_per_sample`、`_compute_paired_loss`、动作推理。
- [LIBERO dataset](../diffusion_policy/dataset/libero_dataset.py) 与 [PushT dataset](../diffusion_policy/dataset/pusht_image_dataset.py)：episode 配对、padding 下的决策时间和 worker 开关。
- [Workspace](../diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py)：阶段、EMA／恢复、跨卡预算与日志；[LIBERO runner](../diffusion_policy/env_runner/libero_socket_runner.py)：实际调用计数、任务聚合和视频重算。

未来服务器验收保留以下最小场景；本次文档编辑不执行训练：

- [ ] 全粗／全细／混合树完整覆盖；有效数量等于 mask 之和，padding 不影响有效输出。
- [ ] paired 只改变选中父节点、保留后代；一臂等于基础树，两臂随机条件一致。
- [ ] Temporal 成功、无匹配、无效占位及全部无效时，实际 Delta、损失分母与计数正确。
- [ ] 不等 rank batch／尾部累积窗口下聚合正确；阶段切换与恢复保持参数更新对象、EMA、LR、lambda 和 worker 配对状态。
- [ ] 离线在线图像／动作转换一致；baseline 与迁移模型使用相同官方评测协议。
- [ ] 有核心 loss、闭环分数、四叉树数量与布局／视频证据；执行与重算来源可区分，未测收益明确标为未验证。

交接报告简要列出实际改动文件、有效配置／版本、指标映射、验证结果和剩余问题。Spatial、Temporal、预算和三阶段均属于完整方法；额外研究诊断可后续补充。
