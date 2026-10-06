# 四叉树多视角扩展计划

更新：2026-10-06。本文整理多视角架构、tensor 维度与 loss 计算方案，尚未实施架构修改。以 TCL 的 static（代码 key 为 `image`）和 gripper 双相机为例。

## 1. 设计原则与训练阶段

**View 与 time 是交叉维度：每个样本随机选择一个 view 做 spatial paired 干预，temporal 在同一 view 内比较，所有拆分收益均由完整多视角动作模型评价。** Temporal 比较同一相机在两个时间窗口中的细化收益，不要求树结构、空间位置或 token 数随时间保持一致，不增加跨相机节点配对。

- **各 view 独立视觉路由，共享低维条件编码。** 同一份低维状态通过 policy 层唯一的 `router_condition_projection` 投影一次，结果供所有 view 使用。每路 `FoveationNet` 的图像编码器、内部条件处理与融合层、split 预测参数按 view 独立；动作视觉编码器 `FoveationObsEncoder` 也按 view 独立。各路复用同一套单视角类。
- 默认每个样本均匀采样一个干预 view；开启 temporal 时按时间对采样，a/b 共享 view 和相对帧索引。只修改所选节点，其余视角、帧和树决策保持基础状态。
- 同一时刻各 view 的视觉特征沿特征维拼接，再拼接低维状态，输入一个 Diffusion 动作模型。
- 每个 view 独立约束平均预算；第一版不研究跨 view 转移预算，不添加跨 view 一致性或收益排序 loss。

现有实现共享路由与动作视觉编码器、每个样本随机选择一个 view，并使用跨 view 平均预算。扩展保留随机选择 view 的方式，重点拆分参数与预算。每个样本始终只评价 stop/split 两臂，避免反事实动作评价数量随 V 线性增长；枚举所有 view 仅作为小规模对照，不作为默认训练流程。

| 阶段           | 训练对象与目标                                                                                                                         |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Stage A        | 随机合法树下训练动作视觉编码器和 Diffusion 动作模型。                                                                                  |
| Stage B paired | 冻结动作路径，以 spatial + temporal（可选）+ budget 更新路由相关参数。动作 diffusion loss 在`no_grad()` 下提供 advantage，不直接反传。 |
| 后期下游适配   | 固定确定性 routing，使用 action diffusion loss 训练动作视觉编码器和动作模型；固定参数不代表所有观测使用同一棵树。                      |

四种 loss 属于整体训练流程，不是同时反传给同一组参数的四个目标。TCL 当前默认关闭 temporal。

## 2. Tensor 维度与特征融合

### 2.1 基础数据流

本节 B 表示观测窗口数量。V 为视角数，T 为窗口内历史帧数（`n_obs_steps`），C/H/W 为图像维度；M 为可拆分内部节点数，K 为完整 patch bank 的候选数，N 为选中 leaves 的 padding 长度，P 为 patch 像素边长；D 为每路每帧视觉特征维度，S 为低维状态维度，L/A 为动作 horizon / 单步动作维度。

首版各 view 使用相同图像尺寸、树配置及特征维度，逻辑 shape 如下：

| Tensor                                   | 逻辑 shape        |
| ---------------------------------------- | ----------------- |
| 图像                                     | `[B,V,T,C,H,W]`   |
| split logits / hard split 决策           | `[B,V,T,M]`       |
| 完整 patch bank                          | `[B,V,T,K,C,P,P]` |
| 选中的 patches                           | `[B,V,T,N,C,P,P]` |
| 有效 token mask                          | `[B,V,T,N]`       |
| hard leaves 数 / soft expected leaves 数 | `[B,V,T]`         |
| 空间 tokens 聚合后的视觉特征             | `[B,V,T,D]`       |
| 同时刻多视角特征拼接                     | `[B,T,V*D]`       |
| 加入低维状态后的动作条件                 | `[B,T,V*D+S]`     |
| diffusion 预测目标（如动作噪声）         | `[B,L,A]`         |

**V 在路由和 loss 统计阶段保留身份，在动作条件处融合进特征维；T 保留为观测时间轴。** `[B,V,T,D]` 必须先转置为 `[B,T,V,D]` 再 reshape 为 `[B,T,V*D]`，不能直接 reshape。

实际实现沿用按 view 的字典和独立模块，每路输入 `[B,T,C,H,W]`、输出 `[B,T,D]`，最后 `cat(..., dim=-1)`。每路 patches 可以有自己的 `N_v`，无需为了显式 V 维统一 padding。现有 `FoveationNet` 入口为 `[B,C,T,H,W]`，单路调用前交换 C/T。

```text
image   [B,T,C,H,W] → FoveationNet_static  → ObsEncoder_static  → [B,T,D]
gripper [B,T,C,H,W] → FoveationNet_gripper → ObsEncoder_gripper → [B,T,D]

cat(两路特征, 低维状态) → [B,T,2D+S] → 一个 Diffusion 动作模型
```

路由的共享边界如下。先将 `router_condition_keys` 指定的低维状态归一化、拼接，通过唯一的 `router_condition_projection` 得到条件特征 `[B,T,D_r]`，其中 `D_r` 为 `router_cond_dim`。所有 view 复用该结果，再由各路 router 结合自己的图像特征生成树决策。

```text
同一份低维状态 → 共享 router_condition_projection → 条件特征 [B,T,D_r]
                                                      ├→ static router  + static 图像特征  → static 树决策
                                                      └→ gripper router + gripper 图像特征 → gripper 树决策
```

“共享低维条件编码”具体指 policy 层的这一个投影模块；`FoveationNet` 内部进一步处理条件、融合图像和预测 split 的参数仍按 view 独立。Stage B 路由训练时，共享投影接收各路路由损失汇总的梯度，在一次 backward 中联合更新。

动作路径将归一化的低维状态按现有 mask 规则处理，与所有 view 的视觉特征拼接一次，形成 `[B,T,V*D+S]`，再由动作 Transformer 的 `cond_obs_emb` 统一投影。动作路径使用自己的条件投影，与路由的 `router_condition_projection` 是两个模块。

### 2.2 展平维度的使用范围

| 表示          | 用途与注意事项                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| `[B,V,T,...]` | 逻辑表示，明确视角、时间与统计范围；实现可用字典 key 表示 V。                                          |
| `[B*V,T,...]` | 适合批量调用共享网络。独立权重方案按 view 调用；展平本身不会产生跨视角融合。                           |
| `[B,V*T,...]` | 当前方案不采用。未来若用于联合 attention，需保留 view/time 身份及正确 mask，不能把换相机当作时间推进。 |

现有跨 view `cat(dim=0)` 按 `[全部 static 样本, 全部 gripper 样本]` 排列，实际索引为 `v*B+b`，对应 `[V,B,...]` 展平，不等于 `[B,V,...]` 直接 reshape。

## 3. View 采样与 paired 批量评价

### 3.1 每个样本只选择一个干预 view

关闭 temporal 时，对 B 个普通窗口各自均匀采样：

```text
selected_view_B  = randint(V, size=B)
selected_frame_B = randint(T, size=B)
```

同一个 batch 的不同样本可以干预不同 view，例如 `[view 2, view 7, view 2, view 9]`。每个样本仅在选中的 view/frame 内采样一个 reachable 节点，构造 stop/split 两臂；其余 view 仍作为完整动作条件的一部分。

默认采用独立均匀采样，不要求每步覆盖所有 view。每路平均获得约 B/V 个样本的 paired 监督。若相机覆盖波动明显，可后续采用与样本内容无关的随机均衡分配，保持每个样本边际上均匀选择 view；第一版不引入按收益优先采样。

### 3.2 开启 temporal：按时间对采样 view

**本节及后续 temporal 公式中，B 表示时间窗口对数量，Q=2 表示窗口 a/b。** T 仍是每个窗口内部的历史帧数。

```text
images_B_Q_V_T_C_H_W : [B,2,V,T,C,H,W]
actions_B_Q_L_A      : [B,2,L,A]
pair_valid_B        : [B]
selected_view_B     : [B]
selected_frame_B    : [B]
```

先为每对采样一个 view 和一个相对帧索引，再供 a/b 共用。例如 a 为第 10/11/12 帧、b 为第 20/21/22 帧，索引 1 对应第 11 和第 21 帧，保证历史位置相同。不能让 a/b 独立选择 view，否则会混合时间变化与相机变化。

对所有输入生成基础树，在所选 view/frame 内按现有随机锚点与主动匹配规则分别选取 a/b 的 reachable 节点，尝试匹配实际 leaves 增量。节点位置、层级不要求相同。仅切换所选父节点，保留后代决策。

实际按 `[全部 a 窗口, 全部 b 窗口]` 合并为 2B 个窗口，view/frame 索引对应重复两遍。该布局为 `[Q,B,...]` 展平，还原逻辑 `[B,Q,...]` 时须转换顺序。

Temporal 主动匹配会影响 spatial 的候选分布；`temporal_loss_weight=0` 仍保留双窗口和匹配流程，不等于关闭 temporal。具体规则见 [temporal 方案](study_temporal_adaptive.md)。

### 3.3 两臂评价与随机数

每个窗口仅构造两个完整多视角动作条件：

| 版本 | 所选 view/frame | 其余 view/frame |
| --- | --- | --- |
| stop | 所选节点强制 stop | 基础树 |
| split | 所选节点强制 split | 基础树 |

所有 view 的特征输入同一个冻结的动作模型，在 `eval()` / `no_grad()` 下评价。a/b 各自使用自己的观测、基础树和动作目标。

- 同一窗口 stop/split 严格共享 diffusion noise、timestep、mask 和输入增强。
- a/b 保留现有语义：共享 timestep，使用各自目标与独立 noise。
- 不同样本或时间对正常独立采样随机数。每个窗口只干预一个 view，无需再讨论同一窗口的跨 view 干预随机数共享。

```text
selected_logits_B_Q : [B,2]
loss_stop_B_Q       : [B,2]
loss_split_B_Q      : [B,2]
advantage_B_Q       = loss_stop_B_Q - loss_split_B_Q
actual_delta_B_Q   : [B,2]
```

动作误差已对动作 horizon / 维度聚合。采样后不再保留完整 V/T/M 轴，由 `selected_view_B` 记录归属。关闭 temporal 时以上结果为 `[B]`。

### 3.4 Forward 组织与计算复用

令 F 为 `V*D+S`。不同样本干预的 view 不同，不影响动作条件的统一 shape，无需按 view 分别调用完整动作模型：

| 模式 | 两臂合并后的动作条件 | 显存不足时的等价组织 |
| --- | --- | --- |
| 无 temporal，B 个窗口 | `[2B,T,F]`，一次动作网络 forward | stop/split 各一次 `[B,T,F]` |
| 有 temporal，B 个时间对 | `[4B,T,F]`，一次动作网络 forward | stop/split 各一次 `[2B,T,F]` |

合并时复制各臂对应的目标、noise、timestep、mask，评价后按一致顺序拆回。一次大 forward 改善批处理效率，不等于减少两臂的计算量；核心节省来自每个样本只采一个 view。

基础路由与视觉编码仍需处理所有 view。可缓存基础视觉特征，按选中 view 分组重编码被干预的帧，再替换回完整动作条件；stop/split 中与基础树一致的一臂可复用基础特征。该复用基于当前动作视觉编码器按帧聚合、冻结且处于 eval 的语义。独立视觉编码器可能仍需按 view 分组调用，不等于完整动作模型重复 forward V 次。

## 4. Loss 与统计

### 4.1 Spatial：所选节点的细化收益

使用 canonical split logits（`split_action_value=0` 时按现有规则反转符号）：

```text
split_prob_B_Q = sigmoid(selected_logits_B_Q)
spatial_terms_B_Q = -split_prob_B_Q * detach(advantage_B_Q)
L_spatial = mean_over_B_Q(spatial_terms_B_Q)
```

Advantage 为正推动 split 概率上升，为负推动下降。所有 2B 个窗口参与平均，包括时间匹配失败和无效配对的占位窗口。关闭 temporal 时结果为 `[B]`，仅沿 B 求平均。

### 4.2 Temporal：所选 view 内的跨窗口收益排序

沿 Q 轴取出 a/b，二者属于同一选中 view：

```text
gain_B = advantage_a_B - advantage_b_B
prefer_a_B = sigmoid(logits_a_B - logits_b_B)

delta_a_B = leaves_split_a_B - leaves_stop_a_B
delta_b_B = leaves_split_b_B - leaves_stop_b_B
matched_B = pair_valid_B & (delta_a_B == delta_b_B)

temporal_terms_B = -matched_B * prefer_a_B * detach(gain_B)
L_temporal = mean_over_all_B_pairs(temporal_terms_B)
```

Matched 是 0/1 权重，实现时转为浮点类型。数量差使用有效 leaves，不用 padding 长度；不将 gain 除以 delta。失败项贡献零，分母仍为全部 B，不按成功匹配数放大。

若 a 的收益大于 b，temporal 推动 `z_a-z_b` 增大，反之减小。它比较各自 stop/split 收益而非原始动作误差；推理时仍允许 a/b 都 split 或都 stop。最终更新由 spatial、temporal、budget 的合成梯度决定。

### 4.3 均匀采样估计 view 平均目标

在相同节点采样和匹配规则下：

```text
目标：mean_over_samples(mean_over_views(loss[sample, view]))
估计：mean_over_samples(loss[sample, selected_view[sample]])
```

均匀采一个 view 的估计在期望上等于原来的 view 平均目标，**不额外乘 V**。Spatial 与 temporal 直接按上述样本维平均，不再先对 batch 中出现的各 view 求均值后等权平均，否则会改变样本权重。分 view 均值仅作诊断。

此结论只针对 view 采样，不表示局部干预目标就是全局最优计算分配的无偏梯度。

### 4.4 Budget：所有 view 都参与

基础路由已提供所有 view 的 soft count，预算不随 paired 的 view 采样而裁剪：

```text
soft_count_B_Q_V_T : [B,2,V,T]
mean_soft_count_V = mean_over_B_Q_T(soft_count_B_Q_V_T)
L_budget_V = budget_objective(mean_soft_count_V, target_V, lambda_V)
```

关闭 temporal 时省去 Q 轴。每路先平均 batch/时间数量，再应用预算 loss，允许困难帧多用、简单帧少用；不改成逐图像惩罚后平均。

沿用现有 soft 数量定义与预算权重。各 view 先使用相同 target schedule 和超参数，但上限型 Lagrangian 的 lambda 独立，按各自超预算量更新，多卡统计保留 V，不允许跨 view 平均抵消超支。

### 4.5 总目标与日志

```text
L_router = paired_weight * L_spatial
         + temporal_weight * L_temporal
         + mean_over_V(L_budget_V)
```

Spatial/temporal 是 view 平均目标的随机估计，budget 是所有 view 的完整平均；budget 项已包含原有权重。关闭 temporal 时省略该项。

计算依赖为“paired 评价 → advantage → spatial/temporal”，复用同一批收益，加上 budget，一次 backward 更新路由相关参数；不分成依次更新模型的多个步骤。

按 `selected_view_B` 分组累计 spatial/temporal 项的和与采样计数、advantage、valid/matched 数，多卡按和与计数汇总。某路没有被采中时标记无样本，不把均值零解释为无收益。Temporal 日志均值按该路全部被采样时间对计数，匹配率按 valid 数计数；训练总 loss 仍使用上面的全 batch 平均。所有 view 的 soft/hard 数量、预算差值和 lambda 均持续记录。

## 5. 方法边界

完整动作评价使每一路的 advantage 依赖其他视角的信息。推理时，每路 router 的输入为本路图像特征和共享的低维条件特征；共享状态投影不包含其他相机的图像信息。若可变遮挡导致路由失败，再考虑共享低成本多视角视觉上下文。

单路干预估计的是其他基础树固定时的局部边际收益，无法完整识别“两个 view 同时细化才有效”的联合收益，也不保证整段任务的最优计算分配。独立预算支持各 view 内的空间/时间分配，不代表跨 view 自由转移预算；后者需要另行改变预算目标。

## 6. 实施约定

- **Checkpoint：** 无需兼容旧版共享 checkpoint 或旧 optimizer/lambda resume；按新架构从 Stage A 开始训练，可直接修改内部实现。
- **Stage A 随机树：** 各 view 独立采样模式（如 `level` / `random`）、random 模式下的细化比例及空间树布局，复用现有单路随机细化逻辑，允许不同总 patch 数。例如同一样本的 static 可选 `level:1`，gripper 可选 `random:0.5`。保留同一 view 的一个窗口内 T 帧共享采样树的现有语义。
- **任务范围：** 以 TCL 为入口，policy 支持通过不同配置选择任意 V，包括单视角；每次训练固定相机集合、数量和拼接顺序，Stage A/B、恢复训练与推理保持一致。LIBERO 提供双相机兼容逻辑，但保持当前默认单相机配置；现有 dataset 的相机 key 映射保持原范围。
- **首版结构范围：** 所有 view 使用相同图像尺寸、树配置及视觉特征维度；各路权重、实际 patch 数和 padding 长度 `N_v` 可以不同。各路预算状态独立，初始 target schedule 和超参数相同。
- **参数共享边界：** 采用第 2.1 节的“各 view 独立视觉路由，共享低维条件编码”：policy 层只保留一个 `router_condition_projection`，各路 `FoveationNet` 和 `FoveationObsEncoder` 分别持有自己的参数。共享投影按原训练阶段管理更新与冻结；下游动作模型保持现有联合条件逻辑。

## 7. 预计涉及的文件与验证

| 文件或范围                                                               | 主要调整                                                                                              |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `diffusion_policy/policy/foveation_dp_transformer_policy.py`             | 按 view 管理网络；按样本/时间对采样 view，批量评价两臂，适配特征复用、分 view 日志、冻结和优化器。 |
| `diffusion_policy/model/roi/foveation_net.py`                            | 复用单视角树、patch 构建和干预规则，适配分 view 预算接口。                                            |
| `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py` | 分 view lambda、多卡统计、日志、调度、EMA、阶段切换及 checkpoint。                                    |
| `diffusion_policy/common/foveation_video_util.py` 与相关 env runner      | 按 view 获取网络和输出，保留现有 TCL val 视频、按 task 上报和 hard 数量统计。                         |
| `configs/tcl_force_foveation_dp.yaml` 与 TCL Stage A/B 单次、batch 脚本  | 保证各阶段架构一致，接入分 view 预算及初始化配置；其他任务配置按第 6 节范围处理。                     |

Dataset 保持各相机图像、低维状态和一份动作目标的联合样本语义；数据划分、归一化和 dropout 语义不随架构改变。检查上述文件的直接调用方，避免单网络属性访问失效，不扩大为无关重构。

代码实施时不新增独立测试脚本，本地做 Python 静态编译、shell 语法与 diff 检查；训练梯度、多卡、新架构 checkpoint 保存/恢复和视频运行验证留到服务器。本次仅更新设计文档。
