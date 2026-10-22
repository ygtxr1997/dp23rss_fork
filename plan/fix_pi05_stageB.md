# π0.5 LIBERO StageB 修复说明与分批实施计划

状态：review 已完成；第 0、1、2A、2B、3、4、5、6、7 批代码或文档修复已实施并完成本地静态检查。另已接入冻结 V-JEPA StageB 实验入口：默认 `vjepa2_1_vit_base_384`、112×112 encoder 输入、768 维特征和本地 `TORCH_HOME`。V-JEPA 服务器 forward、checkpoint、显存和训练稳定性验收仍待执行。第 2 批拆为 forward/shape 与 checkpoint 合同两个子批，便于分别验收。

本文以用户最新要求为准：StageA、StageB 共用一个训练 YAML；保留 router 的 56×56 全局输入。第 0—7 批已按本文实施；不能把本地静态检查等同于 StageB 训练完成。

## 1. 目标、约束与当前结论

当前 π0.5 StageB 的已知代码阻断和目标偏差已按本计划修复；训练数值、CUDA/DDP、真实 checkpoint 和闭环效果仍需服务器验收。其中共享 `MultiViewFoveationNet` 的 Temporal 接口仍由 DP 和 π0.5 共用。

修复后的目标：

- StageA、StageB 共用 `configs/pi05_libero_sft.yaml`，通过启动脚本覆盖阶段参数。
- 保留 π0.5 原生的 `10×32` flow-matching action loss。
- Spatial、Temporal、budget、阶段切换的算法语义与 DP 对齐。
- 继续复用现有四叉树 geometry、patch bank、节点干预、packing、视频和 LIBERO runner。
- 修复共享模块对 DP Temporal 路径造成的回归。
- 正确支持单卡／多卡训练、逐相机指标、EMA 和完整 checkpoint 恢复。

实施约束：

- 不重新收集数据，不重新设计 quadtree，不把 π0.5 改成 DP epsilon-MSE。
- 不继承整个 DP workspace；优先复用已有模块，必要时抽取少量与动作模型无关的公共函数。
- 不新增独立测试脚本。各批执行静态检查，训练验收通过现有入口在服务器完成。
- 新增 public class/function 添加简洁 docstring；tensor/NumPy 变量使用 shape 后缀。注释解释设计原因和 shape，代码保持简短、易读。
- 每批实施前重新阅读相关调用链和当前 diff，保留其他历史未提交改动。工作区包含历史改动，不能把全部 `git diff` 都归因于上一轮。
- 文档分别标记“代码修复完成”“本地静态通过”“服务器验收通过”，不得混写。
- StageA 的分辨率优势继续标记为用户服务器反馈，不能推断 StageB routing 或闭环收益。

### 1.1 已确认的修复选择

- **保留 router 的 56×56 全局输入。** 用户已在方案确认时选择此项。
- 只放宽 pooled-global ResNet forward 的尺寸检查，保留 node-aligned FPN 的对齐约束。
- 默认 router 继续使用 `pooled_global + resnet10`，双相机独立参数，共享 normalized 8D state projection；语言只进入 π0.5 action model。
- downstream 切换采用 **rank 0 EMA router**；关闭 EMA 时才使用 live router。
- Spatial 和 Temporal 采用现有 DP 的目标函数。
- target schedule 采用 DP 的 cosine 关系，以 π0.5 optimizer step 为单位。

### 1.2 指令遵循与设计来源

原任务明确写了“只更新 `plan/extract_pi0.5_code.md`，本轮不修改代码”。按该范围，实际扩展 policy、dataset、workspace、共享模块和启动入口，超出了该轮授权范围；长期 StageB 设计不能替代当轮范围。

独立 StageB YAML 曾出现在旧计划中，因此不是无来源的设计；现在应按用户最新要求合并到一个 YAML。`pooled_global`、共享 8D state projection、语言暂不进入 router、1201/600 步及 loss 权重也有旧计划依据。独立 π0.5 workspace 和 native flow-matching 同样有原设计依据。

需要修正的无说明偏差包括 Spatial surrogate 改为直接乘 logit、Temporal 改为无梯度绝对差、冻结新 global encoder，以及参数覆盖失效。linear target schedule 与 DP 的 cosine 不同；旧计划只规定端点和时长，属于未说明差异，本文明确选择与 DP 对齐。

## 2. Review 问题清单与依据

以下编号用于关联修复批次。严重程度 P1 表示运行阻断或核心训练正确性问题，P2 表示配置、诊断、日志或兼容性缺陷。

| 编号 | 严重程度 | 问题与影响 |
|---|---|---|
| F01 | P1 | normalized state 已补齐为 32D，却输入 `Linear(8,512)`，routing forward 维度不匹配 |
| F02 | P1 | 默认 global level 提供 56×56，pooled ResNet 要求 32 整除，forward 抛错 |
| F03 | P1 | 共享 Temporal 分支把 `[R,M]` 决策传给 `[B,T,M]` leaf-mask 接口，reshape 失败，并影响 DP |
| F04 | P1 | paired packing 与 adapter 连续两次 `[:,0]`，破坏 patch、mask 和 metadata shape |
| F05 | P1 | workspace 用 detached paired/temporal/budget 标量重新组装 loss，再调用 backward |
| F06 | P1 | Temporal seed 使用 `seed + index*1000003`；默认 index=4295 时超出 NumPy uint32 范围 |
| F07 | P1 | Spatial 直接乘 logit，与 DP sigmoid surrogate 不同；Temporal 仅计算 detached advantage 绝对差，没有 router 梯度 |
| F08 | P1 | `freeze_global_encoder=True` 使新 ResNet 无法学习；普通 `train()` 又破坏阶段要求的 eval 状态 |
| F09 | P1 | paired 训练绕过 DDP forward；下游参数在 DDP 注册前被冻结，切换后解冻不受正确同步 |
| F10 | P1 | lambda 更新减去 min 而不是 target；只用最后 microbatch 的双相机均值，缺少跨卡汇总与逐相机独立更新 |
| F11 | P1/P2 | EMA 更新所有分支，冻结 router 继续漂移；切换遗漏 projection optimizer state；worker pairing 状态不及时切换 |
| F12 | P2 | Temporal 分母、matched count、梯度累积／跨卡汇总错误；逐相机指标不足；诊断 routing 缺少 state condition |
| F13 | P2 | checkpoint 扩展加载白名单过宽，full checkpoint 被作为 live 初始化，配置／资产校验与旧 resume 兼容不足 |
| F14 | P2 | A/B 使用不同 YAML；顶层与 `training.*` 参数重复，覆盖镜像字段不生效；`paired_node_sampling` 未接入 |
| F15 | P2 | StageB 脚本未沿用 StageA 的 seed、micro/global batch 环境变量，默认实验条件发生隐式变化 |
| F16 | 文档 | 原任务只授权更新规划文档，实际扩大为代码实现；当前状态描述高估实现完整性 |

### 2.1 关键调用链

实施前必须重读以下文件及函数，行号可能随修复变化：

- [preprocessing.py](../diffusion_policy/model/pi05/preprocessing.py)：`normalize()` 会补齐 state/actions 到 32D。
- [pi05_policy.py](../diffusion_policy/policy/pi05_policy.py)：构造函数、`compute_paired_loss()`、`_route_observation()`、`_adapt_foveation_outputs()`、`set_stage_b_phase()`、`compute_resolution_diagnostics()`。
- [resnet_helper.py](../diffusion_policy/model/roi/resnet_helper.py)：`_reshape_frames()`、`forward()`、`forward_stages()`；pooled forward 已按实际 feature-map 尺寸打包。
- [multi_view_foveation_net.py](../diffusion_policy/model/roi/multi_view_foveation_net.py)：`sample_paired_interventions()` 的新增 Temporal mask 计算，以及已有 metric sums/counts。
- [foveation_net.py](../diffusion_policy/model/roi/foveation_net.py)：`sample_paired_splits()`、`pack_topology()`、`_build_soft_and_hard_leaf_masks()`、`_encode_global_images()`、budget primitives。
- [train_pi05_workspace.py](../diffusion_policy/workspace/train_pi05_workspace.py)：初始化与 DDP 注册、训练 loop、`_update_stage_b_state()`、`_load_stage_a_weights()`、`_signature()`、checkpoint 与日志。
- [libero_dataset.py](../diffusion_policy/dataset/libero_dataset.py)：`LiberoFTDataset(output_mode=raw)` 的 `__getitem__()`、episode ranges、`set_temporal_pairing()` 和 metadata。
- [pi05_training_util.py](../diffusion_policy/common/pi05_training_util.py)：`Pi05EMA`、`Pi05BatchSampler`、RNG 保存恢复及 `distributed_ema()`。
- [DP policy](../diffusion_policy/policy/foveation_dp_transformer_policy.py)：`_compute_paired_loss()`、`train()`、`set_stage_b_phase()`。
- [DP workspace](../diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py)：`_update_budget_lambda()`、`_globalize_elu_budget()`、`_set_stage_b_phase()`、逐相机汇总及 target schedule。
- [DP LIBERO dataset](../diffusion_policy/dataset/libero_dataset.py)：episode 内采样和 worker pairing 开关。

F04 的 shape 错误是 `[B,1,N,3,14,14] → [B,N,3,14,14] → [B,3,14,14]`，mask 同样丢失 N 维。F05 中保留计算图的 `loss` 字段被忽略，而三个用于日志的 detached 字段被用于 backward。

F06 已实际复现：默认 seed=42、index=4295 得到 4295012927，报 `ValueError: Seed must be between 0 and 2**32 - 1`。文档记录训练集有 135270 个窗口，因此不是仅在异常数据下触发。

F12 的计数反例：总 pair=4、有效 pair=2、匹配 pair=1，当前实现用 `valid_count * (matched / all_pairs)` 得到 matched count=0.5。正确 matched count 应直接求和，match fraction 应为 1/2。

F14 已通过 Hydra 组合复现：`training.temporal_adaptive=false` 不改变 workspace 实际读取的顶层 `temporal_adaptive=true`；`training.paired_loss_weight=2.0` 不改变顶层 1.0。

### 2.2 π0.5 与 DP 的 high-level 对照

| 环节 | 当前对照结论 |
|---|---|
| StageA 固定／随机树训练下游视觉与动作模型 | 高层目标一致；π0.5 保留 native flow-matching |
| StageA → StageB checkpoint 初始化 | 都有入口，π0.5 扩展加载契约不完整 |
| Spatial 单节点 stop/split、共享 noise/time | 复用了采样基础，但 π0.5 packing 和目标函数有错误 |
| 同 episode Temporal 配对、同相机匹配 | 意图一致，π0.5 数据采样、匹配接口和目标未正确落实 |
| 独立相机预算与 lambda | FoveationNet 层独立，π0.5 workspace 更新时错误合并 |
| router-only → downstream | 有阶段框架，冻结、DDP、EMA、worker 状态未对齐 |
| scheduler 连续运行、不重新 warmup | π0.5 当前确实沿用同一个 scheduler |
| rollout、视频和 patch-count 汇总 | 已复用现有 runner/evaluation 链，代码方向合理 |

当前 DP LIBERO StageB 脚本默认 `temporal_adaptive=false`、downstream switch 为 `null`；π0.5 的默认 true、step 600 切换来自旧方案，不是 DP 脚本的原样默认值。π0.5 用 optimizer step，DP 相应配置用 epoch，不能直接比较数字。patch14、pooled-global、双真实相机和 native flow loss 是有依据的 π0.5 差异。

### 2.3 复用边界

已复用 FoveationNet 的 geometry、image pyramid、patch bank、固定／随机 topology、`sample_paired_splits()`、`pack_topology()`、budget primitive，以及 MultiViewFoveationNet 的多相机组织。rollout/video 继续使用 LiberoSocketRunner 和公共 foveation video 工具，caption 已标明 recorded-observation routing。

需要消除的冗余主要是：忽略共享 sums/counts 后重新转 CPU list 统计、重复 Temporal leaf accounting、重新实现却偏离 DP 的 phase/lambda/归约机制。仓库的 `model/pi05/patch_sampler.py` 搜索未发现当前 Python 调用方，是遗留清理项；当前主训练路径没有同时调用两套 quadtree sampler。

π0.5 的原始观测处理、SigLIP/Gemma、flow loss、FP32-master optimizer 和模型相关的显存管理继续保持专用实现。

### 2.4 已完成与未完成的验证

review 阶段已完成：9 个相关 Python 文件的内存语法编译、四个 A/B shell 脚本分别执行 `bash -n`、两份 π0.5 YAML 的 Hydra compose/resolve、关键参数覆盖核对、NumPy seed 越界复现，以及 shape／梯度／DDP／EMA／phase 静态追踪。

尚未运行完整 π0.5 forward/backward、CUDA/DDP 训练、真实 StageA checkpoint 加载、LIBERO rollout 或 W&B。上述静态检查不代表训练已通过；明确的 shape、detach、seed 和 loss 问题无需等待服务器实验才能确认。

## 3. 修复后的统一行为与接口

### 3.1 配置与启动入口

唯一训练 YAML 为 `configs/pi05_libero_sft.yaml`。移除独立 StageB YAML，并更新相关引用。

参数职责统一为：

- 顶层：路径、`pretrained_ckpt`、`pretrained_ckpt_mode`、`resume`。
- `model.*`：router 结构、geometry、temperature、最终 patch target、budget 类型及 loss 权重等模型配置。
- `training.*`：StageB 开关、phase 切换步数、Spatial/Temporal 权重、Temporal 开关、pairing seed、target 起点及退火时长、lambda 初值及学习率。
- `runner.*`、`rollout.*`：评测和视频参数。

每项参数只保留一个权威位置。删除旧顶层 StageB 参数及其 `training.*` 镜像，避免配置显示值与运行读取值不同。`training.stage_b` 控制阶段入口；模型预算字段继续采用 Pi05Config 的 `foveation_*` 命名。

`paired_node_sampling` 放在 `training.*`，首版仅支持 `reachable`，传入共享采样路径或在入口明确校验，不接受无效果的配置项。

StageB 默认值：

| 参数 | 默认值 |
|---|---:|
| 总 optimizer steps | 1201 |
| downstream 起点 | 600 |
| target 起点／终点 | 0.75／0.1 |
| target cosine 退火时长 | 600 steps |
| Spatial／Temporal 权重 | 1.0／1.0 |
| budget loss 权重 | 0.1 |
| budget 类型 | `upper_bound_lagrangian_elu` |
| base cost | 0.0 |
| lambda 初值／学习率 | 0.0／0.01 |
| Temporal | true |
| router temperature | 1.0 |

这些是迁移初始值，不是已验证最优超参数。第一轮服务器实验后再依据 flow loss 尺度校准。

两份脚本都支持 `SEED`、`MICRO_BATCH`、`GLOBAL_BATCH`，脚本默认沿用 StageA 的 42、16、128；原始 dense YAML 的 batch 默认值保持其既有含义。Temporal 开启时，global batch 指主窗口／pair 数，实际 action-scoring 窗口数翻倍，日志明确记录。

StageB 保留 `PRETRAINED_CKPT`、`PRETRAINED_CKPT_MODE`、`TEMPORAL_ADAPTIVE`、`TEMPORAL_LOSS_WEIGHT`、`PAIRED_LOSS_WEIGHT`、`BUDGET_LOSS_WEIGHT`、`TARGET_NUM_PATCHES`、`TARGET_NUM_PATCHES_START`、`BUDGET_LAMBDA_INIT`、`BUDGET_LAMBDA_LR`、`DOWNSTREAM_FINETUNE_START_STEP`、`MAX_STEPS`、`ROLLOUT_EVERY`、`N_TEST`、`OUTPUT_DIR`，补齐 target decay、budget type、temperature 等对应覆盖入口。脚本末尾的 Hydra 参数仍可覆盖脚本默认值。

### 3.2 Router、packing 与 loss

统一通过一个 policy 内部 routing helper 构造 normalized state 的前 8D、`[B,1,512]` router condition、两个真实相机的输入和 routing outputs。普通训练、paired、推理、视频与 topology diagnostics 都使用该 helper。

保留共享 FoveationNet 的 `[B,T,...]` 输出；只在进入 π0.5 camera slots 时由 adapter 去掉一次 `T=1`。第三相机继续为无效 placeholder。

接口约定：

- `compute_loss()` 保持 native flow-matching 行为。
- `compute_paired_loss()` 返回可求导的 `total_loss`、loss components，以及用于日志的 detached 统计；保留逐窗口 stop/split loss 和 advantage 供统计。
- `forward()` 提供明确的 native／paired 分派入口；workspace 的所有训练调用均经过 DDP 包装后的 `self.model(...)`。
- workspace 只对可求导的 `total_loss` backward，日志转换时再 detach。返回的纯诊断 routing 数据不能意外保留不参与目标的计算图。
- budget weight 在 FoveationNet 中应用一次，workspace 不重复乘权重。

用代码记号统一 DP 和 π0.5，避免依赖特殊公式渲染：

```text
advantage_B = stop_loss_B - split_loss_B
spatial_terms_B = -sigmoid(selected_logit_B) * detach(advantage_B)

gain_P = advantage_A_P - advantage_B_P
prefer_a_P = sigmoid(logit_A_P - logit_B_P)
temporal_terms_P = -prefer_a_P * detach(gain_P) * matched_P

total_loss = paired_loss_weight * mean(spatial_terms_B)
           + temporal_loss_weight * mean(temporal_terms_P)
           + budget_loss
```

Spatial 对所有窗口求均值；Temporal 对全部主 pair 求均值，无效／未匹配 pair 的项为零。匹配率单独采用 `matched_count / valid_count`，有效数为零时报告 0。全部无效时 Temporal 项仍以 logits 构造零权重项，保证数值有限且梯度为零。

基础树 action-loss 日志通过 `base_split_B` 选择对应的 stop/split arm，不能使用两臂平均代替。统一 advantage 正值表示 split 比 stop 更好。

一次 paired 计算只预处理／增强一次。同一窗口两臂共享 action target、noise、flow time、语言及动作监督 mask；树的有效 patch mask 根据各臂拓扑生成。Temporal A/B 共享 flow time，各窗口仍使用自己的 noise。

### 3.3 DDP、阶段和 EMA

采用 DP 的稳定参数注册方式：

- DDP 构造时注册未来两个阶段需要的参数。
- 阶段切换不通过反复修改注册参数集合实现。
- router-only 的 action scoring 处于 eval，作为 detached advantage 评分，不建立下游参数梯度图。
- downstream routing 处于 eval，不建立 router 梯度图，native action loss 更新下游。
- 覆盖 `train()`，每次恢复模式时重新施加阶段约束。
- 不在需要 router 输入梯度的路径上添加通用 `no_grad()`；上述关闭梯度仅限已定义的 detached 评分和冻结 routing 分支。

这里的“冻结”指不产生参数更新，不要求切换 DDP 已注册参数的 `requires_grad`。inactive branch 的梯度应为 None，不能让旧梯度、动量或 weight decay 继续更新它。

step 语义：`global_step` 表示已完成更新数。完成 600 次 router 更新后，在下一次 forward 前进入 downstream；scheduler 不重建、不重新 warmup。

EMA：

- 仅更新 active branch。
- ResNet 等 active branch 的 buffers 按确定规则复制进 EMA，包含 BatchNorm running stats 和整数计数。
- 切换时将 rank 0 EMA 的 router、global encoder、condition projection 参数及 buffers 广播到 live policy。
- 该确定状态同时保留在 EMA 中，之后不再对 router 分支执行 EMA 插值。
- 清理两个 routing 模块的 optimizer state 和残留梯度，保留 optimizer 参数组与 scheduler。

### 3.4 Temporal、budget 与指标

Temporal 数据接口与 DP 对齐：

```text
batch["temporal_pair"] = {"obs": ..., "action": ...}
batch["temporal_pair_valid"] = BoolTensor[P]
```

使用 episode ranges 和 `searchsorted`，避免为每个窗口保存一个 Python 字典条目。配对随机性由 `(pairing_seed, epoch, primary_index)` 确定，使用 `SeedSequence/default_rng`，不依赖 worker 全局 RNG，也不使用容易溢出的线性 seed。

相同 epoch／index 的配对可复现，不同 epoch 可重新采样；均匀跳过 A 自身。无候选时返回 A 副本和 false。验证集关闭 pairing。

阶段切换时，在 optimizer 边界停止旧 iterator、丢弃预取 batch，从已提交的 sampler 位置重建 iterator，让 workers 获得关闭 pairing 的配置；epoch 和恢复流程同样在构造 iterator 前设置 pairing 上下文。重建不能额外改变训练 RNG／loader generator 的连续运行语义。

Budget：

- 每个相机独立累计 expected patch sum 和 observation count。
- 沿用 DP 的全局均值代理处理 ELU objective。
- 每个 optimizer step 完成后，仅用该 step 的全局统计及该 step 实际使用的 target 更新一次 lambda。
- 更新公式为 `max(0, lambda + lr * (expected_mean-target)/patch_range)`。
- target 起终点先经共享 resolver 转换成绝对 patch 数，再进行 cosine 插值，支持 ratio 和 absolute count。
- 每个新 step 前设置本步 target；日志区分本步使用值与下一步状态。

直接消费共享模块的 `_view_metric_sums/counts` 和 `_budget_*`，避免 tensor→CPU list→tensor 的二次统计。所有比例按全局分子／分母计算，不平均 rank-local ratio。

必需的训练日志包括：

```text
train/total_loss
train/diffusion_loss
train/paired_loss
train/paired_stop_diffusion_loss
train/paired_split_diffusion_loss
train/paired_advantage_mean
train/paired_advantage_abs_mean
train/paired_advantage_positive_fraction
train/temporal_loss
train/temporal_match_fraction
train/temporal_valid_pair_count
train/temporal_matched_pair_count
train/budget_loss
train/budget_lambda
train/expected_num_patches_mean
train/target_num_patches
train/num_patches_mean
train/total_num_patches_mean
train/downstream_finetune_active
train/lr
```

逐相机保留 `train/views/image/*` 和 `train/views/gripper/*` 的 paired、temporal、budget、lambda、expected/actual patch 统计。全局 `budget_lambda` 若保留则明确为两个相机的均值，仅作日志；计算始终使用逐相机 lambda。

## 4. 分批实施顺序

依赖顺序：**第 0 批 → 第 1 批 → 第 2 批 → 第 3 批 → 第 4 批 → 第 5 批 → 第 6 批 → 第 7 批**。

各批可单独 review；中间批次并不意味着完整 StageB 已可训练。Spatial smoke 显式关闭 Temporal，最终入口仍默认开启 Temporal。第 0、1 批已经实施，其余文件和后续行为仍按下列批次逐步修复。

### 第 0 批：解除共享 Temporal 回归，修正文档状态

涉及文件：

- `diffusion_policy/model/roi/multi_view_foveation_net.py`
- `diffusion_policy/policy/pi05_policy.py`
- `plan/extract_pi0.5_code.md`
- `plan/fix_pi05_stageB.md`

实施：

- 删除新加的二维 leaf-mask 调用及重复 Temporal matched 字段计算。
- 保留现有 `sample_paired_splits()` 的实际子树增量匹配行为。
- π0.5 在本批同步改为根据 packed stop/split count 计算 actual delta，Temporal match 不再依赖共享模块字段；第 5 批继续复用该结果完成完整 Temporal objective 和全局计数。
- 将现有文档状态改为“StageB 存在已确认阻断，修复中”，保留 StageA 用户反馈。

验收：

- DP Temporal 调用链恢复到不经过错误新增分支的状态。
- shared API 的相机选择、frame 选择和单节点干预语义不变。
- Python 静态编译通过；DP Temporal 短跑列为服务器回归项。

对应：F03、F16。

### 第 1 批：统一配置与启动脚本

涉及文件：

- `configs/pi05_libero_sft.yaml`
- `configs/pi05_libero_stageB.yaml`，移除
- `scripts/pi05/train_libero_stageA.sh`
- `scripts/pi05/train_libero_stageB.sh`
- `diffusion_policy/workspace/train_pi05_workspace.py`
- `diffusion_policy/model/pi05/config.py`
- 相关文档

实施：

- 按第 3.1 节统一参数位置，workspace 全部读取权威字段。
- A/B 使用同一 YAML。
- 补齐脚本环境变量及帮助注释。
- 静态搜索清理独立 StageB YAML 和旧参数位置的引用。
- 不在本批改变 dense/StageA 的训练目标。

验收：

- dense、StageA all-fine/all-coarse/mixed、StageB Spatial、StageB Temporal 均可 Hydra compose。
- 修改 Temporal、paired weight、budget、phase step 等字段时，解析后的实际使用值同步变化。
- 不支持的 node sampling 显式报错。
- 两个脚本分别执行 `bash -n`。

对应：F14、F15。

### 第 2 批：修复 StageA 初始化与 router forward

当前已拆为第 2A、2B 两个子批，先完成 forward/shape，再完成 checkpoint 合同。

涉及文件：

- `diffusion_policy/policy/pi05_policy.py`
- `diffusion_policy/model/pi05/config.py`
- `diffusion_policy/model/roi/resnet_helper.py`
- `diffusion_policy/workspace/train_pi05_workspace.py`

实施：

- normalized state 只取前 8D，统一 routing helper。
- 明确 global level 为 0，保留 56×56。
- pooled ResNet forward 根据实际 feature-map 大小打包，不要求输入 32 整除；保留 FPN `forward_stages()` 的对齐检查。
- StageB global encoder 可训练，移除错误的硬编码冻结。
- packing 保持 `[B,T,...]`，adapter 只去掉一次 T。
- StageB 初始化只接受 StageA EMA policy 或 run directory；full checkpoint 仅走 `resume`，避免静默读取 live 权重。
- 建立严格加载白名单：已有模型参数和 geometry buffers 必须匹配；允许缺失仅为新增 router、global encoder、condition projection 及其明确关联 buffers。
- StageA 固定树专用 buffer 无法作为 learned router 的活动状态：只允许这一类已知 source-only key 被归档，保存名称和拓扑内容／摘要，报告其退出活动 state。其他 unexpected key 报错。
- 校验 geometry、视觉位置配置、图像朝向、状态／动作契约、tokenizer 和 normalization 资产；已有来源 metadata 提供的数据划分 hash 也必须核对。

来源记录：

- 保存 checkpoint 路径、SHA256、加载／允许缺失／明确归档 key。
- 保存 StageA 模型配置和完整 StageB 配置。
- StageA 完整运行配置优先读取 checkpoint metadata，其次读取邻近 run config；旧单独 EMA 未携带时明确记为 unavailable，不伪造。
- 对缺失的历史 metadata 明确记录可验证边界。

验收：

- 服务器上 StageA EMA 成功构造 StageB。
- state projection shape 为 `[B,1,512]`。
- 56×56 ResNet forward 成功。
- all-coarse/all-fine/mixed 输出的 patches、mask、bank index、level、prefix shape 正确且 loss 有限。
- 现有 DP 的合法 ResNet/FPN 输入输出不变。

对应：F01、F02、F04、F08 的 encoder 部分、F13 的初始化部分。

当前执行状态：第 2A、2B 已实施。代码已修复 state condition 的 8D 输入、56×56 pooled ResNet、paired adapter 的单次 T 维移除、StageB global encoder 的可训练性，以及 StageA EMA 的严格扩展加载白名单和共享配置校验。服务器 checkpoint/forward smoke 待执行。

### 第 3 批：打通 Spatial、DDP 和训练模式

涉及文件：

- `diffusion_policy/policy/pi05_policy.py`
- `diffusion_policy/workspace/train_pi05_workspace.py`
- `diffusion_policy/policy/foveation_dp_transformer_policy.py`

实施：

- 优先按 DP 的 Spatial terms 语义修复 π0.5；公共 helper 暂不抽取，避免在服务器回归前扩大 DP 重构范围。
- 按第 3.2 节修复 Spatial 公式、advantage 符号和基础树 loss。
- 两臂保留相同输入随机条件，仅改变选中相机的一个父节点，保留后代决策。
- 建立明确的可求导 `total_loss` 接口，删除 workspace 对 detached metrics 的训练依赖。
- 所有训练经 DDP forward；注册两阶段参数，实施持久的阶段 train/eval 约束。
- 本批 smoke 使用 `temporal_adaptive=false`、`budget_loss_weight=0`，先验证纯 Spatial。
- 第一轮允许两臂顺序调用原生 action model；不同时引入 SigLIP cache 或复杂性能优化。

验收：

- backward 成功，选中 router 有梯度。
- action/SigLIP 参数不更新，global encoder 梯度路径存在。
- 一臂与基础树一致，两臂只差选中父节点决策。
- 验证／sample／rollout 后恢复训练，冻结分支模式仍正确。
- 双卡或四卡一次 optimizer update 后，router 参数在各 rank 一致。
- DP 公共 loss helper 迁移前后的公式、符号和 reduction 一致。

对应：F05、F07 的 Spatial 部分、F08 的模式部分、F09。

当前执行状态：已实施 π0.5 Spatial surrogate、可求导 `total_loss`、DDP forward 分派和阶段 train/eval 约束；本批没有重复实现 quadtree 或动作评分 helper。

### 第 4 批：修复 budget、lambda 和指标汇总

涉及文件：

- `diffusion_policy/common/foveation_training_util.py`
- `diffusion_policy/workspace/train_pi05_workspace.py`
- `diffusion_policy/policy/pi05_policy.py`
- `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py`

实施：

- 复用 FoveationNet budget primitive。
- 将可以共用的 lambda 数值更新、sum/count 计算抽为小函数；跨卡通信仍由各 workspace 使用其现有后端。
- π0.5 接入逐相机全局 ELU objective 与整个 optimizer step 的 lambda 更新。
- cosine schedule 先解析绝对 patch 数，明确更新顺序。
- 全部 paired/budget 指标累计所有 microbatch 和 rank 的分子／分母。
- 保留每相机指标和总 patch 数，统一日志中本步 target/lambda 的含义。

验收：

- 高于 target 时 lambda 增加，低于 target 时下降并截断到 0。
- 人为构造相机 expected counts 不同，两个 lambda 独立变化。
- 相同数据的单卡／多卡统计结果一致。
- ratio 和 absolute target 均能退火。
- budget 权重只生效一次；仅开启 budget 且目标有非零斜率时，router 可获得对应梯度。默认 lambda=0、base cost=0 的第一步预算梯度可为零，不能将此误判为断图。

对应：F10、F12 的 budget／汇总部分。

当前执行状态：已实施。新增共享的 cosine target 与 projected dual ascent 数值 helper；π0.5 按 optimizer step 累计两个 camera 的 budget sum/count，ELU budget 使用跨 rank 均值代理，日志保留逐 camera 和全局分子／分母语义。DP workspace 使用同一数值 helper。服务器多卡和预算梯度验收待执行。

### 第 5 批：完成 Temporal 数据与目标

涉及文件：

- `diffusion_policy/dataset/libero_dataset.py`
- `diffusion_policy/policy/pi05_policy.py`
- `diffusion_policy/workspace/train_pi05_workspace.py`
- `diffusion_policy/common/foveation_training_util.py`

实施：

- 使用 episode ranges、无溢出的显式 seed 组合和 epoch 上下文。
- 数据接口统一为主 batch 外层 `temporal_pair_valid`。
- 窗口排列保持 `[all A, all B]`；同一 pair 选择同一相机。
- 从 packed arms 获取 actual delta，生成 valid/matched mask，替换第 0 批移除字段的旧消费。
- 接入共享 Temporal loss，保持所有无效窗口参与 Spatial/budget。
- 记录全局及逐相机 valid count、matched count、Temporal loss 和 match fraction。
- worker 配置在 iterator 创建前确定，恢复时重现同样的 partner。

验收：

- index=4295 及更大合法索引不再 seed 越界。
- B 与 A 属于同 episode，且有效 pair 的时刻不同。
- 同 seed/epoch/index、不同 worker 数和 resume 条件下 partner 一致。
- 成功匹配、无匹配、单帧占位、全部无效四种情况计数正确。
- 有效 Temporal 项依赖 router logits；全部无效时有限且梯度为零。
- DP Temporal regression 通过。

对应：F06、F07 的 Temporal 部分、F11 的 pairing 部分、F12 的 Temporal 部分。

当前执行状态：已实施。dataset pairing 改为 episode 内的 SeedSequence 采样，Temporal valid 标记移到 batch 外层；π0.5 paired loss 使用 packed topology 的实际 delta 计算 match，并返回逐 camera metric sums/counts。服务器 Temporal 梯度、worker 一致性和 DP regression 待执行。

### 第 6 批：完成 phase switch、EMA 与 full resume

涉及文件：

- `diffusion_policy/common/pi05_training_util.py`
- `diffusion_policy/workspace/train_pi05_workspace.py`
- `diffusion_policy/policy/pi05_policy.py`
- `diffusion_policy/dataset/libero_dataset.py`

实施：

- 按第 3.3 节实现 active-branch EMA 和 buffers 保存。
- EMA 初始状态在 DDP 参数同步后建立。
- 切换前清空梯度，选定并广播 EMA routing，清理完整 routing optimizer state。
- 切换后重新建立不配对的 iterator，丢弃预取数据，从已提交 sampler 位置继续。
- checkpoint 保存下一次更新所需的 phase、target schedule、逐相机 lambda、temperature、pairing seed/epoch/enabled、episode 索引摘要、optimizer、scheduler、EMA、RNG 和 global step。
- 新版 full checkpoint 使用明确的版本标记。
- 旧 dense/StageA full checkpoint 对缺失 StageB 字段补语义默认值后校验；不凭空放宽原有严格匹配。
- 旧有缺陷的 StageB full checkpoint 不承诺连续恢复，明确提示从 StageA EMA 重新开始。
- 推理 EMA 文件继续可由现有 policy loader 读取，新 metadata 向后兼容。

验收：

- step 599、600、601 的 active branch、phase、scheduler 连续性正确。
- downstream live/EMA router 参数及 buffers 均固定，下游开始更新。
- projection optimizer state 也被清理。
- workers 关闭 Temporal，恢复时不会重新出现 paired windows。
- 切换前后分别执行连续运行与 resume 对照，检查 step、配对、phase、lambda、EMA、optimizer 和下一步更新。
- dense/StageA 历史恢复路径单独检查。

对应：F09 的切换部分、F11、F13 的恢复部分。

当前执行状态：已实施。Pi05EMA 保存 buffers、支持 active branch 排除和 rank 0 routing 复制；phase switch 会清理 routing optimizer state、关闭 Temporal pairing 并重建 iterator；full checkpoint 增加 target step、temperature 和 pairing epoch 元数据。服务器连续运行/resume 对照待执行。

### 第 7 批：诊断、rollout 回归与文档收尾

涉及文件：

- `diffusion_policy/policy/pi05_policy.py`
- `diffusion_policy/workspace/train_pi05_workspace.py`
- `scripts/pi05/train_libero_stageB.sh`
- `plan/extract_pi0.5_code.md`
- `plan/fix_pi05_stageB.md`
- runner/evaluation/video 文件仅在发现具体缺陷时修改

实施：

- topology diagnostics 使用真实 state condition。
- 保留 resolution advantage 与 paired routing advantage 的不同含义。
- 核对 action-execution patch counts 与 recorded-observation routing 视频的来源标注。
- 复用现有双相机视频、成功率、逐相机及总 patch 数日志，rank 0 在同一 W&B step 上报。
- 核对本地 `train.jsonl`、`summary.json`、`results.json` 和视频保留。
- 将每项修复、静态检查和服务器结果写回文档。
- 遗留未调用的 `patch_sampler.py` 记录为独立清理项，不把兼容代码清理混入本轮训练正确性修复。

验收：

- 单卡短跑、四卡短跑、空任务 rank、evaluator 启动失败清理。
- StageA all-fine/all-coarse/mixed 的既有入口回归。
- DP Spatial 与 Temporal 入口回归。
- `test/num_patches_mean_image`、`test/num_patches_mean_gripper`、`test/total_num_patches_mean` 与成功率在同一步上报。
- 最后进行十任务、多 episode 正式 rollout，分别报告成功率、有效 patch 数、padding 长度、延迟和显存。

对应：F12 的 diagnostics 部分，以及整体回归。

当前执行状态：已实施。resolution diagnostics 经过真实 state condition；现有 `LiberoSocketRunner` 已提供 action-execution patch counts、recorded-observation routing 视频标记、逐 camera/总 patch 数和同一步 rollout 日志，本批只完成静态调用链核对，正式 rollout 待服务器执行。

## 5. 执行与验收记录要求

每批完成后，实施 agent 必须报告：

1. 修改／新增／删除的文件及作用。
2. 对应修复的 F 编号。
3. 已完成的静态检查与实际结果。
4. 未运行的服务器验收及原因。
5. 新发现的问题和下一批依赖。

本地检查使用：

- 修改过的 Python 文件静态编译。
- 每个相关脚本分别执行 `bash -n`，不要把多个文件当成一次 bash 调用的位置参数。
- Hydra compose/resolve 和关键覆盖参数检查。
- checkpoint key 白名单、配置和调用链静态核对。
- 不安装完整训练栈，不新增独立测试脚本。

进入正式实验的前提是：router forward、Spatial backward、Temporal 梯度、逐相机 budget、多卡同步、phase switch 和 resume 的最小服务器验收均通过。任何一项尚未通过，文档都保留为“待验收”或“存在阻断”，不能以语法检查替代。

### 5.1 当前执行记录

| 批次 | 代码修复 | 本地静态检查 | 服务器验收 |
|---|---|---|---|
| 第 0 批：共享 Temporal 回归 | 已实施：移除共享二维 leaf-mask 分支，π0.5 从 packed topology count 派生 match | 已完成：Python 静态编译、API 引用检查 | 待执行：DP/π0.5 Temporal 服务器短跑 |
| 第 1 批：统一配置 | 已实施：StageA/StageB 共用 `configs/pi05_libero_sft.yaml`，删除独立 StageB YAML，统一脚本覆盖 | 已完成：Python 静态编译、两份脚本分别 `bash -n`、Hydra compose/`Pi05Config` 组合 | 待执行：真实 StageA/StageB 启动与参数覆盖回归 |
| 第 2A 批：router forward／shape | 已实施：8D condition、56×56 pooled ResNet、单次 T 维 adapter、global encoder 可训练 | 已完成：相关 Python `py_compile`、`git diff --check` | 待执行：真实 StageB router forward smoke |
| 第 2B 批：StageA EMA 初始化合同 | 已实施：EMA-only 入口、共享配置校验、missing/unexpected whitelist、source-only topology 归档 | 已完成：workspace Python `py_compile`、静态 key/config 调用链核对 | 待执行：真实 StageA EMA 加载与 key 报告检查 |
| 第 3 批：Spatial／DDP | 已实施：DP 对齐 Spatial surrogate、可求导 `total_loss`、DDP paired forward、阶段模式约束 | 已完成：相关 Python `py_compile`、两个 pi05 脚本 `bash -n`、`git diff --check` | 待执行：Spatial backward、单卡／四卡 DDP smoke |
| 第 4 批：budget／指标 | 已实施：cosine target、全局 ELU、逐 camera lambda 和 sum/count 汇总 | 已完成：相关 Python `py_compile`、`git diff --check`、调用链静态核对 | 待执行：单卡／多卡 budget 梯度与 lambda 对照 |
| 第 5 批：Temporal | 已实施：无溢出 episode pairing、外层 valid、packed delta match、逐 camera Temporal 统计 | 已完成：dataset/policy/workspace `py_compile`、seed/API 静态核对 | 待执行：Temporal 梯度、worker/resume 一致性、DP regression |
| 第 6 批：phase／EMA／resume | 已实施：active-branch EMA/buffers、EMA routing switch、optimizer 清理、pairing/temperature checkpoint | 已完成：training util/workspace `py_compile`、checkpoint key/phase 静态核对 | 待执行：step 599/600/601 和连续/resume 对照 |
| 第 7 批：诊断／rollout | 已实施：state-conditioned diagnostics；核对现有 runner/video/W&B 链路并更新文档 | 已完成：相关 Python `py_compile`、两个 pi05 脚本 `bash -n`、`git diff --check` | 待执行：单卡／四卡 rollout、空 rank、evaluator failure cleanup、正式十任务结果 |

第 2.4 节记录的是已有 review 验证，不能填入本表作为未来修复后的验收结果。
