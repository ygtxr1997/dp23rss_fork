# π0.5 dense LIBERO 基线与四叉树迁移方案

本文记录 dense π0.5、LIBERO StageA 和 StageB 的方案及实施状态。StageA 的模型、预处理、LIBERO 数据集、SFT workspace、EMA、独立评测和多卡 rollout 已接通；用户已在服务器完成 all-fine、all-coarse 和 mixed-resolution 训练，并确认高分辨率相对低分辨率有优势。该结论来自用户服务器反馈，不伪装成本地读取的实验结果。StageB 已完成共享 Temporal API 回归修复、统一配置入口、router forward/packing、StageA EMA 扩展加载、Spatial surrogate、budget/逐相机汇总、完整 Temporal pairing、phase/EMA/full checkpoint 逻辑和诊断链路；CUDA、DDP、真实 checkpoint、LIBERO evaluator 和最终服务器验收仍待执行。完整闭环收益不能从 StageA 的 resolution advantage 直接推断。

文中“仓库事实”指已核对的现有实现或固定版本源码，“项目选择”指用户选定的初始方案，“待验证”指实现前核对或服务器实验事项。各项是否实现、是否经服务器验证分别记录，不以代码实现代替训练或闭环验证。

## 1. 两步路线与来源

1. **原始 dense π0.5：** 从官方 `pi05_base` 出发，在现有 LIBERO-FT HDF5 上联合微调 LIBERO-10，完成保存、恢复和闭环评测。这是项目数据上的 dense 基线，不直接声称复现公开榜单结果。
2. **π0.5＋四叉树：** StageA 先让下游 SigLIP 适应固定／随机 split mode 产生的混合尺度叶子；StageB 再从 StageA EMA 学习双相机 router、paired/budget 目标和 router-only→downstream phase，具体方案见第 12 节。

现有 DP 已支持多任务联合训练；新增工作的核心是接入 π0.5 的模型、训练、预处理和推理语义，而非重新建设多任务框架。更长期目标仍包括 RoboTwin 2.0 和自定义真机。真机为已通过 DP 验证的自定义单臂，使用 static＋gripper 双相机和末端位姿控制，继续复用采集、通信与控制链，实际入口和字段映射后续复核。

**选型：** 摘取 RLinf 原生 PyTorch π0.5 核心，不引入完整 RLinf 框架；官方 OpenPI 作为权重和行为参考，XPolicyLab 作为 RoboTwin 环境适配参考。这一选择服务于后续修改 SigLIP、视觉 token、prefix mask 和逐样本动作损失。

| 来源 | 用途与版本状态 |
|---|---|
| [官方 OpenPI](https://github.com/Physical-Intelligence/openpi) | 基座选择为 `pi05_base`；实施前固定与 RLinf 匹配的 OpenPI 提交、tokenizer 资产和权重版本／校验值 |
| [RLinf 固定提交](https://github.com/RLinf/RLinf/tree/c70606f08cdca259b8dec03d4430926b5b8fac9d) | 摘取参考为 `c70606f08cdca259b8dec03d4430926b5b8fac9d`；不是运行验证完成的声明 |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | 仅作后续 RoboTwin 适配参考，具体版本和协议待复核 |

以下源码链接均固定在上述 RLinf 提交，实施时据此记录配置解析结果，而非采用浮动主分支：

- [模型核心、配置注册与适配目录](https://github.com/RLinf/RLinf/tree/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/openpi)、[运行依赖约束](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/requirements/embodied/models/openpi.txt)。
- [LIBERO 数据转换](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py)、[LIBERO 输入输出适配](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/openpi/policies/libero_policy.py)。
- [π0.5 模型 YAML](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/embodiment/config/model/pi0_5.yaml)、[LIBERO-10 普通 eval 配置](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/evaluations/libero/libero_10_openpi_pi05_eval.yaml)、[Euler eval 实现](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/openpi/tasks/eval.py)、[LIBERO 环境实现](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/libero/libero_env.py)。
- [权重转换来源与说明](https://github.com/RLinf/RLinf/tree/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/utils/ckpt_convertor/openpi)。

摘取时必须记录匹配的 OpenPI 依赖、tokenizer、转换脚本与命令、原始／转换后权重来源和校验值，保留各来源许可证及版权声明。基座权重转换后的数值一致性仍需验证。

## 2. 代码复用与模型修改边界

### 2.1 dense 模块职责

| 模块 | 新增职责与复用边界 |
|---|---|
| 模型核心 | 从固定 RLinf 提交摘取 SigLIP、Gemma、动作专家、flow matching loss、Euler 推理和必要权重加载代码；不迁入 RL、value head、DAgger、RTC 或完整分布式框架 |
| 独立 π0.5 policy | 统一负责图像处理、语言 tokenization、状态／动作转换、归一化及动作反归一化；提供 `compute_loss`、`predict_action`、`reset`，训练返回原生动作 loss，推理返回可直接送入 LIBERO 控制器的动作 |
| 精简 π0.5 SFT workspace | 复用 Hydra、实验目录、日志和 checkpoint 基础设施；不继承四叉树 workspace 的阶段切换、预算、配对和 DP scheduler，显式接入 π0.5 训练配方 |
| π0.5 dataset adapter | 复用 RoboKit HDF5 reader、窗口采样和现有 episode 划分设施，输出原始图像、原始状态、语言及动作；不先执行 DP 的 mean/std 状态归一化和 min/max 动作归一化 |
| 评测适配 | 复用 `LiberoSocketRunner` 的 HTTP、子进程、并行环境、GPU/EGL 选择、成功率汇总和视频设施；通过显式输入输出适配兼容 π0.5，保留原 DP 行为 |

**第一批开始时的仓库事实与适配原因：** [LiberoFTDataset](../diffusion_policy/dataset/libero_dataset.py) 已支持多 HDF5、任务索引、episode 隔离和统计保存，但数据转换采用 DP 约定。[LiberoSocketRunner](../diffusion_policy/env_runner/libero_socket_runner.py) 当前限定 6 维状态、执行 DP 图像／状态预处理和动作反归一化，且日志及评测配置固定使用 EMA 标记。第二批已增加显式 π0.5 适配分支并修正权重来源记录，详见第 10 节。

后续预计涉及的文件类别为：模型核心目录、新 policy、新 workspace、新 dataset adapter、新 LIBERO π0.5 配置及训练／独立评测启动入口，以及现有 runner 和 LIBERO-FT evaluator 的必要适配。已落地文件和接口分别见第 9、10 节，其余模块留待后续批次。

### 2.2 预处理归属

Policy 必须携带匹配的 tokenizer、图像处理、状态／动作转换、归一化资产和 checkpoint 配置；模型专用转换只执行一次。Dataset 负责读取、窗口和字段映射，runner 负责传输及调用适配，二者不得重复 policy 的归一化或 delta 转换。离线训练与在线评测采用同一转换契约，训练增强的差异需显式记录。

## 3. dense 数据契约

以下为项目选择，字段实际可用性与转换一致性需在服务器验收：

- **任务与语言：** LIBERO-10 十任务联合训练，数据使用现有 LIBERO-FT HDF5。语言指令区分任务；任务 ID 仅用于索引与统计，不替代语言输入。适配层需核对每个 HDF5 对应的语言来源与任务映射。
- **相机：** agentview＋wrist 双相机，原始渲染／数据图像为 256×256，模型默认输入为 224×224；现有训练入口已支持通过 `PI05_IMAGE_SIZE` 设置 `model.image_size`。第三相机按上游约定占位并屏蔽，不作为有效观测，但当前仍占用视觉计算和 prefix tensor 槽位。
- **原始状态：** 保留末端位置 3 维、旋转向量 3 维和双指夹爪位置 2 维，共 8 维。现有 [HDF5 生成代码](../../LIBERO-FT/libero/force/hdf5_replayer.py) 已写入 `ee_states` 和 `gripper_states`；服务器仍需核验实际文件的字段、形状、单位及时间对齐，不能由生成代码推定数据完整。
- **固定 LIBERO 模型配置：** 按 `pi05_libero` 使用 `action_horizon=10`、`discrete_state_input=False`、模型动作维度 32、环境动作维度 7、`extra_delta_transform=False`。该 LIBERO 配置关闭离散状态输入，不照搬通用 π0.5 的状态 tokenization 默认值；数据层仍保留 8 维原始状态，按固定配置转换。
- **划分与统计：** 使用现有 episode 隔离逻辑的普通 train/val 划分，不引入四叉树 A/B 分区。保存源文件和 episode 划分清单，确认十任务覆盖；归一化统计仅来自训练部分，并与 checkpoint 一同保存，不采用现有 DP 统计直接替代 π0.5 统计。
- **时间与图像语义：** 对照上游核对图像方向、resize、增强、观测与动作时间对齐、窗口尾部 padding、有效 mask 和 loss 分母。LIBERO 动作已是 delta，禁止重复旋转、重复归一化或重复 delta 转换。当前 HDF5 与在线采集保留模拟器 RGB 方向，须在 policy 的统一边界落实上游所需方向处理。

**当前分辨率接口：** 修改前的版本等价于 `PI05_IMAGE_SIZE=224`。该参数控制 resize 后实际送入 SigLIP 的图像尺寸：224 对应每图 16×16＝256 个 token，56 对应 4×4＝16 个 token，视觉及语言 prefix 的长度和计算量随之改变。56 路径是原图直接缩小后输入模型，不是降采样到 56 再放大回 224 的固定 token 实验。预训练位置参数仍为 16×16 网格，当前 dense 实现按实际 patch 网格作二维 bicubic 插值；这尚不等于支持不规则混合叶子。本文采用 patch14 对齐的 224／56 作为研究基线，原始数据资产不需因分辨率设置重新生成；已有接口的代码状态不代替各分辨率的服务器效果验收。

## 4. dense SFT 初始配方与 checkpoint

### 4.1 用户选定的初始配方

| 项目 | 初始方案 |
|---|---|
| 训练起点 | 官方 `pi05_base` 及经过一致性验证的 RLinf 兼容权重 |
| 更新参数 | SigLIP、视觉投影、Gemma embedding／主干、动作专家、动作／时间投影全部更新 |
| LoRA／默认冻结 | 均不启用；不得继承模型 YAML 的 `train_expert_only=True` 作为本项目全参数 SFT 的生效配置 |
| 全局 batch | 256；micro batch 与梯度累积按服务器显存配置，记录实际全局 batch |
| 学习率 | `5e-5` |
| warmup | 10,000 optimizer steps |
| 后续 LR | 按所选 π0.5 配置，peak 与 decay LR 均为 `5e-5`，不误写成衰减到零 |
| 训练长度 | 首版规划 30,000 optimizer steps，属于项目初始选择 |
| 梯度裁剪 | 1.0 |
| EMA | 0.999，显式实现并保存恢复 |

上述 batch、LR、warmup 和 EMA 来自 RLinf 注册的 π0.5 LIBERO `TrainConfig`，**不是其 π0 SFT YAML 的实际生效参数**。新 workspace 必须显式接入这些值，不能认为注册配置会自动控制现有训练循环。AdamW 其余参数和 warmup 精确实现跟随固定版本引用的 optimizer；实施前核对并保存解析后的 optimizer、scheduler、参数组及更新范围。步数均指完成梯度累积后的 optimizer step。

精度沿用摘取核心的分模块约定，不统一强制转换为 bf16。优先验证单卡和不分片的数据并行，梯度检查点作为显存选项。A100/A800 优先用于 SFT，3090 可用于推理、评测和小批量调试；不承诺全参数＋EMA 能在某张 GPU 上运行，实际显存及吞吐需测量。四叉树阶段的更新参数、精度、EMA 和超参数仍独立待复核。

### 4.2 保存、恢复与评测权重

默认只保存 latest／best EMA 推理文件及必要资产，减少磁盘占用。可选完整 checkpoint 同时保存 live／EMA 权重、optimizer、scheduler、optimizer step、随机状态、有效配置、数据划分清单和归一化资产，并关联 tokenizer、依赖及基座版本；实际依赖也记录于 run_metadata.json。随机状态需覆盖实际使用的 Python、NumPy、Torch CPU／CUDA 和采样生成器，以支持恢复核验。

区分两个入口：从基座初始化新的 SFT，以及从训练 checkpoint 完整恢复。完整恢复不得重新估计统计、重划分数据、重置 EMA 或调度步数。独立评测默认使用 EMA，并准确标记实际使用的权重类型；使用 live 时也必须如实记录，不能沿用 runner 当前硬编码的 `weights="ema"` 和 EMA 日志标签。

## 5. 推理与闭环评测

**项目选择为 RLinf 普通 eval 口径，复用现有执行设施：**

- 模型生成 10 步 action chunk，每次执行前 5 步后重新观测；LIBERO-10 每任务最多 520 步，每任务 50 个 episode，共 500 个 episode。
- 使用固定配置的普通 Euler flow 推理；上述模型 YAML 的 `num_steps=5` 是 flow 采样步数，与生成 10 步／执行 5 步是不同概念。初始噪声显式受 seed 控制，确定性 Euler 更新不代表策略没有随机性；不得误用 RL 的 SDE 噪声配置。
- 一次观测计算一次 prefix KV，并在同次动作生成的各 flow 步骤中复用；下一次观测重新计算。
- 第二批 [LIBERO-FT evaluator](../../LIBERO-FT/libero/force/force_benchmark.py) 已新增显式状态布局，π0.5 发送附带夹爪位置的 8 维状态，在线观测与录像使用相同拼接函数；DP 保留 6 维末端状态。[通信协议](../../RoboKit/src/robokit/connects/protocols.py) 的状态维度本身可变，无须默认重写传输层。
- π0.5 policy 输出已经反归一化，runner 不再执行 DP 动作反归一化；语言、相机和状态输入同样经显式 π0.5 适配边界传递。
- 核对并记录 RLinf 的 reset、初始状态选择、settling、图像方向、夹爪后处理、终止规则与步数统计，明确 settling 是否计入上限。仅将执行长度改成 5、上限改成 520，不足以宣称协议一致；现有 LIBERO-FT 环境／数据的差异也需如实记录。
- 同时支持训练期间 rollout 和从 checkpoint 独立启动评测，二者使用同一 policy 适配与评测链。记录 seed、任务、checkpoint、live／EMA 类型、有效推理配置、成功数、episode 数和代表性视频。

## 6. dense 实施里程碑

每个里程碑完成对应验收后再进入下一步：

1. **模型与资产加载：** 固定依赖、tokenizer、许可证及模型资产；本批仅以 dense 模型可实例化并严格加载转换后的权重为验收条件，具体命令见模型 README。数值对齐不作为本批验收条件。
2. **先验证评测接口：** 使用已有 LIBERO SFT checkpoint 验证评测链，携带该 checkpoint 匹配的转换和归一化资产。该结果标为接口检查，不作为项目 HDF5 自有数据 SFT 成绩。
3. **接通联合 SFT：** 接通 HDF5、policy 和 workspace，完成小批量更新、各参数组梯度检查和推理保存，再接通十任务评测；完整恢复验收按用户决定延期。
4. **形成项目 dense 基线：** 对训练所得 checkpoint 完成逐任务与总体成功率、episode 数量和代表性视频报告，明确权重类型与评测差异，作为第二步四叉树实验的基线。

## 7. 第二步：随机混合分辨率下游视觉编码器适配

**状态：StageA 代码已接通，用户已完成服务器训练；正式效果以日志和 checkpoint 为准。** 本阶段让 Pi0.5 的 SigLIP 消费固定／随机四叉树采样得到的混合尺度叶子。学习型 router、paired loss、Temporal pairing 和 phase switch 属于 StageB，不应混入 StageA 的 dense 语义。

### 7.1 Stage A 参考配方与现有 DP 编码器

已核对调用链：[remaining6 脚本](../scripts/roi/sh_train_foveation_libero_stageA_remaining6.sh) → [Stage A 脚本](../scripts/roi/sh_train_foveation_libero_stageA.sh) → [LIBERO foveation 配置](../configs/liberoft_foveation_dp.yaml)。未提供环境变量或末尾 Hydra 覆盖时，实际生效的默认配方为：

```bash
FIX_SPLIT_MODE='["80%level:2","20%random:0.05-0.99"]'
```

- 每个样本以 80% 概率使用全细叶子，以 20% 概率从连续区间抽取分裂比例，再生成对应分裂次数的随机合法四叉树；比例表示内部节点分裂次数占可分裂节点总数的比例，不是图像尺寸或高分辨率面积比例。
- 每个相机独立采样模式、比例和树，同一观测窗口的各帧共享树。下游 encoder 不解析 split mode 字符串，只消费采样后的 patches 及元数据。
- 当前 DP 几何为 256／128／64 金字塔、patch16、三层树，网格为 16×16／8×8／4×4。所有叶子的像素 tensor 均为 `3×16×16`，混合尺度指其覆盖原图的范围不同。
- [FoveationObsEncoder](../diffusion_policy/policy/foveation_dp_transformer_policy.py) 的当前调用配置为：共享 Conv stem → 按 level 的残差 adapter → 256 维投影 → APT 插值位置编码＋level embedding → 带 padding mask 的四层 Transformer → CLS pooling。adapter 接在 stem 的 48 通道中间特征图上；此结构与四叉树决策编码器分开。

### 7.2 SigLIP 的目标结构与接口

保留当前 So400m/14 的预训练 stem、27 层 Transformer、最终 LayerNorm 和 `1152→2048` 输出投影，沿用跨相机共享权重。每层为 pre-LN attention 与 MLP 残差结构，16 个 attention head，MLP 为 `1152→4304→1152`。首版叶子几何如下，与原 DP 的三层节点网格一一对应：

| 金字塔图像尺寸 | 叶子像素尺寸 | patch 网格 | 全部选择该层时的 token 数 |
|---|---|---|---|
| 56×56 | 3×14×14 | 4×4 | 16 |
| 112×112 | 3×14×14 | 8×8 | 64 |
| 224×224 | 3×14×14 | 16×16 | 256 |

```text
选中的混合尺度 patches [B, N, 3, 14, 14]
  → 原 SigLIP patch stem → [B, N, 1152]
  → 按叶子取空间位置编码 + level embedding
  → 原 SigLIP Transformer（加入 padding mask）及最终 LayerNorm
  → 原输出投影 → [B, N, 2048] + 有效 mask
  → Pi0.5 多相机视觉 prefix 与语言 token
```

接口与权重复用约定：

- 输入沿用 `FoveationNet` 的 `patches_B_T_N_C_P_P`、`patch_bank_indices_B_T_N`、`levels_B_T_N` 和 `token_valid_mask_B_T_N`；Pi0.5 单帧观测适配为 `T=1`，进入 SigLIP 时移除该时间维。返回叶子 token 序列与有效 mask，不引入 DP 的 CLS pooling。
- 将叶子展为 `[B*N, 3, 14, 14]`，只经过原 `kernel=14, stride=14` 的 stem，再恢复 token 序列。所有叶子共同通过 Transformer，使粗细叶子之间可以 attention；不逐叶子运行完整 SigLIP。
- 保留 stem／位置编码的 FP32 计算及原有分模块精度约定；Transformer 的 Q/K/V、MLP 和输出投影权重形状不因叶子数变化而改变。保留 dense 前向入口作为对照，不重新初始化预训练主干。

### 7.3 多尺度位置编码、mask 与可选 adapter

- **空间位置：** 复用 DP 的 `_build_multiscale_position_bank` 与 gather 思路，以 SigLIP 预训练 `[1, 256, 1152]` 参数作为 16×16 细网格位置表；细层直接使用，粗层以 bicubic、`align_corners=False` 插值到 8×8 和 4×4，再按 `patch_bank_indices` 取出所选叶子的位置编码。不能按总叶子数截断位置表或把不规则叶子强行视为规则网格。
- **尺度信息：** 首版增加零初始化的 `[3, 1152]` level embedding，放在 Transformer 前，与空间编码一起加到内容 token 上。从原始预训练行为开始微调，不将新增尺度编码误记为预训练参数。
- **padding：** 有效 mask 传至每层 SigLIP attention，防止有效 token 读取 padding；prefix mask 使用“相机有效 AND 叶子有效”。占位相机在 prefix 屏蔽，视觉侧避免整行全部 key 被屏蔽而产生 NaN。padding 输出不得参与有效视觉 prefix。
- **顺序：** 沿用 patch bank 的层级优先、层内行优先顺序，全细时恢复原 row-major 顺序。Gemma 动态一维 RoPE 支持变化的序列长度，但不代替二维空间／尺度编码；混合排列造成的顺序分布变化也需通过训练适应。
- **adapter：** 首版不新增 level adapter。后续消融可在 patch 内容 token 上采用按 level 的 `x + Linear(r→1152)(GELU(Linear(1152→r)(x)))`，末层零初始化，再加入位置／level embedding。DP 的 48 通道 Conv adapter 无法原样复制到 SigLIP，瓶颈宽度和收益留作该增强实验的设计事项。

### 7.4 适应训练、比较口径与代码落点

先验证全细路径与 dense 对齐，再验证全粗和随机混合输入；参考 7.1 的 80% 全细＋20% 随机配方进行多拓扑适应训练。微调需允许 SigLIP 主干更新，保持 Pi0.5 原生 flow-matching 时间采样、插值和 velocity target；本节不引入 DP epsilon MSE、学习型路由损失或后续阶段切换，也不宣称形状兼容即可保证任务效果。

该默认配方没有显式 `level:0`，且随机比例不包含 0，因此没有直接覆盖全粗树。全粗前向与独立评测仍须验证；显式加入全粗训练样本属于后续实验选择，本次不改变参考配方的比例。

预处理在整图上统一完成方向、增强和归一化后再构造金字塔，避免不同尺度各自增强导致空间不一致。当前 dense 56 路径是原图直接 antialiased bilinear resize 并 uint8 round，现有 foveation 金字塔则从高分辨率图逐层 area 下采样；两者不严格等价。数值对齐应使用同一份预处理图像和同一尺度像素，任务对照需统一重采样协议或显式记录差异。

未来主要修改落点：

- [siglip.py](../diffusion_policy/model/pi05/siglip.py)：新增混合 patch 入口、多尺度位置 gather、level embedding，贯通 Encoder／Block 的 padding mask，保留 dense 路径。
- [model.py](../diffusion_policy/model/pi05/model.py)：接入单帧叶子 token 和有效 mask，组合相机／叶子 prefix mask，保留共享 SigLIP 及原生动作目标。
- policy、预处理、配置和训练／评测入口负责选择 dense 或混合采样路径、保存有效几何与 split mode，并将新增参数纳入训练／EMA；复用现有 `FoveationNet` 的采样、patch bank 与 packing，以及 DP 编码器的位置处理算法。新增参数的 dense 预训练初始化与混合 checkpoint 严格恢复须分别处理，不能把旧 checkpoint 自动兼容视为已完成。

当前 packing 补齐到 batch 内最大叶子数 `N_max`；80% 全细的配方常使 `N_max=256`，平均有效叶子数下降不代表 dense attention 的实际成本同比下降。本阶段主要验证适应能力；效率报告须区分有效叶子数、补齐长度、视觉／prefix／动作生成及预处理耗时，记录硬件、batch、warmup、同步方式和通信开销。baseline 与研究版本使用相同后端、数据划分和评测协议，官方 OpenPI 成绩不能替代项目 dense 基线。

### 7.5 后续研究边界

完整几何、patch bank、packing、多视图干预、Spatial／Temporal／预算目标和三阶段语义继续参考[四叉树方法快速迁移与复现指南](transfer_quadtree_to_benchmarks.md)。StageB 正按第 12 节分批修复和接入，当前只完成共享 Temporal API 和统一配置入口；服务器验收通过后再扩展真机、RoboTwin 2.0 和动态长度优化。

- 当前 `node_aligned_fpn` 的 patch16 限制和特征／节点对齐仍需后续实际改造；固定／随机采样接入也须避开现有构造约束，不能声称只将当前 FPN 配置改为 patch14 即可运行。此限制不改变下游 SigLIP 的混合叶子编码方案。
- 后续 Spatial 同窗口两臂共享随机条件；Temporal 两窗口各自目标、独立 noise、共享 time，每个窗口内部两臂共享随机条件。逐窗口 loss、预算语义、更新对象、精度、EMA 和调度按该阶段独立验收。
- 未改变相机的视觉特征仅在满足冻结、确定性条件时复用；改变视觉 token 后不能默认共享完整 Gemma prefix KV。仅冻结参数不等于可在需要输入梯度的训练路径上使用 `no_grad()`；缓存和梯度边界后续单独核验。

## 8. 待验证事项与验收

**状态与待验证事项：** dense 与 StageA 的代码、数据、EMA、评测和 rollout 状态以第 9—11 节为准，已通过项不重新标为未实施。StageB 的 server acceptance 单独列在第 12 节；StageA 用户反馈的高分辨率 advantage 诊断不等同于 paired routing 收益。

**初始方案文档验收记录：**

- 已记录 LIBERO-10 联合训练、现有 HDF5、RLinf 评测口径、π0.5 专用训练参数和 EMA 等用户选择。
- 区分仓库事实、项目初始配方和服务器待验证项，保留固定版本源码链接及许可证要求，不宣称训练或闭环已跑通。
- 明确 6／8 维状态差异、policy 预处理归属、runner 动作反归一化和 EMA 标记的适配边界。
- 初始方案阶段仅检查 Markdown 结构、引用与变更范围；后续实现批次的静态检查和服务器验收单独记录。

**后续服务器验收：**

- 第一批仅验证模型实例化与严格权重加载；全参数 SFT 各参数组梯度与实际更新在后续训练批次验证。
- 数据划分隔离、仅训练部分统计、语言映射、离线／在线图像和状态／动作转换一致，以及 padding 与原生 loss 分母。
- 小批量训练、live／EMA、optimizer／scheduler、随机状态、配置和数据资产的保存与完整恢复。
- 单环境与并行评测的一致性，在明确 seed／噪声分配条件下核验 episode reset、动作执行、终止及统计；训练期间 rollout 与独立评测链一致。
- 完整十任务闭环结果，逐任务与总体成功率、episode 数及视频，准确记录 checkpoint 和评测协议。
- 当前第二步先验：相同预处理、权重和确定性前向条件下全细与 dense 的视觉输出及动作路径数值对齐；全粗及随机混合输入前向／反向有效；同一样本单独与 padding batch 编码的有效输出一致；跨相机 mask、prefix 长度及动作生成正确；混合 checkpoint 保存加载保留有效配置、输出与新增参数，训练／EMA 覆盖新增参数。记录数值差异，不承诺跨硬件逐位一致。
- StageB 还需服务器验证 Spatial／Temporal 随机条件、逐窗口 mask／预算语义、双相机独立汇总、三阶段切换、恢复、更新对象、EMA 和调度状态。

**本次文档更新验收：** 仅核对 Markdown 结构、相对链接、源码／脚本配置和变更范围，不新增测试脚本，不执行模型训练或推理。第 7 节为计划，不能据此更新服务器实施状态；未来代码修改本地先做静态编译，运行场景按上述清单在服务器验收。

## 9. 第一批实施状态：模型与 policy

已新增 `diffusion_policy/model/pi05/`（dense 核心、统一预处理、tokenizer、转换与严格加载、来源和许可证）、`diffusion_policy/policy/pi05_policy.py`、`configs/pi05_libero_policy.yaml`、`diffusion_policy/common/pi05_convert_checkpoint.py` 和 `scripts/pi05/convert_checkpoint.sh`、`requirements_pi05.txt` 和 `requirements_pi05_conversion.txt`。

- `compute_loss` 默认返回全 `10×32` mean，支持显式 noise/time 和逐窗口结果；`predict_action` 返回原始控制空间的十步预测和前五步动作；`reset` 不重置 RNG。
- 保存 policy 时携带 tokenizer bytes、统计、有效配置和资产来源；本批不实现训练 optimizer／EMA／scheduler 恢复。
- 已确认固定 `discrete_state_input=False` 下，π0.5 不使用状态 token；8 维状态仍保留在原始输入契约中。
- 预处理参考锁定 RLinf 指定的 `rlinf-openpi==0.1.1` wheel 及 SHA256。该发布包未标明可核实的 OpenPI Git commit，追溯仍待完成，不伪造匹配提交。
- 明确记录 PyTorch antialiased resize 替代 JAX 的待验证差异，以及 RLinf 默认 generator 不随机增强的问题修正。详见 [首批说明与服务器命令](../diffusion_policy/model/pi05/README.md)。
- 本地已做 Python 静态编译检查；未下载大权重，未执行转换、前向／反向、数值对齐或闭环评测。原始／转换权重哈希由服务器转换 manifest 生成，用户已在服务器完成权重转换及严格加载验证，模型加载里程碑验收通过；本批验收仅要求模型正常加载。

第一批未修改现有 DP dataset／runner／workspace 或 LIBERO-FT evaluator。用户现已准备好 `~/pretrained/pi05/converted_libero`，第二批评测适配如下；原始 HDF5 adapter、联合 SFT workspace 与完整训练恢复留待后续。


## 10. 第二批实施状态：评测适配与单任务试跑

### 10.1 已实现的代码

- 修改 `diffusion_policy/env_runner/libero_socket_runner.py`：新增显式 `policy_family="pi05"` 分支，复用 HTTP、进程、EGL 与结果汇总；原始 RGB／8 维状态／真实语言送入 policy，直接传递反归一化动作。`set_context` 支持权重类型及 checkpoint 来源，DP 默认保持 EMA。
- 修改 `diffusion_policy/env_runner/base_image_runner.py` 的类型导入，并对 runner 的 DP dataset 使用延迟导入，使独立 π0.5 评测无需加载旧 DP dataset 依赖。
- 新增 `diffusion_policy/common/pi05_eval_libero.py`、`scripts/pi05/eval_libero.sh`、`configs/pi05_libero_eval.yaml` 和 `requirements_pi05_eval.txt`：从已有转换资产加载 policy，自动启动现有模拟器 conda 环境，无需训练 workspace 或 DP statistics。
- 修改相邻 LIBERO-FT 仓库的 `libero/force/force_benchmark.py`、`eval_socket.py`：增加 `eef_rotvec_gripper` 状态布局、benchmark 真实任务语言、可配置 reset 等待动作以及实际权重类型和评测元数据记录。RoboKit 通信格式不变。
- 增加普通双相机录像，不调用四叉树可视化接口；同步模型 README 的依赖、路径、启动命令和简单验收步骤。

### 10.2 已确认的运行约定

- 使用现有 HDF5 环境配置。默认任务是 `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`，数据目录沿用 `libero_force_256/libero_10`，命令行可覆盖路径和任务列表。
- 本批历史接口验收使用相机渲染 **256×256**，HTTP 保留原始 uint8 RGB；policy 统一旋转 180°、resize 至 **224×224** 后输入模型。每次一帧、双相机、8 维状态，真实任务语言来自 benchmark。当前可配置实际模型输入尺寸的语义见第 3 节，此处 224 验收不代表其他分辨率已通过闭环验证。
- 模型预测 10 步、Euler 5 步、执行前 5 步。核对固定 RLinf `rlinf/envs/action_utils.py` 后确认：OpenPI 在 LIBERO 不额外做夹爪翻转／二值化，本链路同样直接执行 policy 动作。
- 单环境 3 个 episode，保存全部三个双相机视频；policy seed=42，环境 seed 从 130000 递增，初始状态使用 benchmark 索引 0、1、2。每任务初始化一次 policy 随机流，episode reset 不重置噪声流。
- Reset 等待 15 步，位姿动作全零、夹爪 -1；等待不计入最多 520 个策略动作。成功、环境 done 或达到上限立即结束。DP 默认仍为 5 步全零等待。
- 本实现使用 LIBERO-FT HDF5 环境且成功时提前终止，固定 RLinf eval 配置的 `ignore_terminations=True` 行为不同；结果只标为官方 LIBERO checkpoint 的接口检查，不直接比较公开成绩。图像 resize 数值差异仍待验证。
- 结果记录 `weights="pretrained"`、源 URI、转换资产哈希、实际 policy 配置、任务语言、reset／终止约定、seed、逐 episode 结果与视频路径。

### 10.3 本批验收与后续

本地只做 Python 静态编译、差异检查及接口阅读，不新增测试脚本，不运行模型或模拟器。两个仓库都需同步到服务器；实际启动命令见 [模型 README](../diffusion_policy/model/pi05/README.md)。

用户于 **2026-10-08** 反馈：默认 STUDY_SCENE1 任务完成 3 个 episode，成功率 **1.000（3/3）**，单任务闭环已跑通。运行目录为 `outputs/pi05_libero_eval/20261008_215226/eval/epoch_0000/`。记录依据为用户提供的运行日志；未独立读取服务器 JSON 或视频；用户随后确认视频正常，本批单任务接口验收通过。此结果是官方 LIBERO 权重的单任务接口检查，不代表十任务总体成绩。

单任务通过后可扩展任务列表和 episode 数。并行设施保留，但本批不宣称单／并行数值一致；改变 batch 会改变噪声分配。后续继续原始 HDF5 adapter、项目训练集统计、联合 SFT workspace 和完整训练恢复。


## 11. 项目数据与联合 SFT：三批实施方案

用户已确认按三批独立实现、服务器逐批验收。11.1 已完成且用户提供的服务器数据日志验收通过；11.2 训练主循环短跑通过，完整恢复验收暂缓；11.3 已实现，待服务器多卡评测验收。三批均更新本文和模型 README，不新增独立测试脚本，本地只做 Python 静态编译、shell 语法及差异检查。

### 11.1 第一批：原始 HDF5 与训练统计（服务器验收已通过）

新增文件：

- `diffusion_policy/dataset/libero_dataset.py`：`LiberoFTDataset(output_mode=raw)` 与统一数据准备入口。
- `diffusion_policy/common/prepare_data.py`、`scripts/pi05/prepare_libero_data.sh`：正式 LIBERO 数据准备入口与 shell 启动命令。
- `configs/pi05_libero_data.yaml`、`requirements_pi05_train.txt`：十任务默认配置与数据准备依赖。
- 修改 `diffusion_policy/common/libero_task_util.py`，增加与 benchmark 一致的任务语言映射，无模拟器依赖。

实现约定：

- 复用 RoboKit `LiberoH5FrameDataset` 和 `MergeDataset`；不继承 DP dataset 的图像变换、归一化或 normalizer。双相机原始 uint8 `[H,W,3]`，状态为 `ee_states[6] + gripper_states[2]`，原始动作 `[10,7]`，prompt 为真实任务语言。默认 collate 直接符合现有 policy 输入接口。
- 保留所有观测起点；`obs[t]` 对应 `actions[t:t+10]`，末尾重复 episode 最后动作，完整 `10×32` loss。按用户选择忽略 `replay_reset_mask`，仅以完整 episode 为窗口边界。此项写入数据契约。
- 沿用 DP 普通全局划分：排序 `(source, demo_key)` 后，以 `RandomState(42)` 随机排列；验证比例 2%，不设训练 episode 上限，不使用四叉树 A/B 分区。报告各任务覆盖，验证集不保证每任务都有 episode。
- 准备时检查两相机 uint8／共同尺寸、8D 状态和 7D 动作长度与有限值、非空 episode、成功回放文件及显式时间／图像元数据。旧文件缺失元数据时记录模拟器原始 opengl 方向和动作前观测的假设；冲突时报错。
- 每个训练时间步统计一次，避免窗口重叠和尾部填充重复计数；读取 float32 数值，以 float64 NumPy 计算精确线性分位数 q01/q99、mean/std。仅收集低维数组，不加载全部相机数据。不声称与上游直方图或窗口加权统计等价。
- 默认资产目录 `outputs/pi05_data/libero10_seed42/`，保存 `dataset_metadata.json`、`norm_stats.json` 和有效 `config.yaml`；记录相对源路径、episode 长度和划分、语言、输入约定、统计方法与统计文件哈希。已有资产拒绝覆盖。
- Dataset 提供 `get_validation_dataset()`、`get_dataset_metadata()` 和 `close()`，复用已存划分，不拟合新统计。验证统计哈希和所选 episode 长度，构造后关闭 reader 句柄，workers 各自打开文件。原地修改数据后需重新准备资产，不对全部图像内容哈希。

服务器验收已通过：用户日志确认 490 train／10 val episode、135270 train／2820 val 窗口，双相机 `[2,256,256,3]` uint8、状态 `[2,8]` 和动作 `[2,10,7]` float32，语言正确，并打印 `preparation complete`。十任务均有训练数据，验证 episode 覆盖五个任务，符合全局随机划分。记录依据为用户日志，未独立读取服务器资产。继续使用 `outputs/pi05_data/libero10_seed42`，不重新准备。

### 11.2 第二批：混合精度 SFT、EMA 与可选完整恢复（主循环短跑通过，2B 暂缓）

本次确认：单机 4 张 80GB GPU；验证只用 EMA；默认只保留 latest／best EMA 推理文件，完整保存默认关闭，完整恢复代码保留，但按用户决定暂缓 2B 验收。

本批新增：

- `diffusion_policy/workspace/train_pi05_workspace.py`：独立训练循环、DDP、验证、导出与完整恢复入口。
- `diffusion_policy/common/pi05_training_util.py`：确定性 batch sampler、随机状态、EMA、保存与首次更新诊断。
- `diffusion_policy/model/pi05/optimizer.py`：从固定 RLinf 摘取 FP32MasterAdamW，保留许可证和来源。
- `configs/pi05_libero_sft.yaml`、`scripts/pi05/train_libero.sh`：训练配置及 `accelerate launch` 单／多卡启动。脚本顶部 `tmp` 保存命令示例，环境变量设置常用参数，末尾 `"$@"` 接受 Hydra 覆盖；启动器关闭额外混合精度，workspace 继续自行管理 DDP。

修改 `requirements_pi05_train.txt`（补充 BaseWorkspace 导入所需 dill）、`diffusion_policy/model/pi05/sources.json`（优化器源路径和 SHA256）、本文及模型 README。不修改 DP workspace、模型核心、LIBERO-FT 或 RoboKit。

**训练与精度：**

- 复用 `train.py`、Hydra、BaseWorkspace 的入口／目录约定，独立实现训练保存，不使用通用 dill payload；不继承四叉树循环。
- 新训练从 `~/pretrained/pi05/converted_base` 初始化并校验转换 manifest，使用已验收的项目统计。单机 DDP，每卡 micro batch=1，全局 batch=256；四卡累积 64 次。要求全局 batch 整除 world size × micro batch；首版不引入 FSDP／ZeRO。
- 保持既有分模块 bf16／FP32，不全局 autocast／GradScaler／整模型 bf16。主权重与 Adam 状态为 FP32，恢复不降精度；梯度累积／通信沿用参数梯度 dtype，不宣称全程 FP32 梯度。
- AdamW betas=(0.9,0.95)、eps=1e-8、weight decay=1e-10；梯度范数 FP32 汇总、裁剪至 1。首次更新 LR=5e-5/10001，完成 10000 次更新后到 5e-5，之后恒定，默认总计 30000 optimizer steps。
- 全参数允许训练，不冻结／LoRA；默认非重入梯度检查点。部分最终语言输出分支不参与 action loss，DDP 开启 unused-parameter 检测，诊断列出真实无梯度参数，不增加伪造 loss。
- sampler 对窗口均匀打乱，完整全局 batch 组织成 `[accumulation, world_size, micro_batch]`，每轮丢弃不足一组的余数；主循环明确提交已完成更新的位置，预取不改变恢复位置。默认每 rank 2 个 spawn workers，只负责确定性数据读取。
- DDP 调用 policy forward，非最后累积 micro step 的前向／反向使用 no_sync。每次完整更新才推进 optimizer、scheduler、EMA、global_step 和 sampler。

**EMA、验证与导出：**

- rank 0 在 GPU 保存 FP32 EMA shadows，系数 0.999，基于主权重更新；不常驻第二个完整模型。
- 默认每 1000 步及正常结束时计算 EMA loss；固定 seed 选定验证窗口，最多 32 batch、默认 batch size=1、关闭增强。rank 0 临时应用 EMA，live 备份放 CPU，其余 rank 等待；验证后恢复 live／train mode／RNG。
- 每步 JSONL 记录 loss、实际 LR、裁剪前梯度范数、batch、消费位置、吞吐和各 rank 峰值显存；首步诊断记录主要模块梯度、主权重／EMA 坐标变化。无 W&B 依赖。
- 每次验证或 rollout 更新 `latest_ema.pt`，所选指标改善更新 `best_ema.pt`（第三批默认 success_rate），按模型推理 dtype 保存，FP32 模块保持 FP32。文件包含配置、tokenizer、统计和必要来源信息，兼容已有 `Pi05Policy.from_checkpoint()`；不含 optimizer、训练 RNG 或整份数据清单。
- 本批提前完成原 11.3 中的 EMA 推理导出。第二批最初使用有限验证子集的 flow loss 选 best；第三批默认改为总体 rollout 成功率，loss 仅诊断。旧 best 文件保持原语义。

**可选完整恢复：**

- `checkpoint.save_full=false` 默认关闭；开启后按 `checkpoint.full_every=1000` 及正常结束覆盖 `latest_full.ckpt`。同步保存，仅在 optimizer step 边界，临时文件写完后替换；不累积历史文件。
- 完整 payload 保存 live、FP32 主权重／Adam／EMA、scheduler、global_step／best_loss、各 rank Python／NumPy／Torch CPU/CUDA RNG、独立 DataLoader generator、sampler epoch／消费位置，以及配置、数据划分、统计、tokenizer 和基座来源。
- `resume` 仅接受完整文件。恢复通过 mmap 加载，逐 rank 恢复随机流；不加载 base／外部准备资产、不重新划分或估计统计。原始 HDF5 仍需存在，检查数据契约／episode 长度，不对图像内容全量哈希。
- 同一 world size、global／micro batch、seed、模型、优化器、EMA、验证子集设置才能恢复；允许改数据根路径、输出目录、停止步数、日志／保存／验证频率。DataLoader 重建区分 epoch 中途／边界，避免额外消耗采样状态。
- 每次运行／恢复使用新输出目录。历史 best 指标保留；新目录在超越历史 best 后才产生 best 文件，历史 best 文件保留在原目录。默认仅 EMA 保存时，没有完整中断恢复保障。

**实施与验收：**

按 2A（训练／EMA／推理保存）、2B（完整恢复）组织，两个子阶段代码均已实现。本地只做静态编译、shell 语法和差异检查，未运行模型／训练／DDP／恢复测试，不新增独立测试脚本。

服务器先运行四卡 global batch=8、micro batch=1、累积 2 次的两个更新，核对有限 loss、主要模块梯度、主权重／EMA 更新和 EMA 严格加载。完整恢复验收暂缓；以后需要时临时开启完整保存，比较连续 4 步与 2 步保存＋恢复至 4 步，核对后两步采样位置、LR、步数、EMA 次数和 loss；同一硬件／依赖下记录数值差异，不承诺跨环境逐位一致。README 给出正式入口的完整命令。实际 80GB 单卡容量和吞吐须测量，DDP 不合并四卡显存。

### 11.3 第三批：EMA 评测、多卡 rollout、成功率选模和 W&B（已实现，待服务器验收）

本批范围经用户确认扩大为独立评测及训练内自动评测，并参考 `plan/libero_exp.md` 与 DP workspace 的现有多卡实现。完整 checkpoint／live 推理入口延期，2B 不作为前置条件。

- 独立入口支持转换资产或自包含 EMA 文件，checkpoint 优先。严格读取内嵌配置／tokenizer／项目统计；完整训练文件明确拒绝。记录路径、SHA256、训练步数及权重类型。支持单卡和 torchrun 多卡，`all_tasks=true` 读取数据准备配置的十任务名单。
- 训练默认每 1000 optimizer steps 及正常结束评测，重合只执行一次，不在 step 0 评测。默认使用准备数据清单的任务，每任务 3 episodes、每卡 1 个环境；支持显式子集与数量覆盖。
- 各 rank 按 stride 分配任务，四卡十任务为 3／3／2／2；每卡一个任务队列。复用现有 runner 的 `configure_distributed`／`run_tasks`／`summarize_results`，自动端口 HTTP 服务、跨 `libero` 环境 evaluator、PCI/UUID 到 EGL 匹配及进程清理。无需改共享 runner、DP workspace 或 LIBERO-FT。
- 仅 rank 0 常驻 FP32 EMA；评测前每 rank 把 live 参数／buffer 备份到 CPU，安装并广播 rank 0 的 EMA 推理参数；各卡复用模型，结束后恢复 live、mode 和 RNG。CPU 需容纳每 rank 的 live 备份，不新增常驻 GPU 模型。
- Gloo 组负责 CPU 结果／错误同步，超时覆盖最长队列；评测期间没有挂起的 NCCL 等待。空任务 rank 参与同步，可捕获错误统一传播，不接受不完整结果。硬进程退出仍由 torchrun／通信超时兜底。
- 训练结果保存在 `rollouts/step_XXXXXXXX/eval/epoch_XXXX/`，按步数隔离，防止同 epoch 覆盖。
- 默认 `checkpoint.monitor=success_rate`，只保留 latest／best EMA。首次有效成绩保存 best，严格提高才覆盖，平分保留较早权重；validation loss 继续计算但只作诊断。不在线评测时显式设置 `rollout.enabled=false checkpoint.monitor=val_loss`。
- 文件内嵌 monitor、metric、best_metric、训练步数及来源；成功率选模包含评测配置。可选完整状态增加 selection；旧文件的 best loss 不当作成功率，2B 验收仍延期。
- W&B 默认 online、project=`diffusion_policy_libero`，支持 offline／disabled，仅 rank 0 上报；同一步训练／验证／评测统一提交。记录训练、显存、EMA loss、逐任务／总体成功率、耗时、视频及有效配置，不上传模型权重。本地 JSONL 和评测结果继续保留。

**本批变更文件：**

新增 `diffusion_policy/common/pi05_eval_util.py`、`configs/runner/pi05_libero.yaml`。
修改 `diffusion_policy/common/pi05_eval_libero.py`、`diffusion_policy/common/pi05_training_util.py`、`diffusion_policy/workspace/train_pi05_workspace.py`、`diffusion_policy/policy/pi05_policy.py`（加载格式提示及 mmap）、`configs/pi05_libero_eval.yaml`、`configs/pi05_libero_sft.yaml`、`scripts/pi05/eval_libero.sh`、`requirements_pi05_train.txt`、本文及 `diffusion_policy/model/pi05/README.md`。

**验收：**

本地仅 Python 静态编译、shell 语法、Hydra 配置组合及差异检查，不新增独立测试脚本。未运行模型、CUDA、多卡、HTTP、模拟器或 W&B 网络上报。
服务器先完成同一 STUDY 任务 3 episodes，再四卡十任务共 30 episodes；短训练连续两轮 rollout 后恢复更新，核对各卡服务／EGL、完整任务结果、独立目录、latest／best metadata、W&B 同步 step 和视频。另覆盖单卡、空任务 rank、可捕获 evaluator 启动失败的清理和错误传播。不设短训练成功率门槛；正式实验提高 episode 数。

`converted_libero` 与官方统计仍用于既有接口检查，不替代项目 SFT 成绩；数据不需重新准备。README 提供完整启动与覆盖命令。

### 架构修正：正式接入 FoveationNet

Pi0.5 的混合视觉路径由 `Pi05Policy` 正式持有 `MultiViewFoveationNet`，预处理器只负责原始图像、语言、状态和动作的统一变换，不再实现平行 patch sampler。两个真实相机的 routing 输出统一封装为 `Observation.foveation_outputs`，由 `Pi05Model` 消费；四叉树 geometry、固定/随机拓扑和 `vis_ar_tokens()` 复用 `diffusion_policy/model/roi`。训练与 rollout 的图片、视频渲染统一经过 `diffusion_policy/common/foveation_video_util.py`，第三相机保持无效 placeholder。

Stage A 仅启用 fixed/random topology 和 SigLIP 下游适配；StageB 使用 `pooled_global` 双相机 router、paired routing、budget loss 和 phase switch。固定拓扑使用 `router_type=none`，不构造 global encoder/router，随机拓扑使用 Torch RNG 并随已有 checkpoint RNG 恢复。旧 mixed 自定义 sampler checkpoint 不承诺兼容；dense base asset 和 dense rollout 保持原语义。StageA 的 forward、backward、可视化、rollout 和 checkpoint resume 仍以服务器日志为准，StageB 验收单独见第 12 节。
## 12. π0.5 LIBERO StageB 全量计划与当前实现

### 12.1 当前缺口与已实现边界

StageA 已使用 π0.5 模型、预处理、LIBERO dataset、SFT workspace、EMA、独立评测和多卡 rollout；`Pi05Model` 支持 patch14 的混合分辨率 SigLIP，`Pi05Policy` 持有双相机 `MultiViewFoveationNet`。四叉树 geometry、patch bank、固定/随机拓扑、packing、可视化以及 topology、patch count、foveation video 和三层 resolution advantage 诊断均复用现有 `FoveationNet`。用户服务器反馈确认高分辨率相对低分辨率有优势，但本地没有 StageA 输出目录，文档不补写运行时间、checkpoint 路径或成功率。

StageA 和 StageB 共用 `configs/pi05_libero_sft.yaml`，由 `scripts/pi05/train_libero_stageA.sh` 和 `scripts/pi05/train_libero_stageB.sh` 覆盖阶段参数；独立 StageB YAML 已移除。StageB 只从 `latest_ema.pt`/`best_ema.pt` 或 run directory 部分加载 StageA EMA，full checkpoint 只走 `resume`；非 router 的 model contract 必须匹配，新 router、global encoder 和 state condition projection 允许随机初始化，加载报告记录 loaded、allowed missing、source-only topology 摘要和 unexpected key，并保存来源路径与哈希。`Pi05Config` 放宽了 `mixed_foveation + pooled_global` 组合，同时保留 StageA 的 fixed topology + `router_type=none` 语义。共享 Temporal 的错误 leaf-mask 分支已移除，π0.5 的匹配状态改为从 packed topology count 派生；Spatial surrogate、全局 ELU budget、逐 camera 指标、episode-local Temporal pairing、active-branch EMA、phase switch 和 full checkpoint state 已接入 DDP 训练路径。StageB 默认的全局视觉 encoder 现为冻结的 `vjepa2_1_vit_base_384`，只将 56×56 global level 双线性缩放到 112×112 输入 JEPA，patch14 四叉树 geometry 保持不变；ResNet10 仍可通过脚本或 Hydra override 回退。

本轮仍需服务器验收的重点是 paired loss 的数值与梯度、Temporal 配对分母、budget lambda、两相机汇总、阶段切换、EMA/full checkpoint resume 以及多卡 rollout；这些都不能由静态检查代替。

### 12.2 确定设计

- StageB 默认使用 `router_type=pooled_global`、冻结的 `global_encoder_type=vjepa2_1_vit_base_384`、`global_feature_dim=768` 和 112×112 JEPA 输入；两个真实相机共享这一冻结 global encoder，但 router、state condition projection 和 budget 仍按相机独立。ResNet10/512 作为 fallback，语言只进入 π0.5 action model。每个相机独立 budget，总 patch 数仅作汇总日志。
- π0.5 geometry 是 patch14、224×224、三层。现有 `node_aligned_fpn` 明确要求 patch16，因此不能直接照搬到 StageB；patch14 只能使用 pooled-global router。
- 参考迁移默认总步数 1201，router-only 为 step 0–599，step 600 起 downstream adaptation；target patch 从 0.75 退火到 0.1，`paired_loss_weight=1.0`、`temporal_loss_weight=1.0`、`budget_loss_weight=0.1`、`upper_bound_lagrangian_elu`、base cost 0、lambda 初值 0、lambda lr 0.01。这些是初始迁移值，不是已验证最优超参数。
- Spatial paired 对同一观测、动作目标、noise、flow time 和 mask 计算 stop/split 两臂；只有 detached action advantage 通过选中父节点 logit 更新 router。Temporal 使用 `[all A, all B]` 布局，同一 episode 内配对并共享 flow time；无效 pair 继续参加 Spatial/budget，Temporal 分母单独记录。
- router-only 冻结 action/SigLIP 参数但保留输入梯度路径；downstream 冻结 router、关闭 paired/temporal、使用固定 router 的 native flow-matching loss 更新下游。两阶段共用 scheduler，切换时清理冻结分支 optimizer state。

### 12.3 接口、日志、checkpoint 与 rollout

`Pi05Policy` 提供 `compute_paired_loss()`、`set_stage_b_phase()`、`set_target_num_patches()` 和 `set_router_temperature()`；`LiberoFTDataset(output_mode=raw).set_temporal_pairing()` 只在同 episode 真实决策窗口中采样第二窗口，无候选时返回占位副本和 `temporal_pair_valid=false`，并把 pairing seed、episode ranges 和启用状态纳入 checkpoint metadata。workspace 保存 phase、lambda、target schedule、pairing、optimizer、scheduler、EMA、RNG 和 global step。

训练日志至少覆盖 `total_loss`、`diffusion_loss`、`paired_loss`、stop/split diffusion loss、advantage mean/absolute mean/positive fraction、`temporal_loss`、match fraction、valid/matched count、`budget_loss`、lambda、expected/actual patch counts、target、downstream flag 和 lr；每个相机保留 `views/image/*`、`views/gripper/*`。比例和跨卡计数必须按分子/分母汇总。rollout 继续使用 `Pi05Evaluation` 和 `LiberoSocketRunner`，写入逐相机及总 patch 数、成功率、原始双相机视频和四叉树可视化视频，并明确 routing 是 action execution 时生成还是 rollout 后重算。

### 12.4 实施顺序与验收

按以下顺序推进服务器实验：

1. StageB checkpoint 扩展加载、router forward 和 state condition；
2. Spatial paired loss smoke，确认两臂只改选中父节点且 router 有梯度；
3. budget/lambda 和双相机汇总；
4. Temporal dataset pairing、有效/无效 pair 和 temporal loss；
5. router-only→downstream phase switch，确认 router 固定、下游更新、scheduler 不重置；
6. EMA/full checkpoint resume，恢复 phase、lambda、pairing、optimizer、RNG 和 global step；
7. 单卡/四卡 LIBERO rollout、视频和 W&B 日志。

最小 acceptance 包括 all-coarse/all-fine/mixed tree 的有限 patch/mask/prefix/action loss，Spatial 两臂拓扑一致性，Temporal 计数和分母正确，双相机 router/budget 独立，切换前后更新对象正确，空任务 rank/evaluator 失败清理正常。最后才进行十任务、多 episode 正式 rollout，单独报告成功率和计算收益。

本地验证仅做 Python `py_compile`、相关 shell `bash -n`、Hydra 配置组合和 checkpoint key/config 静态检查；CUDA、DDP、W&B、LIBERO evaluator 及 StageA/StageB 训练结果均属于服务器侧验证。
