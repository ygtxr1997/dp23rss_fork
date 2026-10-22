# DP FPN 路由迁移到 π0.5：训练入口与多尺度 SigLIP 输入

状态：批次 1–3 及 DP 双相机、多任务预训练入口补充已实现；已完成本地静态验证，实际训练和迁移验收待服务器执行。

## 1. 目标和范围

目标是用双视角 DP 学习 `ResNet10 + node_aligned_fpn` 路由器，再将路由分支接入 π0.5。路由器只负责四叉树节点的 split 决策；π0.5 仍使用自己的 action policy、语言分支和 flow-matching 目标。

完整流程为 `DP StageA → DP StageB → Pi05 StageA → Pi05 StageB`，入口职责如下：

- `scripts/pi05/pretrain_foveation_libero_stageA.sh`：新增 DP source StageA，一个双相机模型联合训练完整 LIBERO-10；
- `scripts/pi05/pretrain_foveation_libero_stageB.sh`：新增 DP source StageB，继承 DP StageA EMA，训练并导出 routing artifact；
- `scripts/pi05/train_libero_stageA.sh`：Pi05 固定/随机拓扑 StageA，训练下游和 adapter/APT；不加载 DP artifact；
- `scripts/pi05/train_libero_stageB.sh`：加载 Pi05 StageA EMA 和独立 DP artifact。

DP 两个入口使用 `configs/liberoft_foveation_dp_for_pi05.yaml`，source 保持 `256×256 / patch16`；Pi05 保持 `224×224 / patch14`。保留现有 `scripts/roi/` 脚本。此次使用 ResNet10，JEPA 留待独立对照实验：现有 JEPA token 接口不能直接替换 node-aligned FPN 的 ResNet 分层特征。

必须保持的迁移合同：

- source 和 target 都使用真实双视角 `image, gripper`，顺序固定；
- 两个视角的 ResNet10、FPN 和 BN buffers 独立，不复制一套参数给另一视角；
- source 使用 `256×256 / patch16 / 3 levels / global_level=0`；
- Pi05 使用 `224×224 / patch14 / 3 levels`；
- source/target 相机方向一致，专用入口固定为 `rot180` 静态相机、wrist 不旋转；
- router temperature 使用固定值，Pi05 不新增 temperature schedule；
- router condition 使用 DP 的 6 维末端状态统计量和官方 10 维 task one-hot；
- 节点层数、节点顺序、4×4 + 8×8 内部节点网格保持一致。

允许保留的差异：图像增强、数据划分、action policy、action normalization、optimizer、scheduler、EMA、batch size 和训练步数。action normalization 可以继续由各自 action policy 独立管理，不能替代 router condition 的 DP normalization。

## 2. `node_aligned_fpn` 的尺寸适配

DP 和 π0.5 的 patch 尺寸不同，但三层四叉树的节点网格可以保持一致：

| 项目 | DP source | π0.5 target |
|---|---:|---:|
| 原始图像 | `256×256` | `224×224` |
| 下游 patch | `16×16` | `14×14` |
| image pyramid | `64/128/256` | `56/112/224` |
| 内部节点网格 | `4×4 + 8×8` | `4×4 + 8×8` |
| FPN router 输入 | `64×64` | `56×56 → 64×64` |
| split logits | 80 | 80 |

Pi05 路径必须分成两条：

```text
224×224 image
    ├─ patch path: 56/112/224 → 14×14 patch bank → SigLIP
    └─ router path: 56×56 → bilinear resize 64×64 → DP ResNet10 + FPN
```

resize 只作用于 FPN/ResNet 路由支路：`F.interpolate(..., size=(64, 64), mode="bilinear", align_corners=False)`。输入已经是 `64×64` 时跳过插值。不能只把 `patch_size` 改成 14，因为 FPN lateral feature map 的空间尺寸仍需匹配 DP 的节点网格。未来代码应让 `global_encoder_resize=64` 对 `node_aligned_fpn` 路径真正生效，并将几何校验改为检查“路由 feature map 与节点网格对齐”，而不是简单要求 patch16。

不把 ResNet feature map 输入 Pi05 action model；FPN 只产生节点决策，patch bank 仍由 Pi05 自己构造。

## 3. DP source StageA/StageB

专用配置列出与 Pi05 数据准备一致的官方 10 个 HDF5，复用 `LiberoFTDataset` 和 `MergeDataset`，一次运行训练一个共享模型。两个相机的 encoder/FPN 独立，task one-hot 同时提供给 router condition 和 DP 下游策略。默认按窗口 shuffle，不增加 task-balanced sampler；沿用 legacy 划分，StageB 继承 StageA 数据 seed、划分和统计量。

### StageA

专用 StageA 入口固定双视角、独立 ResNet10/FPN、`256×256` 输入、patch16、3 levels、`node_aligned_fpn`、`router_hidden_dim=128`、`router_cond_dim=512` 和固定 router temperature。StageA 保留固定或随机合法 topology，冻结/训练范围沿用 DP 的 source 配方，并记录 source/target 方向、任务映射、condition 统计量和 split metadata。

默认 StageA 使用 `random:0.0,0.25,0.5,0.75,1.0`，训练下游视觉和动作模块，冻结路由分支；默认 901 epochs。图像增强策略可以与 Pi05 不同，不作为迁移失败条件。

### StageB

专用 StageB 入口从 StageA checkpoint 开始，设置 `fix_split_mode=null`、`random_topology_prob=0`，使用 paired router objective。router-only 阶段更新两个独立 ResNet10、两个 FPN 和共享 condition projection，冻结 DP 下游视觉和动作模块，保持 temperature 固定，并导出 EMA 路由参数和 persistent buffers。

专用 DP StageB 默认 300 epochs，预算目标在 200 epochs 内从 0.75 降至 0.1，不自动切换 downstream。temperature 固定为 1.0。脚本参数可以通过环境变量或末尾 Hydra overrides 覆盖。

DP 与 Pi05 的 StageB step/epoch、target schedule、optimizer 和 EMA 可以不同；这些差异由目标 action policy 的训练框架决定，不作为 router artifact 的结构兼容条件。

## 4. Pi05 router condition

Pi05 router 使用独立于 action normalization 的 condition normalization：

```text
raw joint_state_B_8
    ├─ 前 6 维
    ├─ 使用 DP mean/std normalization
    └─ concat official task_onehot_B_10
        ↓
16 → router_cond_dim projection
```

task one-hot 使用完整官方 LIBERO-10 顺序，不使用任务子集索引，不从语言 prompt 推断 task ID；dataset、rollout 和离线视频回放使用相同 task ID。不能从 Pi05 已归一化的 state 重建 DP condition，也不能用 Pi05 action norm 替换 DP condition norm。source artifact 保存 DP state mean/std、拼接顺序和 normalization epsilon，condition projection 从 DP source 继承。两端统一使用 `(state - mean) / max(std, eps)`；统计量来自 `dataset_metadata.statistics["obs.ee_states"]`，不是 DP policy 的 identity normalizer。

## 5. Pi05 SigLIP 多尺度输入改造

Pi05 的下游 action policy、语言模型和 flow-matching loss 保持原生实现，但 SigLIP patch 输入端增加论文创新模块。原计划中“不要修改 SigLIP”的表述删除，改为“保留 SigLIP 预训练主体，在 stem 后扩展 patch token 表示”。

每个有效 `14×14` patch 的处理顺序固定为：

```text
14×14 RGB patch
    ↓ SigLIP stem: 3×14×14 → 1152
    ↓ level-specific residual patch adapter
    ↓ 原始 SigLIP absolute position
    ↓ APT multiscale position
    ↓ level embedding
    ↓ SigLIP Transformer
```

adapter 放在 stem 后、Transformer 前，保持原始 RGB stem 输入和 1152 维 token shape 不变。三个 quadtree level 各有一套 adapter，默认 bottleneck 为 64：`Linear(1152,64) → GELU → Linear(64,1152)`，最后一层 zero-initialize。padding token 不参与 adapter 的有效计算；ST leaf weight 只作用于 content token，不能乘到位置编码上。

APT 指 DP 的 learned fine-grid absolute position map 加多尺度插值机制，不是新的 action token：

1. 在最高分辨率 `16×16` patch grid 上维护可学习的 1152 维 position map；
2. 插值生成 `4×4、8×8、16×16` 三层 position map；
3. 按全局 patch-bank index 拼接统一 position bank；
4. 用 `patch_bank_indices_B_T_N` 为有效 token gather APT position；
5. 保留原始 SigLIP absolute position，之后叠加 APT 和 level embedding；
6. 对 invalid padding token 使用 valid mask 清零位置项。

APT 统一编码 row/column、quadtree level 和 canonical patch-bank 顺序，使 coarse/medium/fine token 使用一致的空间坐标。

Pi05 配置新增或明确：

```yaml
foveation_patch_level_adapter: true
foveation_patch_level_adapter_channels: 64
foveation_apt_position_encoding: true
foveation_level_embedding: true
```

dense Pi05 可以关闭 adapter/APT；mixed foveation 默认打开。APT dimension 固定为 SigLIP stem dimension 1152。

## 6. Artifact、初始化和训练阶段

统一导出 `foveation_pretrained`，包括两套 ResNet10/FPN 参数及 buffers、共享 condition projection、DP state mean/std、task mapping、source/target geometry、resize=64、节点顺序、camera order、固定 router temperature、配置和 split metadata。只迁移各相机 `global_encoder.*`、`router.*` 和共享 condition projection，不复制 DP patch-bank 像素坐标等 geometry buffers；Pi05 自己生成 224/patch14 几何。不保存 DP action policy、DP 下游 visual encoder、optimizer moments 或训练进度。artifact 只做必需字段、camera/node order、geometry 和 tensor shape 校验，不加入版本或 hash 校验。

DP workspace 的 `foveation_export_path` 默认为 null；专用 StageB 脚本启用，默认导出到 run 目录下的 `foveation_pretrained.pt`，可通过 `FOVEATION_EXPORT_PATH` 指定相对 run 的路径或绝对路径。主进程在保存 latest checkpoint 时和训练正常结束时导出当前 EMA；结束时 artifact 可能比最近一次周期 checkpoint 更新。可用原路径覆盖保存，不引入额外版本机制。

Pi05 StageB 显式设置 `foveation_router_hidden_dim=128` 与 DP 对齐；即使冻结 encoder，FPN 两个相机仍保留独立参数和 BN buffers。完整 checkpoint 恢复时同时恢复 artifact condition contract 和固定 temperature。

Pi05 加载顺序：创建 Pi05 policy，加载 Pi05 StageA EMA 下游权重，严格校验并加载两套 ResNet/FPN 和 condition projection，创建或加载 adapter/APT 参数，校验 geometry/camera order/condition/state-dict，最后创建 Pi05 optimizer、EMA、DDP 并设置 phase。

旧 Pi05 base checkpoint 缺少新增 adapter/APT 参数时使用 zero-init；已有 SigLIP、PaliGemma 和 action 参数继续严格加载。resume checkpoint 必须自包含新增参数，不重新读取外部 artifact。

StageA 在固定或随机合法 topology 下训练 Pi05 下游、SigLIP、PaliGemma、adapter/APT 和 level embedding。StageB 的 `router_only` 阶段只更新两个 ResNet、两个 FPN、router 和 condition projection，冻结 Pi05 下游、adapter/APT 和 level embedding；temperature 固定，budget target 在校准期固定。切 downstream 后冻结完整 router、condition projection、adapter 和 APT 并设为 eval，不创建 temporal paired data，不更新 budget lambda，只更新 Pi05 downstream action/SigLIP/PaliGemma。

## 7. 文件范围

实现阶段涉及：

- `diffusion_policy/model/roi/foveation_net.py`：FPN resize64 和路由几何校验；
- `diffusion_policy/model/pi05/siglip.py`：stem 后 level adapter、APT position bank、level embedding 和 mask；
- `diffusion_policy/model/pi05/config.py`、`diffusion_policy/model/pi05/model.py`：配置和 SigLIP 输入；
- `diffusion_policy/policy/pi05_policy.py`、`diffusion_policy/common/foveation_pretrained.py`：condition、artifact 和 phase；
- `diffusion_policy/workspace/train_pi05_workspace.py`：初始化、EMA、DDP、resume；
- `diffusion_policy/dataset/libero_dataset.py`、`diffusion_policy/env_runner/libero_socket_runner.py`：固定 task ID；
- `configs/pi05_libero_sft.yaml`；
- `scripts/pi05/train_libero_stageA.sh`：改造成 Pi05 foveation StageA 入口；
- `scripts/pi05/train_libero_stageB.sh`：改造成 Pi05 foveation StageB 入口。

## 8. 验证和实验对照

本地只做 Python `py_compile`、shell `bash -n`、Hydra 配置组合和 state-dict 静态检查。检查 `56→64` 只发生在 FPN 路径、Pi05 patch path 仍为 14×14、APT 与 4/8/16 patch bank 对齐、adapter/APT mask 正确、旧 checkpoint 新参数 zero-init，以及 warm_start/downstream_only 梯度边界。

服务器验收包括双视角 artifact 导出、两套 ResNet/FPN 独立性、80 个 split logits、14×14 leaves、APT bank 顺序、zero-init 回退原生 SigLIP、adapter/APT 的 level 差异、阶段冻结边界和旧 Pi05 checkpoint 兼容性。

实验对照包括：原生 Pi05 dense、原生 fixed/mixed foveation、FPN router 无 adapter/APT、FPN router + level adapter、FPN router + APT/level embedding、FPN router + adapter + APT、warm_start/downstream_only，以及相同实际 patch budget 的 fixed/learned topology。比较 success rate、实际 patch 数、路由稳定性、收敛速度、token 表示差异和总 GPU-hours；增强、数据划分、action normalization、optimizer 等允许不同的配置不作为 router 迁移失败判据。

## 9. 进度追踪

### 批次 1：基础视觉路径和 Pi05 patch 表示

状态：已完成

准备内容：

- Pi05 target 使用 `224×224 / patch14`，router path 使用 `56×56 → 64×64` resize；
- adapter 放在 SigLIP stem 后、Transformer 前；
- APT 使用 `4/8/16` multiscale position bank；
- 兼容旧配置 `foveation_level_adapter`，新增 canonical adapter/APT 配置名；
- artifact 只保留后续需要的结构字段和 tensor shape 校验，不加入版本或 hash 校验。

完成标准：

- FoveationNet 支持 Pi05 `patch14 + node_aligned_fpn + resize64`；
- SigLIP 支持 level adapter、APT、level embedding 和 valid-token mask；
- dense/base checkpoint 缺失新增参数时可以 zero-init；
- 完成 `py_compile` 和配置构造检查。

当前进度：

- [x] 计划文档加入分批实施和 artifact 简化约束；
- [x] FoveationNet 的 patch14/resize64 路由实现；
- [x] SigLIP adapter/APT 实现；
- [x] 配置和 base checkpoint 兼容；
- [x] 本批静态验证。

已完成文件和行为：

- `diffusion_policy/model/roi/foveation_net.py`：允许 Pi05 `patch14 + resize64`，并在 FPN 路径对 `56×56` 输入执行 bilinear resize；
- `diffusion_policy/model/pi05/siglip.py`：加入 level-specific residual adapter、APT position bank、valid-token mask；
- `diffusion_policy/model/pi05/config.py`、`configs/pi05_libero_sft.yaml`：加入 canonical adapter/APT 配置并兼容旧 adapter 字段；
- `diffusion_policy/model/pi05/model.py`：把 adapter/APT 配置接入 SigLIP；
- `diffusion_policy/model/pi05/checkpoint.py`：官方 base checkpoint 缺少新增 additive 参数时 zero-init。

已完成验证：

- `python -m py_compile` 覆盖本批修改的 Python 文件；
- `Pi05Config` mixed-foveation、旧 adapter 别名和 APT 配置构造检查；
- 未运行 GPU 训练、rollout 或服务器环境验证。

### 批次 2：DP artifact、task condition 和 phase 冻结

状态：已完成

准备内容：

- 独立保存两视角 router、condition normalization、task mapping 和几何字段；
- dataset 显式传递 LIBERO task ID；
- 实现 `6 + 10 = 16` 维 router condition；
- 实现 StageB router-only/downstream 冻结边界。

完成标准：

- artifact 只做必需字段、camera/node order、geometry 和 tensor shape 校验；
- StageB router-only 只更新 router；
- downstream 冻结完整 routing branch 和 adapter/APT；
- full resume 不依赖外部 artifact。

当前进度：已完成。新增轻量 DP routing artifact 工具，dataset/rollout 显式携带 task ID，Pi0.5 condition 改为 DP state normalization 加官方 LIBERO-10 one-hot，StageB router-only/downstream 冻结边界已接入；full resume 将 artifact 的 condition contract 写入 policy extra state。未加入版本或 hash 校验。

本批修改文件：

- `diffusion_policy/common/foveation_pretrained.py`
- `diffusion_policy/dataset/libero_dataset.py`
- `diffusion_policy/policy/pi05_policy.py`
- `diffusion_policy/workspace/train_pi05_workspace.py`
- `diffusion_policy/env_runner/libero_socket_runner.py`
- `configs/pi05_libero_sft.yaml`

已完成验证：上述 Python 文件通过 `python -m py_compile`；未运行 GPU 训练、rollout、多卡和服务器环境验证。

### 批次 3：训练入口、配置整合和完整静态验收

状态：已完成

准备内容：

- 改造现有 Pi05 StageA/StageB shell 入口；
- 固定 StageB 使用 node_aligned_fpn、ResNet10 和 resize64；
- 接入 artifact 路径、provenance 和 resume；
- 完成 dense/StageA/StageB/downstream 配置组合检查。

完成标准：

- 两个 Pi05 shell 入口通过 `bash -n`；
- StageA EMA 可以初始化 StageB；
- StageB 可以加载独立 DP artifact；
- 完成全量静态检查和冻结边界检查。

实际修改：

- `scripts/pi05/train_libero_stageA.sh`：StageA 显式打开 patch-level adapter、APT 和 level embedding，固定随机拓扑概率为 0。
- `scripts/pi05/train_libero_stageB.sh`：要求传入 `FOVEATION_PRETRAINED_PATH`，固定 `node_aligned_fpn`、ResNet10、512 维特征、`64×64` router 输入、无固定 split、adapter/APT 和随机拓扑概率 0，并传入独立 artifact 路径。
- `configs/pi05_libero_sft.yaml`、`diffusion_policy/model/pi05/config.py`、`diffusion_policy/policy/pi05_policy.py`：补齐 `foveation_random_topology_prob` 配置，并允许 node-aligned FPN 在 learned mixed topology 下构造。
- `diffusion_policy/workspace/train_pi05_workspace.py`：将该路由配置纳入 StageA/StageB 合同检查；artifact 加载、provenance、full resume 和阶段冻结逻辑沿用批次 2 实现。

已完成验证：两个 shell 通过 `bash -n`；StageA 固定拓扑和 StageB node-aligned FPN 配置可由 `Pi05Config` 构造；相关 Python 文件通过 `python -m py_compile`；`git diff --check` 通过。未运行 GPU 训练、DP artifact 实际导出、rollout、多卡和服务器环境验证。

### 补充批次：双相机 LIBERO-10 DP source 训练入口与导出闭环

状态：代码已完成，本地静态验证通过；服务器训练验收待执行。

- [x] 新增 `configs/liberoft_foveation_dp_for_pi05.yaml`，继承 DP 基础配置，启用双相机、完整 10 任务、rot180、ResNet10/FPN、resize64 和固定 temperature。
- [x] 新增 `scripts/pi05/pretrain_foveation_libero_stageA.sh` 与 `pretrain_foveation_libero_stageB.sh`，支持 Accelerate 单/多卡、环境变量和 Hydra overrides。
- [x] `train_foveation_dp_transformer_workspace.py` 接入可选 EMA artifact 导出，使用继承的 DP state 统计量。
- [x] `foveation_pretrained.py` 仅导出可迁移分支，记录相机方向、任务映射和路由配置，不加入版本/hash 校验。
- [x] `pi05_policy.py` 严格加载完整 encoder/router 和 condition projection，保留 target geometry，统一 std clamp 并恢复 temperature。
- [x] `multi_view_foveation_net.py` 修正冻结 FPN encoder 被跨相机共享的问题；其他 router 的原共享行为保留。
- [x] Pi05 StageB 脚本设置 router hidden dimension=128；清理 Pi05 YAML 重复的随机拓扑配置项。
- [x] 纠正文档中 DP/Pi05 脚本职责与四阶段依赖关系。

已完成验证：Python 静态编译、三个新增/修改 shell 的 `bash -n`、`git diff --check`；使用本地 robodiff 环境拦截脚本 launch 参数后进行 Hydra 配置组合（不启动训练），确认 DP StageA/StageB 双相机顺序、完整 10 任务、拓扑模式、resize64、固定 temperature、导出路径，以及 Pi05 StageB hidden dimension=128 和 `Pi05Config` 构造。

未运行：实际 HDF5 读取、checkpoint/EMA 初始化和 artifact 导出加载、GPU 训练、梯度/BN 冻结验证、80 logits/patch geometry 运行时检查、rollout 和多卡训练。不新增独立测试脚本。

启动示例（替换 checkpoint 和数据/模型资产路径；两个 DP 阶段保持相同的架构 overrides）：

```bash
CUDA_VISIBLE_DEVICES=0,1 OUTPUT_DIR=outputs/dp_pi05_A \
  bash scripts/pi05/pretrain_foveation_libero_stageA.sh

CUDA_VISIBLE_DEVICES=0,1 OUTPUT_DIR=outputs/dp_pi05_B \
  PRETRAINED_CKPT=outputs/dp_pi05_A/checkpoints/latest.ckpt \
  bash scripts/pi05/pretrain_foveation_libero_stageB.sh

CUDA_VISIBLE_DEVICES=0,1 OUTPUT_DIR=outputs/pi05_A \
  bash scripts/pi05/train_libero_stageA.sh

CUDA_VISIBLE_DEVICES=0,1 PRETRAINED_CKPT=outputs/pi05_A \
  FOVEATION_PRETRAINED_PATH=outputs/dp_pi05_B/foveation_pretrained.pt \
  bash scripts/pi05/train_libero_stageB.sh
```

`PRETRAINED_CKPT` 在 DP StageB 指 DP StageA checkpoint，在 Pi05 StageB 指 Pi05 StageA EMA checkpoint/run。旧的全 FoveationNet artifact 缺少本次明确的 routing contract，需按新入口重新导出。


### TCL 完整统计与归一化合同

TCL 新资产只生成 `dataset_metadata.json`。`statistics.state`、
`statistics.actions` 和启用时的 `statistics.force_torque` 均保存
`q01/q99`、`min/max`、`mean/std`。min/max 和 mean/std 覆盖所有配置源的全部帧，
采用分块 float64 合并，总体标准差 `ddof=0`；action 来自
`extracted_data["rel_actions"]`。quantile 保留原抽样范围、seed 和每源数量。
每个统计块记录 `count`、`quantile_count`，`statistics_scope` 记录统计口径。
`prepare_force_stats=false` 时不生成 force 统计，也不新增模型观测。

资产默认路径固定写入 YAML：`outputs/pi05_data/tcl_foveation_seed42`。
TCL 准备和四个训练入口不再读取 `PI05_TCL_ASSET_DIR` 或 `ASSET_DIR` 来覆盖路径。
自定义目录显式传入，例如：

```bash
bash scripts/pi05/prepare_tcl_data.sh asset_dir=outputs/pi05_data/tcl_full_stats_v2
CUDA_VISIBLE_DEVICES=0 bash scripts/pi05/pretrain_foveation_tcl_stageA.sh \
  asset_dir=outputs/pi05_data/tcl_full_stats_v2
```

准备脚本拒绝覆盖非空目录。旧资产不会自动补算；切换方法缺少统计时应准备新目录。
已有 checkpoint 必须继续使用其原统计和归一化合同，不能用新资产覆盖。
LIBERO 仍使用自己的路径、双文件和原校验流程；
`pretrain_foveation_libero_stageA.sh` 不依赖 TCL 的资产路径或环境变量。

DP 配置显式声明：

```yaml
task:
  dataset:
    norm:
      state: identity
      action: minmax
      force: quantile
```

Pi0.5 TCL 的同一配置位置默认 state/action 为 `quantile`。TCL 支持四种方法，
`mean` 仅作为旧 `mean_std` 别名；旧 `norm_force_type` 与新显式配置冲突时报错。
LIBERO DP 声明 state `mean_std`、action `minmax`、force `quantile`；
LIBERO Pi0.5 声明 quantile，其他配置被拒绝，原计算公式保持不变。

| 新 TCL 方法 | 正向转换 | 反向转换 |
| --- | --- | --- |
| identity | x | y |
| minmax | `2*(x-min)/max(max-min,1e-6)-1` | `(y+1)/2*max(max-min,1e-6)+min` |
| quantile | `2*(x-q01)/(q99-q01+1e-6)-1` | `(y+1)/2*(q99-q01+1e-6)+q01` |
| mean_std | `(x-mean)/max(std,1e-6)` | `y*max(std,1e-6)+mean` |

所有新 TCL quantile 均不裁剪，与 Pi0.5 一致。因此新 force 训练也会保留
q01/q99 之外的值。旧 checkpoint 则保留旧单源 force 裁剪和 epsilon 语义。
归一化合同保存方法、维度、epsilon、裁剪规则和实际使用的统计量。

| 路径 | 统计及归一化执行位置 | Router state |
| --- | --- | --- |
| DP Stage A | dataset 读取资产，按 `task.dataset.norm` 转换 | dataset 按 state 方法转换后的前六维 + task one-hot |
| DP Stage B / resume | 继承 Stage A / 自身 checkpoint 合同和划分 | 同已训练 DP 合同 |
| Pi0.5 Stage A | raw dataset；policy 按配置做 norm/denorm | 动作模型不使用 state token |
| Pi0.5 Stage B / resume | 继承 Stage A / 自身 checkpoint；policy 做 norm/denorm | 原始 TCP 经 DP artifact 的 state 合同转换 |

TCL dataset 提供 `normalize(values, key)` / `denormalize(values, key)`，
`key` 为 `state/action/force`；公共模块提供不打开数据文件的同一转换，支持 NumPy/Torch。
DP 训练、验证和录像观测入口使用同一合同。Pi0.5 preprocessing 复用公共转换，
继续负责 padding 和动作输出，service 不重复 denorm。
Pi0.5 checkpoint 保留 `assets.norm_stats`，并保存 `assets.normalization` 和有效 dataset metadata。

Stage B/resume 检查方法和实际统计，不允许配置或外部资产更新改变权重语义。
旧 DP min/max 字段仍可读；旧 Pi0.5 metadata 完全没有 quantile 时才允许从同目录
`norm_stats.json` 回退，部分缺失或非法 quantile 不回退。
旧 checkpoint/artifact 缺少新合同时按原路径恢复默认算法；LIBERO router 继续使用原 mean/std。

DP service 优先核对 checkpoint 与运行目录 metadata、配置，按合同对输入 TCP norm、
对 action denorm；仅统计缺失时读取训练配置明确记录的 asset。新训练保存解析后的绝对资产路径。
minmax/quantile 的预测仍裁剪到 [-1,1]；identity/mean_std 不作此范围裁剪。
gripper 阈值处理仍保留。这些是部署后处理，最终动作不是未经后处理的严格逆变换。
协议、默认 task ID、TCP 截取、checkpoint 搜索和图像处理不变。

本地只做 Python 静态编译、相关脚本 `bash -n` 和 `git diff --check`，不新增测试脚本。
真实资产生成、四条 GPU 训练路径、不同 norm 方法、EMA/full 恢复、artifact 迁移和
 evaluator 联调需在服务器验证。
