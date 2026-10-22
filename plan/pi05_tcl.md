# Pi0.5 TCL 训练与 DP Foveation 迁移计划

状态：Pi0.5 TCL StageA 和 StageB、DP TCL StageA/StageB、foveation artifact 导出、Pi0.5 GPU 推理服务以及 dataset 统一重构代码已实现；DP/Pi0.5 真实训练、artifact 迁移和真机通信仍待服务器验证。

## 1. 本次目标和范围

最终目标是为 TCL 建立下面的训练流程：

```text
Pi0.5 TCL StageA
    ↓
确认 Pi0.5 TCL StageB 的超参数和冻结边界
    ↓
DP TCL StageA → DP TCL StageB → 导出 foveation artifact
    ↓
Pi0.5 TCL StageB 加载 Pi0.5 StageA 权重和 DP artifact
```

本次文档记录后续实现所需的接口、配置、训练顺序和验收标准。当前仓库已经完成 Pi0.5 TCL StageA/StageB、DP TCL 入口、artifact 迁移代码和 Pi0.5 GPU 推理服务；真实训练、foveation artifact 迁移验证和真机通信仍属于后续服务器工作。

当前已完成的 Pi0.5 StageA 文件包括：

- `diffusion_policy/dataset/tcl_dataset.py` 中的 `TCLFoveationDataset`：统一支持 DP normalized 和 Pi0.5 raw 多任务 TCL 输入、episode-local window、6/7 维 state 和 32×7 action window；
- `diffusion_policy/common/prepare_data.py`：统一准备 TCL/LIBERO 的训练前 statistics 和 metadata 资产；
- `configs/pi05_tcl_data.yaml`、`configs/pi05_tcl_sft.yaml`：数据与 StageA 配置；
- `scripts/pi05/prepare_tcl_data.sh`、`scripts/pi05/train_tcl_stageA.sh`：数据资产准备和 StageA 训练入口；
- `meta/tcl_tasks_10.yaml`：最多 10 个 task slot 的手动 mapping 文件；
- Pi0.5 config、preprocessing、model、policy 和 workspace：支持 6/7 维 state、224×224 TCL 图像、32 步 action horizon、EMA/checkpoint/resume 及离线 validation。

以下功能代码已经完成，但真实训练或联调仍未完成：

- `scripts/pi05/pretrain_foveation_tcl_stageA.sh`（已实现）；
- `scripts/pi05/pretrain_foveation_tcl_stageB.sh`（已实现）；
- `scripts/pi05/train_tcl_stageB.sh`（已实现）；
- `configs/tcl_foveation_dp_for_pi05.yaml` 和 `TCLFoveationDataset` 多任务 adapter（已实现）；
- DP artifact 导出和 `foveation_pretrained.py` 的 TCL mapping 泛化（已实现）；
- `gpu_service_pi05.py`（已实现；远程 evaluator 联调待服务器完成）。

## 2. 已确定的共享训练合同

### 2.1 Pi0.5 动作策略输入

Pi0.5 动作模型第一版只接收双相机图像和语言文本：

- 使用 image 和 gripper 两个 RGB 视角；
- 使用当前窗口的 `language_text`；
- Pi0.5 preprocessing 允许接收 6 维或 7 维 TCL state，但动作模型忽略这些 state；
- 不把 TCP pose、gripper width 或 force 编码为动作模型的 state token；
- 不新增 state projection、state embedding 或其他动作模型输入权重；
- 继续复用现有 Pi0.5 的 SigLIP、PaliGemma 和 action expert 权重。

TCL 的 TCP pose 只供 foveation router 使用，不作为动作模型的直接输入。

### 2.2 Router condition

两套相机的 foveation router 使用相同的 condition contract：

```text
TCP xyz + rotation：6 维
TCL task one-hot：10 维
总输入：16 维
```

router 不使用 force、success 或 gripper width。Pi0.5 动作模型的输入合同和 router condition 合同必须在代码和 checkpoint metadata 中分开记录。

### 2.3 TCL task mapping

后续新增 `meta/tcl_tasks_10.yaml`，作为所有训练和推理入口共用的 task mapping。列表最多包含 10 个任务，实际 task ID 由文件显式定义：

```yaml
tasks:
  - task_id: 0
    task_dir: 1014_lh_tower_three_block
  - task_id: 1
    task_dir: ...
```

`task_dir` 就是 TCL 数据目录中的实际 task directory 名称，例如 `1014_lh_tower_three_block`。训练脚本解析 config 得到实际需要的数据目录后，必须逐项在这个 YAML 列表中匹配；config 中有未匹配的 task directory 时立即报错。这个列表由用户后续手动维护，计划只固定 key 和匹配规则，不预先填写完整任务列表。

实际 router condition 仍保留 10 个 one-hot 位置；未使用的 task ID 可以为空，但实际 task_id 必须来自 YAML，不能从目录排序或语言文本动态推断。dataset、DP workspace、Pi0.5 workspace、artifact 和真机服务都读取同一 mapping，artifact 和 full checkpoint 保存实际使用的 mapping。

本次只记录该文件的接口，不创建该文件。

### 2.4 图像几何

TCL 原始图像和两端模型几何固定为：

```text
TCL 原始图像：任意分辨率，例如 256×320、240×320（均为 H×W）
task.dataset.image_resize_wh_ratio: "3:2"
DP source：Dataset resize → 171×256，上/下补 42/43 行 → 256×256，patch16
Pi0.5 target：Dataset resize → 149×224，Policy 上/下补 37/38 行 → 224×224，patch14
```

两端都使用三层 `4×4、8×8、16×16` patch/node 网格，共 80 个内部 split logits。两个相机直接 resize 到固定内容宽高比，保留完整画面并接受形变；采用 antialiased bilinear interpolation，像素四舍五入为 uint8。目标宽度来自 RGB shape_meta（Pi0.5 Workspace 从 model.image_size 构造），内容高度取 round(width × 2 / 3)，再居中补零；归一化后黑边为 -1。ratio 为 null 时保留旧路径，不同原始尺寸不再影响启用 ratio 后的黑边厚度。训练、验证、temporal pair、诊断和离线视频使用相同处理入口。

Pi0.5 路由支路将 `56×56` 的输入 resize 到 `64×64` 再送入 ResNet10/FPN；这个 resize 只作用于路由支路，不能改变 Pi0.5 的 patch bank 几何。

DP 和 Pi0.5 必须采用一致的相机顺序和相机方向。默认沿用现有迁移合同：static camera 使用 `rot180`，wrist camera 使用 raw orientation。

### 2.5 Action window

Pi0.5 TCL 训练使用 `32×7` action window：

- 训练 `action_horizon=32`；
- action 维度为 7；
- 默认执行前缀 `n_action_steps=24`；
- rollout/test runner 可以覆盖实际执行步数；
- 预测长度和执行长度不绑定。

action projection 仍使用现有 Pi0.5 的内部 padded action representation，因此 action horizon 的扩展不应改变已有权重的参数形状。Gemma action expert 的 action context 支持最多 50 个 action tokens，32 步预测不改变模型结构。Pi0.5 TCL 使用自己的 action statistics；DP 的 action normalization 独立管理，不写入 routing artifact。

DP 和 Pi0.5 使用同一批训练数据，train 默认使用全量数据，val 只采样一部分且允许与 train 重叠。两端的统计计算范围应保持一致；action policy 可以使用各自的 normalization 方式，但 foveation/router 依赖的 normalization 和统计量必须使用 DP 的默认方式，并随 artifact 继承到 Pi0.5。

## 3. 实现顺序

### 3.1 第一阶段：Pi0.5 TCL StageA

未来新增：

- `scripts/pi05/train_tcl_stageA.sh`；
- Pi0.5 TCL 专用 data/config/runner 配置；
- TCL 到 Pi0.5 dataset/workspace 的适配。

数据处理要求：

- 读取 raw RGB 双相机图像；
- 按 image_resize_wh_ratio 统一双相机内容尺寸，DP 在 Dataset 补边，Pi0.5 在 Policy 补边；
- 每个窗口使用当前观测帧的 `language_text`；
- 读取 32 步、7 维 `rel_actions`；
- 保存与 Pi0.5 action policy 对应的 action statistics；
- 传递 6 维 raw TCP pose 和 task ID 给 router；
- 不向动作模型增加 state token。

augmentation 应先作用于 foveation 视觉编码器，输出的 mixed-resolution patches 已经包含 augmentation 结果。DP 和 Pi0.5 可以使用不同 augmentation recipe，但必须分别保证训练和推理一致，且双相机的处理方式一致。

StageA 默认配置：

- image size 224；
- patch size 14；
- 3 个 foveation levels；
- patch-level adapter、APT position encoding、level embedding 开启；
- learned router 关闭；
- topology 使用 `40% level:2 + 60% random:0.05-0.99`；
- global batch 128；
- micro batch 16；
- learning rate `5e-5`；
- warmup 200；
- `max_steps=4001`；
- 默认 `n_action_steps=24`；
- 所有训练、输出、batch 和 rollout 参数支持环境变量或 Hydra override。

第一阶段服务器验收必须确认：真实 TCL HDF5 读取、双相机 batch、language prompt、32 步 action window、mixed-resolution patch bank、训练 checkpoint 和恢复流程均能运行。

### 3.2 第二阶段：Pi0.5 TCL StageB（已实现，待服务器训练）

未来新增 `scripts/pi05/train_tcl_stageB.sh`。StageB 加载两类输入：

1. Pi0.5 TCL StageA 的 EMA/checkpoint；
2. DP TCL StageB 导出的 routing artifact。

加载顺序固定为：

1. 构造 Pi0.5 TCL policy；
2. 加载 StageA 的下游、SigLIP、PaliGemma、adapter、APT 和 level embedding；
3. 校验并加载 artifact 中两套独立 ResNet10/FPN/router 和共享 condition projection；
4. 校验 camera order、task mapping、source/target geometry、node order、pose statistics 和 tensor shapes；
5. 创建 StageB optimizer、EMA 和 DDP；
6. 进入 router-only phase。

StageB 默认配置：

- `max_steps=6001`；
- router-only 阶段持续到 step 3000；
- target patch budget 从 0.75 在 3000 steps 内衰减到 0.1；
- paired router objective；
- router temperature 固定为 1.0；
- temporal adaptive pairing 开启；pair 只要求来自同一 episode，具体匹配方式参考当前 DP StageB 和 Pi0.5 LIBERO StageB，不额外增加跨窗口限制；
- router-only 阶段只更新两个 ResNet/FPN、router 和 condition projection；
- downstream 阶段冻结完整 router、condition projection、adapter、APT 和 level embedding；
- downstream 阶段只更新 Pi0.5 下游动作模型；
- 动作模型始终只使用图像和文本。

### 3.3 第三阶段：DP TCL StageA/StageB（已实现，待服务器训练）

已新增：

- `scripts/pi05/pretrain_foveation_tcl_stageA.sh`；
- `scripts/pi05/pretrain_foveation_tcl_stageB.sh`；
- `configs/tcl_foveation_dp_for_pi05.yaml`；
- `diffusion_policy/dataset/tcl_dataset.py` 中的 `TCLFoveationDataset`。

DP TCL 两个阶段支持与 Pi0.5 相同的多任务输入：

```bash
GPU_RESERVE_GB=8 \
DATA_ROOTS_OVERRIDE='["/path/task_a","/path/task_b"]' \
H5_PATHS_OVERRIDE='["/path/task_a.h5","/path/task_b.h5"]' \
TASK_MAPPING_PATH=meta/tcl_tasks_10.yaml \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/pi05/pretrain_foveation_tcl_stageA.sh
```

StageA 脚本内部仍保留可编辑的 Bash `DATA_ROOTS=(...)` 和 `H5_PATHS=(...)` 默认数组，启动时会自动序列化成 Hydra list。多任务实验使用上面的 `*_OVERRIDE` 字符串覆盖默认数组。

`GPU_RESERVE_GB` 按每张 GPU 预留固定 GiB 显存，默认值为 `0`。预留 buffer 不参与训练计算，也不会写入 checkpoint；多卡运行时每个 Accelerate rank 在自己的 GPU 上分别预留相同大小的显存。

运行 StageB 前，必须先得到 DP StageA checkpoint：

```bash
GPU_RESERVE_GB=8 \
DATA_ROOTS_OVERRIDE='["/path/task_a","/path/task_b"]' \
H5_PATHS_OVERRIDE='["/path/task_a.h5","/path/task_b.h5"]' \
TASK_MAPPING_PATH=meta/tcl_tasks_10.yaml \
PRETRAINED_CKPT=outputs/dp_tcl_for_pi05_stageA/.../checkpoints/latest.ckpt \
FOVEATION_EXPORT_PATH=outputs/dp_tcl_for_pi05_stageB/.../foveation_pretrained.pt \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/pi05/pretrain_foveation_tcl_stageB.sh
```

DP adapter 根据 image_resize_wh_ratio 对 raw uint8 双相机图像直接 resize，再填 0 黑边到 shape_meta 指定的 `256×256`。Pi0.5 Dataset 在 collate 前统一为 `149×224`，Policy 再补边；状态、动作和力的统计资产无需重建。新规则改变已有 checkpoint 的输入图像几何，不视为旧实验的等价续训。train 使用全量 frame，val 使用固定 seed 的子集并允许与 train 重叠；episode 边界只用于 temporal pairing。

DP TCL router condition 固定为 6 维 raw TCP state 加 10 维 task one-hot。DP 继续使用 raw + identity state normalization，导出的 artifact 保存零均值、单位方差的 identity contract；Pi0.5 StageB 必须沿用该 contract，不能使用 Pi0.5 action statistics 替代。

DP source 必须和 Pi0.5 artifact 合同对齐：

- 256×256 / patch16；
- 双相机独立 ResNet10/FPN；
- `global_feature_dim=512`；
- `router_hidden_dim=128`；
- `global_encoder_resize=64`；
- `6 维 TCP pose + 10 维 task one-hot` condition；
- 不使用 force/success/gripper width；
- 使用 `meta/tcl_tasks_10.yaml`；
- camera order、orientation、node order 与 Pi0.5 一致。

DP StageA 默认保留当前 TCL DP 配方，并优先遵循 LIBERO Pi0.5 source 入口的初始化和冻结方式：

- 1500 epochs；
- 随机合法 topology；
- `p_cond_drop=0.3`；
- budget loss 关闭；
- foveation、global encoder、router 和其相关 condition 分支不训练；
- 通过 split mode 提供固定/随机混合分辨率 patch；
- 只训练下游 mixed-resolution visual encoders 和 action transformer policy；
- routing branch 的初始化、torch cache/预训练 encoder 使用方式、freeze/eval 语义和配置优先级，直接对齐 `scripts/pi05/pretrain_foveation_libero_stageA.sh` 及其配置，不另造一套初始化规则。

DP StageB 默认保留当前 TCL DP 配方：

- 900 epochs；
- 从 DP StageA checkpoint 初始化；
- router-only；
- paired router objective；
- temperature 固定为 1.0；
- 现有 target budget 和 lambda schedule 保持不变；
- temporal adaptive 关闭；
- 训练中导出 EMA routing artifact。

artifact 不包含 DP action policy、DP 下游 visual encoder、optimizer moments 或训练进度，只包含 Pi0.5 后续需要的 routing branches、condition projection、DP 默认的 foveation/router normalization statistics、task mapping、geometry 和结构 metadata。Pi0.5 action policy 的 statistics 独立保存在 Pi0.5 checkpoint/assets 中。

### 3.4 第四阶段：Pi0.5 TCL StageB 迁移闭环（代码已实现，待服务器验证）

最终迁移流程必须验证：

1. Pi0.5 StageA 下游权重可以加载；
2. DP artifact 的两套 camera branch 可以严格加载；
3. router condition projection 输入为 16 维；
4. task mapping 与 `meta/tcl_tasks_10.yaml` 一致；
5. 6 维 TCP pose normalization statistics 一致；
6. source/target geometry、camera order 和 node order 一致；
7. 两个相机的 ResNet/FPN 参数和 buffers 保持独立；
8. router-only 阶段的梯度边界正确；
9. downstream 阶段 router、condition projection、adapter 和 APT 均保持冻结；
10. full resume 自包含 dataset metadata、action statistics、task mapping 和 artifact contract。

Pi0.5 StageB 的 EMA checkpoint 可直接交给 `gpu_service_pi05.py` 部署。服务从 self-contained checkpoint 恢复 Pi0.5 preprocessing、action statistics、foveation/router 权重和模型配置，不需要在服务启动时再次加载 DP artifact。

## 4. 相关实现涉及的文件范围

本节汇总相关文件范围，并标出已经实现和仍待实现的部分。

### 4.1 Shell 和配置

- `scripts/pi05/train_tcl_stageA.sh`（已实现）；
- `scripts/pi05/train_tcl_stageB.sh`（已实现）；
- `scripts/pi05/pretrain_foveation_tcl_stageA.sh`（已实现）；
- `scripts/pi05/pretrain_foveation_tcl_stageB.sh`（已实现）；
- `configs/tcl_foveation_dp_for_pi05.yaml`（已实现）；
- `diffusion_policy/dataset/tcl_dataset.py` 中的 `TCLFoveationDataset`（已实现）；
- `diffusion_policy/common/prepare_data.py`（已实现）；
- `scripts/pi05/prepare_tcl_data.sh`、`scripts/pi05/prepare_libero_data.sh`（已实现）；
- Pi0.5 TCL data/model/runner 配置（StageA 已实现）；
- DP TCL source 配置；
- `meta/tcl_tasks_10.yaml`（已创建，实际 task 列表待用户维护）。
- `gpu_service_pi05.py`（已实现；暂按 `tcp_state[..., -1]` 携带 task ID）。

### 4.2 Dataset、workspace 和 policy

- `TCLFoveationDataset` 的 DP normalized/Pi0.5 raw 双模式；
- `LiberoFTDataset` 的 DP normalized/Pi0.5 raw 双模式；
- `diffusion_policy/common/prepare_data.py` 和两个按数据域区分的 prepare shell 入口；
- Pi0.5 preprocessing 的 action horizon 和 TCL action statistics；
- Pi0.5 workspace 的 StageA/StageB initialization、EMA、DDP 和 resume；
- Pi0.5 policy 的 6+10 router condition；
- TCL episode-local temporal pairing；
- DP dataset 的 task ID 和 256×256 letterbox geometry。

### 4.3 Artifact 和 routing

- 将 `foveation_pretrained.py` 从 LIBERO 专用 task mapping 泛化为 TCL mapping；
- 保存和校验 TCL pose statistics；
- 校验 source/target geometry、camera order、node order 和 branch shapes；
- 保留两个相机独立的 ResNet/FPN/router 参数与 buffers。

## 5. 当前 StageA 的服务器运行前置条件与验证状态

本地开发环境缺少 `hydra` 和 `torch`，因此目前只能完成静态 Python 编译、shell 语法检查和不依赖训练依赖的配置 smoke check，无法在本地执行 Hydra 配置组合、真实 TCL HDF5 读取、统计资产生成、GPU/多卡训练、checkpoint 恢复或 artifact 迁移验证。正式服务器必须先完成下面的数据资产准备。

### 5.1 填写 task mapping

运行前，用户必须手动补齐 `meta/tcl_tasks_10.yaml` 中的实际 task directory。`task_dir` 必须与传入 `data_roots` 的 basename 精确一致，例如：

```yaml
tasks:
  - task_id: 0
    task_dir: 1014_lh_tower_three_block
```

配置解析出的每个数据目录都必须在该列表中找到唯一匹配；未匹配、重复或超出 10 个 slot 时，数据准备和训练按设计直接报错。task ID 不从目录排序或语言文本动态推断。

### 5.2 准备 Pi0.5 TCL 统计资产

`prepare_data.py` 是训练前的一次性数据资产准备步骤，不由 StageA/StageB 自动调用。数据源、task mapping、split seed 或统计范围改变时，使用新的 `asset_dir` 重新生成；同一批 StageA、StageB 和 resume 训练复用这份 metadata/statistics。

正式服务器上使用明确的 TCL 入口：

```bash
bash scripts/pi05/prepare_tcl_data.sh \
  data_roots='["/path/task_a","/path/task_b"]' \
  h5_paths='["/path/task_a.h5","/path/task_b.h5"]' \
  task_mapping_path=meta/tcl_tasks_10.yaml \
  asset_dir=outputs/pi05_data/tcl_foveation_seed42
```

LIBERO 使用单独的入口 `scripts/pi05/prepare_libero_data.sh`。旧的
`scripts/pi05/prepare_data.sh` 已删除，避免无法判断数据域；两个 shell 入口最终都调用
`diffusion_policy/common/prepare_data.py`。

其中 `data_roots` 和 `h5_paths` 必须严格一一对应；每个 root 的 basename 通过 `task_dir` 与 mapping 精确匹配。统计资产使用固定 seed，并保存 Pi0.5 自己的 state/action normalization；foveation/router 后续继承 DP 的统计量，不与 Pi0.5 action policy 的统计量混用。

### 5.3 启动 Pi0.5 TCL StageA

统计资产准备成功后，在正式服务器上执行：

```bash
DATA_ROOTS='["/path/task_a","/path/task_b"]' \
H5_PATHS='["/path/task_a.h5","/path/task_b.h5"]' \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/pi05/train_tcl_stageA.sh
```

StageA 从官方 `converted_base` Pi0.5 权重初始化，动作模型仍只使用图像和语言；6/7 维 state 只保留给 TCL/foveation 数据合同。训练前应确认 mapping、source paths、state dimension、32×7 action horizon、224×224 image contract 和统计资产 metadata 一致。

### 5.4 服务器验收项

服务器具备完整依赖后，至少需要验证：

- Hydra 配置组合和环境变量 override 能成功解析；
- 多任务真实 HDF5 读取、双相机 batch、language prompt 和 episode-local action window 正常；
- mixed-resolution patch bank、EMA checkpoint、full checkpoint resume 和离线 validation 正常；
- `converted_base` 的权重加载与 `action_horizon=32`、`n_action_steps=24` 合同兼容。

DP artifact 迁移、Pi0.5 StageB、真实 GPU 训练和远程机械臂推理不属于当前 StageA 的本地验证结果，必须在对应实现完成后单独验收。

DP TCL StageA/StageB 运行前同样必须先补齐 `meta/tcl_tasks_10.yaml`，并保证 `DATA_ROOTS_OVERRIDE` 中每个目录 basename 与 `task_dir` 精确匹配。StageA 的默认 Bash 数组会转换为同样的 Hydra list；StageB 还必须提供 DP StageA 的 `checkpoints/latest.ckpt`；Pi0.5 TCL StageB 再使用 DP StageB 导出的 `foveation_pretrained.pt`。

当前本地环境缺少 Hydra、Torch 和真实 TCL 数据依赖，因此本批没有执行 DP TCL 配置组合、HDF5 读取、checkpoint 初始化、artifact 导出、GPU 训练或迁移验证。上述检查保留到正式服务器执行。

### 5.5 真机服务（服务代码已实现，待远程联调）

训练和离线检查完成后，在正式服务器启动：

- `gpu_service_pi05.py` 参考 `gpu_service.py` 的 GPU 服务端口、请求处理、模型加载和远程通信方式；
- 服务加载 Pi0.5 StageB 的 self-contained `latest_ema.pt` 或 `best_ema.pt`，不在服务端重新加载 DP artifact；
- 请求必须包含双相机视频和 `tcp_state`；当前暂按 `tcp_state[..., :6]` 为 TCP pose、`tcp_state[..., -1]` 为稳定的整数 task ID；
- 服务验证 task ID 必须位于 `[0, 9]` 且存在于 `PI05_TASK_MAPPING_PATH`，不能依赖默认 task 0 或从语言文本反推 task ID；
- Pi0.5 输出完整 32 步、7 维 action，默认通过 `max_cache_action=24` 交给 evaluator 执行。

启动示例：

```bash
export PI05_CHECKPOINT=outputs/pi05_tcl_stageB/.../checkpoints/latest_ema.pt
export PI05_TASK_MAPPING_PATH=meta/tcl_tasks_10.yaml
CUDA_VISIBLE_DEVICES=0 \
  uvicorn gpu_service_pi05:gpu_app --host 0.0.0.0 --port 6081
```

当前 task ID 的 TCP state 位置仍是临时约定；正式 evaluator 联调时如果 state layout 发生变化，需要同步修改服务端解析和本计划说明。

## 6. 各阶段的主要参考文件

本节用于指导后续实现时的阅读顺序。参考文件提供已有接口和训练语义，不能未经核对直接复制，因为 LIBERO 和 TCL 在数据格式、任务 mapping、图像尺寸、action horizon 和 router condition 上存在差异。

### 6.1 Pi0.5 TCL StageA

优先参考：

- `scripts/pi05/train_libero_stageA.sh`
  - 参考 Accelerate 单卡/多卡启动、GPU 列表解析、输出目录、Hydra override 和 StageA topology 配置方式。
  - 需要改成 TCL 配置、TCL dataset、`action_horizon=32` 和默认 `n_action_steps=24`。
  - LIBERO 的 `model.image_size=224`、adapter/APT/level embedding 开关可以作为直接参考。
- `configs/pi05_libero_sft.yaml`
  - 参考 Pi0.5 workspace、optimizer、EMA、checkpoint、mixed foveation 和 router 配置字段。
  - 不能直接复用其中的 LIBERO dataset root、8 维 state、10 个 LIBERO task、runner 和 action statistics。
- `diffusion_policy/workspace/train_pi05_workspace.py`
  - 参考 StageA/StageB 初始化、StageA EMA、完整 checkpoint、EMA、DDP、rollout 和 temporal pairing 的生命周期。
  - 统一使用 `diffusion_policy/dataset/libero_dataset.py` 中的 `LiberoFTDataset(output_mode=raw)`，再接入 TCL dataset metadata。
- `diffusion_policy/policy/pi05_policy.py`
  - 参考 Pi0.5 policy 的 preprocessing、router 调用、task ID 传递、StageB 冻结和 artifact 加载边界。
  - 动作模型保持图像 + 文本输入；新增的 TCL pose 只进入 router condition。
- `diffusion_policy/model/pi05/config.py`
  - 参考 Pi0.5 配置校验和 checkpoint signature。
  - 需要规划如何允许 `action_horizon=32`，同时保持 action projection 参数形状兼容。
- `diffusion_policy/model/pi05/model.py`
  - 参考 action horizon、flow loss、Euler sampling、prefix cache 和 action output 的时间维度处理。
  - 重点确认 32 步 action 不会被固定的 10 步注释、shape 检查或采样逻辑限制。
- `diffusion_policy/model/pi05/preprocessing.py`
  - 参考原始 RGB 的 orientation、letterbox resize、action normalization/decode 和 prompt 处理。
  - Pi0.5 TCL preprocessing 需要允许 6/7 维 state，但动作模型忽略 state；不引入 discrete state token。

数据格式和 prompt 还应参考：

- `diffusion_policy/dataset/tcl_dataset.py`：TCL HDF5 字段、`language_text`、`robot_obs`、`rel_actions`、episode 边界和现有归一化逻辑；
- `diffusion_policy/dataset/libero_dataset.py`：`LiberoFTDataset` 同时提供 DP normalized 和 Pi0.5 raw 窗口、task ID、prompt、action window 和 temporal pair；
- `diffusion_policy/common/prepare_data.py`：训练前一次性生成 TCL/LIBERO split、statistics 和 metadata 资产；
- `configs/pi05_libero_data.yaml`：Pi0.5 数据准备配置字段和资产目录约定。

### 6.2 Pi0.5 TCL StageB

优先参考：

- `scripts/pi05/train_libero_stageB.sh`
  - 参考 StageA checkpoint/run directory、`FOVEATION_PRETRAINED_PATH`、router-only、downstream switch、budget schedule 和 rollout override。
  - TCL 版本需要改成 32 步预测、24 步默认执行、TCL task mapping 和 6+10 condition。
- `diffusion_policy/workspace/train_pi05_workspace.py`
  - 重点阅读 `_load_stage_a_weights`、artifact 加载、`stage_b` 初始化、router-only/downstream 冻结、EMA excluded prefixes 和 full resume。
  - 需要确保 full resume 不重新依赖外部 artifact，并保存 TCL dataset metadata。
- `diffusion_policy/policy/pi05_policy.py`
  - 重点参考 `load_foveation_artifact`、`_router_condition`、`set_stage`、`get_extra_state` 和 `set_extra_state`。
  - LIBERO 的 task mapping 校验需要替换为 `meta/tcl_tasks_10.yaml`，但 condition projection 仍保持 16 维输入。
- `diffusion_policy/model/pi05/config.py`
  - 参考 StageB 的 `foveation_router_type=node_aligned_fpn`、hidden dim、resize64、temperature 和 freeze 配置。
- `diffusion_policy/dataset/libero_dataset.py`
  - 重点参考 episode-local temporal pairing、pairing seed、pairing epoch 和 checkpoint metadata。
  - TCL 版本需要根据 TCL episode 边界实现相同的不跨 episode 约束。
- `plan/inherit_dp_foveation.md`
  - 参考 Pi0.5 加载顺序、router-only/downstream 冻结边界、artifact 必需字段和 geometry 合同。

### 6.3 DP TCL StageA（已实现入口，待服务器训练）

优先参考：

- `scripts/roi/sh_train_foveation_tcl_stageA.sh`
  - 参考 TCL dataset 参数、`p_cond_drop`、`horizon=32`、随机合法 topology、TCL rollout/video 参数和 dataset split seed。
  - 需要改为 Pi0.5 兼容的 256×256 source geometry、ResNet10/FPN、512 feature dim、resize64 和 6+10 condition。
- `scripts/pi05/pretrain_foveation_libero_stageA.sh`
  - 参考 Pi0.5 专用 DP source 的 Accelerate 启动、双相机、完整 task mapping、输出目录和 source 配置覆盖方式。
  - LIBERO-10 文件列表和 LIBERO task metadata 不能直接复制。
- `configs/tcl_force_foveation_dp.yaml`
  - 参考 TCL DP 的 workspace、dataset、horizon、paired/temporal loss、budget 和 checkpoint 设置。
  - 新配置应移除 force/success，并接入显式 TCL task mapping。
- `configs/liberoft_foveation_dp_for_pi05.yaml`
  - 参考 DP→Pi0.5 迁移所需的 ResNet10、`global_feature_dim=512`、`router_hidden_dim=128`、resize64 和固定 temperature。
  - 其 LIBERO shape metadata、数据源和 task mapping 必须替换为 TCL 版本。
- `diffusion_policy/policy/foveation_dp_transformer_policy.py`
  - 参考 router condition key、低维 observation shape、双相机 routing branch、paired router loss 和 task ID 的接入位置。
- `diffusion_policy/model/roi/foveation_net.py`
  - 参考 node-aligned FPN 的 patch16 geometry、global encoder resize、80 logits 和 source shape 校验。

### 6.4 DP TCL StageB 和 artifact 导出（已实现入口，待服务器训练）

优先参考：

- `scripts/roi/sh_train_foveation_tcl_stageB.sh`
  - 参考 StageA checkpoint、router-only、paired objective、temperature、budget schedule 和数据 seed 恢复。
- `scripts/pi05/pretrain_foveation_libero_stageB.sh`
  - 参考 `FOVEATION_EXPORT_PATH`、EMA artifact 导出、Pi0.5 source checkpoint 和专用输出目录。
- `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py`
  - 重点阅读 `_export_foveation_artifact`、StageB router-only 初始化、EMA/live module 选择和保存时机。
- `diffusion_policy/common/foveation_pretrained.py`
  - 参考 artifact 字段、camera branch 提取、state statistics、condition projection、geometry 和 tensor shape 校验。
  - 当前 LIBERO task mapping、256/patch16 source 和 Pi0.5 target 合同需要泛化为 TCL mapping，但本计划的 6+10 condition 形状保持不变。
- `diffusion_policy/model/roi/multi_view_foveation_net.py`
  - 参考双相机独立 encoder/FPN/router 参数和 buffers，避免导出时错误共享两个 camera branch。
- `plan/inherit_dp_foveation.md`
  - 参考 artifact 不应包含的 DP action/downstream/optimizer 状态，以及 source/target geometry 的迁移边界。

### 6.5 Task mapping、图像 geometry 和静态验证

优先参考：

- `diffusion_policy/common/libero_task_util.py`：参考 task name/id 工具的接口形式，但不能复用 LIBERO task 列表；
- `diffusion_policy/dataset/tcl_dataset.py`：确认 TCL canonical task name、episode 归属和任务 ID 生成位置；
- `diffusion_policy/model/pi05/preprocessing.py`：参考 Pi0.5 224×224 letterbox；
- `diffusion_policy/model/roi/foveation_net.py`：参考 DP 256×256、patch16、node grid 和 resize64；
- `tests/test_foveation_net.py`：只参考已有 geometry/state shape 检查思路，不在本次计划阶段新增测试脚本；
- `AGENTS.md`：遵守静态编译、简短实现、tensor shape 注释和不新增独立测试脚本的要求。

### 6.6 LIBERO 继承链作为 source of truth

当 TCL 侧出现初始化、冻结、checkpoint 继承或 artifact 加载顺序不确定时，优先按下面四个 LIBERO 入口和它们使用的配置核对，不单独发明 TCL 版本的规则：

- `scripts/pi05/pretrain_foveation_libero_stageA.sh`：DP source StageA 的初始化、source 配置、固定/随机 topology、输出和冻结语义；
- `scripts/pi05/pretrain_foveation_libero_stageB.sh`：DP source StageB 的 StageA checkpoint 继承、router-only、paired objective 和 artifact 导出；
- `scripts/pi05/train_libero_stageA.sh`：Pi0.5 StageA 的 base asset、mixed-resolution topology、adapter/APT/level embedding 和下游训练；
- `scripts/pi05/train_libero_stageB.sh`：Pi0.5 StageB 的 StageA EMA 加载、DP artifact 加载、router-only/downstream 切换和冻结边界；
- `configs/liberoft_foveation_dp_for_pi05.yaml`、`configs/pi05_libero_sft.yaml`：上述入口实际使用的结构配置和默认值。

TCL 只替换数据来源、task directory mapping、图像几何、action window 和 6+10 condition；能从 LIBERO 方案继承的初始化、checkpoint 加载顺序、EMA、冻结和 artifact 语义都应保持一致。

## 7. 后续实现阶段的验证标准

遵循 AGENTS.md，后续实现阶段本地优先做静态验证，不新增与实现重复的独立测试脚本：

- `bash -n` 检查四个入口；
- `python -m py_compile` 检查修改的 Python 文件；
- Hydra compose 检查四个训练阶段的配置组合；
- `git diff --check`；
- CPU synthetic batch 检查两种 letterbox geometry；
- 检查 32 步 action window 和 24 步 execution prefix；
- 检查 16 维 pose/task condition；
- 检查 80 个 split logits；
- 检查 artifact 保存、加载和严格 shape 校验；
- 检查两相机参数独立性；
- 检查 router-only/downstream 冻结边界；
- 检查 temporal pair 不跨 episode；
- 检查 Pi0.5 原有权重在 action horizon 扩展后仍可加载。
- 检查 6/7 维 TCL state 可以通过 preprocessing，但不会进入动作模型 token；
- 检查 config 中的 task directory 都能在 `meta/tcl_tasks_10.yaml` 中匹配，未匹配项直接报错；
- 检查 DP/Pi0.5 使用同一批数据统计范围，而 action normalization 可以独立；
- 检查 DP StageA 的 foveation/global encoder/router 沿用 LIBERO source 入口的初始化和冻结语义；
- 检查 uint8 黑边、bilinear resize 和双相机 preprocessing 一致性；
- 检查真机服务请求显式携带 task ID。
- 检查 `gpu_service_pi05.py` 能加载 self-contained EMA checkpoint，并返回 `[B,32,7]` action 和默认 `max_cache_action=24`。

本地不运行 GPU 训练、真实 TCL HDF5、多卡训练、rollout、checkpoint/EMA 实际迁移或服务器 artifact 闭环；这些属于后续服务器验收。

## 8. 当前实现的验证范围

本批新增服务代码和文档更新的静态范围包括：

- 文档覆盖已经确定的目标、数据合同、训练顺序、文件范围、服务启动方式和验收标准；
- `gpu_service_pi05.py` 明确使用 self-contained EMA checkpoint、双相机输入、TCP pose/task ID 和 32×7 action contract；
- 当前仍不运行 `py_compile`、Hydra compose、真实 HDF5 读取、GPU 推理、远程 evaluator 或真机通信；这些检查保留到正式服务器。


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
