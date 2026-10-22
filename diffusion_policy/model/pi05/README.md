# Dense π0.5：模型、policy 与 LIBERO 评测

已实现独立 PyTorch dense 模型、原始 LIBERO 预处理、原生 flow loss、Euler 推理、模型资产保存，以及 LIBERO-FT runner／evaluator 适配和独立评测入口。用户已在服务器完成权重转换和模型严格加载验证，并已准备 `~/pretrained/pi05/converted_libero`。单任务闭环完成 3 个 episode、成功 3/3，用户已确认视频正常。本地只做静态检查；原始 HDF5 dataset 与训练集统计准备已通过服务器验收。四卡 SFT 主循环已由服务器 step 15/16 日志确认跑通；完整恢复 2B 暂缓验收。训练 EMA 独立评测、多卡训练 rollout、成功率选模和 W&B 已实现，待服务器闭环验收。

## 来源和实现边界

- 模型及 JAX 权重映射：RLinf `c70606f08cdca259b8dec03d4430926b5b8fac9d`，目录 `rlinf/models/embodiment/openpi` 与 `rlinf/utils/ckpt_convertor/openpi`。
- tokenizer、quantile normalization 和 resize 参考：RLinf 安装脚本指定的 `rlinf-openpi==0.1.1`。wheel SHA256 为 `e7b8e6e61d575487bce63e89dee5e7af7834904ee07152bcbbd5b58c06e4c47d`。发布包没有可核实的 Git commit，`sources.json` 如实记录 null；不能据此声称已锁定对应的官方 OpenPI commit。对照使用这个确切发布包，官方 commit 的追溯仍待完成。
- `sources.json` 保存原始源码哈希；`licenses/` 保留 RLinf、OpenPI 和 Gemma 许可证。模型权重和 tokenizer 由服务器从原始发布源取得，不纳入仓库。
- 运行时无需 RLinf、OpenPI、Transformers、JAX、Orbax。转换才需要 JAX/Orbax，使用延迟导入。
- 保留 `llm.*`、`img.*` 和四个动作／时间投影的上游 state_dict 键名。没有 LoRA、冻结、RL head、RTC、RLT、点云或分布式框架。
- Gemma attention 默认使用 PyTorch 原生 `torch.nn.functional.scaled_dot_product_attention`，由 PyTorch 在 CUDA 上选择 fused/Flash backend；不自动 `torch.compile`，也不修改全局 TF32 设置。需要与原始 eager 算术对照时可设置 `model.attention_backend=math`。SDPA 与 eager 路径共享同一套权重，但不承诺逐位数值一致。

固定模型为 SigLIP So400m/14 → PaliGemma 2b + Gemma 300m 动作专家。每路相机 256 个 token，第三路黑图占位且 mask=False；语言长度 200；预测 10×32 动作，普通 Euler 5 步，返回 10×7 原始动作和前 5×7 执行动作。动作维度 padding 也参与普通 SFT loss 的均值。

`discrete_state_input=False`：语言中没有离散状态，π0.5 suffix 也没有连续状态 token。8 维原始状态仍必须保留，不能私自新增状态条件。`extra_delta_transform=False`：LIBERO 动作已经是 delta。

默认 bfloat16 只应用于指定主干参数，patch embedding、位置编码、Gemma norms 和动作／时间投影保留 fp32。移动设备用 `.to(device=...)`，不要对整个 policy 调用 `.bfloat16()` 或 `.half()`。不套用额外全局 autocast。需要 FP32 对照时构造 `config={'dtype': 'float32'}`。

## 服务器资产准备

建议在独立 Python 3.10+ 环境运行，避免改动现有 DP 训练环境。模型资产统一存放在 `~/pretrained/pi05`（当前用户主目录下）。从仓库根目录执行：

```bash
python -m pip install -r requirements_pi05_conversion.txt
mkdir -p ~/pretrained/pi05/raw ~/pretrained/pi05/assets
# 需要服务器已有 Google Cloud CLI；这两条命令下载官方原始资产。
gcloud storage cp --recursive gs://openpi-assets/checkpoints/pi05_base ~/pretrained/pi05/raw/
gcloud storage cp gs://big_vision/paligemma_tokenizer.model ~/pretrained/pi05/assets/tokenizer.model
bash scripts/pi05/convert_checkpoint.sh \
  --input-model ~/pretrained/pi05/raw/pi05_base \
  --output-model ~/pretrained/pi05/converted_base \
  --tokenizer ~/pretrained/pi05/assets/tokenizer.model \
  --source-uri gs://openpi-assets/checkpoints/pi05_base
```

输出目录必须不存在，防止覆盖既有资产。转换保存 fp32 `model.safetensors`、LIBERO 运行配置、tokenizer 和 `manifest.json`。manifest 包含原始目录内逐文件 SHA256、转换产物 SHA256、源版本、转换命令及实际依赖版本；校验值在服务器取得真实文件后生成，不使用虚构值。转换会用 meta 模型核对所有键和形状；转换完成后，按文末命令实际加载模型即可完成本批验收。

基座转换允许不提供统计。实例化 policy 时统计必须存在，不能拿官方其他机器人统计替代本项目训练集统计。项目统计通过下文的数据准备命令生成。

评测接口检查使用官方 LIBERO SFT 权重和它自己的统计；已有完整 `converted_libero` 时跳过以下下载与转换：

```bash
gcloud storage cp --recursive gs://openpi-assets/checkpoints/pi05_libero ~/pretrained/pi05/raw/
bash scripts/pi05/convert_checkpoint.sh \
  --input-model ~/pretrained/pi05/raw/pi05_libero \
  --output-model ~/pretrained/pi05/converted_libero \
  --tokenizer ~/pretrained/pi05/assets/tokenizer.model \
  --norm-stats ~/pretrained/pi05/raw/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --source-uri gs://openpi-assets/checkpoints/pi05_libero
```

若发布资产目录结构不同，先确认实际 `norm_stats.json` 的所属任务，再调整该参数。统计格式兼容 OpenPI 的 `{"norm_stats": {"state": {...}, "actions": {...}}}`，state 至少 8 维、actions 至少 7 维，均包含 `q01`、`q99`。标准运行依赖单独列于 `requirements_pi05.txt`。

## Policy 输入输出与恢复

```python
from pathlib import Path
import torch
from diffusion_policy.policy.pi05_policy import Pi05Policy

# Python 路径显式展开 ~。
asset_dir = Path('~/pretrained/pi05').expanduser()
policy = Pi05Policy(
    model_path=asset_dir / 'converted_libero',
    tokenizer_path=asset_dir / 'converted_libero/tokenizer.model',
    norm_stats_path=asset_dir / 'converted_libero/norm_stats.json',
).to(device='cuda').eval()

# obs 是原始模拟器方向的 RGB；每次一帧，不带 DP 的 To 维。
# obs['image'], obs['gripper']: uint8 [B,H,W,3]
# obs['joint_state']: float32 [B,8]，xyz + rotation vector + two gripper qpos
# obs['prompt']: list[str]，长度 B，使用真实任务语言
# batch = {'obs': obs, 'action': raw_actions_B_10_7}
noise_B_T_D = torch.randn(
    len(obs['prompt']), 10, 32, device='cuda',
    generator=torch.Generator(device='cuda').manual_seed(42),
)
result = policy.predict_action(obs, noise=noise_B_T_D)
# result['action_pred']: [B,10,7]; result['action']: [B,5,7]，均已反归一化
policy.save_policy(asset_dir / 'libero_policy.pt')
restored = Pi05Policy.from_checkpoint(asset_dir / 'libero_policy.pt', device='cuda')
```

`compute_loss(batch, noise=..., time=..., reduction='none')` 返回 `loss_B`；默认返回标量。`policy.eval()` 下 loss 不做增强；`policy.train()` 下执行增强。显式 noise 为 `[B,10,32]`，time 为 `[B]`。省略时噪声使用设备 RNG，time 使用 CPU Beta(1.5,1.0)，再做 `time * 0.999 + 0.001`。增强使用 CPU RNG，可传 `augmentation_generator=torch.Generator().manual_seed(...)`；它不控制 flow 噪声。预测的 `generator` 必须与模型设备匹配。

训练输入必须提供完整十步窗口；尚无 HDF5 尾部 padding／valid mask 接口。后续 dataset 必须先确定该语义，不可直接传现有 DP 的归一化窗口。训练 loss 的均值包含全部 32 维，不因执行前五步而改变。

policy 保存包含 tokenizer 原始 bytes、归一化资产、配置及来源信息，`state_dict()` 的 `_extra_state` 同样携带这些信息。`from_checkpoint` 不重新访问原资产路径。这里的 `.pt` 仅为 policy 保存，不包含 optimizer、EMA、scheduler 或 RNG；后续 workspace 才实现完整训练恢复。

## 预处理差异与待验证项

1. `render_image_orientation` 控制 static（`image`）输入方向，默认 `rot180`，也可设为 `raw`；wrist（`gripper`）始终保留原始方向。两相机保持 uint8 执行带 padding 的 antialiased linear resize，再转 float32 `[-1,1]`。PyTorch `interpolate(antialias=True)` 替代参考包的 JAX resize；核函数和 uint8 rounding 的数值一致性尚未验证，不能声称逐像素相同。
2. q01/q99 normalize 和 inverse 使用 `+1e-6`，不 clipping，先归一化真实维度再补零。输出取前七维，不额外变更夹爪符号、二值化或 delta。已核对固定 RLinf 的 `rlinf/envs/action_utils.py`：OpenPI 的 LIBERO 动作直接传递，runner／evaluator 同样不额外处理夹爪。
3. 图像增强采用固定 RLinf 核心的逐样本方案：主相机 95% crop/resize、±5° rotate；相机颜色增益分别为 `1+[0,.3)`、`1+[0,.4)`、`1+[0,.5)`。显式修复其 generator=None 时跳过随机采样，以及 wrist 路径依赖局部 TF import 的问题。保留这一配方，不声称等同于官方 JAX 的增强分布。
4. Euler 使用一次 prefix KV，suffix 步骤拼接后产生的 cache 不回写 prefix。每次 `predict_action` 重新计算；`reset` 不重置调用方随机流。

## 服务器验收：模型正常加载（已通过）

用户已反馈服务器权重转换和加载验证通过。以下保留复现命令，从仓库根目录执行：

```bash
python - <<'PY'
from diffusion_policy.model.pi05.config import Pi05Config
from diffusion_policy.model.pi05.model import Pi05Model
from diffusion_policy.model.pi05.checkpoint import load_model_weights

model = Pi05Model(Pi05Config())
load_model_weights(model, '~/pretrained/pi05/converted_base')
model.eval()
print('dense pi0.5 model loaded successfully')
PY
```

命令正常结束并打印成功信息即通过。`load_model_weights` 对模型键和参数形状执行严格兼容检查；仅允许旧 dense 资产缺少新加的零初始化 `img.level_embedding`，其余缺失键、多余键或形状不匹配都会直接报错。此检查在 CPU 上完成，需要足够内存，无需 GPU、观测输入或归一化统计；已有 LIBERO 转换权重时可将路径换为 `~/pretrained/pi05/converted_libero`。

数值对齐、前向／反向、保存恢复和闭环评测不作为本批验收条件。加载成功仅说明模型结构与权重兼容，预处理差异仍保留上述记录。

## 服务器单任务试跑：3 个 episode

用户于 2026-10-08 反馈：默认 STUDY_SCENE1 任务完成 3 个 episode，成功率 1.000（3/3），并确认视频正常，单任务接口验收通过。结果目录为 `outputs/pi05_libero_eval/20261008_215226/eval/epoch_0000/`。此处依据用户运行日志和视频反馈记录，未独立读取服务器结果；以下保留复现命令。默认使用已有的 `~/pretrained/pi05/converted_libero`，要求目录内包含转换产生的 `model.safetensors`、`tokenizer.model`、`norm_stats.json`、`config.json` 和 `manifest.json`；入口校验文件与转换 manifest 一致。

### 同步代码与安装依赖

需要同时同步本仓库和 **LIBERO-FT** 的修改。服务器上的 LIBERO-FT 位于 `~/code/LIBERO-FT/`，需同步其中的 `libero/force/force_benchmark.py` 与 `eval_socket.py`。仅同步本仓库会导致 evaluator 状态布局不兼容。RoboKit 通信协议无需修改，但两端应使用当前支持 Pydantic 2 的协议版本。

在 π0.5 的独立 Python 3.10+ 环境中，从本仓库根目录执行一次：

```bash
conda activate pi05
python -m pip install -r requirements_pi05_eval.txt
# 按实际 RoboKit 目录调整；只注册本地包，所需 runner 依赖由上面的文件安装。
python -m pip install --no-deps -e ../RoboKit
```

评测依赖固定 `av==13.1.0` 并要求使用预编译 wheel，避免自动编译 PyAV。该版本提供 Python 3.10 的 Linux x86_64／aarch64 manylinux2014 包（glibc ≥2.17），无需手动准备 FFmpeg 开发库。如果此前出现 `av-14.4.0.tar.gz`、`libavformat.pc` 找不到等错误，同步新的 requirements 后重试。若当前镜像缺少 wheel，可先从官方 PyPI 安装：

```bash
conda activate pi05
python -m pip install --only-binary=av --index-url https://pypi.org/simple "av==13.1.0"
python -m pip install -r requirements_pi05_eval.txt
```

若仍提示没有匹配的发行包，需检查当前 Python 版本、CPU 架构和系统 glibc；不要去掉 `--only-binary` 再尝试源码编译。PyAV 只用于本地视频编码，不影响模型权重。

模拟器继续使用已有 `libero` conda 环境及其中的 LIBERO-FT、RoboKit、MuJoCo 等依赖，不需要在模拟器环境安装 π0.5 的 PyTorch/JAX。无需 W&B 登录，入口只保存本地结果。

### 启动

从本仓库根目录运行，以下已使用服务器上的 LIBERO-FT 路径：

```bash
conda activate pi05
cd "$HOME/code/dp23rss_fork"
bash scripts/pi05/eval_libero.sh \
  asset_dir="$HOME/pretrained/pi05/converted_libero" \
  libero_root="$HOME/code/LIBERO-FT" \
  runner.conda_env=libero \
  device=cuda:0
```

`.sh` 仅负责切换到仓库根目录并转发命令行参数，Python 工具入口位于 `diffusion_policy/common/pi05_convert_checkpoint.py` 和 `diffusion_policy/common/pi05_eval_libero.py`。也可从仓库根目录通过 `python -m diffusion_policy.common.pi05_convert_checkpoint` 或 `python -m diffusion_policy.common.pi05_eval_libero` 调用，并传入相同参数。评测 shell 默认使用 `~/code/LIBERO-FT`，模块直调时需显式传入 `libero_root`。

入口自动启动 HTTP policy 服务与外部 evaluator，不需要手动开两个终端。默认评测任务为 `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`，HDF5 路径为：

```text
~/code/LIBERO-FT/libero/datasets/libero_force_256/libero_10/
  STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo_wrench.hdf5
```

默认 **单环境、3 个 episode，保存三个视频**。需要调整数据目录或任务时，在命令末尾增加 Hydra 覆盖，例如：

```bash
  runner.dataset_root="$HOME/code/LIBERO-FT/libero/datasets" \
  runner.dataset_subname=libero_force_256/libero_10 \
  'runner.hdf5_fns=[其他任务_demo_wrench.hdf5]'
```

上述为启动命令的追加参数，不是单独执行的命令。更多参数见 `configs/pi05_libero_eval.yaml`：`runner.n_test` 控制 episode 数，`runner.n_test_vis` 控制保存视频数。默认自动匹配 policy 所在物理 GPU 的 EGL 设备，沿用现有 DP 的选择逻辑；如需手动指定，可同时设置 `runner.egl_device_id` 与 `runner.render_gpu_device_id`，不要将 CUDA 逻辑序号直接当作 EGL 序号。

### 图像、动作与 reset 约定

- **相机渲染为 256×256；模型输入为 224×224。** Runner 传原始 uint8 RGB，policy 默认仅将 static 旋转 180°，两相机均 resize 到 224×224 并归一化。每路 256 个视觉 token 来自 16×16 个 patch，与原始图像的 256 像素不是同一概念。
- 每次输入一帧、8 维原始状态和 benchmark 的真实任务语言。状态为 xyz＋rotation vector＋双指 qpos；不使用 DP statistics 或任务 one-hot。
- 预测长度 10、Euler 5 步、执行前 5 步；每个 episode reset 清空动作缓存。策略动作已经反归一化，直接送入 LIBERO。
- Reset 使用 benchmark 初始状态索引 0、1、2，随后等待 15 步，等待动作为 `[0,0,0,0,0,0,-1]`。等待不计入每个 episode 的 520 步策略动作上限。
- Policy seed 默认为 42，每个任务开始时初始化一次随机流，episode reset 不重置它；环境 seed 依次为 130000、130001、130002。改变并行环境数会改变噪声分配，不承诺单／并行逐 episode 完全相同。
- 每个环境在成功、环境 done 或达到步数上限时结束。关闭环境扰动和在线更新。DP 默认仍使用原来的 6 维状态、归一化、5 步全零等待约定。

### 输出与简单验收

结果默认位于 `outputs/pi05_libero_eval/<时间戳>/`：

```text
config.yaml                         # 入口配置
 eval/epoch_0000/
   eval_config.json                 # 有效 policy／环境配置和权重来源
   evaluator.log                    # 模拟器进程日志
   results.json                     # 逐 episode seed、初始状态、步数、结束原因、成功与视频路径
   summary.json                     # 成功数、episode 数、成功率及权重来源
   media/episode_*.mp4               # 双相机并排视频，显示为 policy 的观看方向
```

**完成 3 个 episode、没有接口／非有限动作错误、结果完整且三个视频可播放，即完成本批接口验收。** 不设置成功率门槛；全失败时保留结果和视频分析策略行为。运行中断可先检查终端中的 policy 报错与 `evaluator.log`。

若报 `Evaluator tcp_state shape mismatch`，默认单环境应收到 `[1,1,8]`（xyz＋rotvec＋双指 qpos）。若实际为 `[1,1,6]`，通常是服务器上的 LIBERO-FT evaluator 仍使用旧 DP 状态拼接。确认 `~/code/LIBERO-FT/libero/force/force_benchmark.py` 已包含 `eef_rotvec_gripper` 分支和 `robot0_gripper_qpos` 拼接，并同步 `eval_socket.py`，然后重新启动评测；不要通过把 runner 改回 6 维或补零绕过输入契约。若形状不是上述情况，保留完整的 received／expected 报错进一步定位。

结果记录 `weights="pretrained"`，携带转换资产 SHA256 与源 URI，不标记为 EMA。本次使用官方 LIBERO checkpoint 和 HDF5 中的 LIBERO-FT 环境配置，仅作接口检查。固定 RLinf eval 配置使用 `ignore_terminations=True`，本实现成功后提前结束；加上图像 resize 等尚未验证的差异，不能将结果直接视为 RLinf／官方公开成绩，也不能替代项目数据 SFT 的 dense 基线。

## 项目 HDF5 数据准备（服务器验收已通过）

用户提供的服务器日志已确认：500 个 episode 划分为 490 个训练、10 个验证 episode；train／val 分别有 135270／2820 个窗口。双相机 batch 为 `[2,256,256,3]` uint8，状态 `[2,8]`、动作 `[2,10,7]` float32，语言正确，并打印 `preparation complete`。验证集覆盖五个任务，符合已选择的全局随机划分。已有 `outputs/pi05_data/libero10_seed42` 时无需重新运行准备命令。

在仓库根目录运行，不加载模型、不需要 GPU 或模拟器：

```bash
conda activate pi05
python -m pip install -r requirements_pi05_train.txt -r requirements_pi05_eval.txt
python -m pip install --no-deps -e ../RoboKit
wandb login
bash scripts/pi05/prepare_libero_data.sh
```

该入口只负责 LIBERO，默认读取 `~/code/LIBERO-FT/libero/datasets/libero_force_256/libero_10/` 下的十个成功回放文件，具体清单见 `configs/pi05_libero_data.yaml`。TCL 使用独立的 `scripts/pi05/prepare_tcl_data.sh`；两者共享 `diffusion_policy/common/prepare_data.py`，数据准备在训练前手动执行一次，训练不会自动调用。

输出为：

```text
outputs/pi05_data/libero10_seed42/
  dataset_metadata.json   # 相对数据源、episode 划分、任务语言、长度、输入契约、统计哈希
  norm_stats.json         # 仅训练 episode 的 state[8] / actions[7] 统计
  config.yaml             # 数据准备配置
```

目录已有统计或 metadata 时拒绝覆盖。重新生成或更改划分时指定新的 `asset_dir`，例如：

```bash
bash scripts/pi05/prepare_libero_data.sh \
  dataset_root="$HOME/code/LIBERO-FT/libero/datasets" \
  asset_dir=outputs/pi05_data/libero10_seed42_v2
```

默认全局随机划分沿用 DP 的普通逻辑：排序 `(source, demo_key)`，使用 `RandomState(42)`，留出 2% 验证 episode，不限制训练 episode 数。任务列表不同会产生不同划分；这里复用的是算法，不代表自动复用某个既有 DP 实验清单。终端报告十任务各自的训练／验证数量，全局划分不保证每个任务都有验证 episode。

数据集不做图像旋转、resize、增强或归一化。`obs_before_action` 下，观测 `obs[t]` 对应动作 `actions[t:t+10]`；尾部重复最后动作，仍监督完整窗口。按本次决定忽略 episode 内的 `replay_reset_mask`。状态拼接 `ee_states[6] + gripper_states[2]`，语言与 LIBERO benchmark 一致，不使用带场景名的 reader 文本。

统计使用训练 episode 的所有原始时间步，每步只计一次，不统计重叠窗口与尾部填充。保存 float64 NumPy 精确分位数 `q01/q99` 和 `mean/std`；实际读入的状态和动作为 float32。此统计是项目配方，不声称与上游直方图估计或窗口加权统计数值相同。

命令检查状态／动作形状与非有限值、两相机 uint8 和共同尺寸、episode 长度，以及已声明的图像／时间约定。旧 HDF5 缺少元数据时，会打印并保存“原始 opengl 方向、动作前观测”的假设；明确冲突时报错，不自动翻转或移动动作时间。数据源原地修改后应重新准备资产，当前不会对全部图像内容计算哈希。

**本批简单验收：** 命令打印 `preparation complete`，十任务均有训练 episode，train/val batch 分别显示以下格式，并成功生成上述三个文件：

```text
image / gripper: [B,256,256,3], torch.uint8
joint_state:    [B,8], torch.float32
action:         [B,10,7], torch.float32
prompt:         B 个任务语言字符串（没有 STUDY_SCENE 等场景前缀）
```

默认检查 batch size=2，验证集较小时 B 可以为 1；图像尺寸按 HDF5 实际值保留。这里不要求模型前向／反向，模型输入的 224×224 转换仍由 policy 完成。需要检查多 worker 读取时，可在新的输出目录运行并追加 `num_workers=2`。

后续 workspace 可直接使用：

```python
from diffusion_policy.dataset.libero_dataset import LiberoFTDataset

dataset = LiberoFTDataset(
    dataset_root='~/code/LIBERO-FT/libero/datasets',
    dataset_subname='libero_force_256/libero_10',
    hdf5_fns=None,
    horizon=10, pad_before=0, pad_after=0,
    shape_meta={'obs': {}, 'action': {'shape': [7]}},
    output_mode='raw',
    asset_dir='outputs/pi05_data/libero10_seed42',
)
validation = dataset.get_validation_dataset()
# DataLoader 默认 collate 即可生成 policy.compute_loss 所需的 batch。
```

## 混合精度 SFT、EMA 与可选完整恢复

训练入口复用仓库根目录 `train.py`。Python 工具在 `diffusion_policy/common/pi05_training_util.py`，`scripts/pi05/train_libero.sh` 使用 `accelerate launch` 启动单机单／多卡进程；workspace 仍自行管理 DDP，启动器设为 `--mixed_precision=no`，保持模型原有分模块 bf16／FP32。使用独立 Python 3.10 `pi05` 环境；默认训练期间会自动调用 `libero` 环境中的模拟器：

```bash
conda activate pi05
python -m pip install -r requirements_pi05_train.txt -r requirements_pi05_eval.txt
python -m pip install --no-deps -e ../RoboKit
wandb login
```

**新训练需要 `~/pretrained/pi05/converted_base`，不能使用 `converted_libero` 代替基座。** 若尚未准备 base，先执行本文前面的 base 下载／转换步骤。训练读取 base 的模型、tokenizer、config 与 manifest，归一化使用已验收的项目 `outputs/pi05_data/libero10_seed42/norm_stats.json`。

### 四卡启动和默认配方

在仓库根目录运行，GPU 编号按服务器实际分配修改：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/pi05/train_libero.sh
```

未设置 `OUTPUT_DIR` 时，脚本自动使用 `outputs/pi05_sft/YYYY-MM-DD/HH-MM-SS`，每次启动获得独立目录。只有需要固定目录（例如显式恢复）时才设置 `OUTPUT_DIR`；workspace 会拒绝已有 `train.jsonl` 的目录。脚本顶部 `tmp` 字符串保存可复制的正式训练命令，不会自动执行。可用 `MICRO_BATCH`、`GLOBAL_BATCH`、`MAX_STEPS`、`SEED`、`ROLLOUT_EVERY`、`N_TEST` 环境变量设置常用参数；也可在命令末尾通过 `"$@"` 传入 Hydra 覆盖，同名配置以末尾覆盖为准。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MICRO_BATCH=2 GLOBAL_BATCH=256 \
OUTPUT_DIR=outputs/pi05_sft/libero10_micro2 \
bash scripts/pi05/train_libero.sh runner.n_test=10 logging.name=pi05_micro2
```

四卡 micro batch=2、global batch=256 时累积 32 次。多组训练同时运行时，为各组设置不同的 `MAIN_PROCESS_PORT`（默认 29500）；它与 runner 自动分配的 HTTP 端口独立。已有环境需重新安装训练 requirements 以取得 Accelerate；每次运行使用新的输出目录。

| 项目 | 默认值 |
|---|---|
| 并行 | 单机 DDP，脚本按 `CUDA_VISIBLE_DEVICES` 的卡数启动 |
| micro batch / 全局 batch | 每卡 1 / 256；四卡累积 64 次更新一次 |
| 更新范围 | 全参数允许训练，无冻结、LoRA 或任务均衡 |
| 精度 | 原有分模块 bf16／FP32，无全局 autocast／GradScaler |
| Attention | 默认原生 PyTorch SDPA；可用 `model.attention_backend=math` 回退到 eager 实现 |
| 优化器 | FP32MasterAdamW，betas=(0.9,0.95)，eps=1e-8，weight decay=1e-10 |
| LR | 第一次更新为 5e-5/10001，完成 10000 次更新后达到 5e-5，随后恒定 |
| 训练长度 / 梯度裁剪 | 30000 optimizer steps / 全局范数 1.0 |
| 梯度检查点 | 默认启用，非重入；可用 `model.gradient_checkpointing=false` 关闭 |
| EMA | 0.999；仅 rank 0 在 GPU 保存 FP32 shadows，基于 FP32 主权重更新 |
| 数据加载 | 每 rank 2 workers，确定性读取，policy 在主进程处理增强 |
| 验证 | 每 1000 步及正常结束时，仅 EMA、固定 seed、固定窗口，最多 32 个 batch |
| 在线评测 | 每 1000 步及正常结束；十任务，每任务 3 episodes，每卡 1 个环境 |
| 保存 | latest EMA／按总体成功率选出的 best EMA；完整状态默认不保存 |
| W&B | 默认 online，project=diffusion_policy_libero，仅 rank 0 上报 |

步数均为完成梯度累积后的 optimizer step。全局 batch 必须整除 `卡数 × micro_batch`。每轮均匀打乱训练窗口，丢弃不足一个全局 batch 的余数；四卡不会重复补齐样本。单卡启动只需设置一个 GPU 编号。

四卡 DDP 每卡保留完整模型和优化器，rank 0 额外维护 EMA；四张 80GB 不能合并成单卡 320GB。首先使用下文短跑测量容量与吞吐。当前梯度累积和 DDP 通信沿用参数梯度 dtype；“FP32 master”指主权重与 Adam 状态，不表示全程梯度都是 FP32。

### 输出和 EMA 加载

```text
outputs/pi05_sft/libero10_dense/
  config.yaml
  run_metadata.json                # 有效配方、来源、依赖版本、实际累积次数
  train.jsonl                      # train / validation / rollout / checkpoint 记录
  first_update_diagnostics.json    # 首次更新的模块梯度、主权重/EMA 变化、无梯度参数
  checkpoints/
    latest_ema.pt                  # 最近一次验证或 rollout 的 EMA，自包含推理文件
    best_ema.pt                    # 总体 rollout 成功率最高的 EMA（默认）
    latest_full.ckpt               # 仅 checkpoint.save_full=true 时生成
```

EMA 文件只保存推理 dtype 的权重、模型配置、tokenizer、项目统计和训练来源，不含 Adam、主权重或训练 RNG。FP32 模块仍保存 FP32。`best` 默认按完整任务评测的总体成功率选择，首次有效评测即保存，严格提高才覆盖，平分保留较早权重。validation loss 只作为诊断；当前验证 episode 只覆盖五个任务，不代表十任务成功率。无法在线评测时显式使用 `rollout.enabled=false checkpoint.monitor=val_loss`。记录的每步吞吐仅统计训练更新，不包含验证和保存耗时。

最简单的模型加载验收：

```bash
python - <<'PYCODE'
from diffusion_policy.policy.pi05_policy import Pi05Policy
policy = Pi05Policy.from_checkpoint(
    'outputs/pi05_sft/smoke/checkpoints/latest_ema.pt', device='cuda:0')
print('EMA strict load OK:', policy.device)
print(policy.get_extra_state()['assets']['training'])
PYCODE
```

保存使用临时文件成功写完后替换，保留 latest／best，不累积各步历史文件；写入期间需容纳临时文件。恢复到新目录时会保留历史 best 指标，只有超过该指标才在新目录生成 `best_ema.pt`，历史 best 文件仍留在原目录；`latest_ema.pt` 正常生成。

### 纯训练短跑（已跑通主循环；不含新 rollout）

先以四卡、全局 batch=8（每卡 1、累积 2 次）运行两个 optimizer steps，仅验证少量窗口：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=outputs/pi05_sft/smoke \
bash scripts/pi05/train_libero.sh \
  training.global_batch=8 training.max_steps=2 \
  rollout.enabled=false checkpoint.monitor=val_loss logging.mode=disabled \
  validation.every=2 validation.max_batches=2
```

检查 loss／grad_norm 有限，诊断中主要模块有梯度、主权重与 EMA 发生变化，两个 EMA 文件生成，并能按上面的命令严格加载。某些最终语言输出分支不参与 action loss，无梯度参数会列出；不要求每个参数都非零更新。首次 LR 很小，部分 bf16 live 权重可能暂未跨越量化间隔，应观察 FP32 主权重。

完整恢复 2B 当前暂缓，不是下一步评测的前置条件。默认不保存完整恢复文件。以后需要验收恢复时，使用同一小配置比较连续 4 步与 2+2 步；所有输出目录使用未运行过的新目录：

```bash
# 连续 4 步；第 2、4 步验证，与分段运行频率一致。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=outputs/pi05_sft/resume_reference \
bash scripts/pi05/train_libero.sh \
  training.global_batch=8 training.max_steps=4 \
  rollout.enabled=false checkpoint.monitor=val_loss logging.mode=disabled \
  validation.every=2 validation.max_batches=2

# 前 2 步：临时开启完整保存。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=outputs/pi05_sft/resume_part1 \
bash scripts/pi05/train_libero.sh \
  training.global_batch=8 training.max_steps=2 \
  rollout.enabled=false checkpoint.monitor=val_loss logging.mode=disabled \
  validation.every=2 validation.max_batches=2 \
  checkpoint.save_full=true checkpoint.full_every=2

# 恢复到总计 4 步；默认关闭后续完整保存以节省空间。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=outputs/pi05_sft/resume_part2 \
bash scripts/pi05/train_libero.sh \
  resume=outputs/pi05_sft/resume_part1/checkpoints/latest_full.ckpt \
  training.global_batch=8 training.max_steps=4 \
  rollout.enabled=false checkpoint.monitor=val_loss logging.mode=disabled \
  validation.every=2 validation.max_batches=2
```

核对 `resume_reference/train.jsonl` 与 `resume_part2/train.jsonl` 中第 3、4 步的 `sampler`／`next_sampler`、LR、global_step 和 loss，以及验证记录中的 EMA updates。采样位置／步数／LR 应一致；固定相同硬件和依赖后记录 loss 数值差异，不承诺跨环境逐位一致。不要用更换卡数或 global batch 的方式做完整恢复验收。

### 完整恢复开关与限制

正式训练需要恢复保障时追加：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=outputs/pi05_sft/libero10_with_resume \
bash scripts/pi05/train_libero.sh checkpoint.save_full=true
```

完整文件默认每 1000 步及正常结束时覆盖一份 `latest_full.ckpt`，包含 live、FP32 主权重／Adam／EMA、scheduler、各 rank 随机状态、已消费数据位置、配置和数据资产。文件较大，使用者可自行选择开启；保存失败时不会覆盖上一份有效文件。

`resume` 只接受完整文件；EMA 文件用于推理，不能恢复 Adam 历史。恢复不依赖原始 base 或准备资产目录，仍需要同一份 HDF5 数据，可通过 `dataset_root=...` 调整其位置。world size、global／micro batch、seed、模型与优化器配置、EMA 和验证子集设置必须一致；可以调整输出目录、停止步数、日志／保存／验证频率。更改配置后应作为新实验从 base 启动。中途强制终止只能恢复到最近已成功保存的完整更新边界。

每次运行和 resume 使用新的 `OUTPUT_DIR`，防止日志和结果混淆。恢复自动复用 checkpoint 的划分、统计和 tokenizer，不重新准备数据，也不重置 EMA 或 warmup。现有检查验证 episode 长度与数据契约，不对全部 HDF5 图像内容计算哈希，原始数据应保持不变。

本地仅执行静态检查；四卡训练主循环已有用户服务器日志确认，新 rollout 和恢复对比未在本地执行。

## 训练 EMA 评测、多卡 rollout 与 W&B（已实现，待服务器验收）

默认保存语义已改变：新训练的 `best_ema.pt` 按成功率选择；旧文件不会被改写，旧 `best_ema.pt` 可能仍是按 validation loss 选出的权重。查看文件内 `assets.training` 的 `monitor`、`metric` 和 `global_step` 区分来源。

### 1. 独立评测训练 EMA

先使用已导出的权重跑之前同一个 STUDY 任务 3 episodes（替换实际 checkpoint 路径）：

```bash
conda activate pi05
python -m pip install -r requirements_pi05_train.txt -r requirements_pi05_eval.txt
wandb login
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=outputs/pi05_libero_eval/sft_single \
bash scripts/pi05/eval_libero.sh \
  checkpoint=outputs/pi05_sft/smoke/checkpoints/latest_ema.pt
```

`checkpoint` 优先于 `asset_dir`，使用文件内嵌配置、tokenizer 和项目统计。无需原始 base 或准备资产目录，仍需要 LIBERO-FT 的任务 HDF5。入口只接受训练导出的 EMA 推理文件，不接受 `latest_full.ckpt` 或 live 文件。未指定 `checkpoint` 时，保留 `converted_libero` 官方权重接口检查。

再使用全部 GPU 跑十任务，共 30 episodes：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=outputs/pi05_libero_eval/sft_ten_tasks \
bash scripts/pi05/eval_libero.sh \
  checkpoint=outputs/pi05_sft/smoke/checkpoints/latest_ema.pt all_tasks=true
```

脚本按显式 `CUDA_VISIBLE_DEVICES` 的卡数启动 torchrun；未设置时使用普通 Python 单进程入口。直接使用 `python -m diffusion_policy.common.pi05_eval_libero` 也可。十任务名单来自 `configs/pi05_libero_data.yaml`；`all_tasks=false` 时可通过 `runner.hdf5_fns=[...]` 指定子集。每个 rank 加载同一个文件，记录其 SHA256、原训练步数及实际评测配置，结果仍在 `eval/epoch_0000/summary.json`。

### 2. 训练内多卡 rollout 验收

用新输出目录，连续两次评测并在中间恢复训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=outputs/pi05_sft/rollout_smoke \
bash scripts/pi05/train_libero.sh \
  training.global_batch=8 training.max_steps=2 \
  rollout.every=1 validation.every=1 validation.max_batches=2
```

默认读取准备数据清单的全部任务。四卡十任务按 rank 分配 3／3／2／2 个任务，每卡同时运行一个任务、一个模拟器环境，任务队列串行推进。`runner.n_test` 和 `runner.n_envs` 可调整；首批先保持 3 和 1。任务数少于卡数时空任务 rank 仍参与同步。

每卡在本进程启动 HTTP 服务，`port=0` 自动分配端口，再由 `libero` Conda 环境执行 LIBERO-FT evaluator。无需手动启动服务。沿用 DP 的 PCI/UUID 与 EGL 匹配，支持非连续 CUDA 设备列表。服务器必须已有此前 DP 自动匹配所需的 LIBERO-FT `libero/force/render_device.py` 等更新；本批没有修改 LIBERO-FT。

仅 rank 0 常驻 FP32 EMA。评测前各卡把 live 权重备份到 CPU，rank 0 安装 EMA 并按推理 dtype 广播，各卡直接复用自己的模型；评测结束恢复 live 权重和 RNG。CPU 内存需容纳各 rank 的 live 备份，不增加第二份 GPU 模型。任务执行期间用 Gloo 收集结果，避免其他 rank 长时间挂起 NCCL 等待 kernel。进程／HTTP 清理完成后才继续训练，可捕获错误在各 rank 间传播，不接受不完整评测。

结果按步数隔离：

```text
outputs/pi05_sft/rollout_smoke/
  rollouts/step_00000001/eval/epoch_0000/summary.json
  rollouts/step_00000002/eval/epoch_0000/summary.json
  checkpoints/latest_ema.pt
  checkpoints/best_ema.pt
```

检查两轮各有十任务、30 episodes，无遗漏／重复，四张卡的服务和渲染身份匹配；第 1 轮后能继续第 2 步，视频正常且目录不覆盖。初期成功率可以为零，不设学习效果门槛。首次完整评测即生成 best；只有严格更高的成功率才覆盖。相同步数验证和 rollout 重合时只导出一次，正常结束也只评测一次。

新完整 checkpoint 记录选模方式及最佳值；旧完整文件只带 best loss，切到 success_rate 时从尚未评测开始，不把 loss 当作成功率。恢复后的历史 best 文件仍保留在旧目录。完整恢复验证暂缓。

Stage A 可通过 `FIRST_ROLLOUT_STEP` 在指定 optimizer step 额外执行一次 rollout：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FIRST_ROLLOUT_STEP=50 ROLLOUT_EVERY=400 \
bash scripts/pi05/train_libero_stageA.sh
```

对应配置为 `rollout.first_step=50`，在完成第 50 次 optimizer 更新后评测，
随后仍按原周期在 400、800……步评测，并保留训练最后一步的评测。
默认 `0`（旧配置缺失字段时也为 `0`）不增加额外评测；负值或非整数报错。
该参数不屏蔽指定 step 之前的周期评测；与周期或最后一步重合时只运行一次。
`rollout.enabled=false` 时不会触发，sample 和 validation 的调度不变。
恢复训练沿用全局 step，不补跑已到达或错过的指定 step；指定值超过训练总步数时不额外触发。
额外 rollout 同样参与 best EMA 选模和 latest EMA 导出，不改变完整 checkpoint 的保存周期。

### 3. W&B 与配置覆盖

训练和独立评测默认 `logging.mode=online`，project 为 `diffusion_policy_libero`；可设置 `logging.entity`、`logging.name`、`logging.group` 和 `logging.tags`。仅 rank 0 创建 run，训练中评测写回同一 run；独立评测创建新的评测 run。

- `train/*`：loss、LR、梯度范数、耗时、吞吐；`memory/rank_N/*`：每卡峰值显存。
- `val/*`：EMA validation loss 和历史最低 loss，不用于默认 best 选模。
- `test/*`：总体／逐任务成功率、episode 数、评测耗时及代表性视频。
- `checkpoint/best_success_rate`：历史最佳总体成功率（val_loss 模式对应 best_val_loss）。

所有曲线以 optimizer `global_step` 为横轴，每步统一提交一次，避免同一步验证或 rollout 指标丢失。保留 `train.jsonl`、评测 JSON 和本地视频，不自动上传大体积权重。

无网络可追加 `logging.mode=offline`，完全关闭上报用 `logging.mode=disabled`。在线初始化失败明确报错；不会静默回退。训练日志中的吞吐不含验证、rollout、保存和上报时间。

仅训练、不启动模拟器：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=outputs/pi05_sft/no_rollout \
bash scripts/pi05/train_libero.sh \
  rollout.enabled=false checkpoint.monitor=val_loss logging.mode=offline
```

本地只做 Python／shell 静态检查和 Hydra 配置组合检查；未运行 CUDA、HTTP、模拟器、多卡 rollout 或 W&B 网络上报。服务器还需核对单卡、空任务 rank、连续评测恢复训练，以及 evaluator 启动失败时端口和进程清理。每任务 3 episodes 是接口验收规模，正式成绩应提高 episode 数。官方权重曾获 3/3 不代表项目 SFT 十任务成绩。

### 输入方向与可视化兼容

`model.render_image_orientation` 属于 policy 输入预处理，支持 `raw` / `rot180`。WandB 的 LIBERO static 始终按原图旋转 180° 展示，wrist 保持原方向；训练与 eval 共用 policy 预处理，runner 的旧显示开关不覆盖此规则。

新 checkpoint 保存方向字段；旧 pi0.5 配置缺失该字段时按 `rot180` 规范化后比较。wrist 不再沿用旧版双相机翻转行为，因此不能用本版本声称逐步复现旧版 pi0.5 训练。DP 对应字段为 `policy.render_image_orientation`，默认和旧配置缺省值均为 `raw`；旧 `vis_rot_image` 仅是显示选项，不表示输入旋转。


### Stage A sample diagnostics

每次 `training.sample_every` 触发时，rank 0 使用 EMA/eval policy 计算训练 sample、
各任务 topology 和总体 resolution advantage。pi05 的间隔单位是 optimizer step，DP 是 epoch；
不增加独立诊断频率。`sample_every=0` 或 dense 模式跳过这些诊断。

- `sample/topology_div_<task_name>`：同一任务窗口间叶节点集合的平均 Jaccard distance，
  忽略 padding，在两路真实相机间平均。`sample/topology_div` 为有效任务的等权平均。
  固定完整层级时 diversity 为 0；不足两个窗口的任务不参与 topology 汇总。
- `sample/resolution_advantage_0_to_1`、`sample/resolution_advantage_1_to_2`：
  两路相机同时从粗层切换到细层时的 **粗层 loss − 细层 loss**，正值表示细化有收益。
  pi05 只上报总体曲线，先在任务内平均，再对任务等权平均，不上报分任务的 resolution advantage。
  即使训练固定使用 `level:2`，诊断仍比较全部三个层级。
- pi05 使用与 SFT 相同的完整 `10 × 32` flow-matching velocity MSE（包括 padding 维度）；
  DP 使用自己的 diffusion loss。两者统计逻辑一致，绝对数值尺度不同。

`topology_samples_per_task=64` 从每个任务的训练窗口中无放回采样，不足 64 时使用全部窗口。
同一官方任务的多个文件先合并，不改变原有数据划分或资产格式。
`sample_resample=false` 固定窗口及诊断随机输入；为 true 时每次重新采样窗口，
pi05 使用 `training.seed + global_step`，DP 使用 `training.seed + epoch`。
模型诊断始终按 `training.seed + task_id` 固定随机种子；pi05 使用 eval 图像预处理。
同一窗口的三个层级共享图像、patch bank、噪声和 flow time，诊断结束恢复训练 RNG 和模式。
缓存训练 sample 的图像仍保持固定，resample 开启时可重新采样其展示拓扑。

`sample_resolution_advantage=false` 只关闭分辨率 loss 诊断，保留各任务 topology。
pi05 按任务串行处理，`sample_diagnostic_batch_size=8` 限制 resolution 模型前向批大小；
噪声和时间在每任务级生成后切片，topology 在完整任务窗口集合上计算，包含跨小批的窗口对。
新增配置缺失时使用上述默认值，不新增模型权重或改变训练恢复 signature。
默认每 9 step 执行完整诊断会增加训练耗时，可通过 `sample_every` 降低频率。
显存、耗时和实际 WandB 曲线需要在 CUDA 训练环境验证。


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
