# LIBERO：在 sample 周期监控相邻分辨率优势

更新：2026-09-30。已接入诊断、加权 level/random 及 random 范围采样；本地仅静态检查，数值、多卡和训练效果待服务器验证。
level0 为 coarse，level 越大越 fine；低分辨率 loss 减高分辨率 loss，正值表示高分辨率更好。
历史接入过程见 [libero_exp.md](libero_exp.md)。
Stage A 覆盖缺口、本次范围选择与 Stage B 低预算路由的解释边界见
[Stage A 拓扑覆盖与 Stage B 低预算路由分析](auto_search_libero_budget.md)。

## 1. 指标与统计口径

三层树仅新增两个日志：

```text
sample/resolution_advantage_0_to_1 = L0 - L1
sample/resolution_advantage_1_to_2 = L1 - L2
```

每个任务得到逐窗口、逐层动作 diffusion loss `loss_B_L`，先在任务内按窗口平均，
再对有效任务等权平均得到 `loss_L`，最后相邻相减。K 层生成 K−1 条优势；单层树不生成优势指标。
不输出绝对 loss、每任务 loss、正比例、标准差或耗时指标。只有一个可用窗口的任务仍参与 loss 平均，
但按原规则不参与 topology 多样性计算。任务不足目标样本数时使用全部可用窗口，不复制补齐。

这些指标来自固定训练诊断子集，不等于整轮训练 loss，也不能直接由已有 `train/diffusion_loss` 相减得到。
全图统一 level 同时改变 token 数量与汇聚方式，不能解释为高频像素的独立因果效果，
也不能单独证明 Stage B 混合树中的局部 stop/split 收益可靠。

## 2. 采样、开关与调度

```yaml
training:
  sample_every: 6
  topology_samples_per_task: 64
  sample_resolution_advantage: true
```

LIBERO YAML 默认开启；其他任务未配置时默认关闭。在哪个阶段开启，就使用该阶段训练 dataset，
不固定为 Stage A，也不切换验证集。沿用 `training.seed + task_id` 的任务 seed。

workspace 共用任务诊断循环，每次 sample 周期只调用一次 `dataset.iter_task_diagnostic_batches()`。
每个任务的同一批数据同时用于 topology 和分辨率优势。每次重新读取固定 seed 选出的窗口，
保留固定 seed 下的训练增强；不增加长期图像缓存、采样清单或无增强读取路径。
各 level 共享同一批增强后的图像、动作和低维状态。

沿用现有 sample 调度、主进程执行、EMA 优先与 eval 模式；无 EMA 时使用解除 DDP 包装后的 live 模型。
结果进入 `step_log`，由现有流程写入 WandB 和 JSON。关闭开关时保持原 topology 行为。
保留现有主进程诊断后的同步，诊断内部没有跨 rank collective。

## 3. Policy 计算流程

新增无梯度接口 `compute_resolution_diagnostics(batch, *, seed: int)`，
返回 `loss_B_L` 和原 topology 所需的 `foveation_outputs`。仅支持 `obs_as_cond=true`，
由 workspace 在 eval 模式下调用；不满足条件时明确报错。

1. 按 `n_obs_steps` 截取观测，复用既有动作与观测归一化。
2. 调用一次 `_route_observations()`，同时取得 topology 输出和 `patch_bank_B_T_K_C_P_P`，
   替代原任务 topology 前向，不重复调用 `get_foveation_outputs()`。
3. 对 level `l`，将内部节点索引 `:internal_level_offsets[l]` 设为 split，其余 stop。
   复用 `pack_topology()` 构造全图统一 level，所有视角和观测帧同步使用该层。
4. 复用 `_encode_routed_observations()` 与 `_diffusion_loss_per_sample()`，堆叠各层逐样本 loss。
5. workspace 完成任务内平均、任务间等权平均和相邻相减。

每窗口仅一组 diffusion noise/timestep，各 level 共享 noise、timestep 与 condition mask，
不增加 repeats 配置。diffusion 使用固定任务 seed 的独立 generator，不受随机 topology 消耗随机数的影响。
路由 RNG 使用 fork/restore，保持原任务 topology seed 和结果，不推进训练 RNG。
不临时更改 `fix_split_mode`、budget 或训练 loss，不反传、不更新参数；eval 关闭模型 dropout。

每任务顺序处理，三层对应三次视觉编码和三次单步 denoiser 前向。继续复用同一个 patch bank，
不增加额外完整动作采样。原 sample 动作采样与可视化保持现有流程。

| 三层树配置 | level0 leaves | level1 leaves | level2 leaves |
| --- | --- | --- | --- |
| 256 输入、patch16，每视角每帧 | 16 | 64 | 256 |
| 128 输入、patch16，每视角每帧 | 4 | 16 | 64 |

实际数量以模型的层级网格为准。随机 split 比例不代表全图统一 level，不能将 `random:0.5` 命名为 level1。

## 4. 修改范围与验收

| 文件 | 修改 |
| --- | --- |
| `diffusion_policy/policy/foveation_dp_transformer_policy.py` | 新增共享路由与随机输入的逐层 loss 诊断接口 |
| `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py` | 合并任务诊断循环、任务聚合与 sample 日志 |
| `configs/liberoft_foveation_dp.yaml` | 新增默认开启的布尔开关 |
| `plan/fix_libero_resolution_loss.md` | 收敛采样与指标协议 |

本节范围仅指分辨率诊断，无需修改 dataset、FoveationNet、启动脚本或 checkpoint 参数结构。
后续新增的训练采样支持见第 5 节。
按仓库约定，不新增独立测试脚本，本地只做 Python 静态编译与 `git diff --check`。

服务器验收：

- 每 sample 周期每任务数据只读取一次，两种诊断共享输入；普通训练 step 不执行额外诊断。
- 各 level 叶层级和数量正确；coarse/fine 端点与既有路径在相同随机输入下 loss 一致。
- 核对任务内平均、任务间等权平均及相邻差值正负号；三层只新增两个指定指标。
- 同权重、同 seed 可复现；开启诊断不改变原 topology 结果、训练 RNG、参数或梯度。
- 一个可用窗口参与 loss、不参与 topology；单层树不生成相邻指标。
- 单卡和多卡可运行；记录实际诊断耗时评估开销，不增加耗时日志指标。

## 5. 后续实验与保留背景

已支持按窗口加权混合统一层级和随机树。LIBERO、PushT 通用默认配置保持不变；
`sh_train_foveation_libero_stageA_remaining6.sh` 默认使用以下配置，仍可通过环境变量覆盖：

```yaml
fix_split_mode: ["10%level:0", "25%level:1", "50%level:2", "15%random:0.05-0.99"]
random_topology_prob: 0.0
```

- `level:l` 表示全图统一叶层级，数字越大分辨率越高；也可单独写 `fix_split_mode: "level:1"`。
  mask 与 resolution diagnostics 相同，不以 split 比例近似层级。
- 外层必须是真正的 YAML 列表，每个条目带引号和百分比；百分比支持小数及零值，必须有限、非负，
  合计 100%（浮点容差 `1e-6`）。首版只允许 `level` 和 `random` 子模式，不接受无权重条目。
- 先按外层权重抽模式，再对 `random:` 的离散条目等概率抽样，或在范围内均匀抽连续比例。
  示例中 random 合计 15%，其余 85% 为统一 level。
  百分比表示长期概率，不保证单个 batch 精确配额，也不再作为 loss 权重。
- 同一动作窗口的所有相机、观测帧共享模式；random 比例也跨相机共享，但各相机独立抽空间布局，
  同一相机的布局跨帧共享。每窗口仅走一次训练前向，不增加多 level 监督或排序约束。
- 训练和 eval 都使用该混合策略，沿用调用方 Torch RNG；新模式不新增参数或持久化 buffer。
  单层树仅允许 `level:0`；不支持的格式、非法层级和比例在初始化时报错。

范围也可独立配置为 `fix_split_mode: "random:0.05-0.99"`。
每个 `random:` 接受单比例、离散列表或一个范围，不接受 `random:0.1,0.2-0.5` 之类的混写；
需要多个范围时，将其分别作为外层加权条目。端点必须有限且满足 `0 <= min <= max <= 1`，
如 `random:0.2-0.2` 表示固定比例、随机布局，`random:0-1` 表示全比例范围。
支持科学计数法（如 `random:5e-2-9.9e-1`），保留旧离散列表中的重复比例。
独立范围同样按动作窗口抽比例，跨相机共享比例、独立抽布局，同相机跨帧共享布局；
旧独立离散模式保留原来的逐样本行为与随机调用顺序。

数量沿用 `num_splits = floor(ratio * num_internal_nodes + 0.5)`，
`num_leaves = min_num_patches + 3 * num_splits`，不均匀抽整数数量。
`0.05–0.99` 在当前 256 输入、patch16、三层树下覆盖 `28, 31, ..., 253` 叶子；
其他几何按通用公式计算，单层树保持唯一拓扑。范围模式在训练和 eval 均采样，使用调用方 Torch RNG，
解析结果为普通 Python 属性，不新增模型参数或持久化 buffer。

启动脚本传入包含引号条目的列表，不能将整个列表再作为一个 Hydra 字符串加引号：

```bash
# 仍需设置 CUDA_VISIBLE_DEVICES 及任务原有的启动参数。
FIX_SPLIT_MODE='["10%level:0","25%level:1","50%level:2","15%random:0.05-0.99"]' \
RANDOM_TOPOLOGY_PROB=0.0 \
bash scripts/roi/sh_train_foveation_libero_stageA.sh
```

PushT Stage A 和通用启动脚本接受同样语法，保留原有其他必需环境变量。
实现涉及 FoveationNet 解析及采样、policy 跨相机分组、workspace Stage A 模式提示、两份 YAML 注释、
上述三个启动脚本和本文；dataset、诊断 seed、日志指标及 Stage B 监督目标不变。
服务器还需核对加权频率、跨相机/帧同步、旧模式随机序列、新模式 checkpoint 恢复及 Stage B 严格加载。
范围采样还需核对独立/加权模式、等端点与边界范围、非法输入、合法数量、长期采样频率及固定 seed 可复现。

旧字符串 `random:` 继续对条目等概率采样，允许重复；可保留以下配置作历史对照：

```bash
# 均匀基线
FIX_SPLIT_MODE='random:0.0,0.25,0.5,0.75,1.0'
# 后续 fine 偏重候选：fine 占 60%
FIX_SPLIT_MODE='random:0.0,0.25,0.5,0.75,1.0,1.0,1.0,1.0,1.0,1.0'
```

保持数据清单、初始化、结构、训练 seed、增强与更新次数一致。相邻优势改善可能来自低分辨率退化，
不能只凭差值变大判断 fine 能力提升；若需区分原因，另行做离线绝对 loss 分析，不扩展本轮日志。
全局优势可靠后，再用混合树的局部 paired 诊断及同实际 patch 数的 rollout 验证区域选择。
加权 level/random 支持只改变显式启用后的输入采样，不保证 `level2` 的 loss 低于 `level1`、
或 `level1` 低于 `level0`。各 25% 的均匀模式混合可作为对照；双路监督和排序约束仍为后续独立实验。

### 5.1 2×2 数据分离：已接入，继续保留

| 集合 | 用途 | 每任务 50 条时 |
| --- | --- | --- |
| A_train | Stage A 训练 encoder＋policy | 24 |
| A_val | Stage B 验证 | 1 |
| B_train | Stage B 训练 router，对 Stage A 未见 | 24 |
| B_val | Stage A 验证 | 1 |

每任务完整 episode 随机分半，半集内 `val_ratio=0.02`；四份清单互斥，验证数据不参与参数更新。
小样本划分、每阶段非空检查及 `max_train_episodes=null` 约束保留现有实现。
Stage A 仅以 A_train 拟合统计；Stage B 继承 checkpoint 的清单和统计，切换为 B_train/A_val，不重新划分或拟合。
`SEED` 控制训练；`DATASET_SPLIT_SEED` 控制数据划分，未指定时默认随 SEED。
Stage B 默认 `DATA_SEED_FROM_CKPT=true`，继承 metadata 中的数据 seed，保留本次 training seed；
恢复训练以恢复 checkpoint 的 metadata 为准，配置契约不匹配时报错。
有效 seed 以运行日志和 checkpoint 为准，Hydra 启动配置不一定反映加载后的覆盖。
共享 YAML 默认 legacy，两阶段脚本显式 two_by_two；旧 legacy checkpoint 不可通过重新划分变成未见数据。
这次新增监控复用上述契约，不重新实现数据划分。

### 5.2 不再作为首轮主线的候选

| 候选 | 保留条件与限制 |
| --- | --- |
| 更强预算／双侧／固定预算 | 仅当同预算布局排序可靠、问题主要是数量偏好时启用。上限预算在目标以下没有向上拉力；双侧软数量也不保证 hard leaves 精确达标 |
| 执行动作或可靠 fine teacher 指示 | held-out diffusion 排序也缺少任务价值时再试；teacher 必须先有独立任务表现依据。共享模型对自身 fine 输出的零距离不构成可靠性证据 |
| Episode 级交叉拟合 | 数据分离有效但样本不足时考虑；增加多份 Stage A 成本，需处理信号尺度与最终下游兼容性 |
| 表示、低通对照、观测历史、初始化调整 | 仅在证据指向这些因素时单独做对照，不与首轮采样策略同时改变 |

当前代码已有 LIBERO Temporal 配对入口，旧文“尚未接入”已过时；启用 Temporal 不属于本轮分辨率监控主线。
同样不以 advantage 取负、取绝对值、加常数或追求非零 topology_div 作为有效区域选择的替代验收。

保留 loss 解释边界：diffusion loss 按动作元素平均，并非按视觉 token 数求和；split 用后代替换父 patch。
`pred_action_steps_only` 当前仍按完整张量平均，切换它会改变尺度；若另测执行段，须按有效元素数归一化。
历史记录指出本地 RoboKit 尾部动作 clip 会重复末帧，服务器应核对依赖版本；本轮不同时改窗口或尾部 mask。
embedding、state mask、horizon、输入尺寸均以实际 run 配置为准，Stage A 脚本覆盖可能不同于 YAML 默认。
