# mLIBERO 实验规划：当前方法、Novelty 与下一阶段工作

更新日期：2026-09-21。本文记录研究定位、当前证据与 LIBERO 下一阶段实验计划。用户已报告 256 replay 数据检查通过；训练与评估接入仍待实施。第 6 节汇总已确认选择、推进顺序和必要验收条件，接口细节在对应实现阶段确定。

## 1. 当前方法与项目状态

项目研究 **runtime adaptive resolution selection for robotic manipulation**：
通过四叉树为不同区域选择 coarse/fine 表示，在预算下保留动作需要的细节与全局上下文。
运行时在 policy invocation / action-chunk 边界重选，不在线更新参数或执行 paired 搜索。

当前成功配方：


| 阶段                     | 更新对象                                    | 作用                                                                         |
| ------------------------ | ------------------------------------------- | ---------------------------------------------------------------------------- |
| Stage A：多拓扑预训练    | 视觉编码器与 diffusion policy               | 在随机合法树上建立跨尺度、跨数量的下游能力                                   |
| Stage B：paired + budget | 冻结下游，更新 routing                      | 在固定观测、noise、timestep、mask 下，用 stop/split 损失差学习局部分辨率收益 |
| 后期下游适配             | 固定确定性 routing，更新视觉编码器与 policy | 适应所选低预算表示，进一步改善闭环表现                                       |

固定 routing 是固定参数和确定性推理规则，不是固定所有输入的拓扑。实现与恢复约定见
[fix_action_backward.md](fix_action_backward.md)，项目愿景见
[quadtree_introduction.md](quadtree_introduction.md)。

### 1.1 已完成的 PushT 证据

- `target_num_patches=0.1`，paired 后固定 routing、适配下游，三 seed 平均报告成绩
  **0.679（`test/mean_score`，最大覆盖性能）**。
- Paired 阶段本身明显优于已有直接端到端训练对照，后期下游适配进一步明显提高成绩。
  **Paired 信号贡献已完成验证，不再安排重复的 PushT paired/ST 实验。**
- 当前 16–256 leaves 配置下，目标 0.1 对应 40 leaves，约为 all-fine 的 15.6%。
  实际推理数量需单独记录，不能写成实测 40 leaves 或仅使用 10% tokens。
- 已有固定 ratio 测速使用 `--include-router`，ratio 0.1 相比 all-fine 显著加速。
  当前未加载训练 checkpoint，使用合成输入和固定拓扑；后续补充 learned routing 测速。

近期预算实验另报告 alpha=0 时硬数量均值 40.91、完整推理 wall 延迟 122.786 ms、IoU 0.6903。
这是独立实验记录，与上述三 seed 汇总分开保留；具体对照及 Temporal 初步观察见第 4 节和
[study_temporal_adaptive.md](study_temporal_adaptive.md)。

逐 seed 数值、方差／离散程度、checkpoint 选择规则和具体耗时从已有记录整理，
不补写未知值。直接端到端对照不自动等同于冻结下游的 B0/ST；性能对比按实际更新范围
标注。旧版本 all-fine 成绩不能直接用于证明当前同协议下的性能超越。

**整体判断：** 已有强低预算初步结果，应进入跨任务与真实闭环验证阶段。下一步主要
风险是三维任务推广、移动视角适用性和部署成本，而不是重新证明 PushT paired 有效。

## 2. 当前版本的 Novelty 总结

核心表述：**策略学习的不仅是如何行动，还包括当前动作需要什么空间分辨率；通过
合法区域表示间的 paired 比较学习细分价值，再通过分阶段适配形成低预算操作策略。**


| 层面         | 当前主张                                                                               | 边界                                                                        |
| ------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 选择对象     | 同一区域 coarse 表示与 fine 子树之间的分辨率选择                                       | 不只是已有 token 的保留；但混合尺度表示本身已有研究                         |
| 学习信号     | 固定下游的 paired stop/split action-loss 比较，估计局部额外细节的收益                  | 区分 action relevance 与 resolution utility；不是环境 reward 的直接因果收益 |
| 训练机制     | 多拓扑准备下游、固定下游学习 selector、固定 routing 适配下游                           | 组合形成有效方法；分阶段训练本身不单独主张首次                              |
| 实证优势     | 低预算 PushT 表现、paired 优势、下游适配收益和包含 router 的测速优势                   | 实验优势不是独立机制创新；跨任务与真机尚待验证                              |
| 时间计算分配 | Spatial 学习状态相关的局部细分价值，Temporal 显式比较跨时刻等增量收益；已在 PushT 实现 | 有初步运行观察，Temporal 的独立任务收益仍待受控验证                         |

四叉树、动态 tokenization、动作感知选择、局部反事实比较均有相关研究。贡献在本任务中
选择对象、局部收益信号与训练机制的具体结合，不宣称这些组成部分各自首次提出。
完整覆盖不等于无损，paired 不提供全局最优树或无偏整树梯度保证。

MSViT 是混合尺度近邻，ToMe 是特征合并近邻，VLA-Pruner 是动作感知选择近邻；
比较应统一策略与成本口径，DP 上的适配不能称为原始 VLA 方法的直接复现。
详细比较与外部基线框架见 [related_work.md](related_work.md)。这些外部比较不是
开始 LIBERO 前必须在 PushT 全部完成的清单。

面向 CVPR，重点解释分辨率收益学习和与近邻方法的区别；面向 RSS，重点展示闭环感知
需求、视角互补与真实操作价值。当前不为扩大 novelty 而预先增加未经失败证据支持的模块。

## 3. LIBERO：优先迁移成功配方

### 3.1 现有接入状态（2026-09-21 核查）

- 用户已完成 replay 并检查数据，暂未发现问题。当前入口使用 `camera_hw=(256,256)`，
  输出 `libero_force_256/libero_10`；成功／失败文件分开，纠偏位置有标记。
  此处为用户运行反馈，不表述为本地完成服务器验收。
- [配置](../configs/liberoft_foveation_dp.yaml) 仍为 384 图像、旧 JEPA 配方及五任务列表；
  不应仅改数据路径便作为 PushT 成功配方的迁移版本。模型结构、观测、预算和启动入口需一起对齐。
- [数据集](../diffusion_policy/dataset/libero_dataset.py) 支持两路图像、末端状态和力，
  但当前 `val_ratio` / `max_train_episodes` 未真正限制 episode，验证集重复读取训练文件。
  图像仍有随机裁剪，wrist 有 30% 置零；只关闭 color jitter 不等于确定性验证。
- 当前 action / state / force 的缩放在 Dataset 内完成，policy normalizer 为 identity。
  因此 checkpoint 的模型权重和 identity normalizer 不足以独立恢复部署时的缩放。
- [ROI 服务](../service/gpu_service_roi_libero.py) 可参考 HTTP 结构，但不能直接用作当前
  foveation 服务：它只取第一视角、强制末端状态统计、依赖 ROI 输出，checkpoint 选择也需替换。
- `LIBERO-FT/eval_socket.py` 与 `libero/force/force_benchmark.py` 已有独立环境的 HTTP
  evaluator，但固定使用旧任务映射和 128 分辨率，且每 episode 的 policy/cache reset 需要修复。
- workspace 已提供主进程 `env_runner.run(policy_rollout)` 和 `training.rollout_every`，
  训练期间评估优先复用该入口。当前没有 LIBERO runner 配置。

### 3.2 推进顺序与验收目标


| 顺序 | 工作                                                 | 完成标志                                                             |
| ---- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| 1    | 核对数据、观测、动作与闭环评估，建立 static all-fine | checkpoint 能稳定 rollout，输入、归一化、动作 chunk 和成功率口径一致 |
| 2    | static camera 迁移 A → paired → 下游适配           | 获得同任务、同数据和评估设置下的性能与实际数量记录                   |
| 3    | 加入 static + wrist                                  | 两视角训练／推理链路一致，按总预算或实际总成本比较                   |
| 4    | 扩展正式任务集合及论文比较                           | 明确任务选择、数据版本、多 seed、checkpoint 规则与性能—成本结果     |

开发阶段先从现有数据选一个任务打通闭环，再扩展抽屉／微波炉和多对象操作。
具体任务文件与正式 suite 在实施时确定，首轮不同时展开完整任务与全视角矩阵。

接入时优先核对：

- 使用已检查的 256 replay，核对训练与在线图像的方向、颜色、数值范围、时间顺序和相机 key；
  旧 384 配置不作为本轮输入标准。
- 动作表示、归一化、实际执行的 action chunk 与训练一致；成功率和分任务结果可追溯。
- 已确认使用单任务 policy，配置可切换任务；跨任务汇总时分别训练，不混用语言条件多任务协议。
- 首先复用成功的三阶段机制；PushT 的预算数值和 epoch 不视为未经调整即可通用的默认值。

LIBERO 优先是因为已有数据和代码基础。robomimic/robosuite 可作为备选，尤其适合
Square、Tool Hang 等对齐任务的机制研究；目前不并行铺开多个新 benchmark。

### 3.3 Wrist camera 是否需要额外方法

移动相机不会天然使当前方法失效：每次策略调用都根据新图像重新选择拓扑，不要求固定
像素对应固定世界点。首先直接迁移，再按失败原因考虑扩展：


| 观察到的问题                 | 候选处理                                                 |
| ---------------------------- | -------------------------------------------------------- |
| 单帧歧义或状态不足           | 使用已有末端／夹爪状态条件；共享 router 时考虑 camera ID |
| 遮挡和历史信息缺失           | 短历史或轻量记忆                                         |
| 双视角预算浪费、缺少互补性   | 共享预算和跨视角上下文                                   |
| 确实需要跨帧对应             | 再考虑位姿／深度或光流对齐                               |
| chunk 内画面变化导致观测过时 | 在统一比较协议下分析 chunk 长度与重规划频率              |

固定中心细化／中心裁剪是 wrist camera 的有意义参考；不能仅凭初始树固定整段轨迹
的弱对照证明动态方法必要。不要直接平滑相邻帧同像素的 split 概率，也不预设靠近物体
必然需要更多 tokens：目标的自然放大和接触遮挡会共同改变细节需求。

## 4. 自主时间计算分配：当前机制与迁移边界

具体公式、采样历史与日志定义以 [study_temporal_adaptive.md](study_temporal_adaptive.md) 为准。


| 目标     | 当前作用                                                            | 适用边界                                                                                                          |
| -------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Budget   | `upper_bound_lagrangian_elu`、alpha=0，以全局平均软期望数量控制成本 | 超 target 且 lambda>0 时产生减量压力；达到或低于 target 时梯度为零。lambda 初始为零，硬满树与软期望数量需分开判断 |
| Spatial  | 冻结动作路径，用`a=loss_stop-loss_split` 学习局部细分价值           | 正收益支持 split，负收益支持 stop；满树时测量合并代价，reachable stop 节点也可再次展开                            |
| Temporal | 有效等增量配对比较`a_A-a_B`，提高高收益一侧的相对偏好               | 异号收益时与 Spatial 同向；均正时压低低收益一侧、均负时抬高收益更接近零的一侧，可能与 Spatial 冲突                |

细分收益普遍为正且预算存在数量压力时，Budget＋Spatial 倾向于压缩单位节省量对应动作代价
较低的子树；Temporal 补充等增量的跨时刻取舍。实际更新由共享网络上的各项梯度共同决定。

用户报告的 PushT 初步观察：加入 Temporal 后评测效果无明显变化，`sample/topology_div` 的训练曲线更平稳，
匹配率通常超过 0.99；advantage 均值在零附近、正值较多，绝对均值约 `4e-4`，正值比例由约
0.47 波动升至约 0.54。topology_div 衡量 batch 样本间拓扑差异，高匹配率说明等增量候选易找到。
这些观察尚未证明轨迹路由更平滑或任务收益提升。

三个 advantage 日志统计主进程所在卡当前 forward 的全部采样窗口，Temporal 模式含全部 `2P`
窗口及匹配失败、占位窗口；正值比例为 `mean(a>0)`，零值计入分母，无加权或近零阈值。
Temporal loss 与匹配率另行跨卡聚合。收益解释限定在 reachable 与主动匹配形成的采样分布；
结合原始 loss 尺度，并固定窗口、候选和两臂拓扑重复采样 noise/timestep，才能判断信号稳定性。

**已确认先迁移 A → Spatial B → 下游适配，之后单独接 Temporal。** 当前 Temporal 仅支持
PushT，使用完整 reachable 子树和随机锚点等增量匹配，`single_level` 已移除；LIBERO 尚缺配对接口，
workspace 也有数据类型限制。后续接入保留 Spatial only、双窗口匹配且 Temporal 权重为零、完整
Temporal 三组对照，区分额外窗口／候选分布与 Temporal 目标的贡献。LIBERO 任务更难本身不保证收益。

## 5. 下一步待办与后续真机方向

### 5.1 实施清单

- **主要任务：LIBERO。** 修改上述配置与启动入口，核对数据读取，打通四叉树 policy
  的推理评估服务；先 static，再双视角。方法主体优先保持稳定。
- **实验整理：** 汇总已有 PushT run 的配置、逐 seed 结果、paired 前后／适配前后的
  指标及计时输出，不重新安排已完成的 paired 贡献实验。
- **测速工具：** 后续修改 [pusht_compare_infer_speed.py](../exp/pusht_compare_infer_speed.py)，
  支持加载训练 checkpoint 及其配置／normalizer、真实观测输入、保留 learned routing，
  记录实际 leaves 与 `predict_action` 延迟分布。保留现有固定 ratio 模式，用于控制数量
  的比较；all-fine 采用实际部署路径，统一硬件、精度和 diffusion 推理设置。
- **成本记录：** 区分视觉耗时、完整策略调用、每 action 摊销成本和整段累计成本，
  多视角按总量比较，报告 padding 影响与 p95；checkpoint 测速不阻挡 LIBERO 开始。

实现时新增 public class/function 写简洁 docstring，tensor/NumPy 变量使用 shape 命名。
按仓库约定不新增独立测试脚本，本地做适用的静态检查；训练、DDP 与 rollout 验证在服务器
完成。本文更新本身只涉及文档，不代表上述代码已改动或服务器验证已执行。

### 5.2 真机方向（后续规划）

在单臂、固定相机与腕部相机的假设下，优先考虑小物体精确放置、保留视觉纠偏空间的
插入／对齐、以及抽屉或盖子操作后放置对象三类任务。改变目标位置、容差、朝向和
接近视角，检验细节价值；避免初期用主要受力控或标定限制的极端精密任务。

真机保留 all-fine、全局低分辨率、强简单布局／ROI 和完整方法作为重点比较，主要方法
消融使用已有 PushT 与后续仿真证据。记录成功率、完成时间、失败阶段、每视角数量和
延迟；统一控制频率的策略比较与按实际运行速度的系统比较分开报告。具体硬件、任务
和试验规模在真机阶段确定，不作为本次 LIBERO 接入要求。

## 6. 接下来的实验计划

总体顺序保持：先打通训练与闭环接口，再验证三阶段配方，之后分别扩展双视角、任务和 Temporal。
首轮只要求一个任务形成可复现的完整链路；API 字段、超时数值和命令参数在对应实现阶段细化。

### 6.1 已确认范围


| 项目           | 决定                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------ |
| 数据与训练单位 | 已检查的 256 replay；单任务 policy，配置可切换任务                                               |
| 观测           | 支持 static / static+wrist，先 static；两者均加入现有 6 维末端位姿，不加力                       |
| 方法           | Stage A 多拓扑 → Spatial paired＋预算 → 固定 routing 的下游适配；首版`temporal_adaptive=false` |
| 闭环评估       | 通过 RoboKit HTTP 协议连接独立 LIBERO Python 环境；rank 0 服务当前 EMA，其余 ranks 等待          |
| 进程责任       | runner 启停 evaluator 子进程及 HTTP 服务；正常结束或可处理异常后完成回收，不留僵尸进程           |

下一步需确定首个任务及其数据文件、阶段 epoch、预算调度和评估频率。PushT 的预算数值、
epoch 和历史 checkpoint 仅作参考；本轮不重新设计 router/backbone 或要求重新收集数据。

### 6.2 实验顺序与完成标志


| 顺序                  | 工作                                                                                       | 完成标志                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 1. 训练入口与闭环检查 | 修正数据划分、统计和 256 A/B 配置；以短训练 checkpoint 核对 static all-fine 的独立 rollout | 训练可保存／恢复，输入、动作和 HTTP 路径对齐；每 episode reset 正常          |
| 2. static 完整配方    | 同任务、同数据下运行 all-fine 基线与 A → Spatial B → 下游适配；接入训练内同步评估        | 记录各阶段成功率、实际 patch 数与完整策略延迟，评估权重可追溯                |
| 3. 双视角与任务扩展   | 先加入 wrist，再扩展代表性任务，沿用相同评估协议                                           | 两视角链路一致；按各视角及总数量／成本比较，判断迁移收益                     |
| 4. Temporal 独立研究  | 补齐 LIBERO 配对接口，运行第 4 节三组对照                                                  | 在统一预算及训练安排下区分 Temporal 目标贡献，结合实际数量和训练成本解释结果 |

短训练用于检查接口，任务是否学会由正式闭环评测判断。开发时用少量固定初始状态；正式比较
统一任务、数据、评估初始状态、seed 集合和 checkpoint 选择规则，报告分任务成功率及跨 seed 波动。
先跑通一个任务，再据结果调整预算与训练长度，不同时展开完整预算、视角和 Temporal 矩阵。

### 6.3 必须保留的数据与推理契约

- **划分与统计：** 仅使用成功主文件，接受已标记纠偏轨迹，排除 `_failed.hdf5`。按
  `(源文件, demo_key)` 固定 seed 划分互斥 train/val，实际应用 episode 上限；不隐式丢弃跨纠偏
  action chunk。统计只来自训练部分，划分与统计随 checkpoint 保存，Stage B、resume 和评估复用。
- **观测映射：** `agentview_rgb → image`，可选 `eye_in_hand_rgb → gripper`，
  `ee_states → joint_state`；`gripper` 是 wrist 图像，`joint_state` 是末端 xyz＋axis-angle 的
  6 维位姿。在线分别取 `agentview_image`、`robot0_eye_in_hand_image` 与末端位置／四元数转换值；
  不拼接 force 或新增关节角输入。
- **图像与历史：** 统一 256 RGB、存储方向和 `/127.5 - 1` 缩放；val/在线关闭随机增强与 wrist
  dropout。协议按 view-major、各视角内由旧到新发送；历史不足重复首帧。分别定义 horizon、
  `n_obs_steps` 和执行 chunk，核对 MergeDataset 的 padding 与 action 起点。
- **归一化与动作：** 沿用 Dataset 缩放＋identity normalizer 时，adapter 使用训练统计归一化
  位姿并仅反归一化动作一次；统一常数维处理。返回 `predict_action()['action']` 的 `[B,K,7]`
  raw action，保留现有控制表示，由 evaluator 顺序执行，chunk 结束后再请求。
- **模型与配置：** 对齐当前 256、ResNet10 / node-aligned FPN 配方；Stage A 随机合法树、预算关闭，
  Stage B 加载同任务 A checkpoint、使用 reachable paired 和 learned routing。A/B 的划分、统计、
  相机与结构一致；低维 router 条件建议使用 `joint_state`，在配置阶段确定。
- **协议与评估：** 复用 `RoboKit/src/robokit/connects/protocols.py`，独立 checkpoint 服务和训练
  runner 共用 adapter；init 声明模型与输入要求，每 episode 重置 policy/history/action cache。
  首轮建议可选 PNG 无损传输，保留旧 JPEG 兼容；结束事件先于空图像解码处理。使用明确任务和
  benchmark 初始状态，关闭域扰动、demo 状态纠偏及 online update，成功判定先于超时归类。

256 / patch16 / 三层的范围为每视角 16–256 leaves，target=0.1 对应软目标 40，实际数量另行记录。
双视角报告每视角与总量；软预算不构成每帧硬限制。方法与日志口径沿用第 4 节引用文档。

### 6.4 同步评估的必要约束

已确认 rank 0 绑定当前 EMA、启动独立 LIBERO evaluator，首版串行处理一个 episode。
训练环境和模拟器环境通过协议通信，独立服务加载显式 checkpoint，避免评错权重。

- 服务 ready 后再启动指定 Python 的 evaluator；不 fork CUDA 模型、不启动多 worker，子进程
  不加入训练进程组。日志写文件，避免未消费 PIPE 阻塞。
- runner 集中清理：正常结束 `wait()` 回收；异常／超时先终止独立进程组，必要时 kill，最终
  `wait()`；随后停止 HTTP 服务并 join 线程。请求处理完全停止后才恢复模型训练。
- 清理后向全部 ranks 同步成功或失败状态；评估失败一致退出，不让其他 ranks 永久停在 barrier，
  也不把接口错误记成零成功率。评估时限须留出清理余量并小于 DDP 超时；强制终止由作业启动器兜底。
- 服务器验收覆盖连续两轮评估、启动失败、运行异常和超时／中断，检查无僵尸进程、端口释放、
  无旧请求访问 policy，以及多卡能一致恢复或退出。

### 6.5 预计文件范围与当前入口


| 范围                   | 预计新增／修改                                                                                                                                                                    |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 数据与配置             | `configs/liberoft_foveation_dp.yaml`、`diffusion_policy/dataset/libero_dataset.py`；按需扩展 `RoboKit/src/robokit/datasets/libero/libero_force.py` 的 episode 过滤                |
| A/B 启动               | `scripts/roi/sh_train_foveation_pusht_stageA.sh`、`sh_train_foveation_pusht_stageB.sh` 支持配置覆盖，默认仍为 PushT；新增薄 LIBERO wrapper，整理旧 `sh_train_foveation_libero.sh` |
| 训练与评估服务         | `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py`；新增通用 LIBERO adapter、独立服务及 `diffusion_policy/env_runner/libero_image_runner.py`                |
| 跨环境协议与 evaluator | `RoboKit/src/robokit/connects/protocols.py`、`LIBERO-FT/eval_socket.py`、`LIBERO-FT/libero/force/force_benchmark.py`                                                              |

**下一步从训练入口开始：** 确定一个任务，落实 episode 划分、统计持久化和 A/B 配置继承，再用
短训练 checkpoint 打通闭环。本轮规划不修改 replay 格式，也不提前实现 LIBERO Temporal。
代码实施时本地只做适用的静态检查，不新增独立测试脚本；训练、HTTP、DDP 和 MuJoCo rollout
在服务器验收。本次仅更新本文，未实施上述代码或运行实验。

## 7. Stage A 实施计划更新（2026-09-22）

本节记录最新讨论确定的实施计划；与第 6 节不一致之处以本节为准。当前仅记录计划，尚未修改相关代码。

### 7.1 实施范围、默认任务与配置约定

本轮代码实施目标是打通 **LIBERO Stage A 随机树训练 → 当前 EMA 的 HTTP 推理 → 独立
LIBERO 环境闭环评估 → 训练继续** 的完整流程。参考 PushT 已有训练、验证、采样可视化、
EMA、checkpoint、resume 和 rollout 流程，不只增加一个启动命令。Stage B、下游适配和
Temporal 保留为后续工作，不作为本轮 Stage A 的实施内容。

| 项目 | 本轮约定 |
| --- | --- |
| 默认任务 | `libero_10` 中的 `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`，即现有 YAML 任务列表的第一个任务；每个 policy 只训练一个任务 |
| 数据 | 已检查的 `libero_force_256/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_wrench.hdf5`；通过 config 切换任务和路径 |
| 观测 | 单帧 `n_obs_steps=1`，默认 static `image`＋6 维 `joint_state`；支持 config 启用 static+wrist，不使用 force |
| 时序与动作 | 沿用 `horizon=32`、`n_action_steps=8`、7 维动作；单帧观测不等于只预测或执行一个动作 |
| 模型 | 256 RGB、patch16、三层树、ResNet10 / node-aligned FPN，视觉编码与 policy 配方对齐当前 PushT Stage A |
| Stage A | 使用随机合法树训练视觉编码器与 diffusion policy，关闭预算损失和 Temporal，不进入 router-only paired 训练 |
| 同配置评估 | 训练与评估使用同一份解析后的 config，包括任务、相机、状态、结构、树模式、历史和动作 chunk；Stage A 评估保留配置指定的随机树模式 |
| 评估安排 | `training.rollout_every=100`，每轮 100 个 episode，包含 `epoch=0` 的首次评估 |

新增 `scripts/roi/sh_train_foveation_libero_stageA.sh`，直接使用并更新现有
`configs/liberoft_foveation_dp.yaml`，不另建一份容易漂移的 Stage A YAML。启动脚本参考
`sh_train_foveation_pusht_stageA.sh`，支持 GPU、seed、随机树和模型参数覆盖，以及通过
`"$@"` 传递 Hydra overrides；给必要环境变量提供默认值，保留单卡／多卡启动能力。
不要求为本轮改造 PushT A/B 脚本或新增 LIBERO Stage B wrapper。

Stage A 默认沿用 PushT 随机树组合 `fix_split_mode="random:0.0,0.25,0.5,0.75,1.0"`，
`random_topology_prob=0.0`、`budget_loss_weight=0.0`、`temporal_adaptive=false`；
随机树覆盖每视角 16–256 leaves，额外 topology dropout 关闭，避免引入仅训练时存在的
另一套随机机制。末端状态同时作为 policy 条件，并将已有 `router_condition_key` 对齐为
`joint_state`。需要 all-fine 对照时显式覆盖 `fix_split_mode=finest` 并单独标记实验，
runner 不擅自把随机树改为 all-fine 或 learned routing。

评估触发沿用 workspace 的 epoch 编号：`epoch % 100 == 0`，即 0、100、200……。
当前入口在每个 epoch 的训练循环后调用 rollout，因此 `epoch=0` 指首个编号为 0 的 epoch
训练完成后的首次评估，不另行解释为优化器更新前的初始模型评估。结果记录 epoch、
global step、seed 与 EMA 身份，resume 延续已有 epoch 进度。

### 7.2 HDF5 读取复用与必要的数据修改

**复用结论：** 现有 `LiberoFTDataset → LiberoH5FrameDataset → MergeDataset` 已能读取
replay HDF5 的图像、末端状态和动作，并构造历史与 action chunk。256 图像不需要重写
HDF5 reader、转换 replay 格式或重新采集数据；需要修正的是 episode 选择、增强开关、
时序对齐与统计保存。RoboKit 底层只按需增加可选 episode 过滤，并保持不传过滤条件时的旧行为。

- **文件与划分：** 只使用成功主文件，排除 `_failed.hdf5`；保留已经标记的纠偏轨迹，
  不默认剔除跨纠偏点的 action chunk。按 `(源文件, demo_key)` 和固定 seed 生成互斥
  train/val 清单，在构建窗口和计算统计前应用划分；`max_train_episodes` 真正限制训练
  episode，未选中的训练候选不自动混入验证集。
- **验证增强：** 训练增强由 config 控制；val 与在线推理关闭 color jitter、随机裁剪、
  random shift 和 wrist dropout。验证集复用训练统计，不能重新读取同一批训练 episode
  充当验证集，也不能仅关闭 color jitter 便认为验证已确定化。
- **观测映射：** `agentview_rgb → image`，可选 `eye_in_hand_rgb → gripper`，
  `ee_states → joint_state`。`gripper` 指 wrist RGB，`joint_state` 指 xyz＋axis-angle
  的 6 维末端位姿；在线从末端位置与四元数按同一约定转换，不添加关节角或力输入。
- **图像与窗口：** 保持 replay 的图像方向、RGB 顺序及 `/127.5 - 1` 缩放；核对在线
  `agentview_image`、`robot0_eye_in_hand_image` 与离线存储方向，避免重复翻转。
  单帧配置下 `pad_before=0`，核对 MergeDataset 的末尾 padding 和 action 起点，
  确保窗口不跨 episode，且 `predict_action()['action']` 对应当前观测后的执行片段。
- **统计与恢复：** action 的 min/max 和末端状态的 mean/std 只来自最终训练划分。
  保留 Dataset 缩放＋identity normalizer 的现有路线，将 train/val 清单、统计和必要
  数据元信息随 checkpoint 持久化；resume、后续 Stage B、独立服务与训练内评估读取
  相同内容，不从验证集或评估环境重算。统一常数维处理，动作只反归一化一次。

### 7.3 当前 EMA 服务、跨环境 evaluator 与通信流程

rank 0 在训练进程内通过 HTTP API 提供**当前内存中的 EMA policy**，复用
`env_runner.run(policy_rollout)` 入口；不为了评估重新扫描目录、加载旧 checkpoint 或复制
一份可能过期的模型。评估期间模型处于 eval 模式且禁用梯度，暂停训练更新；runner 与独立
checkpoint 服务共用 LIBERO adapter，后者必须加载显式指定的 checkpoint、config 和统计。

runner 使用配置指定的 LIBERO Python 解释器启动子进程，工作目录设为 **`LIBERO-FT`**，
直接运行并扩展该目录下的 `eval_socket.py` 和
`libero/force/force_benchmark.py` 中的 `LiberoForceSocketEvaluator`。不另写一套平行的
evaluator，也不把 MuJoCo/LIBERO 依赖引入训练进程。新增参数接收明确任务、suite／数据路径、
256 分辨率、相机列表、观测帧数、动作 chunk、100 次评测、seed、步数上限、API 地址及结果
路径；参数从训练 config 传递，避免继续使用硬编码 128 图像、双视角和旧任务映射。
保留原 CLI 参数、旧 task-id 映射及原调用方式的默认行为，新 Stage A 模式显式覆盖这些值。

复用 `RoboKit/src/robokit/connects/protocols.py` 的请求／响应模型，按以下顺序通信：

1. rank 0 绑定 EMA 和训练统计，启动单 worker 的 HTTP 服务，确认 ready 后再启动 evaluator。
2. evaluator 调用 `/init` 获取相机及顺序、分辨率、`n_obs_steps`、状态要求、动作维度和
   `max_cache_action` 等契约；新增元信息兼容旧客户端，Stage A 客户端核对其与 config 一致。
3. 每个 episode 初始化 benchmark 状态后调用 `/reset`，同时清空 policy 状态、服务端
   历史和 evaluator 的图像历史、动作缓存、计数器，不能只在整轮评估开始时 reset 一次。
4. 按 view-major、每视角内由旧到新的顺序发送 RGB，协议 shape 为 `[B,V*T,H,W,3]`，
   原始末端状态为 `[B,T,6]`；首轮 `B=1,T=1`，视角数来自 config。通用历史路径在帧数
   不足时重复首帧。adapter 拆分视角、转为 policy 的 `[B,T,C,H,W]` 并应用训练时缩放。
5. adapter 调用 `predict_action()`，取 `['action']` 并仅反归一化一次，返回 raw
   action `[B,K,7]`；evaluator 顺序执行 K 个动作，chunk 结束后请求下一次观测，
   episode 成功或结束时立即停止执行剩余缓存动作。
6. episode 结束发送 `ROLLOUT_END`，服务先判断结束事件再尝试解码图像，允许空图像载荷；
   evaluator 写出逐 episode 结果和汇总文件，runner 校验完成数量后记录成功率。

RoboKit 增加**可选 PNG 无损传输**：编码端可选择 PNG，解码端同时支持 PNG/JPEG，普通
解码和 buffer 解码路径一致；旧请求和默认 JPEG 用法继续可用。Stage A 显式选择 PNG，
避免压缩改变输入；不改动原有动作语义或要求旧客户端提供新增字段。

评估使用明确任务的 benchmark 初始状态，固定并记录 seed／初始状态索引。100 次 rollout
不宣称是 100 个不同的初始状态，必要时按固定规则复用 benchmark 状态并记录对应关系。
关闭 online update、域扰动和 demo 状态纠偏；每步先检查成功再归类超时，记录分任务
成功率、实际 episode 数和失败原因。接口错误、子进程失败或缺失结果不能记为零成功率。

### 7.4 子进程回收与多卡同步

首版串行运行一个 episode，由 rank 0 独占当前 EMA 的推理；其余 ranks 等待评估和清理
完成，不进入下一轮训练或验证 collective。子进程通过独立进程组启动，不 fork CUDA 模型，
不加入训练进程组；清除继承环境中会触发分布式初始化的训练 rank 信息。stdout/stderr
写入评估日志文件，避免未消费的 PIPE 阻塞，并将日志和结构化结果放在本轮 epoch 目录。

runner 集中管理服务与子进程生命周期，正常或可处理异常都执行清理：

- 正常结束后 `wait()` 回收 evaluator；启动失败、运行异常、超时或中断时，先终止独立
  进程组，必要时 kill，最终 `wait()` 回收。即使 evaluator 尚未成功启动，也关闭已启动服务。
- 停止 HTTP 服务并 join 线程，确保请求处理完全结束、端口释放，再恢复 policy 模式和训练；
  不让旧请求在下一轮训练或评估时继续访问模型。
- rank 0 捕获可处理的评估失败，完成清理后向全部 ranks 同步成功／失败状态；成功时一致
  继续，失败时一致报错退出，不能 rank 0 提前抛出而使其他 ranks 永久停在 barrier。
- 配置服务启动、请求、整轮评估和退出超时；整轮 100 次评估时限加清理余量应小于 DDP
  通信超时，必要时调整进程组超时。不可捕获的强制终止由作业启动器负责终止其余进程。

### 7.5 预计文件范围、实施顺序与验证标准

下表为**后续代码实施**预计范围；本次实际只向 `plan/libero_exp.md` 追加第 7 节。

| 范围 | 预计新增／修改及目的 |
| --- | --- |
| Stage A 入口 | 新增 `scripts/roi/sh_train_foveation_libero_stageA.sh`，参考 PushT 完整流程并传递配置覆盖 |
| 现有 LIBERO 配置 | 修改 `configs/liberoft_foveation_dp.yaml`，统一 256、单任务、单帧状态条件、随机树和评估配置 |
| 数据集 | 修改 `diffusion_policy/dataset/libero_dataset.py`，落实互斥划分、训练上限、确定性验证及训练统计复用 |
| HDF5 底层 | 按需修改 `RoboKit/src/robokit/datasets/libero/libero_force.py`，为现有 reader／MergeDataset 增加可选 episode 过滤，不重写读取链路 |
| Workspace | 修改 `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py`，持久化数据契约、复用 EMA rollout 入口并同步多卡评估状态 |
| Runner 与 adapter | 新增 `diffusion_policy/env_runner/libero_image_runner.py` 及共用 LIBERO adapter，管理服务、预处理、动作反归一化与 evaluator 子进程 |
| 独立服务 | 新增适配 foveation policy 的独立 checkpoint 服务，复用 adapter；现有 `service/gpu_service_roi_libero.py` 仅作为 HTTP 结构参考 |
| LIBERO evaluator | 修改 `LIBERO-FT/eval_socket.py`、`LIBERO-FT/libero/force/force_benchmark.py`，扩展既有入口与 `LiberoForceSocketEvaluator`，保留旧用法 |
| RoboKit 协议 | 修改 `RoboKit/src/robokit/connects/protocols.py`，增加可选 PNG 并兼容 JPEG 和结束事件 |
| 实验文档 | 在 `plan/libero_exp.md` 第 6 节后追加本节记录，保留第 1–6 节，不再使用“修改第 6 节”的条目 |

实施顺序为：先落实入口／config 和数据契约，再完成共享 adapter 与现有 evaluator 扩展，
最后接入 runner、统计恢复和多卡同步。使用同一 config 检查训练、验证、独立 checkpoint
服务和训练内 EMA 评估的输入及动作语义，再开展正式随机树 Stage A 训练。

后续代码实施的验收标准如下：

- **本地静态检查：** 对改动的 Python 文件做静态编译、shell 做语法检查、YAML 做解析
  检查，并运行 `git diff --check`；不新增独立测试脚本，缺少训练依赖时明确未运行项。
- **服务器数据与恢复：** 确认 train/val episode 互斥且上限生效、统计仅含训练数据，验证
  预处理无随机增强；保存／resume／独立服务读取相同划分和统计，图像与位姿／动作缩放一致。
- **服务器闭环：** 完成单任务随机树训练、checkpoint 保存／恢复和 rollout；确认 static
  与配置启用的双视角均遵循同一输入契约，每 episode reset、action chunk 和 PNG/JPEG
  兼容正常。首轮 `epoch=0` 及之后每 100 epoch 均完成 100 次评估，EMA 权重身份可追溯。
- **服务器生命周期与多卡：** 连续两轮评估后仍能恢复训练；覆盖服务或子进程启动失败、
  运行异常、超时／中断，确认无僵尸进程、端口释放、无旧请求访问 policy，各 ranks 一致
  继续或退出，并验证旧 evaluator 调用方式仍可使用。

**本次文档更新的验证边界：** 只检查 Markdown 章节、表格、代码块和 `git diff --check`。
本次不修改训练、配置、数据集、服务、LIBERO-FT 或 RoboKit 代码，不运行训练或新增测试；
上面的静态编译与服务器验收均属于后续代码实施工作，不能视为已经完成。

## 8. Stage A 代码接入记录（2026-09-22）

本节记录第 7 节之后确定的实施细化；范围冲突时以本节为准。本轮仅实现训练内评估，
取消独立 checkpoint 服务与独立 adapter 文件。训练侧新增
`diffusion_policy/env_runner/libero_socket_runner.py` 中的 `LiberoSocketRunner`，它只负责
HTTP 与外部进程管理；环境创建、episode 循环、动作执行和成功判定仍复用 LIBERO-FT 的
`LiberoForceSocketEvaluator`。

### 8.1 启动与评估配置

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/roi/sh_train_foveation_libero_stageA.sh \
  libero_root=/home/geyuan/code/LIBERO-FT \
  task.env_runner.conda_env=libero
```

默认使用炉灶单任务、256 RGB、单帧 static＋6 维末端位姿、随机合法树，预算和 Temporal
关闭。默认 3000 epoch，验证／保存／采样每 10 epoch；每 100 epoch 评估 100 个 episode，
每个最多 400 个动作步。`epoch=0` 是首个训练 epoch 完成后的首次评估。
启用 wrist 时在同一 YAML 的 `shape_meta.obs` 中加入 `gripper` RGB 项；相机列表从该配置推导。

无需手动启动 API 或 `eval_socket.py`。rank 0 服务当前内存中的 EMA，通过
`conda run --no-capture-output -n libero python -u eval_socket.py --eval-config ...`
启动子进程，`cwd` 为配置中的 LIBERO-FT 根目录。两个 Python 环境均需安装更新后的 RoboKit。
`task.env_runner.conda_executable` 可显式设置 Conda 路径；否则使用 `CONDA_EXE` 或 PATH。

HTTP 默认为 `127.0.0.1:0`，由系统分配端口，runner 将实际地址传给 evaluator。
固定端口可用 `task.env_runner.port=7077` 覆盖；它与训练通信的 `MAIN_PROCESS_PORT` 独立。
`startup_timeout=60`、`request_timeout=120`、`eval_timeout=7200`、`shutdown_timeout=30`
均以秒计；`training.ddp_timeout_seconds=10800` 为评估和清理留出余量。

### 8.2 数据与文件责任

| 文件／class | 本轮责任 |
| --- | --- |
| `scripts/roi/sh_train_foveation_libero_stageA.sh`、`configs/liberoft_foveation_dp.yaml` | 单卡／多卡入口、Stage A 配方及 Socket Runner 配置 |
| `diffusion_policy/dataset/libero_dataset.py`：`LiberoFTDataset`、`VideoTransformTHWC` | train/val 互斥、episode 上限、验证增强关闭、训练统计与共享缩放 |
| `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py`：`TrainFoveationDPTransformerWorkspace` | 先恢复数据元信息再建 dataset、checkpoint 保存元信息、EMA 评估与多卡失败同步 |
| `diffusion_policy/env_runner/libero_socket_runner.py`：`LiberoSocketRunner` | 同文件实现 HTTP 接口、输入转换、Conda 子进程、结果检查和回收；不运行模拟器 |
| `RoboKit/src/robokit/datasets/libero/libero_force.py`：`LiberoH5FrameDataset` | 可选 demo 过滤；既有 `MergeDataset` 复用所选 episode 的索引和统计 |
| `RoboKit/src/robokit/connects/protocols.py`：`StepRequestFromEvaluator` | 可选 PNG，保留默认 JPEG，普通和 buffer 解码兼容 |
| `LIBERO-FT/eval_socket.py`：`main()` | 可选 JSON 配置入口，保留旧 CLI |
| `LIBERO-FT/libero/force/force_benchmark.py`：`LiberoForceSocketEvaluator` | 复用原 episode 循环，分离观测历史与动作缓存、每 episode reset、成功优先及结果输出 |

训练划分及统计保存为 checkpoint 的 `dataset_metadata`，并写入运行目录的
`dataset_metadata.json`；resume 和 pretrained 加载复用该信息，缺失时明确报错。
统计只来自选中的训练 episode。验证关闭随机裁剪、random shift、color jitter 和 wrist
置零；在线使用训练统计，动作只反归一化一次。图像保持 replay 方向，策略输入不额外翻转。

每轮评估写入 `eval/epoch_XXXX/`：`eval_config.json`、`evaluator.log`、`results.json`。
配置记录 policy 配方、实际 API 地址、epoch 和 global step；结果记录任务、EMA 身份、
逐 episode seed／初始状态索引、步数和结束原因。训练日志写入 `test/mean_score` 与
`test/n_episodes`，接口错误不会转换成零成功率。

正常和可处理异常都回收 Conda/evaluator 进程组、关闭 HTTP 服务；清理完成后向所有 ranks
同步状态，一致继续或退出。不可恢复的 CUDA 卡死或强制终止由作业启动器兜底。

### 8.3 验证边界

本机只做 Python 静态编译、shell 语法、YAML／OmegaConf 插值、配置参数与 constructor
签名核对，以及改动文件的 diff 检查；不新增独立测试脚本。不将这些检查表述为完成训练、
HTTP、DDP 或 MuJoCo 验证。服务器仍需验收单／双视角、连续两轮评估、checkpoint 恢复、
旧 CLI，以及启动失败、请求异常和超时清理。

## 9. Stage A 多环境并行评测与 W&B 项目分离（2026-09-22）

本节记录第 8 节之后的代码更新；评测并行与日志项目配置以本节为准。训练内 EMA
评测默认使用 8 个模拟器进程，W&B project 改为 `diffusion_policy_libero`。
PushT 配置保持原值，已有 W&B run 不迁移。workspace 直接读取 `logging.project`，
训练脚本已支持配置透传，因此无需修改 workspace 或脚本。

### 9.1 配置与依赖

```yaml
logging:
  project: diffusion_policy_libero

task:
  env_runner:
    n_envs: 8
    n_test: 100
    max_steps: 400
```

`n_envs` 的 constructor 和旧评测 JSON 缺省值为 1，Stage A YAML 显式设为 8。
可通过 `task.env_runner.n_envs=1` 回退原串行路径。并行评测仅支持 Stage A 配置入口；
旧 CLI、扰动评测沿用原逻辑。每 100 epoch 评测一次、首轮 epoch=0、100 个 episode
和单 episode 最多 400 个动作步的约定保持不变。

复用 LIBERO-FT 已有的 `SubprocVectorEnv`，依赖标准库 `multiprocessing` 和既有
`gym==0.25.2`、`cloudpickle==2.1.0`，不新增并行框架。配置入口使用 `spawn`，模拟器在
worker 内创建，保留服务器现有渲染设备配置。

### 9.2 并行调度与通信

每轮创建 `min(n_envs, n_test)` 个 worker，跨批次复用。100 个 episode 按 12 批 8 个、
最后一批 4 个执行，不补充虚假 episode。全局 episode 编号决定 seed
（`test_start_seed + episode`）和初始状态索引（`episode % len(init_states)`）；
保留原有 5 步 settling，不计入动作步数。

各环境维护独立观测历史，合并图像 `[B,V*T,H,W,3]` 和位姿 `[B,T,6]` 后，调用 rank 0
上的同一个 EMA HTTP API，返回 `[B,K,7]` 动作。环境协调器缓存 action chunk，每个
动作步仅推进尚未结束的环境。已结束槽位保留最后观测，后续预测动作丢弃，整批结束后
统一重置 policy。成功优先于同一步达到最大步数。

并行路径在每个 episode 结束时调用私有 `POST /episode_end`，上报 episode 编号、
seed、初始状态索引、步数、success 和 reason，仅更新训练终端 tqdm，不重置 policy。
整批结束时发送 `ROLLOUT_END`，仅重置 policy，不重复计数。串行路径保留原有
`ROLLOUT_END` 的计数和 reset 行为。RoboKit 现有批量编码协议无需修改。

`results.json` 按全局 episode 编号排序，额外记录配置的 `n_envs`。runner 继续检查
结果数量、完整编号、epoch/global step 及并行数量；最终文件作为统计依据。
不同并行数保持 episode seed 和初始状态映射一致，但批量大小改变随机树与扩散采样
的随机数使用顺序，不承诺动作或成功率逐位一致。

### 9.3 文件与生命周期

| 文件／类 | 本轮修改 |
| --- | --- |
| `configs/liberoft_foveation_dp.yaml` | `task.env_runner.n_envs: 8`；`logging.project: diffusion_policy_libero` |
| `diffusion_policy/env_runner/libero_socket_runner.py`：`LiberoSocketRunner` | 批量输入输出检查、传递并行数量、episode 完成通知、区分进度与整批 reset |
| `LIBERO-FT/eval_socket.py`：`_run_configured_eval()` | 并行配置、spawn 启动方式、结果记录及统一资源关闭 |
| `LIBERO-FT/libero/force/force_benchmark.py`：`LiberoForceSocketEvaluator`、私有 `_VectorForceEnv` | 复用原模拟器和观测转换，新增批次循环、独立历史、active 环境推进及回收 |
| `plan/libero_exp.md` | 追加本节记录，保留前面章节 |

正常或可处理异常时关闭环境池和 HTTP session。所有 worker 留在 evaluator 进程组内，
沿用 runner 的超时和进程组终止机制；worker 卡死或清理阻塞由评测总超时兜底。
清理后才向其他 ranks 同步评测状态，不将异常转换为零成功率或部分成功结果。
既有 request/eval/shutdown 和 DDP timeout 保持不变。

### 9.4 验证边界

本机仅进行 Python 静态编译、YAML／OmegaConf 解析、配置与 constructor 核对、Markdown
及 diff 检查，不新增测试脚本，不运行训练、HTTP、DDP 或 MuJoCo 验证。

服务器后续需验收：单环境及旧 CLI 兼容；8 环境完成 100 次评测及最后 4 个尾批；提前
结束的环境不再推进且不干扰其他槽位；单／双视角及 action chunk 正确；连续两轮评测
后恢复训练；worker／HTTP 异常和超时清理；新的 LIBERO run 上传至独立 W&B 项目。
用同一 checkpoint 对比 1 与 8 环境的耗时、吞吐和资源占用，不预设线性加速目标。

### 9.5 评测期间的 GPU 等待与渲染设备配置

后续排查发现，其他训练 ranks 提前进入默认 NCCL 的状态广播时，可能在整个 rank-0
评测期间运行等待 kernel，造成 GPU 高利用率，并与模拟器渲染竞争资源。
`TrainFoveationDPTransformerWorkspace` 现为 rollout 创建独立 Gloo 通信组，所有 ranks
在相同位置创建和销毁该组；评测前同步各自的 CUDA 工作，评测结果和异常通过 CPU
状态广播传播。训练梯度通信仍使用原通信组，评测清理和错误同步语义保持不变。

新增 `task.env_runner.egl_device_id: null` 和 `render_gpu_device_id: null`：前者显式
设置 evaluator 子进程的 `MUJOCO_EGL_DEVICE_ID`，后者经 JSON 和 evaluator 传给
`SingleForceEnv`，覆盖 HDF5 的同名渲染参数。null 保留已有环境变量或 HDF5／后端选择；
未指定 `MUJOCO_GL` 时子进程默认使用 egl，显式设置 EGL device 时强制使用 egl。
环境变量在启动 Python 子进程前设置，spawn workers 继承相同设置。

EGL 枚举编号与 CUDA 逻辑编号／物理 GPU 编号不保证一致，不自动用 rank 或 CUDA 编号
推算。用户需依据服务器实际 robosuite／EGL 后端确认编号和参数优先级。
runner 在终端及 `eval_config.json` 记录 policy device 和渲染环境设置，`SingleForceEnv`
在 evaluator 日志记录 worker PID 和请求的渲染参数；这些是请求配置，不声称验证了
实际物理设备映射。更新文件包括 workspace、runner、LIBERO YAML、`eval_socket.py`、
`force_benchmark.py` 和本记录。

应重启训练任务加载修复，先保持设备参数为 null 观察 CPU 等待带来的变化，再按已确认
的映射显式绑卡。首个 episode 的 tqdm 估计包含环境初始化，且整批并行会集中完成多个
episode，不能用 `1/100` 时的 ETA 判定总耗时。服务器验收应检查至少前两批的耗时、
非推理／非渲染 GPU 的利用率、实际渲染进程位置及连续评测后恢复训练。
本机仅做静态检查，未验证服务器 GPU 利用率、设备映射或性能改善。

## 10. Stage B：Spatial paired、下游适配与完整评测视频

### 10.1 配方与启动

新增 `scripts/roi/sh_train_foveation_libero_stageB.sh`，必须设置 `PRETRAINED_CKPT`，
使用 Stage A EMA、checkpoint 的数据划分与归一化统计，不重建 policy service。
首版仅接入 Spatial，曾在 workspace 构建模型前拒绝 `policy.temporal_adaptive=true`。
2026-09-24 已接入 LIBERO Temporal，当前入口与验收边界见第 10.1.1 节；
下面的默认值表及 epoch 900 切换描述保留首版历史记录，当前参数以脚本和 YAML 为准。

| 项目 | Stage B 默认值 |
| --- | --- |
| 总训练与阶段 | 1500 epoch（0–1499）；0–899 routing；900–1499 下游适配 |
| 预算调度 | epoch 0–900 余弦下降，归一化目标 0.75 → 0.1，即每视角软目标 196 → 40 |
| Spatial | paired、reachable，权重 1.0；learned routing，random_topology_prob=0 |
| 预算 | upper_bound_lagrangian_elu；weight=0.1，base cost=0，tau=0.025 |
| lambda | 初值 0，学习率 0.01，复用多卡统计与 checkpoint 恢复 |
| 模型 | 4 层视觉编码器、CLS pooling、conv patch embedding、adapter channels=16 |
| 在线评测 | 每 100 epoch，100 episodes，8 环境，每 episode 最多 400 步 |
| checkpoint | `test_mean_score` 最大化 top-3，文件名同字段，继续保存 last |
| 视频 | 前 4 个 episode，20 FPS、CRF 22，离线路由 batch size 16 |
| W&B | diffusion_policy_libero |

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PRETRAINED_CKPT=/path/to/stageA/checkpoints/epoch=xxxx.ckpt \
bash scripts/roi/sh_train_foveation_libero_stageB.sh \
  libero_root=/path/to/LIBERO-FT task.env_runner.conda_env=libero
```

脚本支持单卡／多卡、`MAIN_PROCESS_PORT`，以及总 epoch、下游起点、预算起点／终点／
衰减长度、paired／预算参数和录像参数的环境变量。末尾 Hydra 参数可以覆盖任务、
渲染设备、训练和评测设置。共享 YAML 默认仍关闭下游切换、预算调度、视频和在线
patch 统计，保留 Stage A 行为。epoch 0 在首轮训练后评测；epoch 900 训练前切换，
复用当前 optimizer 和 LR scheduler，不重置优化进度。

### 10.1.1 LIBERO Temporal 接入（2026-09-24）

Stage B 现支持 `TEMPORAL_ADAPTIVE=true`，复用 PushT 的 reachable 完整子树、随机锚点
等 patch 增量匹配和 Temporal loss。默认仍为 false；`TEMPORAL_LOSS_WEIGHT` 默认 1.0。
权重为 0 仍读取双窗口并执行主动匹配，不能视作原单窗口 Spatial 基线。

`LiberoFTDataset.set_temporal_pairing(enabled)` 控制配对：A 是原样本，B 从同一 HDF5、
同一 demo 的其他真实决策时刻均匀采样，可早可晚，不限制时间间隔。区间由 MergeDataset
及 reader 的 prefix/demo_lengths 构建，按 max_len 裁剪，不跨文件或数据分区。
单帧有效区间复用 A 的已增强数据作占位，`temporal_pair_valid=false`。
历史和动作窗口继续采用原有边界重复，不因为窗口含 padding 而排除真实决策帧。
A/B 各自增强、各自动作目标；同一窗口 stop/split 两臂共享增强、noise 和 mask。

当前 Stage B 脚本使用 `two_by_two`：训练 B_train、验证 A_val，沿用 Stage A 保存的
归一化统计；Temporal 不重划数据、不改变 metadata 或 checkpoint 参数结构。
共享内存开关支持 persistent workers，索引在 workers 启动前准备。router 阶段开启，
下游适配阶段关闭；resume 根据已恢复阶段和 epoch 初始化。验证、在线评测和视频保持
单窗口，不要求未来观测，不改变 RoboKit 协议。

训练 batch 不自动减半。默认 batch=64 时，开启后处理 64 对、128 个窗口；预算统计
计入双窗口，继续复用多卡汇总和 lambda 更新。需要时通过末尾 `dataloader.batch_size=32`
手动调小。当前脚本默认 300 epochs、无下游切换，预算在 200 epochs 内从 0.75 衰减到
0.1；128 图像、patch16、3 levels 对应每视角软目标 49 → 10，不是严格数量。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PRETRAINED_CKPT=/path/to/stageA/checkpoints/latest.ckpt \
TEMPORAL_ADAPTIVE=true TEMPORAL_LOSS_WEIGHT=1.0 \
bash scripts/roi/sh_train_foveation_libero_stageB.sh \
  libero_root=/path/to/LIBERO-FT task.env_runner.conda_env=libero
```

三组对照保持同一 checkpoint、seed、预算、batch 和数据设置：单窗口 Spatial 使用
`TEMPORAL_ADAPTIVE=false`；双窗口匹配消融使用 true、weight=0；完整 Temporal 使用
true、weight=1。horizon、视觉结构等仍需与所加载 Stage A checkpoint 匹配。
保持 batch 相同意味着双窗口组每 epoch 处理更多窗口，解释消融时需说明此计算量差异。

日志复用 `train/temporal_loss`（全部配对的未加权平均）和
`train/temporal_match_fraction`（有效配对内成功匹配比例）。匹配失败或无效配对的
Temporal 贡献为零，Spatial 和预算仍保留。高匹配率不说明收益排序正确；Temporal
继续依赖指示模型的局部相对收益，不能自动修复分辨率损失反转。实验还需比较成功率、
实际 patch 数及预算下的分配质量，不能只凭拓扑多样性判定有效。

涉及文件：`diffusion_policy/dataset/libero_dataset.py`、
`diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py`、
`configs/liberoft_foveation_dp.yaml`、`scripts/roi/sh_train_foveation_libero_stageB.sh`、
本文及 `plan/study_temporal_adaptive.md`。policy、FoveationNet 和外部 RoboKit 无需改动。

本地验证范围仅为 Python 静态编译、bash 语法、配置接口及 diff 检查；不新增测试脚本，
不运行训练或仿真。服务器仍需验收：同 demo 不同时间的配对、多文件同名 demo 隔离、
max_len 截断及单帧占位、验证集隔离、单/双视角和 n_obs_steps>1、单/多卡统计、
persistent workers 下阶段切换及各阶段 resume，以及成功率、在线 patch 数和视频。

### 10.2 Stage A 配置检查

`TrainFoveationDPTransformerWorkspace._load_pretrained_weights()` 在 LIBERO router-only
加载 EMA 前比较 policy、foveation net、noise scheduler 的结构与推理配置，以及相对
数据来源、shape_meta、horizon、padding 和归一化契约。旧配置省略的可选参数使用对应
组件 constructor 默认值；不同字段逐项报告 A/B 值。保留严格 state dict 加载和 dataset
metadata 校验。允许变更数据根目录、routing 模式、冻结状态、预算、优化器、训练时长
及日志配置。Stage A `fix_split_mode` 写入启动日志，固定树 checkpoint 可加载，但提示
未经过多分辨率联合训练。

### 10.3 实际在线 patch 统计

runner 使用本次 `predict_action(..., return_foveation_outputs=True)` 返回的
`num_patches_B_T`，在 action 响应旁添加可选 `num_patches: {rgb_key: [B,T]}`。
不额外运行 router；旧客户端仍可只读取 action。

evaluator 仅在请求新 action chunk 时累计，缓存动作执行期间不重复统计，排除已经结束
的并行 slot；尾批使用实际 batch size。每个 episode 的 `patch_counts` 按视角保存 calls、
observation_count、patch_sum、mean。`results.json` 顶层同时保存统计口径与累计值。
重复出现在不同推理历史窗口的观测按实际被推理次数计入，reset padding 也属于实际输入。
总体均值按有效推理观测数加权，不直接平均 episode 均值。

| 日志 | 口径 |
| --- | --- |
| `test/num_patches_mean` | 所有视角、所有有效推理观测的 hard leaf 均值 |
| `test/num_patches_mean_<rgb_key>` | 对应视角的加权均值 |
| `test/total_num_patches_mean` | 同一推理观测的各视角 hard leaf 总数均值 |

启用统计却缺少字段或 shape 不符时明确报错。软目标不代表实际 hard leaf 数；视频重算
数量不会进入这些在线指标。本轮不新增耗时指标。

### 10.4 完整观测与离线视频

配置驱动的 evaluator 独立于旧 CLI 视频／扰动选项，记录 episode 0–3（不足 4 个则全部），
成功和失败均保留。序列包含 reset 稳定后的初始观测、每个实际执行动作后的观测及终止帧，
不记录终止后的冻结 slot。RGB 保持模拟器原始方向，配对同一时刻的 6D 末端位姿；临时 NPZ
保存在 `eval/epoch_XXXX/observations/`，results 的 episode、seed、observation_path 关联记录。

训练侧在 evaluator 和 HTTP 服务退出后逐 episode 渲染。图像采用在线输入尺寸，位姿使用
同一统计归一化，重建逐步观测历史并在开头重复首帧。仅调用同一 EMA 的
`get_foveation_outputs()` 和 `vis_ar_tokens()`，不运行 diffusion 动作预测。
`torch.random.fork_rng` 隔离路由随机数，Python／NumPy 状态也恢复；其他 ranks 在原 Gloo
边界等待视频完成后恢复训练。录像开关不向在线推理增加随机操作。

原始与四叉树 MP4 使用相同完整帧序列，写入 `eval/epoch_XXXX/media/`。展示方向一致，
`vis_rot_image` 仅将静态相机 image 旋转 180 度，与 workspace 约定一致，不改变输入。
从 PushT 提取的共享 `draw_patch_history()` 只显示截至当前帧的数量历史；PushT 保留
`_draw_patch_history` 入口与原绘制行为。视频成功生成后删除对应 NPZ；失败时保留 NPZ
便于排查，并关闭视频资源。results 中的临时路径在成功清理后不再存在。

W&B 使用 `test/sim_video_<seed>` 与 `vis/test_foveation/video_<seed>`。四叉树 caption
包含 `recorded-observation routing`、episode、seed、success，说明它是当前 EMA 对录像
观测的重算，不能解释为每帧的树均参与了实际 action chunk 生成。

### 10.5 文件与验证边界

本轮文件：Stage B shell、LIBERO YAML、训练 workspace、LIBERO socket runner、共享
`foveation_video_util.py`、PushT runner、LIBERO-FT 的 `eval_socket.py` 和
`libero/force/force_benchmark.py`，以及本节。保留两个仓库已有的无关改动。

本机完成 shell 语法、Python 静态编译、Hydra Stage A／Stage B／尾部覆盖组合与插值解析、
阶段切换和预算终点一致性、top-k 指标／方向／文件名及两个仓库的 diff whitespace 检查。
未新增独立测试脚本，未启动训练、socket 服务、MuJoCo 或实际视频编码／W&B 上传。
未定义符号检查：本次新增代码无报错；LIBERO-FT HEAD 已有旧路径的 `eval_socket.py`
中 `policy` 和 `force_benchmark.py` 中 `make_env` 两处未定义引用，本轮未修改这些路径。

服务器后续验收（本机未执行）：

- 同任务 Stage A checkpoint 可启动；任务、结构或数据契约不符时逐字段拒绝。
- routing 阶段动作路径固定；900 切换后 routing 参数／状态固定，下游更新；resume 恢复阶段与 lambda。
- 1 环境、8 环境、提前成功、最后不足一批的 episode，在线统计均只计 active 推理。
- 固定粗／细树诊断下，单视角 hard leaf 数分别为 16／256。
- 前 4 个 episode 的原始／四叉树视频覆盖完整操作、帧数为 steps+1、包含终止状态，W&B 前缀正确。
- 相同评测配置开关录像不改变动作和成功结果；连续两轮评测恢复训练，异常路径能回收模拟器、HTTP 和视频资源。


### 10.6 自动匹配训练与 EGL 物理设备

`task.env_runner.auto_render_device=true` 为默认值。四个设备配置字段均为 null 时，
runner 从实际 EMA 的 CUDA 逻辑设备通过 CUDA Driver API 查询 PCI bus ID 和物理 GPU UUID，随
`render_device_pci_bus_id`、`render_device_uuid` 写入 evaluator JSON，不根据 rank 或可见设备字符串猜编号。
串行评测使用执行评测的 EMA 所在卡；分布式评测使用各 rank 自己的 EMA 所在卡。
`training.distributed_rollout` 仍默认 false，开启后不再要求手工提供设备映射。

单卡显式 `egl_device_id` 或 `render_gpu_device_id` 沿用原手工行为；分布式手工覆盖必须
同时填写 `egl_device_ids` 和 `render_gpu_device_ids`，按 local rank 排列。只提供一份列表
会报错。`auto_render_device=false` 完全关闭自动路径，恢复原来的设备配置要求。

自动模式仅为 evaluator 子进程设置空的 `CUDA_VISIBLE_DEVICES`、`MUJOCO_GL=egl` 和
`PYOPENGL_PLATFORM=egl`，清除继承的 `MUJOCO_EGL_DEVICE_ID`。训练进程环境不变。
evaluator 通过 HTTP 请求训练进程推理，本地 rollout 使用 CPU；空 CUDA 列表避免
robosuite 1.4.0 将 EGL 编号与 CUDA 可见列表混用的导入检查，不负责选择 EGL 设备。

LIBERO-FT 的 `libero/force/render_device.py` 使用 MuJoCo 的 EGL binding 枚举设备，在提供
`EGL_NV_device_cuda` 的 NVIDIA 入口中，优先用 `EGL_EXT_device_persistent_id` 的设备 UUID
匹配 CUDA UUID。缺少 UUID 时才使用 NVIDIA 入口的 DRM/sysfs PCI 身份；不会仅因为
Mesa 入口指向相同 PCI 地址就选中它。唯一匹配后先验证 `eglInitialize` 和 NVIDIA vendor，
再在导入模拟器前设置 `MUJOCO_EGL_DEVICE_ID`，并将同一 EGL index 通过现有
`render_gpu_device_id` 参数覆盖 HDF5 的设备设置。spawn workers 继承相同环境。
匹配失败或有歧义直接报错，不回退到 GPU 0；错误沿已有 rollout 错误同步路径传播。

请求 PCI 地址和 UUID 记录在 `eval_config.json`；匹配结果及最终环境记录在 evaluator 日志和
`results.json` 的 `render_device` 字段。没有新字段的旧 evaluator 配置和旧 CLI 保留
原来的手工路径。训练仓库与 LIBERO-FT 必须一起更新后再使用自动模式。

支持边界为 Linux/NVIDIA 普通非 MIG GPU，需要 NVIDIA EGL 的 UUID 扩展，或 DRM 身份扩展
及可读的 sysfs。不支持的环境使用经验证的手工映射。自动模式不修改 robosuite 安装文件，
不新增 Python 依赖；映射不能解决操作系统权限、容器设备隔离或图形驱动缺失。

本地只做 Python 静态编译、配置核对与 diff 检查，不新增测试脚本，不运行 GPU/仿真。
服务器验收需覆盖：单卡、非连续/重排 CUDA 列表及 GPU UUID；串行/多 rank、多个环境
worker 与空任务 rank；手工覆盖和自动匹配失败传播；开启视频后完成汇总并恢复训练。
核对日志中的 PCI 地址与服务器实际设备，并确认安装的 robosuite 后端遵循显式 EGL 选择。


#### EGL 列表超过 10 个设备时的兼容修正

MuJoCo 2.3.7 的 `egl_ext.eglQueryDevicesEXT(max_devices=10)` 默认截断设备列表。
独立枚举得到的 index 14 即使正确，也会被 MuJoCo 导入阶段的 0–9 范围检查拒绝。
映射代码现在预加载当前环境安装的 `mujoco/egl/egl_ext.py`（避免提前执行创建
EGL_DISPLAY 的包初始化），将无参数枚举改为先查询实际数量、再读取完整列表。
匹配器、MuJoCo 和 robosuite 共用这一模块，不再分别加载 EGL 来推断相同编号。

自动模式设置 `LIBERO_FULL_EGL_ENUMERATION=1`，使 spawn 重新加载 evaluator 入口时，
在导入模拟器之前应用同一进程内修正。不会修改 site-packages 文件；手工模式不主动
启用这条兼容路径。匹配日志额外记录 `egl_device_count`，保留 PCI 和 DRM node 信息。
服务器应补充验收：设备总数超过 10、目标 index >=10、多 worker，以及连续评估。
本地仅静态编译和 diff 检查，未运行 EGL/MuJoCo。

#### 区分物理 GPU 身份与可用的 NVIDIA EGL 入口

`egl_device_count=21`、index 14 对应正确 PCI，但访问 `/dev/dri/renderD133` / `card6`
被拒绝，说明枚举上限已解除，失败发生在 display 初始化阶段。这个现象可能来自同一
GPU 的 Mesa 入口，也可能是真正的设备访问权限限制；仅凭日志不能确认后端。
此前仅检查 DRM/PCI，没有检查驱动与初始化，是实现遗漏。

当前增加 CUDA/EGL UUID 匹配、NVIDIA 入口筛选和 display 预检查。成功记录包含
`match_method`、`gpu_uuid`、`egl_vendor` 和 `display_initialized`；匹配失败列出候选
身份，初始化失败保留原 EGL 错误。预检查不验证完整 OpenGL context、MuJoCo 渲染或
spawn worker，这些仍需服务器验收。原先的完整枚举及 spawn 修正继续保留。

同步 runner、LIBERO-FT 的 `eval_socket.py` 和 `libero/force/render_device.py` 后，
可先在服务器的 libero 环境、LIBERO-FT 目录单独检查日志中的目标卡，无需启动训练：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="" python - <<'PY'
import subprocess
from libero.force.render_device import match_egl_device

pci = "0000:ce:00.0"
gpu_uuid = subprocess.check_output([
    "nvidia-smi", "--id=" + pci, "--query-gpu=uuid", "--format=csv,noheader",
], text=True).strip()
print(match_egl_device(pci, gpu_uuid))
PY
```

若 GLVND 加载了错误 vendor，可在确认文件存在且指向 NVIDIA 驱动后，在全新进程设置
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 再运行检查。
实际路径随安装环境变化；切换 vendor 后 EGL 编号可能改变，需要重新匹配。
该环境变量会由 runner 继承并记录，不在代码中硬编码系统路径。

若 NVIDIA 入口仍无法初始化，检查目标设备节点的属组/ACL、作业或容器的设备访问规则，
以及 NVIDIA EGL/OpenGL 库。容器需提供 graphics 能力；仅有 CUDA compute 能力并不充分。
手工填写 EGL index 同样不能绕过这些限制。OSMesa 可作为安装支持下的 CPU 排障途径，
但不满足使用训练 GPU 渲染的目标，也不适合作为当前多环境评测的默认替代。

参考规范：
- [EGL_EXT_device_persistent_id](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_persistent_id.txt)
  规定物理 UUID 可用于跨 API 识别，同时提醒同一设备可由多个驱动提供不同 EGL 入口。
- [GLVND vendor selection](https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/libeglvendor.c)
  在首次加载时读取 `__EGL_VENDOR_LIBRARY_FILENAMES`。

本次只做静态编译和 diff 检查；未在本地运行上述命令或 GPU/仿真。


## 11. Task 条件与 task 内 topology 诊断（2026-09-24）

本次实现新增固定 10 维 task one-hot，并将 LIBERO 的 `sample/topology_div` 改为
固定诊断样本上的 task 内差异。只记录这个 topology 指标，不增加 global、per-task 或覆盖率日志。

### 11.1 固定任务身份与输入路径

完整名称表固化在 `diffusion_policy/common/libero_task_util.py`，采用官方
`libero_suite_task_map.py` 的 `libero_10` 原始顺序（LIBERO-Long）。不随实验子集、
HDF5 文件顺序或 benchmark 的 `task_order_index` 重新编号。

| ID | 任务 |
| --- | --- |
| 0 | LIVING_ROOM_SCENE2：alphabet soup、tomato sauce 放入篮子 |
| 1 | LIVING_ROOM_SCENE2：cream cheese、butter 放入篮子 |
| 2 | KITCHEN_SCENE3：开炉子并放上 moka pot |
| 3 | KITCHEN_SCENE4：黑碗放入底层抽屉并关抽屉 |
| 4 | LIVING_ROOM_SCENE5：两只杯子分别放到左右盘子 |
| 5 | STUDY_SCENE1：书放入收纳架后部 |
| 6 | LIVING_ROOM_SCENE6：白杯放盘上，布丁放盘右侧 |
| 7 | LIVING_ROOM_SCENE1：alphabet soup、cream cheese 放入篮子 |
| 8 | KITCHEN_SCENE8：两只 moka pot 放上炉子 |
| 9 | KITCHEN_SCENE6：黄白杯放入微波炉并关门 |

当前 YAML 中的四个任务对应 ID `[3, 2, 9, 5]`。训练四个任务时仍使用完整 10 维 one-hot。

Dataset 顶层 `task_id` 是整数标签，collate 后为 `[B]`，用于分组。
启用模态后，`obs.task_id` 为 float32 `[B,T,10]`，每帧重复同一 one-hot，使用 identity
normalizer。完整任务表保存在 checkpoint 的 preprocessing contract 中，恢复和 rollout
校验映射一致性。启用 one-hot 时未知任务报错；未启用时，其他 LIBERO source 可使用
`10 + source_index` 作为仅供诊断的分组 ID，不占用固定 0–9 编码。

Policy 的 `router_condition_key` 兼容 null、原单字符串和有序字符串列表。
配置默认 `[joint_state, task_id]`，将 `[B,T,6]` 状态和 `[B,T,10]` task 拼为
`[B,T,16]`，复用原 router 条件投影。Action policy 将 task 与视觉和其他低维特征拼接，
不新增独立 task encoder。Temporal 两侧保留相同任务身份，诊断只读取单窗口。

`mask_policy_state` **仅屏蔽 action policy 的机器人状态，不影响 router**：

| 输入 | Router | Policy，mask=false | Policy，mask=true |
| --- | --- | --- | --- |
| 图像 | 使用 | 使用 | 使用 |
| joint_state | 使用 | 使用 | 清零 |
| task_id | 使用 | 使用 | 使用 |

LIBERO socket runner 根据当前评估任务名称构造同一 one-hot，覆盖在线动作推理和录像
拓扑重建；各 rank 使用自己的任务上下文，不用分配队列位置代替 task ID。外部模拟器
协议不变。本次未扩展旧的独立 GPU service 脚本，Stage A/B 评估走 socket runner。

### 11.2 唯一 topology 指标

历史记录中的 `sample/topology_div` 是首个训练 batch 上包含跨 task 样本对的距离。
本次之后，LIBERO 在每个 `training.sample_every` 时机，从当前训练 split 为每个 task
固定选取 `training.topology_samples_per_task` 个不同窗口，默认 **每 task 64 个**。
Stage B 使用 B_train；不使用 val 数据。窗口遵守 max_len，数量不足时使用全部窗口，
不重复补齐，少于两个窗口的 task 排除。无有效 task 时跳过指标并给出控制台说明。

```text
D_task = mean(JaccardDistance(leaves_i, leaves_j))
         over i < j within the same task, view and relative observation frame
sample/topology_div = mean(D_task over eligible tasks)
```

忽略 token padding，保持原叶集合距离定义；task 间等权平均，不按窗口数或样本对数加权。
同一 task 内拓扑一致、不同 task 拓扑不同时，新指标为 0。

索引通过 training.seed 和固定 task ID 的局部 RNG 选择；现有训练增强（包括 wrist dropout）
使用隔离且固定的 CPU RNG，因此跨 epoch 的诊断输入固定。相同数据、seed 和配置下，恢复
时重建相同输入。逐 task 读取并调用 `get_foveation_outputs()`，跳过 action diffusion 和
visual token encoder；额外路由 RNG 同样隔离。主进程沿用 EMA 优先的 eval 模式，不缓存
全部任务图像，不修改 Temporal pairing 开关。

PushT 保持原首个固定训练 batch 的指标。原 sample action MSE 和可视化也保持原采样方式。
LIBERO 新旧日志同名但统计口径不同，比较历史实验时需区分版本。Task 内非零差异仅说明
路由随输入变化，不等于节点选择对动作或成功率有用。

### 11.3 配置与 checkpoint 使用

基础 YAML 直接定义 `shape_meta.obs.task_id={shape:[10],type:low_dim}` 和
`policy.router_condition_key=[joint_state,task_id]`；直接运行 train.py 和 Stage A/B 脚本
均默认启用。脚本不再提供 USE_TASK_ID 判断或注入 task 配置。
`TOPOLOGY_SAMPLES_PER_TASK=64` 独立控制每任务诊断规模，
不改变训练 batch size。原有末尾 Hydra overrides 仍有效。

```bash
# 先训练带 task 输入的新 Stage A；其他结构/数据参数按实验设置。
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/roi/sh_train_foveation_libero_stageA.sh

# Stage B 必须使用对应的新 Stage A checkpoint，并匹配其结构、horizon 等参数。
CUDA_VISIBLE_DEVICES=0,1 \
  PRETRAINED_CKPT=/path/to/task_conditioned_stageA/checkpoints/latest.ckpt \
  TOPOLOGY_SAMPLES_PER_TASK=64 \
  bash scripts/roi/sh_train_foveation_libero_stageB.sh
```

用户现有 Stage B `PRETRAINED_CKPT` 默认赋值原文保留。新增 task 改变 policy 条件维度和
router 投影形状，旧无 task checkpoint 不能直接严格加载到新模型；使用新 checkpoint，
或在末尾追加 Hydra 参数 `'~shape_meta.obs.task_id' policy.router_condition_key=joint_state`
恢复旧结构继续实验。A/B 的 mask 设置也需匹配。

### 11.4 涉及文件与验收

新增 `diffusion_policy/common/libero_task_util.py`；修改 LIBERO dataset、foveation policy、
训练 workspace、LIBERO socket runner、LIBERO foveation YAML、Stage A/B 脚本及本文。

本地仅静态编译、Hydra 配置解析、bash 语法和 diff 检查，不新增测试脚本、不启动训练。
服务器仍需验收：子集/文件换序时固定 ID；Temporal 两侧编码；mask 作用范围；训练与
rollout/视频一致；各 task 不等样本量、max_len、单样本 task、多相机/多帧的聚合行为；
恢复后的固定诊断输入；DDP 和阶段切换；关闭模态的旧 checkpoint 与 PushT 兼容性。
