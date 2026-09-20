# LIBERO 实验规划：当前方法、Novelty 与下一阶段工作

更新日期：2026-09-18。本文汇总当前实验反馈与后续方向，不表示尚列为待办的代码已实现。

## 1. 当前方法与项目状态

项目研究 **runtime adaptive resolution selection for robotic manipulation**：
通过四叉树为不同区域选择 coarse/fine 表示，在预算下保留动作需要的细节与全局上下文。
运行时在 policy invocation / action-chunk 边界重选，不在线更新参数或执行 paired 搜索。

当前成功配方：

| 阶段 | 更新对象 | 作用 |
|---|---|---|
| Stage A：多拓扑预训练 | 视觉编码器与 diffusion policy | 在随机合法树上建立跨尺度、跨数量的下游能力 |
| Stage B：paired + budget | 冻结下游，更新 routing | 在固定观测、noise、timestep、mask 下，用 stop/split 损失差学习局部分辨率收益 |
| 后期下游适配 | 固定确定性 routing，更新视觉编码器与 policy | 适应所选低预算表示，进一步改善闭环表现 |

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

逐 seed 数值、方差／离散程度、checkpoint 选择规则和具体耗时从已有记录整理，
不补写未知值。直接端到端对照不自动等同于冻结下游的 B0/ST；性能对比按实际更新范围
标注。旧版本 all-fine 成绩不能直接用于证明当前同协议下的性能超越。

**整体判断：** 已有强低预算初步结果，应进入跨任务与真实闭环验证阶段。下一步主要
风险是三维任务推广、移动视角适用性和部署成本，而不是重新证明 PushT paired 有效。

## 2. 当前版本的 Novelty 总结

核心表述：**策略学习的不仅是如何行动，还包括当前动作需要什么空间分辨率；通过
合法区域表示间的 paired 比较学习细分价值，再通过分阶段适配形成低预算操作策略。**

| 层面 | 当前主张 | 边界 |
|---|---|---|
| 选择对象 | 同一区域 coarse 表示与 fine 子树之间的分辨率选择 | 不只是已有 token 的保留；但混合尺度表示本身已有研究 |
| 学习信号 | 固定下游的 paired stop/split action-loss 比较，估计局部额外细节的收益 | 区分 action relevance 与 resolution utility；不是环境 reward 的直接因果收益 |
| 训练机制 | 多拓扑准备下游、固定下游学习 selector、固定 routing 适配下游 | 组合形成有效方法；分阶段训练本身不单独主张首次 |
| 实证优势 | 低预算 PushT 表现、paired 优势、下游适配收益和包含 router 的测速优势 | 实验优势不是独立机制创新；跨任务与真机尚待验证 |
| 时间计算分配 | 保留为核心研究目标，学习何时增加或减少总体计算 | 当前数量波动和预算控制不构成完成证据 |

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

### 3.1 现有接入状态

- [配置](../configs/liberoft_foveation_dp.yaml) 使用 LIBERO-FT 的 `libero_force_384/libero_10`
  下五个任务；当前启用 static `image`，`gripper` 与低维状态仍被注释，`n_obs_steps=1`。
  这是定制数据／任务子集，不直接写成完整标准 LIBERO suite 结果。
- [数据集](../diffusion_policy/dataset/libero_dataset.py) 已有按配置读取 `eye_in_hand_rgb`
  与 `ee_states` 的路径；[policy](../diffusion_policy/policy/foveation_dp_transformer_policy.py)
  已有多 RGB 输入和 `router_condition_key` 接口。
- [现有 ROI 服务](../service/gpu_service_roi_libero.py) 只取第一视角，并使用 ROI 输出字段；
  需要核对并适配四叉树策略的部署接口，不能仅打开训练配置就视为双视角闭环已完成。
- [LIBERO 启动脚本](../scripts/roi/sh_train_foveation_libero.sh) 尚需补齐与 PushT 成功路线
  对应的 Stage A / Stage B 入口和参数传递。

### 3.2 推进顺序与验收目标

| 顺序 | 工作 | 完成标志 |
|---|---|---|
| 1 | 核对数据、观测、动作与闭环评估，建立 static all-fine | checkpoint 能稳定 rollout，输入、归一化、动作 chunk 和成功率口径一致 |
| 2 | static camera 迁移 A → paired → 下游适配 | 获得同任务、同数据和评估设置下的性能与实际数量记录 |
| 3 | 加入 static + wrist | 两视角训练／推理链路一致，按总预算或实际总成本比较 |
| 4 | 扩展正式任务集合及论文比较 | 明确任务选择、数据版本、多 seed、checkpoint 规则与性能—成本结果 |

开发阶段先选择代表性抓取放置和抽屉／微波炉交互任务，不把完整长程任务和全视角实验
矩阵同时作为起点。具体任务文件与正式 suite 在 LIBERO 实施时根据现有数据确定，
不在此虚构新的任务条目。当前五任务中已有抽屉、微波炉和多对象操作可供选择。

接入时优先核对：

- 图像方向、颜色、数值范围、时间顺序和相机 key 一致；确认 384 图像来自原生渲染／
  重渲染还是低分辨率上采样，后者不能作为额外视觉细节的证据。
- 动作表示、归一化、实际执行的 action chunk 与训练一致；成功率和分任务结果可追溯。
- 明确 per-task policy 或有显式任务条件的共享 policy。同场景多目标不能靠背景猜任务；
  使用自定义任务条件时准确标注，不与标准语言条件多任务协议混用。
- 首先复用成功的三阶段机制；PushT 的预算数值和 epoch 不视为未经调整即可通用的默认值。

LIBERO 优先是因为已有数据和代码基础。robomimic/robosuite 可作为备选，尤其适合
Square、Tool Hang 等对齐任务的机制研究；目前不并行铺开多个新 benchmark。

### 3.3 Wrist camera 是否需要额外方法

移动相机不会天然使当前方法失效：每次策略调用都根据新图像重新选择拓扑，不要求固定
像素对应固定世界点。首先直接迁移，再按失败原因考虑扩展：

| 观察到的问题 | 候选处理 |
|---|---|
| 单帧歧义或状态不足 | 使用已有末端／夹爪状态条件；共享 router 时考虑 camera ID |
| 遮挡和历史信息缺失 | 短历史或轻量记忆 |
| 双视角预算浪费、缺少互补性 | 共享预算和跨视角上下文 |
| 确实需要跨帧对应 | 再考虑位姿／深度或光流对齐 |
| chunk 内画面变化导致观测过时 | 在统一比较协议下分析 chunk 长度与重规划频率 |

固定中心细化／中心裁剪是 wrist camera 的有意义参考；不能仅凭初始树固定整段轨迹
的弱对照证明动态方法必要。不要直接平滑相邻帧同像素的 split 概率，也不预设靠近物体
必然需要更多 tokens：目标的自然放大和接触遮挡会共同改变细节需求。

## 4. 自主时间计算分配：保留的核心研究目标

现有预算约束让每个观测的期望 patch 数接近固定目标，对高于和低于目标均惩罚。
运行时空间布局变化、leaf 数波动，均不等于学会根据任务需要改变总计算量。

一个待研究的简单方向是在现有目标预算项之外，加入非负 patch 使用量的 L1/L2 成本
正则，使较低计算量获得更低损失。概念上为：

```text
L_router = w_pair * L_paired + L_target_budget + lambda_cost * R(patch_usage)
```

`R` 的具体范数、归一化、使用可微期望数量的方式、权重及跨样本／时间聚合范围尚未
确定；这不是已经实现的训练目标。固定目标项与低使用量正则存在竞争，可能只是把所有
状态的数量整体压低。若总量是固定的，单纯线性的数量求和也不会决定如何在时间上分配。

后续需在匹配平均或整段任务成本下，区分“固定数量、动态空间布局”与“状态相关数量”
的收益；分析数量变化是否反映任务需求，而不只反映图像纹理、相机运动或执行时间。
这一研究目标与 LIBERO 迁移并行推进，不成为新 benchmark 的前置条件。

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
