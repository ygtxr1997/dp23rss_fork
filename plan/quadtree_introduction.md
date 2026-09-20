# Detail on Demand: Runtime-Adaptive Resolution Selection for Robotic Manipulation

## 1. 项目定位与科学问题

本项目研究**面向机器人操作的运行时自适应视觉分辨率选择**：
将视觉细节分配给真正影响动作决策的图像区域，并随操作过程自主调整总体视觉计算量，
在保持闭环操作性能的同时减少计算。

核心科学问题是：**机器人如何从操作任务信号中学会判断不同位置需要多精细的视觉信息，并在操作过程中自主决定何时投入更多或更少的视觉计算？**

- **空间分配（spatial allocation）：** 在同一时刻，为图像不同位置分配不同分辨率。操作相关的物体、局部结构和交互区域可以获得更多细节，其他区域保留较粗表示，同时维持全场景上下文。
- **时间分配（temporal allocation）：** 随机器人接近、接触和操作物体，模型根据当前信息需求自主增减视觉 token 总量。精细对齐等阶段可能需要更多计算，较粗信息足够时则减少计算；具体需求由任务学习，不预设距离与分辨率的单调关系。

两者共同回答“细节分配到哪里，以及什么时候需要更多细节”。研究目标是让同一策略在运行时自主调整局部分辨率和总计算投入，而无需人为逐时刻指定预算。

可用于项目申请的英文表述：

> We will study task-driven adaptive resolution selection for robotic manipulation, learning to allocate fine visual detail to manipulation-relevant regions and autonomously adjust overall visual computation as the robot approaches and interacts with objects, with the goal of preserving manipulation performance while reducing computation.

## 2. 核心方法：用四叉树统一空间与时间分配

- **混合分辨率表示。** 从图像金字塔构建多尺度 patch bank。每个节点选择保留当前粗尺度 patch，或将其细分为四个子节点。最终叶子完整、无重叠地覆盖图像，多个不连续区域可以同时细化。各叶子的像素 patch 大小相同，但覆盖原图的面积不同。
- **任务相关路由。** 空间对齐的 router 根据节点对应区域的视觉特征及全局上下文预测细分决策。所选 patches 经带位置与尺度信息的视觉编码器聚合后，作为 diffusion policy 的观测条件。路由从动作预测收益中学习，无需额外的显著性或交互区域标注。
- **联合分配机制。** 叶子的空间分布与层级决定各区域的分辨率，叶子总数决定当前使用的视觉 token 数。模型通过选择树结构共同调整两者：同样数量的叶子可以形成不同空间布局，不同时刻也可以使用不同数量的叶子，无需预设独立的空间与时间决策模块。

每个叶子对应一个视觉 token，因此叶子数直接表达 token budget，并作为视觉计算成本的代理。实际效率还受 router、图像金字塔、batch padding 和动作生成开销影响，需要测量端到端延迟与 FLOPs。

## 3. 学习难点与技术路线

空间分配的关键是识别局部细化对动作决策的价值，避免与图像内容无关的固定布局，或动作网络与 router 共同适应后退化为全粗表示。当前成功路线分为三个阶段：

1. **Stage A：多拓扑预训练。** 用多种随机合法树训练视觉编码器与动作 policy，建立跨尺度、跨 token 数量的输入覆盖。
2. **Stage B：paired 选择学习。** 冻结下游，在相同观测、diffusion noise、timestep 和 mask 下比较 stop/split 的动作预测损失，用局部分辨率收益与 budget 约束训练 routing。
3. **后期下游适配。** 固定确定性 routing，再训练视觉编码器与动作 policy，适应所选低预算表示；固定 routing 参数不意味着所有观测使用相同的树。

PushT 上 paired 信号的贡献已完成验证；paired 阶段本身优于已有直接端到端训练对照，后期适配进一步明显提升表现。下一步迁移该配方到 LIBERO，不重复安排这一贡献的冗余实验。

时间分配进一步要求模型判断额外视觉计算是否值得，从而在保持任务性能的同时减少整段操作的计算成本。候选方向是在现有目标预算约束外增加 patch 使用量的 L1/L2 成本正则，使较低计算量获得较低损失。具体范数、权重和聚合范围尚未确定；固定目标约束与降低使用量的正则存在竞争，加入正则本身不等于学会按需求分配计算。局部细化收益也不保证整段任务的最优计算分配。

## 4. 当前贡献与后续验证重点

1. **动作收益驱动的空间分辨率选择与分阶段学习：** 用合法区域表示间的 paired 比较学习细分价值，通过多拓扑预训练与固定 routing 后的下游适配形成有效低预算策略。PushT 已提供实验支持，下一步检验三维操作中的推广能力。
2. **低预算闭环表现与实际加速：** 已有三 seed 的 PushT 结果与包含 router 的固定 ratio 测速；继续整理实际数量、checkpoint 推理延迟和跨任务性能—成本曲线。
3. **需求驱动的时间计算分配（核心研究目标，尚未完成）：** 在相近平均或整段任务成本下，与逐时刻固定预算比较，验证状态相关的总计算量变化是否有益。该目标不作为开始 LIBERO 的前置条件。

## 5. 当前进展与研究边界

截至 2026-09-18 的实验反馈：使用 Stage B paired 优化后固定 routing、适配下游，在 `target_num_patches=0.1` 下，**三个 seed 的平均报告成绩为 0.679（`test/mean_score`，最大覆盖性能）**。逐 seed 数值、离散程度与 checkpoint 选择规则留待已有实验记录汇总，不在此推定。

当前 16–256 leaves 配置下，归一化目标 0.1 对应 `16 + 0.1 × (256 - 16) = 40` 个目标 leaves，约为 all-fine tokens 的 15.6%；这不是实际推理数量的测量值，也不是仅保留 10% tokens。不能直接以历史版本 all-fine 分数证明当前同协议下的性能超越。

[测速脚本](../exp/pusht_compare_infer_speed.py) 的已有实验启用了 `--include-router`，ratio 0.1 相比 all-fine 显著加速。当前测速使用合成输入和固定 ratio 拓扑，尚未加载训练 checkpoint；后续补充真实观测下 learned routing 的实际数量和延迟分布，不将此作为 LIBERO 的前置条件。

**自主时间分配是本项目的核心研究目标，仍需完善学习目标并验证。** 当前期望叶子数约束主要服务于数量控制；不同输入产生不同叶子数，本身不能证明模型已学会根据操作需求分配计算。后续需要验证这些变化是否反映额外细节的任务价值，以及是否改善整段任务的性能与计算权衡。

下一阶段优先迁移到 LIBERO：先完成 static camera 的可靠闭环，再加入 wrist camera。移动视角不天然要求新模块，是否增加时序、状态或几何设计由实际失败决定。完整总结与待办见 [LIBERO 实验规划](libero_exp.md)。

## 6. Abstract（当前进展与研究目标版）

Visual detail is unevenly relevant across a scene, and reliable manipulation can require different spatial precision as interaction unfolds. We study Detail on Demand, a framework for runtime-adaptive resolution selection using quadtree partitions that retain coarse scene context while refining selected regions. A task-driven selector predicts the partition at each policy invocation, and the resulting mixed-resolution tokens condition a diffusion policy. Training combines downstream pretraining across diverse tree topologies, paired coarse-versus-fine action-loss comparisons with a frozen downstream model, and downstream adaptation with fixed routing. On PushT, the current pipeline achieves a reported three-seed mean score of 0.679 at a normalized target budget of 0.1. Fixed-ratio timing experiments including the router show substantial acceleration relative to all-fine processing; checkpoint-based timing with learned routing remains to be added. Next, we will evaluate transfer to LIBERO and moving wrist-camera observations. Autonomous temporal computation allocation remains a core research goal: we aim to learn when additional detail justifies its cost, beyond selecting spatial layouts under a target budget.

---

设计与实验依据：[FoveationNet](foveation_net.md)、[空间路由问题](fix_router_collapse.md)、[动作梯度与分阶段训练](fix_action_backward.md)。本文提炼研究定位；具体实现与实验状态以代码和后续记录为准。
