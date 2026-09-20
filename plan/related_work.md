# Runtime Adaptive Resolution Selection：论文定位、Related Work 与实验

拟定标题：**Runtime Adaptive Resolution Selection for Robotic Manipulation**。

本文研究：在任务执行过程中，根据当前观测和机器人状态，为不同图像区域选择当前
动作需要的分辨率。四叉树表达合法的多尺度空间分区，paired stop/split 比较提供
分辨率选择的学习信号。预算是感知选择的约束；减少 tokens 和计算量是结果之一，
不把论文主线定义为压缩已有模型。

**当前状态（2026-09-18）：** PushT 上 paired 信号贡献已完成验证，不再安排重复的
paired/ST 实验。成功配方为多拓扑预训练 → 冻结下游学习 paired routing → 固定 routing
适配下游。`target_num_patches=0.1` 下三 seed 平均报告成绩为 **0.679**
（`test/mean_score`，最大覆盖性能）；包含 router 的固定 ratio 测速已显示显著加速。
当前主要工作转向 LIBERO，详见 [实验规划与 novelty 总结](libero_exp.md)。

## 1. 核心故事与术语

### 1.1 为什么需要 runtime selection

**同一个区域，在任务不同阶段所需要的视觉精度不同。** 接近物体可能需要整体位置
关系，建立接触可能需要局部边界，推动和调整时细节需求又会随接触位置和姿态变化。
这是论文要验证的假设，不应把示意性的阶段解释直接写成已经证明的行为。

核心问题：**为了做出当前动作，策略需要在哪里看得更细，在哪里只需粗略观察？**

一个区域对动作重要，并不意味着它始终需要最高分辨率。论文应区分：

- **Action relevance：** 某个已有区域／token 对动作是否重要。
- **Resolution utility：** 为该区域增加分辨率，是否改善当前动作预测。

统一术语：

| 用途 | 推荐表述 |
|---|---|
| 整体问题／方法 | runtime adaptive resolution selection |
| 局部选择 | region-wise resolution selection |
| Router | resolution selector |
| 学习方式 | action-guided resolution selection |
| 动态性的来源 | state-dependent / task-progress-dependent resolution selection |

Allocation 可以解释计算资源的分布，但核心方法名称使用 selection。

### 1.2 对齐当前实现

- 选择的是各区域的 coarse/fine 表示，不只是整图在几个输入尺寸之间切换。
- Runtime 表示部署执行时根据新观测重新选择，不表示在线更新模型参数。
- 当前选择频率是 **policy invocation / action-chunk 边界**，不是每个环境 action 后
  都重新调用策略。逐帧四叉树诊断视频不等于策略逐帧在线重选分辨率。
- Router 根据观测和状态预测分裂决策；当前没有显式任务阶段变量，也没有跨时间
  规划树结构。阶段相关的选择应由轨迹实验展示。
- 当前策略先得到完整拓扑，再送入下游视觉 token encoder。表述为
  “before the downstream visual token encoder”，因为 router 本身仍使用 global encoder/FPN。

## 2. 与 Token Pruning / Merging 的联系和实质差异

### 2.1 承认相似性，不依赖名称划界

选择父区域 coarse 表示替代 fine 子树，在结构上可以视为**受空间层级约束的粗化／
合并**。树规定了哪些表示可以互相替换：父子互斥、空间连续、层级嵌套、完整覆盖。
但 merging 也可以加入空间或层级约束，因此“用了树”不是充分的 novelty。

两类方法都涉及有限视觉计算下的信息选择。不宣称动机完全无关；本文从执行过程中的
动态感知需求定义问题，与压缩方法在机制和结果上存在交集。

### 2.2 核对后的代表方法

| 方法 | 实际机制 | 应正面比较的内容 |
|---|---|---|
| VLA-Pruner | 语义 prefill attention 与时间平滑的 action attention 各选候选 tokens，取并集后以特征多样性过滤 | 已有动作感知选择；区别在分辨率收益、选择对象与训练方式 |
| ToMe | 在网络层内根据特征相似性匹配并聚合已有 tokens | 特征聚合与图像区域尺度选择；相似性与 paired 任务反馈 |
| DynamicViT | 从中间特征学习重要性，逐层删除 tokens | 已有 token 的保留决策与区域尺度选择 |
| EViT | 根据 class-token attention 保留重要 tokens，并融合低关注 tokens | 不能归为简单删除背景；比较聚合形式与空间结构 |
| MSViT | 条件 gating 为区域选择 token 尺度 | 表示上的直接近邻；重点比较层级结构与动作收益学习 |

VLA-Pruner 已考虑语义与动作的不同视觉需求，且不是简单的单一 attention top-k。
ToMe 主要使用特征相似性，其他 pruning 也可采用学习式评分、梯度或多样性，不能
把整个领域概括为 attention top-k。Pruning/merging 本身也可以随输入和时间变化。

### 2.3 选择对象与表示形成方式

**Subset selection：** VLA-Pruner 从已有 tokens 中保留一个子集。

**Feature merging：** ToMe 将已有 token 特征聚合为更少的特征。

**Region-wise resolution selection：** 本文在一个父区域的 coarse 表示与覆盖同一区域
的 fine 子树表示之间选择。Coarse patch 来自图像金字塔相应区域，再经 patch embedding，
不是几个已编码 fine tokens 的平均。未细分区域仍有独立的 coarse 表示。

完整空间覆盖不等于信息无损。它明确保留区域级上下文，但不能单凭这一点声称优于
所有 merging 方法；MSViT 也已研究混合尺度表示。

### 2.4 选择依据：相关性与细分收益

Attention 衡量模型对已有表示的关注，不直接测量增加区域分辨率的收益。本文比较：

```text
Delta_resolution = L_stop - L_split
paired_loss = -mean(p_split * detach(Delta_resolution))
```

固定下游、观测、diffusion noise、timestep 和 mask，在当前树上下文中改变一个可达
节点的 stop/split。Stop 将节点作为叶子；split 展开节点，其后代沿用已有决策，因此
不一定只产生四个最终 leaves。

该信号直接评估一次局部分辨率替换对动作预测损失的影响。它仍是采样上下文中的局部
反馈，不是全局最优树保证，也不是直接测量环境 reward 的因果收益。

### 2.5 更新过程：训练学习，运行时预测

| 时段 | 本文的操作 |
|---|---|
| 训练 | 采样可达节点，评估 stop/split 两种拓扑，更新共享 selector 参数 |
| 策略推理 | 根据观测预测分裂决策，构造合法空间分区，不在线执行两次损失评估 |
| 相邻策略调用 | 根据新观测重新预测，不是显式修改上一时刻的树或在线搜索最优树 |

相对 ToMe 的层内匹配合并，本文学习一个直接预测空间分区的策略；相对 VLA-Pruner
的无需训练部署，本文需要 selector 训练和多尺度下游准备。这是设计取舍，不宣称
无条件替代即插即用方案。推理时不需要历史 action-attention 作为评分输入。

## 3. 核心卖点、Novelty 与行文

### 3.1 审稿人最可能的质疑

> 动态多尺度 tokenization 已有，动作相关 pruning 也已有；本文是否只是把二者结合，
> 再用四叉树实现？

这个质疑必须通过方法与实验回应，不能靠把 merging 改称 selection 回避。

| 组成部分 | 单独作为 novelty 的判断 |
|---|---|
| 随输入动态选择分辨率 | 已有相关研究，不单独主张首次 |
| 四叉树组织空间尺度 | 成熟结构，需证明在本问题中的价值 |
| 用动作信息选择视觉表示 | VLA-Pruner 等已考虑，不单独主张首次 |
| Paired stop/split 学习局部分辨率价值 | 当前最值得深入的方法贡献，但局部反事实比较本身并非全新思想 |
| 多拓扑预训练 → 固定下游学习 selector → 固定 routing 适配下游 | 当前成功配方，已有 PushT 实验支持；分阶段训练本身不单独主张首次 |
| 随交互需求变化的选择改善闭环表现 | 重要实证贡献，需动态对照与干预支持 |

核心卖点：**策略不只学习如何行动，也学习当前行动需要什么空间分辨率。**

方法贡献应具体落在：**通过合法区域表示间的 paired 比较学习动作所需的分辨率，
再通过分阶段适配将这种选择转化为有效的低预算操作策略。** 四叉树、多尺度表示、
动作相关选择和分阶段训练各自并非首次；novelty 在具体选择对象、收益信号和训练机制
的结合。低预算分数与加速是实证优势，不单独构成机制创新。

### 3.2 方法论证链

1. Stage A 提供多种拓扑与尺度的训练覆盖，让下游能够使用不同表示。
2. Stage B 固定下游，使不同分辨率的比较具有相对稳定的评价依据。
3. Paired 比较真实 stop/split 表示，补充 ST 代理梯度或相似性评分未直接测量的
   局部任务反馈；已有 PushT 实验完成了 paired 信号贡献的验证。
4. Selector 在每次策略调用时直接预测分辨率，闭环实验验证动态选择的价值。
5. 固定 routing 后适配视觉编码器与 policy，进一步明显提升表现；它属于完整方法，
   但整理结果时仍与 paired 阶段本身的贡献分开描述。

最新实验反馈表明，paired 阶段本身优于已有直接端到端训练对照，后期适配进一步改善
结果。“直接端到端”不自动等同于冻结下游的 B0/ST，具体训练范围按已有实验记录标注。
这些结果支持当前配方，但不推出 ST 必然失败、attention 无效或 paired 在所有任务上
必然更优。自主时间计算分配仍是核心研究目标，尚未完成验证，也不阻挡 LIBERO 迁移。

### 3.3 Introduction 草稿

> Robotic manipulation requires different visual precision as an interaction unfolds.
> A region that can be represented coarsely during approach may require finer detail
> when establishing contact. We study runtime adaptive resolution selection:
> choosing the spatial resolution of image regions according to the policy's current
> action requirements. Our selector predicts a hierarchical spatial partition at
> each policy invocation and learns local refinement decisions through paired
> evaluations of action-prediction loss.
> We first prepare the downstream model across diverse tree topologies, learn the
> selector with the downstream model frozen, and then adapt the downstream model
> to the selected representations with routing fixed.

### 3.4 与 merging 的正面比较草稿

> Our coarse-region selection can be interpreted as a spatially constrained
> coarsening operation, making token merging a closely related mechanism. Our
> formulation selects among image-region representations at different resolutions,
> rather than aggregating existing token features. The selector is trained through
> paired coarse-versus-fine evaluations of action-prediction loss and predicts a
> resolution partition at each policy invocation. We focus on when and where finer
> visual resolution benefits the current action.

### 3.5 与动作感知 pruning 的比较草稿

> Recent VLA pruning methods, such as VLA-Pruner, incorporate both semantic and
> action attention to retain relevant visual tokens. We address a complementary
> question: how much spatial resolution does an action-relevant region require?
> Our selector learns refinement decisions from paired changes in action-prediction
> loss, distinguishing token relevance from the marginal utility of spatial detail.

避免：“首次考虑执行中的视觉关注变化”“pruning 必然降性能”“merging 不动态”
“完整覆盖就是无损”“我们有动作监督而 VLA-Pruner 没有”。超过 all-fine 是实验结果，
不是独占的机制特征；VLA-Pruner 也报告过剪枝后提升。

## 4. Related Work 组织：三条主线

### 4.1 Adaptive Spatial Resolution and Tokenization

重点讨论 MSViT、FOVEA 和 AdaFocus：

- MSViT 是混合尺度表示的直接近邻，必须正面对比。
- FOVEA 以数据集／帧级先验驱动图像变形与区域放大，不笼统归为 learned ROI。
- AdaFocus 学习局部选区，结合轻量全局网络与高容量局部网络。

落点：已有研究建立动态空间观察与尺度选择；本文研究动作任务中的层级分辨率选择
及其局部收益学习。QuadTree Attention 可简短说明：它用树组织 coarse-to-fine
attention 搜索，与选择输入区域尺度不同。

### 4.2 Action-Aware Token Selection and Reduction

以 VLA-Pruner 为关键近邻，讨论语义相关性、动作相关性与 temporal attention；再用
DynamicViT、EViT、ToMe 覆盖删除、融合、合并。不要只引用视觉分类领域方法而遗漏
机器人动作感知选择。

落点：承认共同的信息选择问题，说明选择对象、评分依据、表示形成位置和训练取舍。
本节区分 **VLA token pruning** 与 **VLM-based 语义定位**：前者是核心近邻，后者
是可选定位基线，不能把它们一起降为低优先级。

### 4.3 Adaptive Perception for Visuomotor Policies

以 Diffusion Policy 为策略背景，结合主动视觉、foveation、机器人 ROI、对象中心
表示和任务条件感知。解释为何感知选择需要服务于当前动作，不写成泛泛的 VLA 综述。
规则、几何投影和 VLM 定位可在此讨论，不为每一种基线单开小节；具体机器人近邻
文献需随最终任务范围补齐。

CVPR 可按 4.1 → 4.2 → 4.3，RSS 可按 4.3 → 4.1 → 4.2。这是建议的叙述重心，
不是会议硬性格式。每节先讲已有进展，再具体说明本文区别。

## 5. 实验：已有证据、结果整理与跨任务推广

### 5.1 证据状态与后续分析

| 核心问题 | 当前状态／分析方式 | 要隔离的贡献 |
|---|---|---|
| Runtime selection 是否必要？ | 每次策略调用重选，对比初始观测选择后整条轨迹固定、固定布局和强规则树 | 动态选择，不只是平均 token 数变化 |
| Paired 依据是否有效？ | PushT 已完成验证，整理现有对照的预算、更新范围与结果，不重复安排实验 | 选择学习方法，不把所有提升归于树结构 |
| 分辨率是否跟随动作需求？ | 接近、接触、调整阶段的轨迹分析，配合局部 stop/split 干预 | 动作需求与分辨率变化的对应，而非漂亮可视化 |

其余动态性分析按论文论证需要补充，不作为开始 LIBERO 的前置条件。整理结果时
比较相近实际 patch 数；动态树 count 有变化时报告其分布，必要时增加 count-matched
对照。Attention/similarity 驱动的树使用相同合法分区约束，明确分数如何汇总到节点，
不将自行构造的适配基线称为原论文复现。

按需开展的轨迹分析应检查同一对象在不同阶段的选择，不只是细分区域随对象平移。高 attention
与高细分收益是否相同，可在固定 checkpoint、匹配树上下文中测量：对同一批区域
比较 attention 排名与重复 noise/timestep 下的 paired 收益。单次 noisy loss 不足以
代表稳定收益；局部损失结果再与闭环表现结合。

### 5.2 外部基线优先级

以下是正式论文的外部比较框架，不是必须先在 PushT 全部重跑才能进入 LIBERO 的清单。
下一步先建立 LIBERO all-fine 与当前三阶段方法的可靠闭环，再扩展适配于该设置的基线。

| 优先级 | 基线 | 回答的问题 |
|---|---|---|
| P0 | All-fine、多档全局缩放 | 为什么需要空间非均匀的分辨率选择？ |
| P0 | MSViT／明确标注的 MSViT-style 适配 | 相对已有 mixed-scale tokenization，新增价值是什么？ |
| P0 | ToMe | 相对通用特征 merging，是否有表示与任务收益？ |
| P0（兼容 VLA 设置） | VLA-Pruner | 相对已有动作感知选择，是否改善闭环表现与成本权衡？ |
| P1 | 学习式 global + local ROI | 层级全图分区相对少量局部裁剪的价值 |
| P1 | DynamicViT | 补充删除式选择对照 |
| P2 | VLM 定位驱动细分 | 在语言／多对象任务中，语义定位与动作收益选择的区别 |

当前 Diffusion Policy 不原生具有 VLM prefill/action-decode 接口，不能声称直接运行
VLA-Pruner。DP-only 实验可做明确命名的 attention-based 适配并说明差异；若扩展到
兼容 VLA backbone，优先直接评估原方法。无论是否能原样运行，related work 都应
正面讨论它。不能把 DP 与另一套 VLA 的分数直接比较来证明选择机制优劣。

MSViT 的适配保留核心 gating、尺度选择与正则化；简化实现称 MSViT-style。
规则树可围绕 agent 或 agent 与物体细分；使用 simulator 真值位置／接触信息的版本
标注额外状态信息。VLM 定位框或 mask 可转成四叉树优先级，但当前 PushT 不优先。

### 5.3 分清三种训练比较

以下用于组织已有实验和后续跨任务比较，不重新打开已完成的 PushT paired 验证。

- **最终系统比较：** 各方法按相应训练方案完成训练，统一数据、评估 seeds、下游
  容量与推理设置，说明训练成本、初始化和额外预训练来源。
- **Selector 贡献：** 固定同一 Stage A 下游，比较强制 all-fine、固定／规则／随机树
  与不同 learned selectors；与“单独训练的 all-fine 模型”分开标注。
- **后期适配贡献：** 单独比较是否启用 downstream finetuning，并给 all-fine
  相同下游更新次数和 LR 的继续训练对照。

训练实现及恢复约定见 [fix_action_backward.md](fix_action_backward.md)。

### 5.4 结果组织与效率测量

最新 PushT 结果为目标预算 0.1 下三 seed 平均报告成绩 0.679。当前 16–256 leaves
配置的目标为 40 leaves（all-fine 的约 15.6%），不等同于实测数量或 10% token 保留率。
逐 seed 数据、离散程度与 checkpoint 选择规则从已有记录汇总；不能直接用历史版本
all-fine 成绩推导当前同协议下的性能超越。

已有 [测速实验](../exp/pusht_compare_infer_speed.py) 启用了 `--include-router`，
ratio 0.1 相比 all-fine 显著加速。当前使用合成输入、固定 ratio 树，尚未加载训练
checkpoint；后续工具支持真实观测与 learned routing 的数量、padding 和延迟分布。
该补充连接同一 checkpoint 的性能与成本，不否定现有测速，也不作为 LIBERO 前置条件。

主结果围绕 **动态选择、选择依据和闭环表现** 组织；性能—成本曲线提供效率证据，
不让整篇论文退化为不同 pruning ratios 的排行榜。

报告相同评估设置下的多 seed 与窗口表现，不仅报告最佳 checkpoint。成本包括
router/global encoder/FPN、视觉编码和动作预测，区分每个 action chunk 的推理延迟
与每个环境 action 的摊销成本。报告实际 token 数、视觉耗时、端到端延迟与 FLOPs；
不同方法各层长度不同，不能用最终 token 数替代公平成本比较。

“Stage B 超过同一下游强制 all-fine”支持选择有用，但单独不足以证明方法 novelty。
最关键的证据是：**相同结构下选择学习方法有收益，相近成本下层级分辨率选择有收益，
而这种收益与任务执行过程中变化的视觉需求相对应。**

## 参考文献入口

- [VLA-Pruner: Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference — 最新核对版本 v5](https://arxiv.org/html/2511.16449v5)
- [MSViT: Dynamic Mixed-Scale Tokenization for Vision Transformers](https://arxiv.org/abs/2307.02321)
- [FOVEA: Foveated Image Magnification for Autonomous Navigation — ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Thavamani_FOVEA_Foveated_Image_Magnification_for_Autonomous_Navigation_ICCV_2021_paper.html)
- [Adaptive Focus for Efficient Video Recognition（AdaFocus）— ICCV 2021](https://arxiv.org/abs/2105.03245)
- [DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification — NeurIPS 2021](https://arxiv.org/abs/2106.02034)
- [Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations（EViT）— ICLR 2022](https://arxiv.org/abs/2202.07800)
- [Token Merging: Your ViT But Faster（ToMe）— ICLR 2023](https://arxiv.org/abs/2210.09461)
- [QuadTree Attention for Vision Transformers — ICLR 2022](https://arxiv.org/abs/2201.02767)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion — RSS 2023](https://arxiv.org/abs/2303.04137)
