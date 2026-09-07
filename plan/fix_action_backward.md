# Quadtree Router 的 Action-Only Collapse

核心问题：budget loss weight 为 0 时，为什么 router 即使从 all-fine 初始化也可能
长期走向 all-coarse？

**当前判断：已有 3-seed 实验支持 fine 路径的训练与闭环收益；新增固定中间树在
epoch400 后的两项误差平均介于 coarse/fine 之间。** 混合尺度本身并非必然无益，
随机切换 topology 的代价更值得优先验证；ST 信用分配与共同适应仍未完成因果定位。

## 一、已有现象与代码事实

| 实验记录 | 现象 | 可支持的判断 |
|---|---|---|
| 开启 target decay | 目标区变 coarse，部分边缘仍 dense | count 约束不保证任务相关的空间分配 |
| budget weight=0 | 早期约 16 patches，曾出现 256，中期回升，长期接近 coarse | 存在明显波动和长期 coarse 倾向，机制待验证 |
| all-fine 初始化 | 初始 deterministic tree 为 all-fine，随后转向 coarse | 初始化不足以避免该现象，不能据此确认 branch starvation |
| target 0.25 / 0.75，budget weight=0 | 不同 run 最终约 18 / 50 patches | 单次 run 差异不构成 target 的因果证据 |

已有基线（前三种 `random:*` 各 3 seeds；`fixed:0.25` 为新增结果）：

| `fix_split_mode` | Patch 数 | `sample/train_action_mse_error` | PushT 评测分数（报告为 IoU） |
|---|---|---|---|
| `random:0.0`（all-coarse） | 16 | epoch100 后约 0.0065 | 0.465 |
| `random:0.25`（固定 count，随机树） | 76 | 未观察到低于 all-coarse | 0.526 |
| `random:1.0`（all-fine） | 256 | epoch100 后约 0.0045，3 seeds 均呈优势 | 0.587 |
| `fixed:0.25`（固定 count 与布局） | 76 | epoch400 后平均介于 coarse/fine 之间 | 尚未报告 |

两组观察需区分训练区间：

- 此前 `train/diffusion_loss`：`random:0.25 > random:0.0 > random:1.0`；
  随机中间树的训练 loss 更高，闭环分数却高于 coarse，两者排序不完全一致。
- 新增 epoch400 后，`sample/train_action_mse_error` 与 `train/diffusion_loss`
  均呈 `random:0.0 > fixed:0.25 > random:1.0` 的平均趋势，曲线仍有交叉。
  这是后期区间趋势，不是逐时刻严格排序，也不能直接合并成四组的同窗口排序。

epoch50 前 coarse/fine 差异可能不明显，不能据此提前淘汰 fine。表中为用户报告的
汇总值；验证 diffusion loss、`fixed:0.25` 的 seed 数与闭环表现仍待补齐。

代码已确认：

- `budget_loss_weight == 0.0` 时跳过 target 对应的 budget loss；target 不直接进入
  action loss、router 初始化、sampling 或 optimizer。
- `random_topology_prob` 保持当前 router count，且替换帧不提供直接的 router action
  梯度；count=16 时只有 all-coarse，无法借此探索更深子树。
- learned router 的 train count 来自 Logistic hard sample；默认 split 编码下 eval
  按 `logit > 0` 构树。平均 count 不能反推出 logit、饱和程度或 deterministic tree。
- `random:0.25` 在当前 80 个内部节点中固定执行 20 次合法 split，得到 76 leaves；
  训练和 eval 每次 forward 都重新采树，同一样本的观测帧共享一棵树。它不等于
  每个节点以 25% 概率 split，也不等于固定一种中间 topology。
- `fixed:0.25` 同样为 76 leaves，但只在初始化时采一棵合法树；布局作为 buffer
  随 checkpoint 保存，所有样本、帧和 train/eval 共用，忽略 `random_topology_prob`。
  当前布局随初始化 RNG 状态确定，固定布局的采样不会推进后续模型初始化的 RNG。
- `sample/train_action_mse_error` 使用缓存的首个训练 batch，以 eval 模式采样动作，
  默认使用 EMA；含 diffusion sampling 随机性，随机树还增加 topology 随机性。
  它不是全训练集误差、验证误差或反传的 diffusion loss。
- gradient norm 只有大小，没有 split/stop 方向信息。

若上述评测值来自当前 runner 的 `test/mean_score`，其定义是每条 rollout 的最大
归一化 coverage reward 再取均值，并非严格的交并比 IoU；正式记录需注明指标来源。

实验需保存 resolved config、seed、checkpoint、代码版本、launcher 参数和初始化来源。
当前本地 FPN output head 没有显式 all-fine 初始化，相关记录需对应实际实验版本。

## 二、核心机制与待验证假设

### 2.1 当前 ST 不直接比较两种合法 topology

当前路径：

```text
hard decision → hard leaves → 只 gather 选中 leaves
→ content embedding × ST weight → 加 position/level embedding
→ visual Transformer → diffusion loss
```

关键代码：

- `diffusion_policy/model/roi/foveation_net.py`：`_sample_split_decisions`、
  `_build_soft_and_hard_leaf_masks`、`_pack_selected_patches`。
- `diffusion_policy/policy/foveation_dp_transformer_policy.py`：
  `FoveationObsEncoder.forward` 的 content ST gate。

当前梯度衡量的是“缩放已选 token 的 content 是否有利”，没有直接计算替代 topology
的内容、位置、序列长度和 attention normalization。它可能与真实 split 收益方向
不一致，但**有偏不等于必然偏向 stop**，需在同一 checkpoint 上测量。

### 2.2 训练覆盖不足与共同适应

默认各尺度共享 patch encoder 和 visual Transformer，因此不能说 fine encoder
完全没训练。可能缺少训练的是 fine 输入分布、层级表示和长序列交互；启用独立
level adapter 时，还可能出现对应参数更新不足。

某 root hard stop 后，该样本该子树的 child logits 没有直接 action-loss 梯度，
但共享 FPN/head 的更新仍可间接改变它们。由此可能形成：

```text
fine 出现减少 → policy 更适应 coarse → 重开 fine 暂时不利 → 更少 split
```

这应称为“难以恢复的 coarse 状态”，尚不能认定为严格吸收态；未数值饱和时仍可能
通过随机 sampling 重开，需要 reopen rate 和 logit 分布验证。

### 2.3 固定中间树有收益，随机切换的代价需单独检验

all-fine 的训练 diffusion loss、后期采样 MSE 更低，闭环分数更高，使“该设置下
coarse 已足够”的解释变弱，也支持避免按早期成熟度选择 topology。但不同模型参数
下的整体优势，不等于当前 supernet 中每次局部 split 都有正收益，不能单独定位 ST。

`fixed:0.25` 后期两项误差均低于 coarse，说明至少该固定混合布局能够被有效学习，
削弱了“中间尺度或混合表示必然无益”的解释。结合此前 `random:0.25` 的结果，
反复切树带来的跨结构适应、梯度冲突，以及固定布局允许的路径专门化更值得关注。
但具体空间布局与各层叶数也会影响收益，需同训练区间、多布局对照才能归因于随机性。

新的整体排序既不保证任意局部 split 都改善 loss，也不证明局部优化存在或不存在
上升区间：比较的是不同模型参数下的结果，仍需同一 policy 的 paired 子树比较。
`fixed:0.25` 的闭环表现尚未知，不能从训练误差推断其优于 `random:0.25`。

PushT 当前确实开启 crop/shift，且未同步修改 state/action。先关闭空间增强做对照：
若增强模拟相机扰动，保留世界坐标标签可以合理；若模拟任务坐标变换，才需要一致
处理位置、目标与 action，不能直接把“同步修改 action”作为通用修复。

### 2.4 论文启发：先成熟分支，再学习选择，但仍需检验梯度

AdaFocus 的 policy 指选择网络，对应本文 router，而非动作预测 policy。以下依据
本地 quadtree 论文原文；分阶段训练值得优先验证，尚不能保证单独解决 collapse。

| 论文与位置 | 可迁移的机制及边界 |
|---|---|
| AdaFocus V1，§3.2–3.3，PDF p4–5 | 随机 crop 训练识别网络 → 冻结网络、PPO 学选择 → 固定选择、微调识别。扩展 AdaFocus+ 已有是否启用局部分支的决策，奖励为启用/跳过的置信度差减计算代价 |
| AdaFocus V2，§3.3，PDF p5；V3，附录 C | V2 用辅助监督、额外随机 crop loss 和 router 输入 stop-gradient 稳定端到端训练；V3 在部分数据集仍采用先随机 policy 训练 backbone、再单独训练选择网络。说明关键是覆盖与稳定监督，而非必须三阶段 |
| MSViT，§2.2、附录 C.3/F | 预训练 backbone、共享尺度参数、gate LR warmup 与 GBaS；附录直接分析了裁掉未选 token 会丢失 coarse/fine 梯度项、增加噪声，但实验精度影响较小，不能据此认定 hard gather 必然 collapse |
| FVC，§4.1、§5(C)，PDF p6–8 | 先训练基础压缩网络，再加入 Gumbel 分辨率/划分选择；不同分支恢复到同一特征网格。也是可学习分辨率对照，但优化 rate-distortion，未证明 action-only 不会退化 |
| Q-attention QTE，§IV、Algorithm 2 | coarse 输入可能无法判断细节价值，通过展开子树累积深层 value；支持诊断多层 split 收益，不能直接把其空间 Q 值当作 diffusion advantage |
| APT，§3.2，PDF p4–5 | resized coarse embedding 加零初始化的高分辨率聚合残差，有助于保留成熟表示；它也计算高分辨率子 patch embedding，与当前 sparse encoder 的成本不同 |
| MRViT/Quadformer §3.2；STTS §3.2 | 固定 leaf count / Top-K 将位置选择与总量分开，适合作为空间分配对照；结构上阻止 count collapse，不代表已修好自由 stop/split |

AdaFocus+ 与 FVC 补充了基础 AdaFocus/MSViT 的比较范围；这些论文都没有证明纯任务
loss 天然保证合理 routing。理想嵌套函数族只保证充分优化后 fine 的最优 loss 不高于
coarse，允许两者相等，不能推出当前共享参数下每次 split 都有收益。

## 三、最小诊断与训练方案

### 3.1 从零训练多拓扑动作 policy

| 比较 | 回答的问题 |
|---|---|
| 分别训练 coarse / fixed:0.25 / fine | 已观察到固定拓扑的后期误差层次，闭环与多布局差异如何？ |
| 同 count 的 fixed:0.25 / random:0.25 | 稳定布局与反复切树的训练、泛化代价如何？ |
| 同一多拓扑 supernet 上 paired stop / split | 当前共享 policy 下，这次 split 是否有收益？ |

复用已有固定/随机树 checkpoint，在 epoch400 后的共同窗口比较均值与波动，补齐
验证 loss、固定树 rollout 和 seed 结果；保存 tree mask 与各层叶数，复核多个布局。

阶段 A 按用户选择 **from scratch**，不加载已有 fine checkpoint。绕过 router，
仅用独立 topology sampler 训练 patch encoder、visual Transformer 和动作模型。
当前代码已支持逐样本选择 ratio、观测帧共享树的语法：

```yaml
fix_split_mode: "random:0.0,0.25,0.5,0.75,1.0"
```

- 每个样本先从列表等概率选择一个 ratio，再按该 count 随机生成合法树；同一样本
  的观测帧共享 ratio 和布局。列表表示离散候选，不是逐节点 split 概率。
- 当前 80 个内部节点对应 split 数 `[0,20,40,60,80]`、leaf 数 `[16,76,136,196,256]`。
  首版每种 ratio 占 20%，每个样本只跑一棵树，不依赖 router count 或 budget target。
- 目标是各 ratio 下 diffusion loss 的加权平均。不同 count 和新布局由同一套参数
  学习；采样比例固定，避免 router 因某路径早期较差而减少其训练机会。

此前单一 `random:0.25` 较差，不足以否定此方案：它未覆盖其他 count，也不能判断
加入 coarse/fine 训练后的迁移效果。但更多随机树不保证更容易优化，混合训练仍可能
有梯度冲突。等比例采样不等于各层得到等量训练，也不保证相同 epoch 内达到各自
专用模型水平；需分 ratio 记录验证 diffusion loss、采样 MSE 和 rollout。

进入 router 阶段前，用独立于训练采样的固定评测树集检查各 ratio、新布局的表现及
fine 收益是否形成，不只看混合平均 loss。若 fine 明显欠训练，可提高其采样权重，
或参考 AdaFocus V2 增加同 batch 的 fine loss；仍从零开始，额外 loss 不作为首版必需。
鲁棒不等于强制不同树预测一致；warmup 由各路径表现决定，不硬编码为 100/400 epoch。

当前按 batch 最大 leaf 数 padding，混合 batch 出现 fine 时通常补至 256 tokens，
平均 count=136 不代表相应计算节省。若训练成本成为瓶颈，再考虑按 ratio 分组或
分 microbatch 计算；首版优先保持采样与训练逻辑简单。

### 3.2 定义清楚 paired 子树比较

冻结同一个 policy，每个样本选择少量节点，每个 pair 只干预一个节点的决策：

```text
T_stop ：节点成为 parent leaf，隐藏后代
T_split：节点展开，后代继续按当前 router 的决策构树
```

两棵树共享其他节点的随机决策。只有最后一个可分裂层，`T_split` 才必然是四个
child leaves；上层若强制四个 child 都 stop，会遗漏深层细节的价值。

另加少量 `parent vs fully-fine subtree` 诊断，检测“单层展开无益、继续展开有益”
的情况。它衡量潜在子树收益，与当前 router 的条件收益是不同目标。

不能只从当前树 reachable nodes 采样，否则长期关闭的子树仍缺少 router 训练。
探索比较需允许强制祖先展开，并与当前树分布下的比较分别记录；policy sampler
本身不会自动解决 router 的深层覆盖。

第一版让冻结的 policy 处于 `eval()`，关闭 dropout，并显式共享图像增强、action
目标、diffusion timestep/noise。不同序列长度下，同一 RNG seed 不保证对应 token
共享 dropout mask。对关键诊断重复若干 noise/timestep，估计收益及其不确定性。

### 3.3 paired router loss 及其边界

以下采用 canonical split logit（默认 `split_action_value=1`）。定义：

$
p_n=\sigma(l_n),\qquad A_n=L_{\text{stop}}-L_{\text{split}}
$

$
L_{\text{router},n}=-p_n\operatorname{sg}(A_n),\qquad
\frac{\partial L_{\text{router},n}}{\partial l_n}
=-p_n(1-p_n)A_n
$

它与混合 stop/split loss 的局部梯度相同。冻结 policy 和比较上下文后，正 advantage
给该 logit 提供提高 split 的方向，但仍有以下限制：

- sigmoid 饱和会使恢复缓慢，仍需探索；非饱和正 bias 只是初始化策略。
- 共享参数和多个节点的梯度冲突，意味着每个节点的实际 logit 不保证同时改善。
- 若估计原随机树目标，需考虑到达概率与节点采样概率；按 level 均衡或强制祖先
  展开属于目标重加权/辅助探索，不能直接称为原目标的无偏梯度。
- action 收益相等时没有细分偏好；若要求默认 fine，需显式引入 prior/margin。

paired loss 可以继续使用 hard gather，反事实信号来自另一棵树的 forward，
无需先实现 full-bank attention。

### 3.4 权重交接与分阶段优化

| 阶段 | Topology 来源 | 更新参数 | 目标 |
|---|---|---|---|
| A：从零训练动作 policy | 多 ratio 独立随机树 | `obs_encoder`、diffusion `model` | diffusion loss，形成多拓扑能力 |
| B0：优先做最小 ST 对照 | 现有 router Logistic sampling | router encoder、FPN/head、condition projection | 保持现有 diffusion loss + ST，检验成熟且冻结的动作路径是否已足够 |
| B：按需加入 paired 与独立探索 | router 决策与独立探索，构造 paired 子树 | 同 B0 | 测量收益并与 ST 对照，进一步定位信用分配/探索问题 |
| C：受控微调 | learned tree + 独立探索树 | 先固定 router 微调动作 policy，需要时交替更新 | 适应部署拓扑并保留其他路径能力 |

A→B0 优先加载经各 ratio 验证的阶段 A 完整 EMA 模型权重，保持同一初始化来源：

- `obs_encoder.*`：patch embedding、visual Transformer、position/level/CLS 与 norm。
- `model.*`：diffusion Transformer；同时恢复 normalizer 并保持数据、几何与 diffusion 配置一致。
- 当前从零初始化的 `foveation_net.global_encoder`、`foveation_net.router` 和
  `router_condition_projection` 在 A 中被绕过，可直接沿用保存的未训练初始化，无需
  再重置，但不能称为已预训练。阶段 A 的 EMA 存储会省略共享的 global encoder，
  此部分需从同一 checkpoint 的 live model 补齐；其余权重保持 EMA 来源。

用新阶段配置构造模型，设置 `fix_split_mode=null`、`budget_loss_weight=0`、
`random_topology_prob=0`；对于当前 ResNet router，显式解除其 encoder/head/condition
projection 的冻结。新建 optimizer、LR schedule 与 EMA，不沿用 A 的完整 resume
状态；“完整 EMA 权重”不包括沿用 A 的随机模式、冻结状态和 optimizer。

B0 冻结 `obs_encoder` 和 diffusion `model` 的参数并保持 eval，router 路径保持 train。
继续使用现有单次 forward、diffusion loss 与 ST backward；**不能对冻结动作路径使用
`no_grad` 或 detach embedding**，仍需经其输入梯度更新 router。首轮不加 paired loss、
独立 sampler 或新结构，保留现有 Logistic sampling，先检验这条最小训练路径。

先测同一 A checkpoint 在固定 coarse/fine/中间树下的 loss，再观察 B0 的任务表现、
各层 logits 与 live/EMA deterministic count。若 B0 能保留有效 fine 收益，说明这套
预训练与冻结策略可能已足够；若仍 coarse，不能单凭 count 判定 ST 有错，需再测局部
paired advantage，并区分信用分配、探索饱和和局部收益不足。

需要进一步定位时，从相同 A checkpoint 和 router 初始化开展 B 对照：

- paired：两个动作网络 forward 可用 `no_grad`，共享输入与 diffusion noise/timestep；
  loss 差 detach 后构造 §3.3 的目标，只对 router 图反传。
- ST 对照：沿用 B0 的梯度路径。与 paired 比较时匹配 policy 模式、采样上下文及噪声；
  独立探索单独做消融，不能同时修改探索和 estimator 后把差异全部归因于 ST。

B0/B 只更新 router 参数，前向仍包含完整动作网络。确认 router 能利用正收益后进入 C，
先固定 router、小 LR 微调动作网络，再按需交替 policy steps / router steps。每类
step 只更新对应参数，保留独立探索；同时端到端更新全模型可作后续消融，不是默认方案。

实现边界：现有 `freeze_obs_encoder()` 会连 router 一起冻结，不能直接用于 B；
`pretrained_ckpt` 当前仅在 policy 存在 `load_weight_from_ckpt` 时执行，而该类尚无
此方法，需补显式阶段加载。router encoder 无独立任务监督，不能照搬 AdaFocus V2
在 router 输入处 stop-gradient 的做法。

## 四、后续结构消融

仅在诊断支持时逐项加入，不与第一轮 estimator 对照捆绑：

| 改动 | 作用与边界 |
|---|---|
| Full-bank + 可微 attention participation | 另一条梯度代理路线；保留 bank 但仍 hard mask 无法恢复未选 token 信号，也不自动得到无偏反事实梯度 |
| Coarse base + fine residual / Haar | 降低替换粗表示的风险；只有关闭细节能恢复完整 coarse 计算，才具有嵌套保证，零值 detail token 仍可能改变 attention |
| Attention logit 加 `log(area)` | 修正叶子数量造成的面积偏置，不保证不同 key/value 下区域 attention mass 不变；不能直接套用到重叠的 Haar detail token |
| 更高分辨率 router 输入、独立 level head/bias | 针对细节不可见或跨层梯度冲突做消融 |

单独调整零权重 target、初始化 bias、temperature、split action 编码或 LR，
不能验证或修复 ST 的反事实信用分配问题。

## 五、判据与执行顺序

关键指标：

- 各 topology 的验证 diffusion loss、重复采样 action MSE、各 seed rollout 分数；
  记录共同后期窗口的均值与波动、checkpoint epoch、固定布局及各层叶数。
- 分层 paired advantage 的均值、不确定性与正值比例；区分当前子树与 fully-fine 比较。
- 同一 checkpoint、匹配上下文下 ST 与 paired 的 **logit 梯度**方向一致率，
  剔除近零或符号不确定的收益，并记录实际 logit 更新方向。
- 每层训练覆盖、logit 分布、reopen rate，以及 train sampled / live deterministic /
  EMA deterministic count。

执行顺序：

```text
固定/随机实验作为性能参照 → 从零用多 ratio 随机树训练动作 policy，分 ratio 验证
→ 加载 A 的 EMA、冻结动作路径，仅用现有 ST 训练 router（B0）
→ 按需测量 paired 收益，再分别对照 estimator 与独立探索
→ 交替优化或单项结构消融 → 再引入 budget
```

**失败判据：存在稳定、可重复的 split 收益，router 却持续合并对应区域。**
若收益接近零或为负，不能仅凭 coarse 判定失败。只有控制其他条件后，paired 相比
ST 更稳定地利用正收益，才支持 ST 是重要成因，仍需结合 rollout 判断实际价值。

## 六、首轮实现涉及的文件

- `diffusion_policy/model/roi/foveation_net.py`：多 ratio 解析与逐样本 count 采样；后续增加显式 topology 覆盖、paired 子树。
- `diffusion_policy/policy/foveation_dp_transformer_policy.py`：逐样本 loss、显式 noise/timestep 输入；动作路径与 router 的独立冻结及参数分组。
- `diffusion_policy/workspace/train_foveation_dp_transformer_workspace.py`：分 ratio 验证、显式阶段权重加载与状态初始化、冻结诊断和 estimator 对照。
- `configs/pusht256_foveation_dp.yaml`：from-scratch、多 ratio 采样、诊断与对照配置。
- `diffusion_policy/dataset/pusht_image_dataset.py`：暴露关闭 crop/shift 的开关。

后续交替训练再增加 optimizer 分组与训练调度；full-bank 和 attention 改动不属于首轮。
本文仅整理分析与方案，尚未修改训练代码。
