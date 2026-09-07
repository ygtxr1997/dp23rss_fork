# Foveation Router：从 pooled global template 到 node-aligned input conditioning

## 1. 当前状态与结论

当前 PushT 配置的 quadtree 几何为：

- 输入图像：`256 x 256`。
- 三个 patch level：`4 x 4`、`8 x 8`、`16 x 16`。
- 需要预测 split 的 internal node 数：`16 + 64 = 80`。
- 最少/最多 leaf 数：16/256。
- `target_num_patches=0.25` 对应 76 个 leaf，也就是 20 次 reachable split。

已经完成的实现：

1. 增加 router 总梯度和跨样本 Jaccard topology diversity 诊断。
2. 将 ST gate 从 RGB patch 输入移动到 Conv content embedding 之后、位置和 level
   embedding 之前，恢复 action/diffusion loss 到 router 的梯度路径。
3. 按 `training.router_grad_norm_every` 分解记录 diffusion、budget 和 total router
   gradient norm，以及 diffusion/budget gradient cosine similarity。
4. task routing 保持 noisy ST sample，budget 单独改用
   `sigmoid(split_logits)` 递归计算 clean expected leaf count。

当前观察：

- `grad/router_grad_norm` 不再降为 0，通常在 `0.001 ~ 0.005` 波动。
- 初始 `num_patches_mean` 已经在目标 76 附近波动。
- 当前 `sample/topology_div` 来自 EMA eval forward，初始约 0.05，最大约 0.08，
  有时降为 0；不同图像的 deterministic hard topology 仍然高度相似。

当前结论：

- 初始 count 已接近预算目标，因此预算匹配初始化不再是主线阶段。
- clean expected budget 去除了 budget 对 sampling variance 的直接惩罚；是否仍压制
  action gradient，可直接通过分量 gradient 诊断判断，不预先加入 hold/ramp。
- 当前最明显的结构瓶颈是：一个 mean-pooled global vector 直接输出 80 个具有固定
  空间含义的 logits，node 没有读取对应图像区域的 spatial feature。
- 下一项最高优先级是 node-aligned spatial router，而不是继续调整初始化或 budget
  schedule。

本计划当前不研究同一图像重复采样的 topology 差异。训练时保留现有 Logistic noise，
但不增加 `topology_div_noise`、重复采样 Monte Carlo baseline 或相关成功标准。

## 2. 最终目标与范围

目标 routing 形式为：

```text
split_logit[level, row, col]
= f_theta(
    spatial_feature[level, row, col],
    global_context
  )
```

期望最终 eval 行为：

- 相同图像得到确定性 topology。
- 不同 agent/block 布局产生不同 topology。
- 图像内容按 quadtree cell 平移后，split map 出现对应的空间响应。
- 细粒度 leaf 更倾向 action-relevant 区域，同时 expected/hard count 保持预算约束。

训练目标仍只使用：

```text
action/diffusion loss + budget loss
```

不加入 saliency teacher、interaction ROI supervision、topology diversity loss 或
entropy bonus。交互区域标注如可获得，只用于离线评价。

## 3. 公共诊断指标

### 3.1 Deterministic input conditioning

优先记录：

- `sample/topology_div_input_live`：live model 在 eval hard threshold 下，不同图像的
  topology Jaccard distance。
- `sample/topology_div_input_ema`：EMA model 的对应指标，作为最终推理行为诊断。
- `router/logit_inter_sample_std`：同一 node 在不同 batch 图像间的 logit 标准差。
- `router/hard_sign_agreement`：同一 node 在不同图像间 hard sign 的一致率。
- `router/reachable_ratio`：各 level 实际被父节点到达的比例。

现有 `sample/topology_div` 应解释为 EMA deterministic input diversity，不能解释为
sampling exploration。

按 quadtree level 辅助记录：

- `router/split_prob_mean`：`sigmoid(split_logits)` 的均值。
- `router/split_entropy`：只用于判断 logits 是否整体饱和，不作为重复采样指标。
- `router/logit_abs_p50/p90/p99`。

解释规则：

- logit 在不同图像间有变化，但 hard sign 几乎完全一致：输入信号尚不足以改变 topology，
  或被 EMA threshold 平滑掩盖。
- inter-sample std 很低且 hard sign agreement 很高：router 主要表现为固定 node bias。
- input diversity 增加但空间平移响应错误：模型在使用输入，但不是 node-aligned 的空间
  条件化。

### 3.2 Count 和梯度

记录：

- clean expected leaf count。
- train hard count mean/std。
- eval hard count mean/std。
- `grad/router_diffusion_grad_norm`。
- `grad/router_budget_grad_norm`。
- `grad/router_diffusion_budget_cosine`。
- `grad/router_grad_norm`。

当前几何下的预期：

- clean expected count 接近 76。
- hard count 的长窗口均值接近 76，但允许 batch 间波动。
- diffusion gradient 保持有限、非零。
- count 接近目标后，budget gradient 不应长期主导 diffusion gradient。
- 若 budget/diffusion norm 相近且 cosine 长期接近 -1，说明预算仍在抵消任务梯度。

梯度分解按 `training.router_grad_norm_every` 低频记录，不扩展到每一步。

### 3.3 空间响应评价

不向训练增加空间监督，只做 evaluation：

1. 对输入做与 quadtree cell 对齐的平移，比较平移前后的 per-level split logit map。
2. 在固定 PushT 轨迹上可视化 agent、block 和接触区域移动时的 tree topology。
3. 若 simulator state 或离线 mask 易于取得，记录 interaction ROI 内外的 fine-leaf
   density ratio。

### 3.4 任务与预算保护指标

同时比较：

- `train/diffusion_loss`。
- `sample/train_action_mse_error`。
- rollout reward/success。
- clean expected count、hard count mean/std 和 `train/budget_loss`。

router architecture 实验不能以明显牺牲 action/eval 指标或预算稳定性换取 topology
diversity。

## 4. 当前主要假设

### 假设 A：随机 output head 形成固定 node template

当前主要路径为：

```text
256 x 256 image
-> global_level=0 的 64 x 64 image
-> stride-32 ResNet
-> 2 x 2 spatial tokens
-> AdaLN-Zero Transformer
-> mean pooling
-> 一个 global vector
-> output-head MLP 输出 80 个固定位置含义的 logits
```

初始化时 AdaLN-Zero blocks 近似 identity。不同 node 的初始 logit 差异主要来自
output head 最后一层不同输出通道的随机 weight/bias；它们容易形成固定的 node-specific
空间模板。不同图像只能通过同一个 pooled global descriptor 调整这 80 个输出，缺少
直接的 node-to-region 对齐。

### 假设 B：clean expected budget 已解决当前 estimator 的直接方差惩罚

sampled-relaxed budget 的逐样本 MSE 同时惩罚 expected count error 和 sampling
variance。clean expected budget 改为：

```text
clean_split_prob = sigmoid(split_logits)
clean_expected_count = tree_recursion(clean_split_prob)
```

它只约束期望预算，不改变 task 使用的 noisy ST routing。阶段 2 完成后不继续通过
初始化或 schedule 间接解决空间条件化问题。

### 假设 C：node-aligned feature 是当前最高优先级缺失项

ResNet 已经产生 spatial feature，但当前 64 x 64 输入、2 x 2 token 和最终 mean
pooling 丢失了 node 对应区域的信息。仅扩大 pooled router、增加 MLP 深度或替换
backbone，但仍由一个 global vector 输出 80 logits，都不能解决这个对应关系。

### 假设 D：ST estimator 可能仍无法比较 counterfactual topology

hard gather 只构造当前选中的 parent 或 children。ST gate 不完整表示：

- 未选择的 parent/children feature。
- position/level embedding 的离散替换。
- token count 和 attention token set 的变化。

因此，即使 node-aligned router 能对图像位置响应，action gradient 仍可能无法可靠比较
stop 与 split。只有空间结构修复后仍出现这一现象，才升级 estimator。

## 5. 实验阶段

每个阶段只改变一个主要机制。除阶段明确列出的变量外，保持数据、seed、action model、
patch encoder、LR、temperature schedule、budget weight 和 target 不变。

### 阶段 1：恢复并分解 router gradient

#### 状态

实现已完成：

- ST gate 已移动到 content embedding 之后。
- 已记录 total router grad norm。
- 已分解 diffusion/budget router grad norm 和 cosine similarity。

#### 短实验成功标准

- diffusion gradient 明确非零。
- total gradient 不是长期完全来自 budget。
- 能判断 budget 是否持续压制或抵消 diffusion gradient。

如果 diffusion gradient 仍接近 0，先核查 gate 和 task gradient path，不进入架构实验。

### 阶段 2：budget 改用 clean expected split probability

#### 状态

实现已完成。task routing 保持原 noisy ST sample；budget 使用：

```text
clean_split_prob = sigmoid(split_logits)
```

递归计算 expected leaf count。clean probability 不除以 temperature，因为 temperature
不改变 hard split 的真实采样概率。

配置：

```yaml
policy:
  foveation_net:
    budget_probability_type: clean_expected
```

`sampled_relaxed` 只保留为旧实现对照。

#### 唯一对照变量

```text
sampled_relaxed budget / clean_expected budget
```

#### 短实验成功标准

- clean expected count 和 hard count 长窗口均值接近 76。
- budget gradient 相比旧实现更稳定，并在 count 接近目标后减弱。
- diffusion gradient 保持有限、非零。
- diffusion loss、action MSE 和 rollout 指标不明显劣化。

阶段 2 不要求 deterministic input topology diversity 单独改善。

### 阶段 3：改成 node-aligned spatial router

这是阶段 2 后的下一项最高优先级工作。

#### 推荐结构：轻量 FPN router

不优先使用完整 U-Net。当前只需要 `4 x 4` 和 `8 x 8` 两个 internal-node
logit map：

```text
256 x 256 image
-> lightweight CNN encoder / feature pyramid
-> global bottleneck context
-> top-down fusion
-> node-aligned 4 x 4 feature -> shared 1 x 1 head -> 16 logits
-> node-aligned 8 x 8 feature -> shared 1 x 1 head -> 64 logits
-> level-major, row-major concatenate -> 80 logits
```

设计要求：

- router 使用与原图坐标对齐的 dense feature，不再从 64 x 64 输入只保留 2 x 2 token。
- 每个 logit 由对应 node 的 spatial feature 产生。
- 每个 level 使用空间共享的 `1 x 1` head，避免用 80 组独立 output rows 表示固定位置
  prior。
- decoder/top-down path 将 bottleneck global context 融合回 spatial feature，使模型
  同时看到局部物体和 agent-block 等全局关系。
- 输出仍为 `split_logits_B_T_M`，保持 level-major、row-major node 顺序。
- 保持 Logistic sampling、hard topology、ST task gradient、clean expected budget、
  temperature schedule 和 policy 输出接口不变。

完整 U-Net 只在轻量 FPN 确认缺少局部细节时作为后续对照。若使用 JEPA/ViT，必须保留
带 2D position 的 dense tokens，并通过 spatial reshape 或 level/coordinate node query
产生 logits，不能再次 mean pooling。

#### 唯一对照变量

```text
current pooled ResNet router / node-aligned FPN router
```

#### 成功标准

- cell-aligned 平移后，deterministic per-level split map 出现对应平移。
- 不同 agent/block 布局产生明显不同的 live/EMA deterministic topology。
- inter-sample logit std 增加，hard sign agreement 不再接近固定模板。
- 交互区域 fine-leaf density 高于非交互区域，且不依赖训练监督标签。
- clean expected/hard count、diffusion loss、action MSE 和 rollout 指标不明显劣化。

### 阶段 4：必要时升级 routing gradient estimator

只有在阶段 3 满足以下条件时进入：

- split logit map 已能随图像位置正确响应；
- deterministic input topology 已不再是固定模板；
- 但 action loss 仍不能稳定提高 action-relevant 区域的细粒度 leaf density。

优先方案：训练时对每个 internal node 同时构造 parent feature 和 aggregated child
feature，形成层级 soft mixture；推理仍使用 hard tree，使 action loss 能直接比较：

```text
stop:  parent representation
split: aggregated children representation
```

备选方案按顺序为：

1. 全部 multiscale patch 的 soft counterfactual feature，推理仍 hard gather。
2. 固定 split budget 的 differentiable tree/top-k selection，用于协调
   count-preserving topology swap。

不使用 saliency teacher 或 auxiliary spatial supervision。

#### 成功标准

- action-only gradient 能稳定改变对应区域的 split probability。
- 不同输入形成稳定、可解释且 action-relevant 的 topology。
- 与 hard-only ST 相比，action/eval 指标改善且额外显存可接受。

## 6. 降级为条件性备用的实验

### 6.1 Budget-calibrated initialization

当前初始 `num_patches_mean` 已在 76 附近，因此不作为主阶段。只有出现以下情况时才
重新考虑：

- node-aligned FPN 初始化后的 clean expected count 明显偏离 76；
- 初始 logits 已明显饱和；
- 某个 level 出现强烈的固定 split/stop bias。

若需要，使用当前几何的预算匹配 probability：

```text
16 * (1 + 3p + 12p^2) = 76
p ~= 0.448
initial split bias = logit(p) ~= -0.21
```

只调整 output head 初始化，不同时改变 budget schedule。

### 6.2 Budget hold + ramp

不预先加入训练 schedule。只有短实验明确显示以下情况时才启用：

- `grad/router_budget_grad_norm` 长期显著大于
  `grad/router_diffusion_grad_norm`；
- 两者 cosine 长期接近 -1；
- budget 更新阻止 inter-image logit variation 或空间响应形成。

若 count 已接近 76、budget gradient 已减弱，则不做 hold/ramp。

## 7. 决策规则

```text
阶段 1~2：短实验确认 gradient 与 clean budget
  |
  +-- diffusion gradient 接近 0
  |      -> 先修复 task gradient path，不进入 router architecture 实验
  |
  +-- budget gradient 长期压制/抵消 diffusion
  |      -> 先检查 clean budget；证据充分时才启用条件性 hold/ramp
  |
  +-- count 约为 76、diffusion gradient 非零，
         但 deterministic topology 相似或平移响应弱
         -> 阶段 3 node-aligned FPN

阶段 3：node-aligned spatial router
  |
  +-- spatial response、input diversity 和 action/eval 指标改善
  |      -> 保留 FPN，继续训练与评估
  |
  +-- spatial response 正确，但不聚焦 action-relevant 区域
  |      -> 阶段 4 counterfactual estimator
  |
  +-- 初始化 count 或 logits 明显异常
         -> 仅启用条件性 budget-calibrated initialization
```

## 8. 当前推荐执行顺序

```text
1. 完成阶段 1 的短梯度分解实验
2. 完成阶段 2 的 clean expected budget 对照确认
3. 优先实现并对照 node-aligned lightweight FPN router
4. 只有空间响应正确但 action allocation 仍失败时，升级 gradient estimator
5. initialization 和 budget hold/ramp 仅按诊断触发，不作为固定阶段
```

不要在同一次实验中同时改变 budget probability、spatial architecture 和 gradient
estimator。否则无法判断 deterministic topology 的改善来自哪一项机制。

每个阶段统一追加：

```text
实现/实验状态：
唯一修改变量：
运行命令和 checkpoint：
deterministic input topology：
spatial translation response：
count 和 action/budget gradient：
train/eval 指标：
观察、结论与下一阶段决策：
```
