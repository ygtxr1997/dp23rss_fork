# Mixed-resolution patch 的位置编码调研与 APT 方案开发计划

## 1. 目标与结论

本文面向 `FoveationNet -> FoveationObsEncoder` 的 mixed-resolution patch
序列，确定下一版位置编码方案。当前实现使用“连续几何 MLP + 离散 level
embedding”；下一版以 APT 的方案为主：

> 在最高分辨率、最小 patch 对应的二维网格上学习一张 position embedding
> map，再把它插值到每个较粗尺度；根据 Quadtree 实际选出的 leaf，从对应
> 尺度的位置图中 gather PE。

关键结论：

* PE 参数和每个尺度的候选 PE map 不读取图像内容；固定 checkpoint、输入尺寸
  和 patch 几何时，它们是确定的。
* 不同图像会通过 Router 产生不同的 Quadtree leaf 集合，因此实际 gather 出来
  的 PE 序列、序列长度和尺度组成通常不同。这是 tokenizer 布局导致的间接
  内容依赖，不是 position encoder 直接依赖像素。
* 不使用动态序列中的 ordinal index 作为空间位置。token 在 packed sequence 中
  排第几个，不应改变其图像空间 PE。
* 首版使用 learned 2D absolute PE + bicubic interpolation；不实现 pairwise
  relative position bias 和 RoPE，二者保留为后续实验方向。
* 本次只借鉴 APT 的位置编码，不实现其多尺度 patch aggregation、ZeroMLP、
  FlashAttention sequence packing 或基于 entropy 的 tokenizer。

## 2. 关键调研结果

参考论文位于本机 `'/Users/greensongavin/Library/CloudStorage/OneDrive-个人/论文/Embodied AI/quadtree/'` 目录。

### 2.1 APT：最细网格 PE 向各尺度插值

论文：*Accelerating Vision Transformers with Adaptive Patch Sizes*，ICLR
2026，`2026ICLR_APT_cmu.pdf`，重点见 Section 3.3。

APT 使用大小为 `p, 2p, 4p, ...` 的 patch。首先在最小 patch `p` 对应的
`H/p x W/p` 网格上定义基础位置编码；对大小为 `s*p` 的 patch，把基础 PE
map 采样/插值成 `H/(s*p) x W/(s*p)`，然后从该尺度网格中选择 leaf 的 PE。
论文把该方式描述为兼容 learned、sinusoidal 和 RoPE。

优点：

* 所有尺度共享一个基础 PE 参数，不需要每个尺度独立学习一张表；
* 天然匹配规则 dyadic Quadtree；
* 容易复用普通 ViT 的二维 learned APE；
* 位置生成复杂度与候选 patch bank 大小线性相关，不引入 `N x N` bias。

### 2.2 MSViT：只学习 fine PE，coarse PE 确定性插值

论文：*MSViT: Dynamic Mixed-scale Tokenization for Vision Transformers*，
ICCV Workshop 2023，`2023ICCVW_MSViT_qualcomm.pdf`，重点见 Section 2.1
和 Equation (4)-(5)。

MSViT 只学习 fine-scale PE `rho_f`，并通过线性插值得到 coarse-scale PE：

```text
rho_c = interpolate(rho_f)
```

内容相关 gate 决定某个区域使用一个 coarse token，还是对应的一组 fine
tokens。因此 `rho_f/rho_c` 本身对样本共享，但最终选出的 PE 子集随 gate
结果变化。该工作进一步支持“共享 fine PE、确定性生成其他尺度”这一设计。

### 2.3 Mixed-Resolution ViT / Quadformer：按 patch 中心编码二维位置

论文：*Vision Transformers with Mixed-Resolution Tokenization*，CVPR
Workshop 2023，`2023CVPRW_MRViT_telau.pdf`，重点见 Section 3.1。

该工作不使用 vanilla ViT 的一维序列位置表，而是分别 embedding patch 中心的
`x/y` 坐标，再拼接成二维 PE。中心坐标定义在最小 patch 决定的网格中。论文
没有额外编码 `w/h` 或 Quadtree depth；尺度主要由 patch resize、中心所在的
dyadic lattice 和 leaf 布局间接体现。

### 2.4 固定网格与自适应网格的区别

| 场景 | 候选 PE | 同尺寸不同图像的实际 PE tensor |
| --- | --- | --- |
| 普通固定网格 ViT | 固定 | 相同 |
| APT/MSViT | 各尺度 PE map 固定 | leaf 尺度或位置不同时不同 |
| Quadformer 中心坐标 PE | 给定坐标的 PE 固定 | Quadtree 中心集合不同时不同 |
| 内容生成式 PE，例如 PEG/CPE | PE 函数直接读取 token feature | 即使几何布局相同也可能不同 |

本项目采用前三类中的 APT 路线，不采用内容生成式 PE。

## 3. 本项目的 APT 式位置编码设计

### 3.1 当前几何和 level 约定

设最高分辨率输入为 `(H, W)`，原生 patch 大小为 `p`，共有 `L` 个
resolution levels。当前 `FoveationNet` 约定：

* `level = 0` 是最低分辨率、覆盖范围最大的 patch；
* `level = L - 1` 是最高分辨率、覆盖范围为 `p x p` 的最细 patch；
* level `l` 相对于最高分辨率的覆盖倍率为
  `s_l = 2 ** (L - 1 - l)`；
* 该 level 的网格为：

```text
H_l = H / (p * s_l)
W_l = W / (p * s_l)
```

现有 `_validate_geometry()` 要继续保证 `H/W` 可以被
`p * 2 ** (L - 1)` 整除。

### 3.2 基础参数与多尺度 PE map

在 `FoveationObsEncoder` 中新增唯一可学习的二维基础参数：

```text
fine_pos_embed_1_D_Hf_Wf: [1, visual_dim, H / p, W / p]
```

初始化使用 ViT 常见的 truncated normal，`std=0.02`。对每个 level `l`
构造：

```text
pos_l_1_D_Hl_Wl = interpolate(
    fine_pos_embed_1_D_Hf_Wf,
    size=(H_l, W_l),
    mode="bicubic",
    align_corners=False,
    antialias=False,
)
```

其中最高分辨率 level 直接使用基础参数，不做无意义的 identity interpolate。
首版固定 interpolation 语义，不把 `mode/align_corners` 暴露成训练配置，避免
checkpoint 在不同运行中悄然改变坐标语义。

把每个 level 从 `[1,D,H_l,W_l]` 执行
`permute(0,2,3,1).reshape(H_l*W_l,D)`，明确按 row-major 展平，再按当前 patch
bank 的 level 顺序从低到高拼接：

```text
position_bank_K_D = cat([
    flatten(pos_0),
    flatten(pos_1),
    ...,
    flatten(pos_{L-1}),
])
```

这里的 `K = sum_l(H_l * W_l)` 必须与 `FoveationNet` 的 patch bank 完全一致。

### 3.3 leaf PE 的索引接口

当前 `_pack_selected_patches()` 已经计算 `selected_indices_B_T_N`，但在
`forward()` 内部取出用于 metadata 后没有返回。下一版把它作为显式接口保留：

```text
patch_bank_indices_B_T_N: long
```

约定：

* valid token 的值是 `[0, K)` 内的 patch-bank 全局索引；
* padding 位置写为 `-1`；
* encoder 先依据 valid mask 把 padding index 替换成安全值再 gather，随后用
  `token_valid_mask_B_T_N` 把 padding PE 清零；
* 该索引等价于 `(level, row, col)`，但直接复用现有 bank index 可以避免根据
  浮点中心坐标反推 row/column，也避免复制一套 level offset 逻辑。

位置编码 gather 为：

```text
position_B_T_N_D = position_bank_K_D[patch_bank_indices_B_T_N]
position_B_T_N_D *= token_valid_mask_B_T_N[..., None]
```

不能使用 packed token 的序列下标 `0..N-1` 查 PE。不同 leaf 数量会改变后续
token 的序列下标，但不应改变其真实空间位置。

### 3.4 token 组成

APT 主方案的默认 token 为：

```text
token = patch_projection(native_pixel_patch) + apt_position_embedding
```

当前的 `geometry_projection([cx, cy, h, w])` 和 `level_embedding` 从 observation
encoder 中移除，避免主实验同时混入三种几何信号，无法判断收益来源。配置中
保留下面的实验标识，但首版只接受该取值：

```text
position_encoding_type: apt_interpolated
```

本轮科研代码不兼容旧 geometry checkpoint，也不增加 `strict=False` 或权重迁移
逻辑。所有新训练配置显式设置 `apt_interpolated`。如果实验显示插值 PE 对尺度
区分不足，第二优先级消融是 `APT PE + level_embedding`；不要直接把 geometry
MLP、level embedding 和 APT PE 全部默认打开。

继续保留 `centers_xy_norm_B_T_N_2`、`sizes_hw_norm_B_T_N_2` 和
`levels_B_T_N` 输出，用于可视化、诊断和后续相对位置实验，即使 APT 主路径不
直接消费 centers/sizes。

### 3.5 梯度、dtype 和缓存

* `F.interpolate` 和 gather 都必须保留到 `fine_pos_embed` 的梯度。
* 训练阶段每次 forward 从基础参数生成多尺度 map，不能缓存 detached tensor，
  否则 optimizer 更新后会使用旧 PE。
* 每个 forward 只生成一次 `position_bank_K_D`，所有 `B/T` 和多相机样本共享；
  不为每张图重复 interpolate。
* 首版不做持久 eval cache。PE map 很小，优先保证 `.to(device/dtype)`、EMA 和
  checkpoint load 后始终正确；确认性能需要后再增加带显式失效条件的 cache。
* AMP 下 gather 后的 PE 在相加前转换成 `pixel_BT_N_D` 的 dtype，避免一次
  float32 PE 加法把整条 token path 提升到 float32。
* padding PE 必须为零，且 padding token 继续由
  `src_key_padding_mask=~token_valid_mask` 屏蔽。
* learned fine PE 单独进入 `weight_decay=0.0` 的 optimizer group；其余 observation
  encoder 和 Router 参数保持现有 weight decay 行为。

### 3.6 固定值与动态值的精确定义

对固定模型参数和输入几何，`position_bank_K_D[k]` 是固定的。如果两张不同图像
产生完全相同的 `patch_bank_indices` 和 valid mask，则它们拿到完全相同的 PE；
patch pixel feature 和 Transformer 输出仍然不同。

如果两张同尺寸图像产生不同 Quadtree，则：

* `fine_pos_embed` 参数相同；
* 每个 level 的候选 map 相同；
* gather 的 index、token 数量和最终 PE sequence 不同。

因此实现和测试中应把“候选 PE bank 固定”和“实例化 PE sequence 动态”分开。

## 4. 开发步骤

### Phase 1：补齐 FoveationNet 的 leaf 索引接口

文件：`diffusion_policy/model/roi/foveation_net.py`

1. 在 `_pack_selected_patches()` 中保留 bank index。
2. `forward()` 构造 metadata 时使用 safe index，但同时返回 padding 为 `-1` 的
   `patch_bank_indices_B_T_N`。
3. 更新返回值注释，明确 patch bank 是“level 从低到高、level 内 row-major”。
4. 不改变 leaf 选择、ST weight、patch 排序和 budget loss 行为。

### Phase 2：实现 APT position bank 和 gather

文件：`diffusion_policy/policy/foveation_dp_transformer_policy.py`

1. `FoveationObsEncoder.__init__()` 接收 `input_hw` 和
   `position_encoding_type`。
2. 创建 `fine_pos_embed_1_D_Hf_Wf`，并保存/推导所有
   `level_grid_hw` 和 level offsets。
3. 新增私有函数，建议命名：
   * `_build_multiscale_position_bank()`；
   * `_gather_position_embeddings()`。
4. 检查生成的 `K` 与输入 bank index 上界一致；报错信息包含 `input_hw`、
   `patch_size`、`num_levels` 和各 level grid，方便 debug。
5. 用 `pixel + apt_position` 替换
   `pixel + geometry_projection + level_embedding`，并删除旧位置模块。
6. 保持 TransformerEncoder、padding mask、普通 mean pooling 和输出 shape
   不变。

### Phase 3：配置和模型集成

文件：

* `configs/liberoft_foveation_dp.yaml`；
* 其他启用 FoveationPolicy 的配置；
* `plan/foveation_net.md`。

步骤：

1. 新配置显式加入 `position_encoding_type: apt_interpolated`。
2. policy 根据 RGB `shape_meta` 把 `(H, W)` 传给共享的
   `FoveationObsEncoder`；多相机继续要求相同 `C/H/W`，共享同一套 PE。
3. policy 初始化时验证 observation encoder 与 FoveationNet 的 `level_grid_hw`
   和总 bank size 完全一致。
4. 将 `fine_pos_embed_1_D_Hf_Wf` 从 observation encoder 的 weight-decay 参数组
   中单独移到 `weight_decay=0.0`。
5. 在 `plan/foveation_net.md` 中把“连续几何 MLP + level embedding 为首版
   方案”标注为旧版，并链接到本文，防止两份设计文档冲突。
6. APT checkpoint 只保证同配置严格恢复；修改现有配置后使用新实验目录，不能
   从旧 geometry run 执行 resume。

### Phase 4：验证

按当前科研代码约定，优先复用已有 `tests/test_foveation_net.py` 或 debug
入口，不为该改动单独新增测试脚本。至少验证：

1. **全粗、全细、mixed leaf shape**：三种路由结果都能生成正确的
   `[B,T,N,D]` PE，并得到 `[B,T,visual_dim]` 输出。
2. **index 对齐**：每个 level 的左上、中心、右下 token 与
   `position_bank[offset_l + row * W_l + col]` 一致。
3. **内容不变性**：修改 patch pixels、保持 bank indices 不变，gather 的 PE
   完全相同。
4. **布局敏感性**：保持输入 `H/W` 不变，修改 leaf indices 后实际 PE sequence
   相应变化。
5. **padding 隔离**：修改 padding index/patch 值不影响 eval 输出；padding PE
   为零，attention 和 mean pooling 均忽略 padding。
6. **梯度**：一次 backward 后 `fine_pos_embed.grad` 非空、finite；Router 的
   ST 梯度路径仍由现有 weighted patch branch 保持。
7. **非方形输入**：使用可整除的 `H != W` geometry 检查 x/y 和 flatten 顺序
   没有互换。
8. **AMP/device**：CUDA autocast 下 token dtype 符合预期，没有因 PE 加法整体
   升为 float32。
9. **优化器**：fine PE 只出现在一个 no-weight-decay 参数组中。
10. **checkpoint**：APT live/EMA state dict 可严格 round trip；不测试旧 geometry
    checkpoint 兼容。
11. **静态检查**：对修改文件执行 `python -m py_compile`。

## 5. 验收标准

* 新训练配置默认运行 `apt_interpolated` 主方案。
* 模型只有一张 learned fine-grid PE 参数，各粗尺度由它确定性插值得到。
* leaf PE 通过 patch-bank 空间索引 gather，不依赖 packed sequence ordinal。
* 相同 bank index 在不同图像间得到相同 PE；不同自适应布局得到相应的不同
  PE sequence。
* valid token 数、padding mask、ST leaf gradient、mean pooling 和下游
  `TransformerForDiffusion` 接口均不发生行为回归。
* 不新增 `N x N` 的位置 bias，位置编码的额外时间/显存开销相对 patch
  Transformer 可忽略。

## 6. 风险与后续消融顺序

1. **尺度表达不够显式**：APT 插值 PE 主要表达空间位置，尺度只通过所选网格
   隐式体现。第一消融为 `APT PE + learned level embedding`。
2. **learned APE 绑定训练分辨率**：首版固定 `input_hw`。以后跨输入分辨率加载
   checkpoint 时，在 load 阶段把 fine PE 插值到新的 fine grid，并记录原始
   grid shape，不能静默 reshape。
3. **粗尺度插值过度平滑**：先保持 bicubic 基线；如效果不佳，再比较 area
   pooling、bilinear 和直接按连续 patch center 采样，不在首版同时引入。
4. **需要更完整的区域关系**：若 dense/robotics 任务需要显式理解覆盖范围，
   后续比较 `(cx, cy, w, h)` Fourier/MLP 或 box-relative attention bias。
5. **RoPE**：若未来替换为 2D RoPE，应根据连续 patch center 重新生成 rotary
   phase，并单独加入 scale/depth 信号；不要按动态序列下标应用 1D RoPE。
