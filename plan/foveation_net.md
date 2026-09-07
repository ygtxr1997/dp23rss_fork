## 1. 代码风格

* 在关键位置加上注释
* 变量如果是 tenser, numpy 类型，要用 `x_B_C_T_H_W` 的命名方式
* 减少非必要的异常处理，因为作为科研代码，用户运行时会尽可能保证变量合法，只在必要的地方报错方便 debug 即可
* 在未说明时，不要新增 test 脚本，只做静态编译检查即可，因为本地 coding 环境不可运行，不像实际服务器上的训练环境

## 2. 实现目标

### 2.1 FoveationNet

代码位置：diffusion_policy.model.roi.foveation_net.py

功能概述：给定原始高分辨率图像 x_high 的 uint8 tensor (B,C,T,H,W)，为方便处理可能需要都先转成 [-1,1] float 型，
从原始分辨率得到 l-1 个不同的下采样低分辨率图, 记分辨率从低到高的图为 {x1,x2,...,xl}, x_l 即 x_high,
下采样倍率固定为 0.5, 代表 xi 的分辨率是 x_{i+1} 的一半。
其中某个图像 xg 会输入 VJEPA V2.1 特征提取器得到 vg，g in [1,l]。
vg 进入一个 from scratch 训练的 Transformer + MLP 组成的 Adaptive Resolution (AR) Router，
AR Router 保留通过 AdaLN 注入 text embedding（以及其他条件 embedding）的能力。

AR Router 不再预测连续区域的中心和尺度，而是对每一帧预测多级四叉树内部节点的 stop/split logits。
AR tokenization 采用 Multi-resolution Patch Bank + Patch Selection：
先将所有分辨率图像分别按固定 patch_size 做原生 patchify，组成多分辨率 patch bank；
再从最低分辨率 x1 开始，根据层级路由决定保留当前 patch，或将其替换成高一层分辨率的 2x2 子 patch。
最高分辨率 x_l 的 patch 作为树的最后一层。

训练时每个内部节点使用带 Straight-Through 的 Gumbel-Sigmoid 做二元 stop/split 决策。
构造 soft leaf probability q、合法的 hard leaf mask H
以及 forward 等于 H、backward 使用 q 梯度的 ST mask m。
具体定义，q:软的叶节点选择概率，用于提供梯度, 计算 budget;
H:hard 的 0/1 叶节点 mask，决定 forward 真正选择哪些 patch;
训练mask m:straight-through mask，forward 等于 H_i，backward 按 q_i 求导;
R代表分裂复制操作, 用于为每个子节点计算父节点的 soft 概率。
H 用于从 patch bank 中 hard gather 叶节点 patch，gather 后的 m 继续与所选 patch 相乘交给下游任务，
使主任务梯度可以回传到 router；q 同时用于可微的软期望 token-budget loss。
推理时使用确定性的 hard decision。

例如 x3(x_high)/x2/x1 分别为 1024/512/256，patch_size=16，则 patch bank 分别包含 4096/1024/256 个 patch，
首版不自动进行宽高 padding，要求最高分辨率宽高能被 patch_size * 2**(num_levels-1) 整除，从而保证每层 patch 网格严格为上一层的 2 倍。
AR Router 对 256-res 层的 256 个节点和 512-res 层的 1024 个节点预测 split logits；
一个节点 stop 时输出当前 patch，split 时由对应的四个高分辨率子 patch 继续决策或输出。
Router 的输出维度应该可以通过输入分辨率、patch size以及层数计算出来。
最终叶节点完整覆盖图像，但不同区域可以使用不同分辨率，不要求高分辨率区域连续。

FoveationNet 输出所选的原始像素 patch，以及它们在原图坐标系中的位置、分辨率级别和 padding valid mask。
每帧选中的叶节点数 N 可以不同，batch 内按 Nmax padding，下游模型负责 patch embedding，并使用位置和分辨率信息理解 mixed-resolution patch sequence。

参数列表和默认值：
* 第一类，输入相关：
    * input_hw: Tuple[int,int], default=(1024,1024), 代表输入图像的高和宽
    * num_levels: int, default=3, 即l，代表总共多少个分辨率的图像，1个原始高分辨率+ l-1 个下采样低分辨率
    * global_level: int, default=0, 即g, 代表哪个分辨率的图像会输入到 VJEPA V2.1 的特征提取器中提取特征，g in [0,l-1]，0 代表使用最低分辨率图
* 第二类，模型复杂度：
    * global_encoder_type: str, default='vjepa2_1_vit_giant_384', 代表选择的 jepa 权重版本
    * torch_home: Optional[str], default=None，指定本地 torch hub 缓存路径；为 None 时由 torch hub 下载
    * router_layers/hidden_dim/... 等等 transformer 最常用的参数，默认值可以参考 vit 模型的一些设置
* 第三类，输出相关：
    * patch_size: int, default=16, 代表 AR tokenization 使用的 patch 大小，16x16
* 第四类，训练相关：
    * freeze_global_encoder: bool, default=True, 代表是否冻结 VJEPA V2.1
    * freeze_router: bool, default=False, 代表是否冻结 AR Router
    * router_temperature: float, default=1.0, 代表 Gumbel-Sigmoid 的训练温度，后续可以配置退火策略
    * target_num_patches: Optional[Union[int, float]], default=None, 代表期望的 mixed-resolution patch 数量, 如果值为[0,1]的float，则代表理论最大最小中间的某个线性插值
    * budget_loss_weight: float, default=0.0, 代表 soft token-budget loss 的权重

返回dict：
* patches_B_T_N_C_P_P：已乘 ST weight 的 float patch
* token_valid_mask_B_T_N：变长序列 padding mask, 1 代表有效 token, 0 代表 padding
* num_patches_B_T
* levels_B_T_N：0 为最低分辨率
* boxes_xyxy_B_T_N_4：原图像素坐标，采用半开区间 [x0,y0,x1,y1)
* centers_xy_norm_B_T_N_2：归一化中心坐标
* sizes_hw_norm_B_T_N_2：归一化覆盖范围
* st_leaf_weights_B_T_N
* split_logits_B_T_M, M代表只有前 L-1 层需要预测 split 的patch数，“按层从低到高、层内 row-major”的固定顺序
* budget_loss

接口函数：
* forward(self, x_B_C_T_H_W, cond_B_T_D=None), 内部用注释 `## 1.xxx`, `## 2.xxx` 等标记步骤
* budget_loss, 静态函数
* set_router_temperature(self, router_temp: float)
* vis_ar_tokens, 把当前 batch 的 AR token 选择情况用不同分辨率的原图像素可视化成 np.ndarray uint8，用半透明的不同颜色标出对应的patch来自哪个分辨率，方便 debug

内部函数:
* _validate_geometry()
* _build_image_pyramid()
* _patchify_pyramid()
* _encode_global_images()
* _build_router_condition()
* _sample_split_decisions()
* _build_soft_and_hard_leaf_masks()
* _pack_selected_patches()
* _build_patch_metadata()

.py 文件中包含的 class:
* FoveationNet
* AdaptiveResolutionRouter
* AdaLNTransformerBlock

已经确定的设计:
* 对 x1 下采样成多分辨率时，在GPU上执行效率更高
* AdaptiveResolutionRouter, 自己定义 Transformer block，复用 PyTorch 底层算子, 保持一定灵活性
* 使用多分辨率原生 patch bank 和层级四叉树路由，不使用 grid_sample 或连续 crop
* 每个内部节点只预测 stop/split，使用 ST Gumbel-Sigmoid
* hard leaf mask H 负责 patch 筛选，ST mask m 负责保留 router 的主任务梯度，soft leaf probability q 负责预算等辅助损失
* 输出原始像素 patch，由下游负责 embedding；同时输出原图坐标、分辨率级别和 valid mask
* mixed-resolution patch 数 N 是动态的，batch 内使用 padding 和 valid mask 对齐
* Gumbel temperature 要支持模拟退火，提供 set_router_temperature() 接口供 workspace 调用设置
* 目标 token budget 计算方式为 sum(q_k), budget loss 暂定为和目标期望值的 MSE
* VJEPA 权重通过 JEPAHelper 加载；helper 只保留 encoder 和 normalization buffers，predictor 权重加载后立即释放
* AR Router 保持 VJEPA 的每帧 N 个 visual token 数量不变，先在 N 维执行 Transformer，再 mean pooling 并通过 node-specific MLP 从 D 映射到固定顺序的 M 个内部节点 split logits
* `_encode_global_images()` 返回和输入时间维对齐的 `[B,T,N,D]` visual token；VJEPA 原始输出按空间网格 reshape，并将每个时间块重复回填到对应输入帧
* 暂时假设所有分辨率都可以被 16 整除，方便设计

与下游的接口约定：
* FoveationNet 只负责输出 patch-bank index、原图坐标、归一化中心、归一化覆盖尺寸、分辨率 level 和 valid mask，不在内部生成下游位置编码
* 当前下游使用 APT 式 learned fine-grid absolute PE，并为各粗尺度插值得到 PE map；完整设计见 `plan/position_embedding.md`

可以参考的代码：
* @diffusion_policy.model.roi.roinet.py
* @configs/liberoft_roi_dp.yaml


### 2.2 FoveationPolicy

代码位置：diffusion_policy.policy.foveation_dp_transformer_policy.py

功能概述：利用 FoveationNet 输出的 mixed-resolution patch sequence，作为输入给下游的 diffusion policy (DP) 模型，
数量动态变化的 patches + 几何信息 + padding mask 由新的 FoveationObsEncoder 处理，并聚合为每帧固定维度的 visual latent。
TransformerForDiffusion 继续接收固定形状的 `[B,T_obs,cond_dim]` observation condition，其时序结构、位置编码、causal mask 和 diffusion sampling 行为保持不变。

已经确定的设计：
* DP 模型复用 TransformerForDiffusion，忽略参考代码 roi_dp_transformer_hybrid_policy 的 MoE/Merge 和 ROI-specific 逻辑
* 动态 token 数只存在于 FoveationObsEncoder 内部；聚合后的固定维度 observation feature 再交给 TransformerForDiffusion
* shape_meta 中的每个 RGB key 都执行 foveation；多路 RGB 共享同一个 FoveationNet 和同一个 FoveationObsEncoder 参数，但每张图像仍独立产生 split decision
* 首版要求所有 RGB key 的 C/H/W 和 foveation geometry 相同，否则直接报错
* 多路 RGB 按 shape_meta 中固定的 key 顺序组织，沿 batch 维拼接后一次调用共享 FoveationNet，再拆回各相机并拼接每路 visual latent
* shape_meta 中的 low_dim observation 按固定 key 顺序展平，并与所有 RGB visual latent 拼接
* 没有文本或其他条件 embedding 时，policy 向 FoveationNet 传 `cond_B_T_D=None`，由 FoveationNet 在正确 device/dtype 上生成严格的 hidden-space 全零条件；不在 dataset/dataloader 中新增全零 tensor
* policy 暴露 `set_router_temperature(router_temp)`，只负责转发给共享 FoveationNet；具体模拟退火曲线和调用时机由 workspace/config 控制
* FoveationNet 输出的 patch 已经乘过 ST leaf weight，FoveationObsEncoder 不再重复乘权重，从而保留 diffusion task loss 到 Router 的梯度路径

FoveationObsEncoder 结构：
* 默认参数量和计算量要控制在和现有 ObsEncoder 差不多的水平，大约是几十M
* pixel embedding 由 `patch_embedding_type` 选择：`linear` 保留共享的 ViT-style Flatten + Linear baseline；`conv` 对每个 leaf 独立应用共享的两层 stride-2 Conv stem，再 Flatten + Linear 到 visual_dim。Conv 只建模 leaf 内部局部性，不跨越 leaf 边界
* 在最高分辨率 patch 网格学习一张二维 absolute PE，各粗尺度使用 bicubic interpolation 生成对应 PE map
* 根据 FoveationNet 返回的全局 patch-bank index gather leaf PE；padding index 为 `-1` 并由 valid mask 屏蔽
* pixel embedding 与 APT position embedding 相加后，输入使用 `batch_first=True` 的 TransformerEncoder；不再叠加旧 geometry MLP 或 level embedding
* 不添加 CLS token；`src_key_padding_mask` 由 `~token_valid_mask_B_T_N` 构造，TransformerEncoder 输出后仅对 valid patch token 做普通 mean pooling，得到该帧固定维度 visual latent
* 普通 mean pooling 对每个有效 leaf token 使用相同权重，不做面积加权、attentive pooling 或其他额外 token 重加权；mixed-resolution 下高分辨率区域因 token 数量更多而自然获得更高的 token-level 权重
* 首版不实现 token 两两之间的 pairwise relative position bias；后续若实验需要，再考虑自定义 attention bias 或 RoPE
* 单路输出形状为 `[B,T,visual_dim]`；最终 `obs_feature_dim = num_rgb * visual_dim + sum(low_dim_dims)`

训练和损失：
* compute_loss 保留现有 diffusion policy 的 action normalization、noise sampling、condition mask、prediction target 和 diffusion MSE 逻辑
* 多路 RGB 合并进 FoveationNet 的 batch 维，因此其 budget loss 自然对 batch、时间和 RGB 路数取 mean，新增相机不会自动放大正则项
* `total_loss = diffusion_loss + budget_loss`，其中 FoveationNet 返回的 budget_loss 已经乘过 budget_loss_weight
* 日志至少返回 total_loss、diffusion_loss、budget_loss 和实际 num_patches 的 mean；需要可视化时才返回按 RGB key 组织并 detach 的 foveation outputs，普通路径不保留完整 patch 输出
* train 模式使用 ST Gumbel-Sigmoid，eval 模式使用确定性 hard decision；temperature 不改变 eval 时以 logit 0 为阈值的 hard split 结果

保留的 public policy 函数：
* conditional_sample()
* predict_action()
* reset()
* set_normalizer()
* get_optimizer()
* compute_loss()
* forward()
* freeze_obs_encoder()
* print_training_status()
* 新增 set_router_temperature()

不保留的参考 policy 逻辑：
* `_build_roi_obs()`、ROI crop/coord-conv 和 ROI area/diversity loss
* `set_roi_area_loss_step()`
* MoE、teacher merge 和 domain adaptation 相关函数及参数

参考代码：
* @diffusion_policy.policy.roi_dp_transformer_hybrid_policy.py


### 2.3 FoveationWorkspace

代码位置：diffusion_policy.workspace.train_foveation_dp_transformer_workspace.py

功能概述：训练框架、train/val 日志

VJEPA 权重共享约束：
* VJEPA 始终冻结，`freeze_global_encoder` 必须为 True，不加入 optimizer，也不参与 EMA 更新
* 多路 RGB 共享同一个 FoveationNet，因此同一个 policy 内只持有一个 VJEPA module 实例
* live policy 和 EMA policy 必须引用同一个 VJEPA module object，不能各自持有数值相同但物理独立的权重副本
* workspace 创建 EMA policy 时，可以 deepcopy 其他模块，但随后必须把 EMA FoveationNet 的 global_encoder 重新绑定到 live policy 的同一实例；不能因为 `.to(device)`、train/eval 切换或 rollout 再创建副本
* checkpoint 只保存一份 VJEPA 权重，EMA state 中排除重复的 VJEPA keys；加载 checkpoint 后重新建立共享引用
* 初始化和 checkpoint 恢复后都要验证 live/EMA global_encoder 的对象 identity 相同、所有参数 `requires_grad=False`，并由 FoveationNet.train() 保证该模块始终处于 eval 模式
* 如果未来需要解冻 VJEPA，必须重新设计 optimizer、EMA 和共享策略，不能沿用当前约束

temperature 调度约定：
* workspace 在训练 forward 前通过 unwrap 后的 live policy 调用 `set_router_temperature()`；启用 EMA 时同步记录相同 temperature，保证 train/val 日志可比较
* 退火起始值、结束值、调度总步数和调度类型放在 config 中，policy 本身不维护 global_step 或退火曲线

其他相关配套文件：
* configs/liberoft_foveation_dp.yaml (参考 configs/liberoft_roi_dp.yaml)
* scripts/roi/sh_train_foveation_libero.sh (参考 scripts/roi/sh_train_roi_libero.sh)

参考代码：
* @diffusion_policy.workspace.train_roi_dp_transformer_workspace.py
