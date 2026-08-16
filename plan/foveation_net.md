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

需要进一步确定的设计：
* 下游如何融合原图二维坐标、分辨率级别与现有位置编码或 RoPE (下游任务再考虑，这里只要保证尽可能输出可能有用的tensor或信息即可)

可以参考的代码：
* @diffusion_policy.model.roi.roinet.py
* @configs/liberoft_roi_dp.yaml


### 2.2 FoveationPolicy

代码位置：diffusion_policy.policy.foveation_dp_transformer_policy.py

功能概述：利用 FoveationNet 输出的 mixed-resolution patch sequence，作为输入给下游的 diffusion policy (DP) 模型，
此时由于 DP 的输入不再是完整分辨率图像，而是数量动态变化的 patches + 位置编码 + padding mask，这里使用的 DP 结构需要做相应修改

已经确定的设计：
* DP 模型需要尽可能服用 TransformerForDiffusion，但可以忽略参考代码 roi_dp_transformer_hybrid_policy 的 MoE/Merge 相关逻辑

需要进一步确定的设计：
* mixed-resolution patch sequence 是原始像素，这里要如何用提取成 embedding, 用什么 encoder 提取？
* 位置编码先用朴素的相对位置编码，这个能否从 foveation net 的输出计算出来？后续可以考虑用更复杂的相对位置编码或 RoPE
* 需要设置 FoveationNet 的模拟退火 Gumbel temperature，是否需要在 policy 中暴露接口
* 保留哪些函数名对应的函数

参考代码：
* @diffusion_policy.policy.roi_dp_transformer_hybrid_policy.py


### 2.3 FoveationWorkspace

代码位置：diffusion_policy.workspace.train_foveation_dp_transformer_workspace.py

功能概述：训练框架、train/val 日志

其他相关配套文件：
* configs/liberoft_foveation_dp.yaml
* scripts/roi/sh_train_foveation_libero.sh

参考代码：
* @diffusion_policy.workspace.train_roi_dp_transformer_workspace.py
