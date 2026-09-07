### 用单 Policy + Backbone 工厂切换 Transformer / MoE 的落地
  方案

  #### Summary

  采用组合+工厂，不做整类 copy，也不做 policy 级继承分叉。
  保留现有 DiffusionTransformerHybridImagePolicy 作为统一入口，
  在内部按配置构建 backbone（TransformerForDiffusion 或
  MoEForDiffusion），从而保证训练、测试、推理、checkpoint 逻辑
  基本复用。

  #### Key Changes

  1. Backbone 统一接口与构建

  - 在 diffusion_transformer_hybrid_image_policy.py 内新增私有
    构建函数（例如 _build_backbone(...)），根据配置项
    backbone_type: transformer|moe 分支创建模型。
  - transformer 分支保持现状。
  - moe 分支支持两种初始化：
      - from_teacher_models（主路径）：读取 teacher_ckpts 列
        表，实例化多个 teacher 并合并为 MoE。
      - 无 teacher 时按 ffn_expand_factor 初始化 MoE（兜底）。
  - 外部训练/推理调用仍统一走 self.model(noisy_trajectory,
    timesteps, cond)；MoE 的 aux loss 通过属性或可选返回值读
    取，不改主调用签名。

  2. Checkpoint/加载架构

  - 保留现有 load_weight_from_ckpt 作为“整策略权重加载”入口。
  - 新增 MoE 专用加载流程（例如
    _build_moe_from_teacher_ckpts）：
      - 逐个读取 teacher_ckpts，只提取 backbone 权重构建
        teacher。
      - 用 MoEForDiffusion.from_teacher_models(...) 生成
        self.model。
      - router 保持随机初始化（符合你当前策略）。
  - 对已有单 ckpt 恢复训练路径保持兼容（resume 继续用现有
    workspace 机制）。

  3. 训练损失集成

  - 在 compute_loss 中加入可配置 MoE aux loss：
      - total_loss = diffusion_loss + moe_aux_loss_weight *
        moe_aux_loss
      - 配置默认开启且可关（你选的策略）。
  - 若 backbone 非 MoE 或没有 aux，自动退化为原 loss，保证
    transformer 实验不受影响。

  4. 配置层改动（最小但明确）

  - policy 配置新增：
      - backbone_type: transformer|moe（默认 transformer 以保兼
        容）
      - teacher_ckpts: []（MoE 推荐路径）
      - moe_aux_loss_weight（默认小值）
      - moe_topk, router_noisy_std, ffn_expand_factor（float 或
        list[float]）
  - 保留 pretrained_ckpt 语义用于整策略恢复，不复用它承载多
    teacher。

  #### Test Plan

  1. 回归兼容

  - backbone_type=transformer 下，训练 1 step + 推理 1 batch，
    loss 与输出 shape 与当前一致。

  2. MoE 构建与训练

  - 提供 2~3 个 teacher ckpt，检查：
      - model 可构建；
      - experts 参数按 teacher 映射成功；
      - router 为随机初始化；
      - train step 正常反传，loss 不为 NaN。

  3. 推理与采样链路

  - predict_action / conditional_sample 在 MoE 下全流程可跑，输
    出 shape 与 action 切片逻辑不变。

  4. checkpoint 恢复

  - MoE 训练保存后可恢复继续训练；
  - transformer ckpt 恢复路径不回归。

  #### Assumptions / Defaults

  - 采用你确认的默认决策：多 ckpt 合并 + 单 policy + 模型工厂 +
    aux loss 可配置启用 + teacher_ckpts 列表配置。
  - teacher 模型除 FFN 维度外，主干结构需一致（层数、heads、
    embedding 等）；同一 teacher 的 FFN 维度默认各层一致。
  - 当前先不拆分独立 MoE policy 类，避免双轨维护；后续若 DA/
    HDFree 分支出现明显分歧，再做二次拆分。