# MoE Merging Coding Plan

## Overview

Merge multiple domain-specific Diffusion Policy experts into a single generalist model:
- **Shared**: Visual encoder (frozen after merge)
- **Attention layers**: Average weight merge (then finetune)
- **FFN layers**: Replace with MoE (domain-aware router, expert weights frozen)
- **Other layers**: Finetune after merge

---

## File Plan

### New Files

| File | Description |
|------|-------------|
| `diffusion_policy/model/diffusion/transformer_for_diffusion_moe.py` | MoE transformer (MoEFFN + avg-merged attention) |
| `diffusion_policy/model/merge_utils.py` | Checkpoint merging logic |
| `diffusion_policy/policy/diffusion_transformer_moe_policy.py` | MoE policy (wraps MoE transformer) |
| `diffusion_policy/dataset/multi_domain_dataset.py` | Multi-domain dataset with domain_id |
| `diffusion_policy/workspace/train_moe_workspace.py` | Training workspace for MoE |
| `configs/pusht256_moe.yaml` | Config for MoE training |

### Modified Files

| File | Change |
|------|--------|
| `gpu_service_pusht.py` | Load MoE checkpoint (interface unchanged) |

---

## Self-Attention vs Cross-Attention: 是否需要不同方案？

### 结构回顾

每个 Decoder Layer（`norm_first=True`，Pre-LN）的前向流程：

```
x  →  LayerNorm1 + Self-Attention(Q=K=V=x)          →  残差
   →  LayerNorm2 + Cross-Attention(Q=x, K=V=memory)  →  残差
   →  LayerNorm3 + FFN(linear1 → GELU → linear2)     →  残差
```

其中 `memory` = encoder 输出 = 视觉条件（`cond_obs_emb` 投影后经过 encoder）。

### Self-Attention

- Q/K/V 全部来自 **noisy action sequence** 本身
- 功能：建模 action token 间的时序依赖
- **Domain 相关性低**：action 内部结构跨 domain 高度一致（都是 2D Push-T 动作）
- ✅ **直接平均 merge，之后 finetune**

### Cross-Attention

- Q 来自 action sequence，K/V 来自 **memory（视觉编码器输出）**
- 功能：将 domain-specific 视觉特征"注入"到 action 生成过程
- **Domain 相关性高**：不同 domain（texture/light/goal）的视觉特征分布差异
  正体现在 K/V 上；各 expert 的 K/V projection 学到了各自的视觉-动作对齐方式
- ❗ 直接平均会损失 domain-specific 的视觉-动作对齐能力

### Cross-Attention 的处理方案

| 方案 | 说明 | 取舍 |
|------|------|------|
| **A. 平均 merge + finetune** | 最简单，参数量不增加 | 丢失对齐信息，需更多 finetune 步骤恢复 |
| **B. cross-attn 也做 MoE** | 与 FFN 共用同一 router | 参数量翻倍，实现复杂 |
| **C. 平均 merge + IA³ adapter（推荐）** | K/V projection 各加一对可训练缩放向量；复用代码库已有 `ia3_kv` 实现 | 低参数量，与现有 DA 基础设施复用 |

**选用方案 C**：cross-attention 平均 merge 后，在每层 `multihead_attn` 的 K/V projection
上挂 domain-conditioned IA³ adapter（`ia3_k`, `ia3_v` ∈ `n_emb`），由 `domain_id`
做监督信号（训练时已有 `domain_id` in batch）。这与 `is_da` 代码路径完全对齐，无需额外基础设施。

### 参数归属更新（完整版）

| 参数 | 来源 | 训练状态 |
|------|------|----------|
| `obs_encoder.*` | 复制第一个 expert | **Frozen** |
| `model.*.self_attn.*` | 跨 expert 平均 | Finetune |
| `model.*.multihead_attn.*`（权重本身） | 跨 expert 平均 | Finetune |
| `model.*.multihead_attn.ia3_k/v` | **新建**，零初始化 | Finetune（低参数适配） |
| `model.*.linear1/2.*`（FFN） | 每个 expert 独立保留 → MoEFFN | **Frozen**（expert 权重） |
| `model.*.ffn.router.*` | **新建**，随机初始化 | Train from scratch |
| `model.*.norm*.*/ln_f.*` | 跨 expert 平均 | Finetune |
| `model.encoder.*/input_emb/pos_emb/cond_*` | 跨 expert 平均 | Finetune |
| `model.head.*` | 跨 expert 平均 | Finetune |
| `normalizer.params_dict.*` | 跨域重新统计（推荐）或取第一个 | — |

---

## Phase 1: `transformer_for_diffusion_moe.py`

Reference: `diffusion_policy/model/diffusion/transformer_for_diffusion.py`

### `MoEFFN`

```python
class MoEFFN(nn.Module):
    def __init__(self, n_emb, n_experts, expert_ffn_state_dicts):
        # expert_ffns: frozen, loaded from checkpoints
        self.expert_ffns = nn.ModuleList([...])  # frozen
        for p in self.expert_ffns.parameters():
            p.requires_grad = False
        # router: takes visual feature (cond), outputs expert weights
        self.router = nn.Linear(n_emb, n_experts)

    def forward(self, x, router_input):
        # router_input: (B, n_emb) — from visual encoder output
        weights = F.softmax(self.router(router_input), dim=-1)  # (B, n_experts)
        out = sum(weights[:, i:i+1, None] * self.expert_ffns[i](x)
                  for i in range(self.n_experts))
        return out
```

**Notes:**
- Router input is the visual encoder feature (domain-aware signal)
- Expert FFN weights are frozen after loading from checkpoints
- Load balancing loss computed externally in policy

### `TransformerForDiffusionMoE`

- Copy structure from `TransformerForDiffusion`
- Replace each decoder layer's FFN with `MoEFFN`
- Self-attention layers initialized from avg-merged weights, finetune
- Cross-attention layers initialized from avg-merged weights + IA³ adapter on K/V (复用 `ia3_kv`), finetune
- Add `router_input` parameter to `forward()`

---

## Phase 2: `merge_utils.py`

```python
def merge_experts_to_moe(checkpoint_paths: list[str], cfg) -> TransformerForDiffusionMoE:
    """
    1. Load N expert checkpoints
    2. Visual encoder: copy from first expert (shared, frozen)
    3. Self-attention / norm / emb / head layers: average weights across experts
    4. Cross-attention: average weights → init ia3_k/ia3_v as ones (trainable)
    5. FFN layers: collect all expert FFN state_dicts → pass to MoEFFN (frozen)
    6. Return initialized TransformerForDiffusionMoE
    """
```

**Key logic:**
- Separate parameter names into four buckets:
  - `obs_encoder.*` → copy from first expert, frozen
  - `model.*.self_attn.*` → avg merge
  - `model.*.multihead_attn.*` → avg merge for weight tensors; init new `ia3_k/v`
  - `model.*.linear[12].*` → pass per-expert state_dicts to `MoEFFN`, frozen
  - everything else (`norm`, `emb`, `head`, `encoder.*`) → avg merge
- Avg merge: `merged[k] = mean([ckpt[k] for ckpt in checkpoints])`
- FFN: pass list of state_dicts to `MoEFFN` constructor

---

## Phase 3: `diffusion_transformer_moe_policy.py`

Reference: `diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py`

**Changes from base policy:**

1. `__init__`: Accept pre-built `TransformerForDiffusionMoE` (from `merge_utils`)
2. `compute_loss`: Add load balancing loss
   ```python
   diffusion_loss = F.mse_loss(pred, noise)
   lb_loss = load_balance_loss(router_weights)  # entropy regularization
   return diffusion_loss + self.lambda_lb * lb_loss
   ```
3. `get_optimizer`: Freeze `obs_encoder` and `expert_ffns`, train router + attention (incl. ia3_k/v)
   ```python
   trainable = [p for n, p in model.named_parameters()
                if 'obs_encoder' not in n and 'expert_ffns' not in n]
   ```
4. `predict_action`: No change (interface identical to base policy)

---

## Phase 4: `multi_domain_dataset.py`

Reference: `diffusion_policy/dataset/pusht_image_dataset.py`

```python
class MultiDomainDataset(BaseImageDataset):
    def __init__(self, domain_datasets: list[PushTImageDataset], balanced=True):
        # domain_datasets: one per expert/domain
        ...

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        batch['domain_id'] = self.domain_id  # int
        return batch
```

- Supports domain-balanced sampling (equal samples per domain per batch)
- `domain_id` passed through to router for optional supervised routing

---

## Phase 5: `train_moe_workspace.py`

Reference: `diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py`

**Changes:**

1. `__init__`: Call `merge_experts_to_moe()` to initialize model (not random init)
2. `run`: Use `MultiDomainDataset`
3. Freeze params before optimizer creation:
   ```python
   for n, p in self.model.named_parameters():
       if 'obs_encoder' in n or 'expert_ffns' in n:
           p.requires_grad = False
   ```
4. Log router entropy per domain for monitoring

---

## Phase 6: `configs/pusht256_moe.yaml`

Reference: `configs/pusht256_dp.yaml`

**New fields:**
```yaml
task:
  expert_checkpoints:
    - outputs/expert_texture/checkpoints/latest.ckpt
    - outputs/expert_light/checkpoints/latest.ckpt
    - outputs/expert_goal/checkpoints/latest.ckpt

policy:
  _target_: diffusion_policy.policy.diffusion_transformer_moe_policy.DiffusionTransformerMoEPolicy
  n_experts: 3
  lambda_lb: 0.01  # load balancing loss weight

workspace:
  _target_: diffusion_policy.workspace.train_moe_workspace.TrainMoEWorkspace
```

---

## Phase 7: `gpu_service_pusht.py`

Minimal change — only update checkpoint path and policy class import:
```python
# Change policy class to DiffusionTransformerMoEPolicy
# predict_action interface is identical, no other changes needed
```

---

## Development Order

```
1. merge_utils.py              ← verify checkpoint loading + avg merge
2. transformer_for_diffusion_moe.py  ← verify forward pass shape
3. diffusion_transformer_moe_policy.py  ← verify loss + optimizer
4. multi_domain_dataset.py     ← verify domain_id in batch
5. train_moe_workspace.py + pusht256_moe.yaml  ← end-to-end training
6. gpu_service_pusht.py        ← verify inference interface
```

---

## Frozen / Trainable Parameter Summary

| Component | Frozen | Trainable |
|-----------|--------|-----------|
| `obs_encoder` (visual encoder) | ✓ | |
| `model.*.self_attn.*` (self-attention) | | ✓ (finetune) |
| `model.*.multihead_attn.*` weight tensors (cross-attention) | | ✓ (finetune) |
| `model.*.multihead_attn.ia3_k/v` (cross-attn IA³ adapter) | | ✓ (finetune, new) |
| `model.*.norm*` (layer norms) | | ✓ (finetune) |
| `model.*.ffn.expert_ffns` (FFN expert weights) | ✓ | |
| `model.*.ffn.router` (MoE router) | | ✓ (train from scratch) |
| `input_emb`, `pos_emb`, `cond_*`, `encoder.*`, `head` | | ✓ (finetune) |
