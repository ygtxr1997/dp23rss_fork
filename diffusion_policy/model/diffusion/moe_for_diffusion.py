import types
import math
import copy
from typing import Union, Optional, Tuple, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


logger = logging.getLogger(__name__)


class FFNExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 p_drop: float = 0.1,
                 activation: str = "gelu", bias: bool = True):
        """
        Refer to: `TransformerDecoderLayer._ff_block()`
        Note: in original `decoder_layer`, dim_feedforward=4*n_emb
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(p_drop)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x_B_T_D):
        x_B_T_D = self.linear2(self.dropout(self.activation(self.linear1(x_B_T_D))))
        return x_B_T_D  # dropout3 will be applied in the decoder layer


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int,
                 k: int = 2, noisy_std: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)  # 随机初始化

        self.num_experts = num_experts
        self.k = k
        self.noisy_std = noisy_std

    def forward(self, x_B_T_D):
        logits_B_T_E = self.gate(x_B_T_D)
        if self.training and self.noisy_std > 0:  # only apply noise during training
            logits_B_T_E = logits_B_T_E + torch.randn_like(logits_B_T_E) * self.noisy_std
        probs_B_T_E = F.softmax(logits_B_T_E, dim=-1)  # [B,T,E]
        topv, topi = torch.topk(probs_B_T_E, k=self.k, dim=-1)  # [B,T,k], sorted inside k-dim
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)  # norm to sum to 1

        # load-balance aux loss（简版）
        B, T, _ = probs_B_T_E.shape
        hard = F.one_hot(topi, num_classes=self.num_experts).float().sum(dim=-2)  # [B,T,E]
        load = hard.mean(dim=(0, 1))  # token 分配比例
        importance = probs_B_T_E.mean(dim=(0, 1))  # soft 概率比例
        aux_loss = self.num_experts * torch.sum(load * importance)
        return topi, topv, aux_loss


class MoEFFN(nn.Module):
    def __init__(self,
                 d_model: int,
                 p_drop: float = 0.1,
                 num_experts: int = 3,
                 topk: int = 2,
                 dim_feedforward_list: List[int] = None,
                 activation: str = "gelu",
                 bias: bool = True,
                 ):
        super().__init__()
        if dim_feedforward_list is None:
            raise ValueError("dim_feedforward_list must be provided.")
        else:
            assert len(dim_feedforward_list) == num_experts, \
                (f"dim_feedforward_list length ({len(dim_feedforward_list)}) "
                 f"must equal num_experts ({num_experts})")
            dim_feedforward_list = [int(x) for x in dim_feedforward_list]

        self.dim_feedforward_list = dim_feedforward_list
        self.experts = nn.ModuleList([
            FFNExpert(d_model, d_ff_i, p_drop, activation=activation, bias=bias)
            for d_ff_i in dim_feedforward_list
        ])
        self.router = TopKRouter(d_model, num_experts, k=topk)

    def forward(self, x):
        topi, topv, aux_loss = self.router(x)  # [B,T,k], [B,T,k]
        out = torch.zeros_like(x)

        # 简单实现：对每个专家全量前向，再按 token 权重聚合（易读，后续可做稀疏优化）
        for e, expert in enumerate(self.experts):
            weight_e = (topv * (topi == e).float()).sum(dim=-1, keepdim=True)  # [B,T,1]
            if torch.count_nonzero(weight_e) == 0:
                continue
            out = out + weight_e * expert(x)
        return out, aux_loss


class MoETransformerDecoderLayer(nn.Module):
    """
    Most code reference to `torch.nn.TransformerDecoderLayer`, but with MoE FFN and some simplifications
    """
    def __init__(self,
                 # original `TransformerDecoderLayer` params
                 d_model: int,
                 nhead: int,
                 ## dim_feedforward: int = 2048, # NOTE: replaced by dim_feedforward_list for MoE
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 batch_first: bool = True,
                 norm_first: bool = True,
                 bias: bool = True,
                 # MoE-specific params
                 dim_feedforward_list: List[int] = None,
                 num_experts: int = 3,
                 topk: int = 2,
                 ):
        super().__init__()
        if dim_feedforward_list is None:
            raise ValueError("dim_feedforward_list must be provided for MoETransformerDecoderLayer.")
        dim_feedforward_list = [int(x) for x in dim_feedforward_list]
        inferred_num_experts = len(dim_feedforward_list)
        if inferred_num_experts != num_experts:
            raise ValueError(
                f"num_experts ({num_experts}) must match len(dim_feedforward_list) ({inferred_num_experts})"
            )

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)

        self.moe_ffn = MoEFFN(
            d_model=d_model,
            p_drop=dropout,
            num_experts=num_experts,
            topk=topk,
            dim_feedforward_list=dim_feedforward_list,
            activation=activation,
            bias=bias,
        )

        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

        self.norm_first = norm_first
        self.last_aux_loss = None
        self.dim_feedforward_list = dim_feedforward_list

    def _sa_block(self, x, attn_mask=None):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ca_block(self, x, mem, memory_mask=None):
        x = self.multihead_attn(x, mem, mem, attn_mask=memory_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x):
        y, aux = self.moe_ffn(x)
        self.last_aux_loss = aux
        return self.dropout3(y)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Original `TransformerForDiffusion` only uses: tgt, memory, tgt_mask, memory_mask
        So we removed: memory_key_padding_mask, tgt_is_causal, memory_is_causal
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._ca_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._ca_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))
        return x


class MoETransformerDecoder(nn.Module):
    """
    Refer to: `nn.TransformerDecoder`
    """
    def __init__(self, decoder_layer: MoETransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.last_aux_loss = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Original `TransformerForDiffusion` only uses: tgt, memory, tgt_mask, memory_mask
        So we removed: memory_key_padding_mask, tgt_is_causal, memory_is_causal
        """
        x = tgt
        aux_losses = []
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            if layer.last_aux_loss is not None:
                aux_losses.append(layer.last_aux_loss)
        if len(aux_losses) > 0:
            self.last_aux_loss = torch.stack(aux_losses).mean()
        else:
            self.last_aux_loss = None
        return x

    def unfreeze_cross_attention(self):
        for layer in self.layers:
            for p in layer.multihead_attn.parameters():
                p.requires_grad = True

    def unfreeze_self_attention(self):
        for layer in self.layers:
            for p in layer.self_attn.parameters():
                p.requires_grad = True

    def unfreeze_mlp(self):
        for layer in self.layers:
            for p in layer.moe_ffn.parameters():
                p.requires_grad = True

    def unfreeze_norms(self):
        for layer in self.layers:
            for p in layer.norm1.parameters():
                p.requires_grad = True
            for p in layer.norm2.parameters():
                p.requires_grad = True
            for p in layer.norm3.parameters():
                p.requires_grad = True

    def freeze_experts_only(self):
        for layer in self.layers:
            for p in layer.moe_ffn.experts.parameters():
                p.requires_grad = False


class MoEForDiffusion(ModuleAttrMixin):
    """
    MoE version of TransformerForDiffusion:
    - Shared SA/CA/Norm from averaged teacher models.
    - MoE FFN experts copied from each teacher FFN.
    - Router parameters stay randomly initialized unless explicitly loaded.
    """

    # [MODIFIED from TransformerForDiffusion.__init__]
    def __init__(self,
            # original `TransformerForDiffusion` params
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0,
            # DA params (not used, just for `TransformerForDiffusion` compatibility)
            is_da: bool = False,
            # MoE compatible params
            ffn_expand_factor: Union[float, List[float]] = 4.0,
            # MoE-specific params
            num_experts: int = 4,
            moe_topk: int = 2,
            router_noisy_std: float = 0.0,  # for training stability
        ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:  # default:0
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )

            # [] Difference: using MoETransformer
            if isinstance(ffn_expand_factor, (list, tuple)):
                expand_list = [float(x) for x in ffn_expand_factor]
                if len(expand_list) != num_experts:
                    raise ValueError(
                        f"num_experts ({num_experts}) must match len(ffn_expand_factor) ({len(expand_list)})"
                    )
            else:
                expand_list = [float(ffn_expand_factor)] * num_experts
            resolved_dim_feedforward_list = [int(v * n_emb) for v in expand_list]

            decoder_layer = MoETransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward_list=resolved_dim_feedforward_list,
                dropout=p_drop_attn,
                num_experts=num_experts,
                topk=moe_topk,
                norm_first=True
            )
            if router_noisy_std > 0:
                for layer in [decoder_layer]:
                    layer.moe_ffn.router.noisy_std = router_noisy_std
            self.decoder = MoETransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:  # not used
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.cond_dim = cond_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_emb = n_emb
        self.p_drop_emb = p_drop_emb
        self.p_drop_attn = p_drop_attn
        self.causal_attn = causal_attn
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.n_cond_layers = n_cond_layers
        self.encoder_only = encoder_only

        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.ffn_expand_factor = ffn_expand_factor
        self.dim_feedforward_list = resolved_dim_feedforward_list if not encoder_only else None
        self.last_moe_aux_loss = None

        self.apply(self._init_weights)
        logger.info(
            "number of parameters (MoE): %e", sum(p.numel() for p in self.parameters())
        )

    # [REUSED from TransformerForDiffusion.unfreeze_module]
    @staticmethod
    def unfreeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    # [MODIFIED from TransformerForDiffusion.freeze_backbone]
    def freeze_backbone(self, unfreeze_params: str):
        for _, param in self.named_parameters():
            param.requires_grad = False

        if hasattr(self, "decoder") and self.decoder is not None:
            if "ca" in unfreeze_params:
                self.decoder.unfreeze_cross_attention()
            if "sa" in unfreeze_params:
                self.decoder.unfreeze_self_attention()
            if "mlp" in unfreeze_params:
                self.decoder.unfreeze_mlp()
            if "norm" in unfreeze_params:
                self.decoder.unfreeze_norms()
            if "router" in unfreeze_params:
                for layer in self.decoder.layers:
                    for p in layer.moe_ffn.router.parameters():
                        p.requires_grad = True
            if "expert" not in unfreeze_params:
                self.decoder.freeze_experts_only()

        if "head" in unfreeze_params:
            self.unfreeze_module(self.head)
            self.unfreeze_module(self.ln_f)
        if "embed" in unfreeze_params:
            self.unfreeze_module(self.input_emb)
            self.pos_emb.requires_grad = True
        if "cond" in unfreeze_params:
            self.unfreeze_module(self.encoder)
            if self.cond_obs_emb is not None:
                self.unfreeze_module(self.cond_obs_emb)
            if self.cond_pos_emb is not None:
                self.cond_pos_emb.requires_grad = True

        cnt = 0
        for p in self.parameters():
            if p.requires_grad:
                cnt += 1
        print(f'[DEBUG][MoE] trainable: {cnt}, unfreeze_params={unfreeze_params}')

    # [REUSED from TransformerForDiffusion.trainable_params]
    def trainable_params(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    # [MODIFIED from TransformerForDiffusion._init_weights]
    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerEncoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            FFNExpert,
            TopKRouter,
            MoEFFN,
            MoETransformerDecoderLayer,
            MoETransformerDecoder,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, MoEForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None and module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    # [REUSED from TransformerForDiffusion.get_optim_groups]
    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif ".ia3_" in pn:  # for IA3 Adapter
                    decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    # [REUSED from TransformerForDiffusion.configure_optimizers]
    def configure_optimizers(self,
            learning_rate: float=1e-4,
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    # [MODIFIED from TransformerForDiffusion.forward]
    def forward(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor]=None,
        return_aux_loss: bool=False,
        **kwargs):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        input_emb = self.input_emb(sample)

        if self.encoder_only:
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x

            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            self.last_moe_aux_loss = self.decoder.last_aux_loss

        x = self.ln_f(x)
        x = self.head(x)
        if return_aux_loss:
            aux = self.last_moe_aux_loss
            if aux is None:
                aux = x.new_zeros(())
            return x, aux
        return x

    # [NEW] helper: merge parameters from source modules into target
    @staticmethod
    @torch.no_grad()
    def _merge_module_weights(
            target: nn.Module,
            sources: List[nn.Module],
            merge_mode: str = "average"):
        assert len(sources) > 0
        src_sds = [m.state_dict() for m in sources]
        dst = target.state_dict()
        for k, v in dst.items():
            vals = [sd[k] for sd in src_sds]
            if v.dtype.is_floating_point:
                if merge_mode == "average":
                    dst[k] = torch.stack(vals, dim=0).mean(dim=0).to(v.dtype)
                else:
                    raise ValueError(f"Unsupported merge_mode={merge_mode}")
            else:
                dst[k] = vals[0]
        target.load_state_dict(dst, strict=True)

    # [NEW] build MoE model from teacher `TransformerForDiffusion` models
    @classmethod
    @torch.no_grad()
    def from_teacher_models(cls,
            teachers: List[TransformerForDiffusion],
            moe_topk: int = 2,
            router_noisy_std: float = 0.0,
            freeze_experts: bool = True,
            target_obs_encoder: Optional[nn.Module] = None,
            teacher_obs_encoders: Optional[List[nn.Module]] = None,
            obs_merge_mode: str = "average",
            is_main_process: bool = False,
        ):
        """
        target_obs_encoder: the model that will be assigned averaged teacher obs_encoders' weights
        """
        assert len(teachers) > 0, "teachers should not be empty"
        if not (1 <= moe_topk <= len(teachers)):
            raise ValueError(
                f"moe_topk should satisfy 1 <= moe_topk <= num_teachers, "
                f"got moe_topk={moe_topk}, num_teachers={len(teachers)}"
            )
        if (target_obs_encoder is None) != (teacher_obs_encoders is None):
            raise ValueError(
                "`target_obs_encoder` and `teacher_obs_encoders` should be both provided or both None."
            )

        base = teachers[0]
        assert not base.encoder_only, "MoE builder currently supports decoder path only (time_as_cond=True)"
        compare_keys = [
            "T", "T_cond",
            "input_dim", "output_dim", "horizon", "n_obs_steps", "cond_dim",
            "n_layer", "n_head", "n_emb",
            "p_drop_emb", "p_drop_attn",
            "causal_attn", "time_as_cond", "obs_as_cond", "n_cond_layers",
            "encoder_only",
        ]

        teacher_factors: List[float] = []
        for ti, t in enumerate(teachers):
            for k in compare_keys:
                if getattr(t, k) != getattr(base, k):
                    raise ValueError(
                        f"Teacher {ti} mismatch on `{k}`: got {getattr(t, k)} vs base {getattr(base, k)}."
                    )
            if t.encoder_only:
                raise ValueError(f"Teacher {ti} is encoder-only; decoder path is required for MoE builder.")
            if len(t.decoder.layers) != t.n_layer:
                raise ValueError(
                    f"Teacher {ti} decoder layer count mismatch: len(decoder.layers)={len(t.decoder.layers)} vs n_layer={t.n_layer}."
                )
            if t.decoder.layers[0].self_attn.num_heads != t.n_head:
                raise ValueError(
                    f"Teacher {ti} attention head mismatch: decoder has {t.decoder.layers[0].self_attn.num_heads}, n_head={t.n_head}."
                )

            f = float(t.ffn_expand_factor)
            expected_dim = int(round(f * t.n_emb))
            actual_dim = int(t.decoder.layers[0].linear1.out_features)
            if expected_dim != actual_dim:
                raise ValueError(
                    f"Teacher {ti} `ffn_expand_factor` and decoder FFN dim mismatch: "
                    f"round({f} * {t.n_emb})={expected_dim} vs actual={actual_dim}."
                )
            teacher_factors.append(f)

        dim_feedforward_list = [int(round(f * base.n_emb)) for f in teacher_factors]
        for li in range(base.n_layer):
            for ei, t in enumerate(teachers):
                got = int(t.decoder.layers[li].linear1.out_features)
                expect = dim_feedforward_list[ei]
                if got != expect:
                    raise ValueError(
                        f"Teacher {ei} has inconsistent FFN dim across layers: "
                        f"layer0={expect}, layer{li}={got}"
                    )

        model = cls(
            input_dim=base.input_dim,
            output_dim=base.output_dim,
            horizon=base.horizon,
            n_obs_steps=base.n_obs_steps,
            cond_dim=base.cond_dim,
            n_layer=base.n_layer,
            n_head=base.n_head,
            n_emb=base.n_emb,
            p_drop_emb=base.p_drop_emb,
            p_drop_attn=base.p_drop_attn,
            causal_attn=base.causal_attn,
            time_as_cond=base.time_as_cond,
            obs_as_cond=base.obs_as_cond,
            n_cond_layers=base.n_cond_layers,
            ffn_expand_factor=teacher_factors,
            num_experts=len(teachers),
            moe_topk=moe_topk,
            router_noisy_std=router_noisy_std,
        )
        if model.dim_feedforward_list != dim_feedforward_list:
            raise ValueError(
                "Built MoE model dim_feedforward_list does not match teachers. "
                f"Expected {dim_feedforward_list}, got {model.dim_feedforward_list}."
            )

        cls._merge_module_weights(model.input_emb, [t.input_emb for t in teachers], merge_mode="average")
        model.pos_emb.copy_(torch.stack([t.pos_emb for t in teachers], dim=0).mean(dim=0))
        cls._merge_module_weights(model.ln_f, [t.ln_f for t in teachers], merge_mode="average")
        cls._merge_module_weights(model.head, [t.head for t in teachers], merge_mode="average")

        if model.cond_obs_emb is not None:
            cls._merge_module_weights(model.cond_obs_emb, [t.cond_obs_emb for t in teachers], merge_mode="average")
        if model.cond_pos_emb is not None:
            model.cond_pos_emb.copy_(torch.stack([t.cond_pos_emb for t in teachers], dim=0).mean(dim=0))
        cls._merge_module_weights(model.encoder, [t.encoder for t in teachers], merge_mode="average")

        for li, moe_layer in enumerate(model.decoder.layers):
            teacher_layers = [t.decoder.layers[li] for t in teachers]
            cls._merge_module_weights(moe_layer.self_attn, [l.self_attn for l in teacher_layers], merge_mode="average")
            cls._merge_module_weights(moe_layer.multihead_attn, [l.multihead_attn for l in teacher_layers], merge_mode="average")
            cls._merge_module_weights(moe_layer.norm1, [l.norm1 for l in teacher_layers], merge_mode="average")
            cls._merge_module_weights(moe_layer.norm2, [l.norm2 for l in teacher_layers], merge_mode="average")
            cls._merge_module_weights(moe_layer.norm3, [l.norm3 for l in teacher_layers], merge_mode="average")

            # For MoE FFN experts, we copy each expert from a different teacher's FFN (1-to-1), without averaging.
            for ei, expert in enumerate(moe_layer.moe_ffn.experts):
                src = teacher_layers[ei]
                if expert.linear1.weight.shape != src.linear1.weight.shape:
                    raise ValueError(
                        f"Expert/teacher FFN shape mismatch at layer={li}, expert={ei}: "
                        f"expert_linear1={tuple(expert.linear1.weight.shape)}, "
                        f"teacher_linear1={tuple(src.linear1.weight.shape)}"
                    )
                expert.linear1.weight.copy_(src.linear1.weight)
                expert.linear1.bias.copy_(src.linear1.bias)
                expert.linear2.weight.copy_(src.linear2.weight)
                expert.linear2.bias.copy_(src.linear2.bias)

                if freeze_experts:
                    for p in expert.parameters():
                        p.requires_grad = False

        model.last_moe_aux_loss = None
        if target_obs_encoder is not None:
            if len(teacher_obs_encoders) != len(teachers):
                raise ValueError(
                    f"teacher_obs_encoders length mismatch: got {len(teacher_obs_encoders)} vs teachers={len(teachers)}."
                )
            cls._merge_module_weights(
                target_obs_encoder,
                teacher_obs_encoders,
                merge_mode=obs_merge_mode
            )
            if is_main_process:
                print(f"[MoE Builder] Merged teacher obs_encoders into target_obs_encoder with merge_mode={obs_merge_mode}.")

        return model
