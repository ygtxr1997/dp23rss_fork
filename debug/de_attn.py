import torch
from torch.nn import MultiheadAttention
from time import time
import torch.nn.functional as F



# 模拟输入
batch_size = 64
seq_len = 2048
embed_dim = 2048
query = torch.rand(batch_size, seq_len, embed_dim).cuda()
key = torch.rand(batch_size, seq_len, embed_dim).cuda()
value = torch.rand(batch_size, seq_len, embed_dim).cuda()

# 原生 MultiheadAttention
mha = MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True).cuda()

# 测试 PyTorch 原生实现
start = time()
for _ in range(1):
    output, attn_weights = mha(query, key, value)
    torch.cuda.synchronize()
print(f"Native MultiheadAttention time: {time() - start:.6f} seconds")

# 自定义实现
class CustomMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.key_scale = torch.nn.Parameter(torch.ones(embed_dim).cuda())
        self.value_scale = torch.nn.Parameter(torch.ones(embed_dim).cuda())

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # q_proj = F.linear(query, self.q_proj_weight, self.in_proj_bias[:self.embed_dim])
        # k_proj = F.linear(key, self.k_proj_weight, self.in_proj_bias[self.embed_dim:2 * self.embed_dim])
        # v_proj = F.linear(value, self.v_proj_weight, self.in_proj_bias[2 * self.embed_dim:])

        # # 自定义修改
        # k_proj = k_proj * self.key_scale
        # v_proj = v_proj * self.value_scale

        return F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            )

custom_mha = CustomMultiheadAttention(embed_dim=embed_dim, num_heads=8).cuda()

start = time()
for _ in range(1):
    output, attn_weights = custom_mha(query, key, value)
    torch.cuda.synchronize()
print(f"Custom MultiheadAttention time: {time() - start:.6f} seconds")


proj_weight = torch.randn(embed_dim, embed_dim).cuda()

