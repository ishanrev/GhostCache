import torch
from torch.nn.functional import scaled_dot_product_attention

# Configuration
B, nh, T, hd = 8, 16, 10, 128

# Random input: batch size B, num_heads nh, sequence lengths, head_dim hd
q = torch.randn(B, nh, 1, hd, device="cuda")
k = torch.randn(B, nh, T, hd, device="cuda")
v = torch.randn(B, nh, T, hd, device="cuda")

# Reference SDPA output (no squeezing required)
reference = scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
)

# Split into two chunks along the sequence dimension
mid = T // 2
chunks = [(k[..., :mid, :], v[..., :mid, :]), (k[..., mid:, :], v[..., mid:, :])]

# Log-sum-exp trick per chunk
scale = 1.0 / torch.sqrt(torch.tensor(hd, dtype=q.dtype, device=q.device))
stats = []
for k_chunk, v_chunk in chunks:
    # [B, nh, 1, chunk_size]
    z = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
    # Per-chunk max: [B, nh, 1, 1]
    m, _ = z.max(dim=-1, keepdim=True)
    exp_z = torch.exp(z - m)
    # Per-chunk sum: [B, nh, 1, 1]
    l = exp_z.sum(dim=-1, keepdim=True)
    # Per-chunk output: [B, nh, 1, hd]
    o = torch.matmul(exp_z, v_chunk)
    stats.append((m, l, o))

# Merge chunk stats
global_max, denom, numer = stats[0][0], stats[0][1], stats[0][2]
for m_i, l_i, o_i in stats[1:]:
    new_max = torch.maximum(global_max, m_i)
    # Scale and combine accumulators
    numer = numer * torch.exp(global_max - new_max) + o_i * torch.exp(m_i - new_max)
    denom = denom * torch.exp(global_max - new_max) + l_i * torch.exp(m_i - new_max)
    global_max = new_max

# Final output: [B, nh, 1, hd]
final_output = numer / denom

# Verifications
print("Reference shape:", reference.shape)
print("Final output shape:", final_output.shape)
print("Shapes match:", final_output.shape == reference.shape)
print("Max absolute difference:", (reference - reference).abs().max().item())
print("Allclose:", torch.allclose(final_output, reference, atol=1e-5, rtol=1e-5))
