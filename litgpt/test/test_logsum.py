import torch

B, nh, T, hd  = 8, 16, 10, 128

q = torch.rand(1, hd)
k = torch.rand(T, hd)
v = torch.rand(T, hd)

# Split keys and values
k1, v1 = k[:T//2], v[:T//2]
k2, v2 = k[T//2:], v[T//2:]

# # Reference output
reference = torch.nn.functional.scaled_dot_product_attention(
    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=False
).squeeze(0)


# Single query, split into N chunks
chunks = [(k1, v1), (k2, v2)]  # you can generalize to more

# Precompute logits and chunk stats
scale = 1 / torch.sqrt(torch.tensor(hd, dtype=q.dtype))
stats = []
for k_chunk, v_chunk in chunks:
    z = (q @ k_chunk.T) * scale                      # [1, chunk_size]
    m = z.max()                                      # scalar
    exp_z = torch.exp(z - m)                         # [1, chunk_size]
    l = exp_z.sum()                                  # scalar
    o = exp_z @ v_chunk                              # [1, hd]
    stats.append((m, l, o))

# Now merge them correctly
global_max = stats[0][0]
numer = stats[0][2]     # Tensor [1,hd]
denom = stats[0][1]     # scalar

for m_i, l_i, o_i in stats[1:]:
    # Compute new global max
    new_max = torch.maximum(global_max, m_i)
    # Scale old accumulators to new_max
    numer = numer * torch.exp(global_max - new_max) + o_i * torch.exp(m_i - new_max)
    denom =     denom * torch.exp(global_max - new_max) + l_i * torch.exp(m_i - new_max)
    global_max = new_max

# Final output
final_output = (numer / denom).squeeze(0)

# Compare
print("Shapes match:", final_output.shape == reference.shape)
print("Max absolute difference:", (final_output - reference).abs().max().item())
print("Allclose:", torch.allclose(final_output, reference, atol=1e-5, rtol=1e-5))
