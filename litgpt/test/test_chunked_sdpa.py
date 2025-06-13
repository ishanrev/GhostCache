import torch
import chunked_sdpa  # Your compiled extension module

def test_chunked_sdpa_equivalence():
    torch.manual_seed(0)
    
    B, H, L, D = 2, 2, 4, 8  # Batch, Heads, SeqLen, Dim
    query = torch.randn(B, H, L, D, dtype=torch.float32, device="cpu")
    key = torch.randn(B, H, L, D, dtype=torch.float32, device="cpu")
    value = torch.randn(B, H, L, D, dtype=torch.float32, device="cpu")
    
    # Native SDPA with math backend (no flash or mem-efficient)
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    )
    
    # Your rudimentary implementation
    out, weights = chunked_sdpa.chunked_sdpa(query, key, value, None, 0.0, False, None, None, False)
    
    print(ref_out.shape)
    # Compare outputs
    torch.testing.assert_close(out, ref_out, rtol=1e-7, atol=1e-7)
    # torch.testing.assert_close(weights, ref_weights, rtol=1e-4, atol=1e-4)
    print("✅ chunked_sdpa matches PyTorch math SDPA output.")

if __name__ == "__main__":
    test_chunked_sdpa_equivalence()
