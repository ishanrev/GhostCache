import torch
from torch.nn.functional import scaled_dot_product_attention
from ghost import OffloadManager, offload, streamed_sdpa_cuda

def test_streamed_sdpa_with_static_prefix():
    """
    Simulates a 5-step autoregressive decode with:
      - Prefix KV cache maintained up to 'threshold', then kept static for streamed_sdpa.
      - Normal SDPA uses a dynamic KV cache that grows each step.
      - Compares streamed_sdpa (with static prefix + offload manager) against standard SDPA.
      - Reports a final allclose check and max absolute difference.
    """
    torch.manual_seed(0)
    device = "cuda"
    B, H, D = 7, 4, 8
    total_steps = 100
    threshold = 3  # number of tokens to keep in in-memory prefix

    manager = OffloadManager(B, H, D, torch.float32, 30, 30)
    prefix_k, prefix_v = [], []
    normal_k, normal_v = [], []
    last_out_stream = None
    last_out_normal = None

    for step in range(total_steps):
        print(f"\n--- Decode Step {step + 1} ---")

        # Simulate model outputs
        query = torch.randn(B, H, 1, D, dtype = torch.float32, device=device)
        key   = torch.randn(B, H, 1, D, dtype = torch.float32, device=device)
        value = torch.randn(B, H, 1, D, dtype = torch.float32, device=device)

        # Append to normal cache every step
        normal_k.append(key)
        normal_v.append(value)

        # Build or freeze prefix cache
        if step < threshold:
            prefix_k.append(key)
            prefix_v.append(value)

        # After threshold, offload new KV pairs only to manager
        if step >= threshold:
            offload(manager, key, value, torch.tensor([step], device=device))

        # Prepare inputs
        # For streamed: static prefix + offloaded in manager
        k_prefix = torch.cat(prefix_k, dim=2)
        v_prefix = torch.cat(prefix_v, dim=2)

        # Run streamed SDPA workflow
        out_stream = streamed_sdpa_cuda(
            manager,
            query,
            k_prefix,
            v_prefix,
            None, 0.0, False, None, None, False
        )

        # Run standard SDPA workflow on full dynamic cache
        k_stack = torch.cat(normal_k, dim=2)
        v_stack = torch.cat(normal_v, dim=2)
        out_normal = scaled_dot_product_attention(
            query, k_stack, v_stack,
            attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # Store last outputs
        last_out_stream = out_stream
        last_out_normal = out_normal

        # Verify step-by-step
        try:
            torch.testing.assert_allclose(out_stream, out_normal, atol=1e-6, rtol=1e-5)
            print(f"Step {step + 1}: PASS")
        except AssertionError as e:
            print(f"Step {step + 1}: FAIL\n{e}")

    # Final overall comparison
    max_diff = (last_out_stream - last_out_normal).abs().max().item()
    allclose = torch.allclose(last_out_stream, last_out_normal, atol=1e-6, rtol=1e-5)
    print(f"\nFinal allclose: {allclose}")
    print(f"Final max absolute difference: {max_diff}")

if __name__ == "__main__":
    test_streamed_sdpa_with_static_prefix()
