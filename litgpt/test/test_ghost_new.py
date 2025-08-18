import math
import torch
from torch.nn.functional import scaled_dot_product_attention
from ghost import OffloadManager, offload, streamed_sdpa_cuda

def test_streamed_sdpa_prefill_decode():
    torch.manual_seed(0)
    device = "cuda"
    B, H, D = 7, 8, 128
    
    # how many tokens to prefill (static prefix), then how many to decode
    prefill_steps = 10
    decode_steps = 1000

    # manager needs capacity = prefill + decode
    max_total = prefill_steps + decode_steps
    manager = OffloadManager(B, H, D, torch.float32, 500, 100)

    # will hold our “static” prefix
    prefix_k, prefix_v = [], []
    # will hold the full dynamic cache for normal SDPA
    normal_k, normal_v = [], []

    last_out_stream, last_out_normal = None, None

    # --- Prefill Phase: build up prefix_k/v and normal_k/v (no SDPA calls) ---
    for step in range(prefill_steps):
        key   = torch.randn(B, H, 1, D, dtype=torch.float32, device=device)
        value = torch.randn(B, H, 1, D, dtype=torch.float32, device=device)

        prefix_k.append(key)
        prefix_v.append(value)

        normal_k.append(key)
        normal_v.append(value)
    k_prefix = torch.cat(prefix_k, dim=2)
    v_prefix = torch.cat(prefix_v, dim=2)
    # --- Decode Phase: one step at a time ---
    for step in range(decode_steps):
        global_step = prefill_steps + step
        print(f"\n--- Decode Step {step + 1} (global token {global_step}) ---")

        # simulate next token’s projections
        query = torch.randn(B, H, 1, D, dtype=torch.float32, device=device)
        key   = torch.randn(B, H, 1, D, dtype=torch.float32, device=device)
        value = torch.randn(B, H, 1, D, dtype=torch.float32, device=device)

        # append to the full dynamic cache
        normal_k.append(key)
        normal_v.append(value)

        # offload only the *new* KV into manager
        offload(manager, key, value, torch.tensor([global_step], device=device))

        # prepare inputs for streamed SDPA:
        #   - static prefix up to `prefill_steps`
        #   - offloaded tail from manager
        

        out_stream = streamed_sdpa_cuda(
            manager,
            query,
            k_prefix,
            v_prefix,
            None, 0.0, False, None, None, True
        )

        # standard SDPA over entire dynamic cache
        k_stack = torch.cat(normal_k, dim=2)
        v_stack = torch.cat(normal_v, dim=2)
        out_normal = scaled_dot_product_attention(
            query, k_stack, v_stack,
            attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # compare
        try:
            torch.testing.assert_allclose(
                out_stream, out_normal,
                atol=1e-6, rtol=1e-5
            )
            print(f"Step {step + 1}: PASS")
        except AssertionError as e:
            print(f"Step {step + 1}: FAIL\n{e}")

        last_out_stream = out_stream
        last_out_normal = out_normal

    # final summary
    max_diff = (last_out_stream - last_out_normal).abs().max().item()
    allclose = torch.allclose(
        last_out_stream, last_out_normal,
        atol=1e-6, rtol=1e-5
    )
    print(f"\nFinal allclose: {allclose}")
    print(f"Final max absolute difference: {max_diff}")


if __name__ == "__main__":
    test_streamed_sdpa_prefill_decode()
