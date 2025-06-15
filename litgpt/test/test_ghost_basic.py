import torch
from torch.nn.functional import scaled_dot_product_attention
from ghost import OffloadManager, offload, streamed_sdpa
from normal_sdpa import normal_sdpa
def test_streamed_sdpa_chunked_flow():
    """
    Simulates a 3-step autoregressive decode where:
      - Each step's (K, V) pair is passed as the 'in-memory' chunk to streamed_sdpa.
      - All *previous* KV pairs are offloaded and managed by OffloadManager.
      - Verifies streamed_sdpa matches torch's standard SDPA at each step.
    """
    torch.manual_seed(0)
    device = "cuda"
    B, H, D = 3, 2, 8  # Batch, Heads, Head dimension

    manager = OffloadManager()
    all_kv = []  # To keep track of all (key, value) pairs for baseline

    for step in range(3):
        print(f"\n--- Decode Step {step + 1} ---")

        # Simulate current step's model outputs
        query = torch.randn(B, H, 1, D, device=device)
        key   = torch.randn(B, H, 1, D, device=device)
        value = torch.randn(B, H, 1, D, device=device)

        # Offload all *previous* KV pairs into the manager
        if step > 0:
            prev_k, prev_v = all_kv[-1]
            off_tensor = offload(prev_k, prev_v, torch.tensor([step - 1], device=device))
            manager.add_reference(off_tensor)

        # Run chunked/streamed SDPA
        if False:
         [ out_stream, bruh] = normal_sdpa(
              query,
              key,           # acts as the "in-RAM" starter chunk
              value,
              None,          # no attention mask
              0.0,           # dropout probability
              False,         # not causal for simplicity
              None,          # no dropout mask
              None,          # no custom scale
              False          # disable GQA
        )
        else:
          
          out_stream = streamed_sdpa(
              manager,
              query,
              key,           # acts as the "in-RAM" starter chunk
              value,
              None,          # no attention mask
              0.0,           # dropout probability
              False,         # not causal for simplicity
              None,          # no dropout mask
              None,          # no custom scale
              False          # disable GQA
          )

        print(out_stream.shape)
        # Baseline SDPA over all tokens so far (including current)
        all_kv.append((key, value))
        k_stack = torch.cat([kv[0] for kv in all_kv], dim=2)
        v_stack = torch.cat([kv[1] for kv in all_kv], dim=2)
        out_baseline = scaled_dot_product_attention(
            query, k_stack, v_stack,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        # Verify outputs match
        try:
            torch.testing.assert_allclose(out_stream, out_baseline, atol=1e-6, rtol=1e-5)
            print(f"Step {step + 1}: PASS")
        except AssertionError as e:
            print(f"Step {step + 1}: FAIL\n{e}")

    print("\nAll steps matched between streamed_sdpa and standard SDPA.")

if __name__ == "__main__":
    test_streamed_sdpa_chunked_flow()
