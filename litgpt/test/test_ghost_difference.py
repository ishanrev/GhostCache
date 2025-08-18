import time
import torch
import pandas as pd
from torch.nn.functional import scaled_dot_product_attention
from ghost import OffloadManager, offload, streamed_sdpa_cuda
from normal_sdpa import normal_sdpa
import math
# Configuration
device = "cuda"
torch.manual_seed(0)
B, nh, D = 32, 32, 128    # smaller dims for faster benchmarking
n_embd = nh * D
threshold = 5          # start offloading after this many tokens

def get_dynamic_num_blocks():    
    free, total_bytes = torch.cuda.mem_get_info()
    free = 0.6*free
    tensor = torch.tensor([], dtype = torch.float32, device = "cuda")
    dtype_num_bits = tensor.element_size()
    print(tensor.dtype)
    num_blocks = math.floor(free/(2 * 2 * B * n_embd * dtype_num_bits)) #  Since we are using a double buffer strategy in this case we need to employ this rule 
    return num_blocks

# This is wrong on so many levels I need to consider kv + I need to consider batch size

def profile_streamed(length):
    max_blocks = get_dynamic_num_blocks()
    print(max_blocks)
    manager = OffloadManager(B, nh, D, torch.float32, max_blocks, 30)
    q = torch.randn(B, nh, 1, D, device=device)
    total_offload = 0.0
    total_sdpa = 0.0

    for step in range(length):
        if step%100 == 0:
            free, total_bytes = torch.cuda.mem_get_info()
            print(f" {free/(1000000000)} free GB available")
        print(step)
        k = torch.randn(B, nh, 1, D, device=device)
        v = torch.randn(B, nh, 1, D, device=device)

        # Offload phase (only after threshold)
        torch.cuda.synchronize()
        start_off = time.perf_counter()
        offload(manager, k, v, torch.tensor([step], device=device))
        # manager.add_reference(off_t)
        torch.cuda.synchronize()
        total_offload += time.perf_counter() - start_off

        # streamed_sdpa phase
        torch.cuda.synchronize()
        start_sdpa = time.perf_counter()
        streamed_sdpa_cuda(
            manager, q, k, v,
            None, 0.0, False, None, None, False
        )
        torch.cuda.synchronize()
        total_sdpa += time.perf_counter() - start_sdpa

    total_time = total_offload + total_sdpa
    # print("Number of chunks = ", manager.num_chunks_)
    # print("Chunk filled number = ", manager.chunk_size_)
    return total_offload, total_sdpa, total_time

def profile_normal(length):
    kv_list = []
    q = torch.randn(B, nh, 1, D, device=device)
    total_stack = 0.0
    total_sdpa = 0.0

    for step in range(length):
        print(step)

        k = torch.randn(B, nh, 1, D, device=device)
        v = torch.randn(B, nh, 1, D, device=device)
        kv_list.append((k, v))

        # stacking phase
        torch.cuda.synchronize()
        start_stack = time.perf_counter()
        k_stack = torch.cat([kv[0] for kv in kv_list], dim=2)
        v_stack = torch.cat([kv[1] for kv in kv_list], dim=2)
        torch.cuda.synchronize()
        total_stack += time.perf_counter() - start_stack

        # sdpa phase
        torch.cuda.synchronize()
        start_sdpa = time.perf_counter()
        scaled_dot_product_attention(
            q, k_stack, v_stack,
            attn_mask=None, dropout_p=0.0, is_causal=False
        )
        torch.cuda.synchronize()
        total_sdpa += time.perf_counter() - start_sdpa

    total_time = total_stack + total_sdpa
    return total_stack, total_sdpa, total_time


def profile_normal_sdpa(length):
    kv_list = []
    q = torch.randn(B, nh, 1, D, device=device)
    total_stack = 0.0
    total_sdpa = 0.0

    for step in range(length):
        k = torch.randn(B, nh, 1, D, device=device)
        v = torch.randn(B, nh, 1, D, device=device)
        kv_list.append((k, v))

        # stacking phase
        torch.cuda.synchronize()
        start_stack = time.perf_counter()
        k_stack = torch.cat([kv[0] for kv in kv_list], dim=2)
        v_stack = torch.cat([kv[1] for kv in kv_list], dim=2)
        torch.cuda.synchronize()
        total_stack += time.perf_counter() - start_stack

        # sdpa phase
        torch.cuda.synchronize()
        start_sdpa = time.perf_counter()
        normal_sdpa(
            q, k, v,
            None, 0.0, False, None, None, False
        )
        torch.cuda.synchronize()
        total_sdpa += time.perf_counter() - start_sdpa

    total_time = total_stack + total_sdpa
    return total_stack, total_sdpa, total_time

# Benchmark across various decode lengths
lengths = [2000]
results = []

for L in lengths:
    torch_stack_time, torch_sdpa_time_norm, torch_total_norm = profile_normal(L)
    # normal_stack_time, normal_sdpa_time_norm, normal_total_norm = profile_normal_sdpa(L)
    print('---------------')
    # print(torch_total_norm)
    print("done with first")
    offload_time, sdpa_time_str, total_str = profile_streamed(L)
    print(total_str)

    results.append({
        # "decode_length": L,
        # "torch_stack_time_s": torch_stack_time,
        # "torch_sdpa_time_s": torch_sdpa_time_norm,
        # "torch_stack_pct": 100 * torch_stack_time / torch_total_norm,
        # "torch_sdpa_pct": 100 * torch_sdpa_time_norm / torch_total_norm,
        # "normal_stack_time_s": normal_stack_time,
        # "normal_sdpa_time_s": normal_sdpa_time_norm,
        # "normal_stack_pct": 100 * normal_stack_time / normal_total_norm,
        # "normal_sdpa_pct": 100 * normal_sdpa_time_norm / normal_total_norm,
        # "sdpa_comparison": normal_sdpa_time_norm/torch_sdpa_time_norm
        "streamed_offload_time_s": offload_time,
        "streamed_sdpa_time_s": sdpa_time_str,
        "streamed_offload_pct": 100 * offload_time / total_str,
        "streamed_sdpa_pct": 100 * sdpa_time_str / total_str,
        "speedup": torch_total_norm / total_str
    })

# Display results using pandas
df = pd.DataFrame(results)
print(df)
