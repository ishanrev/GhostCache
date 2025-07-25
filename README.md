# 🔁 Disaggregated KV Caching for LLM Inference (WIP)

## Overview

This project explores **disaggregated KV cache storage** to enable **long-context LLM inference** beyond GPU memory limits. It introduces a tiered caching architecture that offloads **Key/Value tensors** from GPU to CPU (and eventually to disk), while maintaining throughput through asynchronous overlap of compute and I/O.

⚠️ **Under active development.** Expect ongoing optimizations, new features, and performance experiments.

---

## ✨ Key Features (Planned & In Progress)

* ✅ GPU → CPU offloading of KV cache blocks
* ✅ **Double-buffered async decode** using CUDA streams for overlapping data transfer and attention compute
* 🔄 Three-stage pipeline: **load → compute → merge**
* 🚧 KV Block Manager to minimize fragmentation across varying sequence lengths
* 🚧 **Log-sum-exp partial attention** across streamed KV blocks
* 🔄 Future: Disk-tier offload for ultra-long contexts (100K+ tokens)
* ⚙️ Aim: virtually **unlimited context** on constrained GPUs, with bounded throughput slowdown (<2×)

---

## 🔬 Architecture Summary

```
[GPU KV Buffer] ←→ [Pinned CPU Memory] ←→ [Disk Backend (planned)]
       ▲                 ▲
 Memory │           Async │
 Stream │ memcpyAsync   Transfer
       │                 │
 [Attention Engine] ←→ [KV Block Manager]
```

* **KVBlockManager:** Manages allocation, eviction, and tier transitions
* **Attention Engine:** Custom kernel leveraging `cudaMemcpyAsync` and double buffers
* **OffloadManager:** Schedules data movement across CUDA streams and CPU memory

---

## ⚙️ Tech Stack

* 🧠 **Frameworks:** PyTorch, C++ extensions
* 🚀 **Parallelism:** CUDA streams, double buffering, asynchronous memory transfers
* 🛠️ **Languages:** Python, C++
* 📦 **Deployment:** Docker, AWS

---

## 🧪 Benchmarks (Ongoing)

| Context | VRAM Usage | Throughput (tokens/s) | Offload Stage              |
| ------- | ---------- | --------------------- | -------------------------- |
| 8K      | Baseline   | TBD                   | N/A                        |
| 16K     | –30%       | TBD                   | GPU → CPU                  |
| 32K+    | TBD        | TBD                   | GPU → CPU → Disk (planned) |

⚡ **Goal:** Keep throughput slowdown under 2× while scaling context length.

---

## 📌 Development Roadmap

1. Stabilize **double-buffered async decode** and benchmark against baseline
2. Implement **log-sum-exp accumulation** in streamed attention kernel
3. Add **disk-tier offload** and adaptive caching policies
4. Integrate a demo notebook showcasing 32K+ token inference

---

## 📬 Contact & Collaboration

Maintained by **Ishan Revankar**
🔗 [LinkedIn](https://www.linkedin.com/in/ishanrev/)
📫 Open an issue or reach out for ideas and contributions!

> ⚠️ Work in progress—expect breaking changes and updates frequently.
