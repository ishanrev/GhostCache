# 🔁 Disaggregated KV Caching for LLM Inference (WIP)

> ⚠️ **Unofficial, personal project.**  
> This is a personal research project built **on top of** the [LitGPT](https://github.com/Lightning-AI/lit-gpt) codebase.  
> It is **not** an official LitGPT or Lightning AI repository.

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

```text
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

## 🧱 Base Implementation (LitGPT)

This project started from the [LitGPT](https://github.com/Lightning-AI/lit-gpt) codebase, which provides:

- The core **LLM inference and training** implementation  
- Model loading, configuration, and sampling utilities  
- Baseline attention and KV cache handling logic  

On top of that, this repo adds:

- Disaggregated KV cache design and offload pipeline  
- Custom attention flow for **streamed / partial KV blocks**  
- Experimental offload + scheduling logic for long-context inference  

All original credit for the base LLM engine belongs to the LitGPT authors and Lightning AI; this repository only adds additional experimental components and modifications.

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

⚡ **Goal:** Keep throughput slowdown under 2× while scaling context length.

---

## 📌 Development Roadmap

1. Stabilize **double-buffered async decode** and benchmark against baseline  
2. Implement **log-sum-exp accumulation** in streamed attention kernel  
3. Add **disk-tier offload** and adaptive caching policies  
4. Integrate a demo notebook showcasing 32K+ token inference  

---

## 📜 License & Attribution

This project retains the original **LitGPT** license and copyright
notices where applicable (see `LICENSE` in the root of this repo).

- The **base LLM implementation** and many utilities are from  
  [LitGPT](https://github.com/Lightning-AI/lit-gpt) by Lightning AI and its contributors.  
- New code and modifications in this repository are intended to be
  compatible with the same license unless otherwise noted.  

If you use this work in your own projects, please also credit LitGPT and Lightning AI.

---

## 📬 Contact & Collaboration

Maintained by **Ishan Revankar**  

🔗 [LinkedIn](https://www.linkedin.com/in/ishanrev/)  
📫 Open an issue or reach out for ideas and contributions!

> ⚠️ Work in progress—expect breaking changes and updates frequently.
