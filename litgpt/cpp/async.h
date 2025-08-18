#pragma once

#include <torch/extension.h>
#include <torch/types.h>                        // Scalar types, optional<Tensor>
#include <torch/library.h>                      // TORCH_CHECK, TORCH_WARN_ONCE
#include <c10/core/SymFloat.h>                  // c10::SymFloat
#include <c10/core/SymInt.h>                    // sym_size for symbolic shape handling
#include <c10/util/Optional.h>                  // std::optional used in PyTorch style
#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>
// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>
#include <pybind11/pybind11.h>
//Local imports
#include "offload_manager.h"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h> // gives you at::cuda::CUDAStream
#include <ATen/cuda/CUDAEvent.h>

// A lot of the operations are static so I dont really need a class based appraoch right I dont 
// right I dont really want to update state or anything. can we this is happening in vram 
//  so it still shouldnt be a problem
void combine(

    torch::Tensor local_max, 
    torch::Tensor local_output, 
    torch::Tensor local_sum, 
    const at::Tensor& numerator, 
    const at::Tensor& denominator,
    const at::Tensor& global_max,
    int T
  
  );


// torch::Tensor streamed_sdpa(OffloadManager& manager,
//     const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
//     const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
//     const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa);

    
torch::Tensor streamed_sdpa_cuda(OffloadManager& manager,
    const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa);

    
void bind_async_operations(pybind11::module_& m);
    
