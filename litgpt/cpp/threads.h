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
#include "async.h"
#include "chunked_sdpa.h"

void loader_thread(
  std::queue<std::reference_wrapper<OffloadedKVTensor>>& loading_q,
  std::queue<std::pair<torch::Tensor, torch::Tensor>>& compute_q,
  std::queue<std::reference_wrapper<OffloadedKVTensor>>& offload_q,
  std::mutex& compute_mutex,
  std::condition_variable& compute_cv,
  std::mutex& offload_mutex,
  std::condition_variable& offload_cv,
  std::atomic<bool>& done_loading

);

void compute_thread(

  std::queue<std::pair<torch::Tensor, torch::Tensor>>& compute_q,
  std::mutex& compute_mutex,
  std::condition_variable& compute_cv,
  const torch::Tensor& query_,
  const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
  const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa,

  torch::Tensor& numerator,
  torch::Tensor& denominator,
  torch::Tensor& global_max,

  std::atomic<bool>& done_loading
);

void offload_thread(
  std::queue<std::reference_wrapper<OffloadedKVTensor>>& offload_q,
  std::mutex& offload_mutex,
  std::condition_variable& offload_cv,
  std::atomic<bool>& done_loading

);