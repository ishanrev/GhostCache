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



std::optional<at::Tensor> convert_boolean_attn_mask_(const std::optional<at::Tensor>& attn_mask, caffe2::TypeMeta dtype, double neg_inf);

std::optional<at::Tensor> convert_boolean_attn_mask(const std::optional<at::Tensor>& attn_mask, caffe2::TypeMeta dtype) ;


inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    std::optional<double> scale);

std::tuple<at::Tensor, at::Tensor> pre_process_group_query_attention_input(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const bool enable_gqa) ;

// std::tuple<at::Tensor, at::Tensor, at::Tensor> chunked_sdpa(
//         const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
//         const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
//         const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa);


void chunked_sdpa(
    const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa,
    at::Tensor& local_max, at::Tensor& local_output, at::Tensor& local_sum, int T);