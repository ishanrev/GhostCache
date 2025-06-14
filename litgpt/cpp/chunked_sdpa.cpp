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



std::optional<at::Tensor> convert_boolean_attn_mask_(const std::optional<at::Tensor>& attn_mask, caffe2::TypeMeta dtype, double neg_inf) {
  
  if (!attn_mask.has_value()) {
    return std::nullopt;
  }
  
  
  if (attn_mask->dtype() == at::kBool) {
    return at::where(*attn_mask, 0.0, at::scalar_tensor(neg_inf, at::TensorOptions().dtype(dtype).device(attn_mask->device())));
  }
  
  return attn_mask;
}

std::optional<at::Tensor> convert_boolean_attn_mask(const std::optional<at::Tensor>& attn_mask, caffe2::TypeMeta dtype) {
  return convert_boolean_attn_mask_(attn_mask, dtype, -std::numeric_limits<double>::infinity());
}


inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    std::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

std::tuple<at::Tensor, at::Tensor> pre_process_group_query_attention_input(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const bool enable_gqa) {

  if (!enable_gqa) {
    return std::make_tuple(key, value);
  }
  const auto q_num_heads = query.sym_size(-3);
  const auto k_num_heads = key.sym_size(-3);
  const auto v_num_heads = value.sym_size(-3);

  bool all_equal = q_num_heads == k_num_heads && k_num_heads == v_num_heads;
  bool key_divisible = q_num_heads % k_num_heads == 0;
  bool value_divisible = q_num_heads % v_num_heads == 0;
  TORCH_CHECK(all_equal || (key_divisible && value_divisible),
              "Number of heads in key and value must divide the number of heads in ");

  if (all_equal){
    return std::make_tuple(key, value);
  }
  auto repeat_key_shape = query.sym_size(-3) / key.sym_size(-3);
  auto repeat_value_shape = query.sym_size(-3) / value.sym_size(-3);

  at::Tensor key_repeated = key.repeat_interleave_symint(repeat_key_shape, -3);
  at::Tensor value_repeated = value.repeat_interleave_symint(repeat_value_shape, -3);
  return std::make_tuple(std::move(key_repeated), std::move(value_repeated));
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> chunked_sdpa(
        const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
        const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
        const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa) {
  C10_LOG_API_USAGE_ONCE("torch.sdpa.math_fallback");
  
  auto& ctx = at::globalContext();
  auto origin_dtype = query_.scalar_type();

  
  auto attn_mask = attn_mask_;
  
  bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
  const auto scaling_factor =
      calculate_scale(
          query_, is_negative_scaling ? std::abs(scale.value()) : scale)
          .sqrt();

  const auto query = query_ *
      (is_negative_scaling ? c10::SymFloat(0.0) - scaling_factor
                           : scaling_factor);

  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");

    
    const auto L = query.sym_size(-2), S = key.sym_size(-2);
    attn_mask =
        at::ones_symint({L, S}, query.options().dtype(at::kBool)).tril();
    attn_mask = convert_boolean_attn_mask(attn_mask, query.dtype());
    }


    
    auto [key_expanded, value_expanded] = pre_process_group_query_attention_input(query, key, value, enable_gqa);

    auto attn = at::matmul(query, key_expanded.transpose(-2, -1) * scaling_factor);
    if (attn_mask.has_value()) {

      attn.add_(*attn_mask);
      
    }

    // Custom chunking mechanisms.
    auto local_max = attn.max();
    auto exp_attn = torch::exp(attn-local_max);
    auto local_sum = exp_attn.sum();
    auto local_output = at::matmul(exp_attn, value_expanded).to(origin_dtype);

    
    // attn = at::_safe_softmax(attn, -1);

    return std::make_tuple(local_max, local_output, local_sum);
}



