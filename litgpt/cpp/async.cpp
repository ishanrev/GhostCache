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
// A lot of the operations are static so I dont really need a class based appraoch right I dont 
// right I dont really want to update state or anything. can we this is happening in vram 
//  so it still shouldnt be a problem
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> combine(

    torch::Tensor local_max, 
    torch::Tensor local_output, 
    torch::Tensor local_sum, 
    torch::Tensor numerator, 
    torch::Tensor denominator,
    torch::Tensor global_max

){
  torch::Tensor new_max = torch::maximum(global_max, local_max);
  torch::Tensor new_numerator = numerator * torch::exp(global_max - new_max) + local_output * torch::exp(local_max - new_max);
  torch::Tensor new_denominator = denominator * torch::exp(global_max - new_max) + local_sum * torch::exp(local_max - new_max);
  return std::tuple {
      new_max, new_numerator, new_denominator
  };
}

std::pair<torch::Tensor, torch::Tensor> retrieve(OffloadedKVTensor& offloaded_tensor){

    auto& tensor_ref = offloaded_tensor.ref;
    switch(offloaded_tensor.location){
      // For ram stored tensors the tensors will definitely be in memory.
      
      case StorageType::RAM:
        if(tensor_ref.in_memory.has_value()){

          auto [key, value] = tensor_ref.in_memory.value();
          return std::make_pair(key.cuda(), value.cuda());
        }
      
      break;

    }
}

torch::Tensor streamed_sdpa(OffloadManager& manager,
    const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa){
  
      // First obtain the attention output of the prestored kv tensors that haven ofloaded and 
      // are currently supported within the VRAM (GPU RAM).

    auto [global_max, numerator, denominator] = chunked_sdpa(
        query_, key, value, attn_mask_,  dropout_p,  is_causal, dropout_mask, scale,  enable_gqa
    );
    // Currently following a linear chunking method that does not follow any sort of threading, three stage
    // loading or anything but it will be implemented once the base implementation gets done for sure.

    for(auto& offloaded_tensor: manager.get_reference_list()){

        auto [off_key, off_value] = retrieve(offloaded_tensor);
        // TODO: off_key and off_value  tensor shape checks to ensure everything is running smoothly


        // Here my assumption are that is_causal will probably be false, scale is chill it dont matter, neither does enable_gqa and dropout p is also chill. I just need to focus on the mask eralistically
        auto [local_max, local_output, local_sum] = chunked_sdpa(
            query_, off_key, off_value, attn_mask_,  dropout_p,  is_causal, dropout_mask, scale,  enable_gqa
        );
        std::tie(global_max, numerator, denominator) = combine(
          local_max, local_output, local_sum, numerator, denominator, global_max
        );
    }

    auto final_output = (numerator/denominator);


    return final_output;  

}


// void bind_async_operations(pybind11::module_& m){

//    m.def("streamed_sdpa", &streamed_sdpa, "Streamed SDPA");

// };
