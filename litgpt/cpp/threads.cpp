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





void offload_tensor_again(OffloadedKVTensor& tensor_to_offload){
  

  switch(tensor_to_offload.location){
    case StorageType::RAM: 
      auto [k,v] = tensor_to_offload.ref.in_memory.value();

      auto k_new = k.cpu();
      auto v_new = v.cpu();
      
      tensor_to_offload.ref.in_memory = std::make_pair(k_new, v_new);

      break;
  }

  return;

}


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
return std::make_tuple (
    new_max, new_numerator, new_denominator
);
}

std::pair<torch::Tensor, torch::Tensor> retrieve(OffloadedKVTensor& offloaded_tensor){

  auto& tensor_ref = offloaded_tensor.ref;
  switch(offloaded_tensor.location){
    // For ram stored tensors the tensors will definitely be in memory.
    
    case StorageType::RAM:
      if(tensor_ref.in_memory.has_value()){

        auto [key, value] = tensor_ref.in_memory.value();
        return std::make_pair(key.cuda(), value.cuda());
      }else {
        throw std::runtime_error("Tensors not in memory");
      }
    break;
    default:
      throw std::runtime_error("Unsupported storage type");

  }
}


void loader_thread(
  std::queue<std::reference_wrapper<OffloadedKVTensor>>& loading_q,
  std::queue<std::pair<torch::Tensor, torch::Tensor>>& compute_q,
  std::queue<std::reference_wrapper<OffloadedKVTensor>>& offload_q,
  std::mutex& compute_mutex,
  std::condition_variable& compute_cv,
  std::mutex& offload_mutex,
  std::condition_variable& offload_cv,
  std::atomic<bool>& done_loading

){

  // Will add the Out of memory management lock soon once basic testing is done
  // And after that we will retest it using insane number of tokens to see if it can stabilize for long context
  // And hopefully the original sdpa support runs out 

  while(!loading_q.empty()){

    
    auto offloaded_tensor_safe_ref = std::move(loading_q.front());
    auto& offload_tensor = offloaded_tensor_safe_ref.get();
    loading_q.pop();
    std::unique_lock<std::mutex> offload_lock(offload_mutex);
    offload_q.push(offloaded_tensor_safe_ref);
    offload_lock.unlock();
    
    auto kv_pair = retrieve(offload_tensor);
    offload_cv.notify_one();


    std::unique_lock<std::mutex> lock(compute_mutex);
    compute_q.push(kv_pair);
    lock.unlock();
    compute_cv.notify_one();

    

  }

  {

    std::lock_guard<std::mutex> finished_lock_compute(compute_mutex);
    // std::lock_guard<std::mutex> finished_lock_offload(offload_mutex);
    done_loading = true;
    
  }
  
  compute_cv.notify_all();
  // offload_cv.notify_all();

}
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
){

  while(true){

    std::unique_lock<std::mutex> compute_lock(compute_mutex);
    compute_cv.wait(compute_lock, [&] {return !compute_q.empty() || done_loading;});

    if(compute_q.empty() && done_loading){
      break;
    }

    auto [off_key, off_value] = std::move(compute_q.front());
    compute_q.pop();
    compute_lock.unlock();


    auto [local_max, local_output, local_sum] = chunked_sdpa(
        query_, off_key, off_value, attn_mask_,  dropout_p,  is_causal, dropout_mask, scale,  enable_gqa
    );

    std::tie(global_max, numerator, denominator) = combine(
        local_max, local_output, local_sum, numerator, denominator, global_max
    );


  }

}
void offload_thread(
  std::queue<std::reference_wrapper<OffloadedKVTensor>>& offload_q,
  std::mutex& offload_mutex,
  std::condition_variable& offload_cv,
  std::atomic<bool>& done_loading

){

  while(true){

    std::unique_lock<std::mutex> offload_lock(offload_mutex);
    offload_cv.wait(offload_lock, [&] {return !offload_q.empty() || done_loading;});

    if(offload_q.empty() && done_loading){
      break;
    }

    auto offloaded_tensor_safe_ref = std::move(offload_q.front());
    auto& offload_tensor = offloaded_tensor_safe_ref.get();
    offload_q.pop();
    offload_lock.unlock();

    //Actually offload the tensor if that makes sense

    offload_tensor_again(offload_tensor);
  }

}