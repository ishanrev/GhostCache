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

#include <thread>
#include <mutex>
#include <condition_variable>

#include "threads.h"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h> // gives you at::cuda::CUDAStream
#include <ATen/cuda/CUDAEvent.h>
#include <array>
#include <c10/cuda/CUDACachingAllocator.h>
// cuda_mem_util.h
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <iostream>
#include <iomanip>
#include <string>

inline void print_allocated_mem(const std::string& prefix = "", int device = -1) {
  if (device < 0) {
    device = c10::cuda::current_device();
}

auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);

// `allocated_bytes[0]` holds the "current" stat struct
uint64_t bytes = stats.allocated_bytes[0].current;

double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);

if (!prefix.empty()) {
    std::cout << prefix << " ";
}
std::cout << "allocated = " << std::fixed << std::setprecision(5)
          << mb << " MB" << std::endl;
}




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

){

  

  auto new_max_slice   = torch::maximum(global_max,   local_max);       // [B,nh,T]
  auto exp1_slice      = torch::exp(global_max - new_max_slice);
  auto exp2_slice      = torch::exp(local_max  - new_max_slice);

  auto numer_update    = numerator * exp1_slice.unsqueeze(-1)
                      + local_output * exp2_slice.unsqueeze(-1);      // [B,nh,T,hd]
  auto denom_update    = denominator * exp1_slice
                      + local_sum    * exp2_slice;      // [B,nh,T,hd]

  // 3) Copy back into the full buffers
  global_max.copy_(new_max_slice);

  numerator.copy_(numer_update);
  
  denominator.copy_(denom_update);
}

torch::Tensor retrieve(OffloadedKVTensor& offloaded_tensor, int end){

  auto& tensor_ref = offloaded_tensor.ref;
  switch(offloaded_tensor.location){
    // For ram stored tensors the tensors will definitely be in memory.
    
    case StorageType::RAM:
      if(tensor_ref.in_memory.has_value()){
        if (!offloaded_tensor.ref.in_memory.has_value()) {
          throw std::runtime_error("KV buffer is unset before retrieval");
        }
        auto kv = tensor_ref.in_memory.value();
        // key = key.index({Slice(), Slice(), Slice(), Slice(0, end), Slice()});
        // kv = kv.index({Slice(), Slice(), Slice(), Slice(0, end), Slice()});
        
        return kv;
      }else {
        throw std::runtime_error("Tensors not in memory");
      }
    break;
    default:
      throw std::runtime_error("Unsupported storage type");

  }
}


void dynamic_chunk_copy(

    void* host_ptr,
    void* device_ptr,
    int64_t B, 
    int64_t nh,
    int64_t max_chunk,
    int64_t hidden_dim,
    int64_t end,
    int64_t element_size,
    cudaStream_t streamLoad
){

  auto rows = 2 * B * nh;
  size_t row_pitch = hidden_dim * element_size;

  cudaPitchedPtr srcPtr = make_cudaPitchedPtr(
    (void*)host_ptr,
    row_pitch,
    hidden_dim,
    max_chunk

  );
  cudaPitchedPtr dstPtr = make_cudaPitchedPtr(
    (void*)device_ptr,
    row_pitch,
    hidden_dim,
    max_chunk

  );

  cudaExtent copyExtent = make_cudaExtent(
    row_pitch,
    end,
    rows
  );
  

  cudaMemcpy3DParms params = {};
  params.srcPtr = srcPtr;
  params.dstPtr = dstPtr;
  params.extent = copyExtent;
  params.kind = cudaMemcpyHostToDevice;
  
  cudaMemcpy3DAsync(&params, streamLoad);
  
}


torch::Tensor streamed_sdpa_cuda(OffloadManager& manager,
  const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
  const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
  const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa){
    
    // First obtain the attention output of the prestored kv tensors that haven ofloaded and 
    // are currently supported within the VRAM (GPU RAM).
  c10::InferenceMode guard(true);    
    // Reference the intermediate tensors now itself.
  int device_index = c10::cuda::current_device();
  print_allocated_mem("On entering CUDA sdpa main function ", device_index);
  auto local_max = manager.local_max;
  auto local_output = manager.local_output;
  auto local_sum = manager.local_sum;
  auto global_max = manager.global_max;
  auto numerator = manager.numerator;
  auto denominator = manager.denominator;
  auto toMB = [](size_t bytes){ return static_cast<double>(bytes) / (1024.0*1024.0); };

  int first_T = key.size(2);
  int offset = 0;
  // if (first_T>1){
    
  // }
    
    // attn_mask_.value().narrow(-1, offset, first_T)
  
  // Currently following a linear chunking method that does not follow any sort of threading, three stage
  // loading or anything but it will be implemented once the base implementation gets done for sure.

  if(manager.get_reference_list().size()>0){

    // cudaStream_t streamLoad, streamCompute;

    // cudaStreamCreate(&streamLoad);
    // cudaStreamCreate(&streamCompute); 
    

    cudaEvent_t loaded[2];
    cudaEvent_t done[2];
    cudaEventCreate(&loaded[0]);
    cudaEventCreate(&loaded[1]);
    cudaEventCreate(&done[0]);
    cudaEventCreate(&done[1]);



    auto torch_stream_compute = c10::cuda::getStreamFromExternal(manager.streamCompute, device_index);
    
    
    print_allocated_mem("Before first SDPA", device_index);
    {
          
      c10::cuda::CUDAStreamGuard guard(torch_stream_compute);
      chunked_sdpa(
        query_, key, value, attn_mask_,  dropout_p,  is_causal, dropout_mask, scale,  enable_gqa,
        global_max, numerator, denominator, first_T
      );
      offset += first_T;

    }
    print_allocated_mem("After first SDPA", device_index);
    
    
    std::array<torch::Tensor, 2> staging;
    auto& reference_list = manager.get_reference_list();
    size_t N = reference_list.size();
    staging[0] = retrieve(reference_list[0], manager.chunk_size_[0]);
    

    void* host_ptr   = static_cast<void*>(staging[0].data_ptr());
    void* device_ptr = static_cast<void*>(manager.gpu_buf[0].data_ptr());
    int64_t KV         = staging[0].size(0);  // should be 2
    int64_t B          = staging[0].size(1);
    int64_t nh         = staging[0].size(2);
    int64_t max_chunk  = staging[0].size(3);
    int64_t hidden_dim = staging[0].size(4);
    int64_t end        = manager.chunk_size_[0];
    int64_t elem_size  = staging[0].element_size();
    // Fire off the pitched 3D copy
    dynamic_chunk_copy(
      host_ptr,
      device_ptr,
      B, nh,
      max_chunk,
      hidden_dim,
      end,
      elem_size,
      manager.streamLoad
    );
    cudaEventRecord(loaded[0], manager.streamLoad);
    print_allocated_mem("After first copy", device_index);
    
    for(int x = 1; x <= N; x++){

        int cur = x % 2, prev = 1-cur;

        if(x<N){
          cudaStreamWaitEvent(manager.streamLoad, done[cur], 0);
          staging[cur] = retrieve( reference_list[x], manager.chunk_size_[x]);
          
          host_ptr   = static_cast<void*>(staging[cur].data_ptr());
          device_ptr = static_cast<void*>(manager.gpu_buf[cur].data_ptr());
          end = manager.chunk_size_[x];
          dynamic_chunk_copy(
            host_ptr,
            device_ptr,
            B, nh,
            max_chunk,
            hidden_dim,
            end,
            elem_size,
            manager.streamLoad
          );

          cudaEventRecord(loaded[cur], manager.streamLoad);
          print_allocated_mem("After copy chunk" + x, device_index);
          
        }
        cudaStreamWaitEvent(manager.streamCompute, loaded[prev], 0);
        {
          
          c10::cuda::CUDAStreamGuard guard(torch_stream_compute);
          auto T = manager.chunk_size_[x-1];
          auto k_buffer = manager.gpu_buf[prev][0].narrow(2, 0, T );
          auto v_buffer = manager.gpu_buf[prev][1].narrow(2, 0, T );
          // attn_mask_.value().narrow(-1, offset, T)
          chunked_sdpa(
            query_, k_buffer, v_buffer, attn_mask_,  dropout_p,  is_causal, dropout_mask, scale,  enable_gqa,
            local_max, local_output, local_sum, T
          );

          combine(
            local_max, local_output, local_sum, numerator, denominator, global_max, T
          );

          offset+=T;

          cudaEventRecord(done[prev], manager.streamCompute);
          print_allocated_mem("After compute chunk" + x , device_index);
          
        }
      

    }

    cudaStreamSynchronize(manager.streamCompute);
    cudaStreamSynchronize(manager.streamLoad);

  }

  
  auto final_output = (numerator/denominator.unsqueeze(-1));


  return final_output;  

}





// torch::Tensor streamed_sdpa(OffloadManager& manager,
//     const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
//     const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
//     const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa){
  
//       // First obtain the attention output of the prestored kv tensors that haven ofloaded and 
//       // are currently supported within the VRAM (GPU RAM).

//     auto [global_max, numerator, denominator] = chunked_sdpa(
//         query_, key, value, attn_mask_,  dropout_p,  is_causal, dropout_mask, scale,  enable_gqa
//     );
//     // Currently following a linear chunking method that does not follow any sort of threading, three stage
//     // loading or anything but it will be implemented once the base implementation gets done for sure.
//     int x = 0;
//     for(auto& offloaded_tensor: manager.get_reference_list()){

//         auto [off_key, off_value] = retrieve(offloaded_tensor, manager.chunk_size_[x]);
//         // TODO: off_key and off_value  tensor shape checks to ensure everything is running smoothly
  
//         // Here my assumption are that is_causal will probably be false, scale is chill it dont matter, neither does enable_gqa and dropout p is also chill. I just need to focus on the mask eralistically
//         auto [local_max, local_output, local_sum] = chunked_sdpa(
//             query_, off_key, off_value, attn_mask_,  dropout_p,  is_causal, dropout_mask, scale,  enable_gqa
//         );
//         std::tie(global_max, numerator, denominator) = combine(
//           local_max, local_output, local_sum, numerator, denominator, global_max
//         );
//         x = x + 1;
//     }

//     auto final_output = (numerator/denominator);


//     return final_output;  

// }


// void bind_async_operations(pybind11::module_& m){

//    m.def("streamed_sdpa", &streamed_sdpa, "Streamed SDPA");

// };

// Novel implementation that supportes the true threaded liike asy nc structure,


// TODO: Remove the redundant offload pipeline because its not really ever necessay if 
//       we maintain a simple async loading and processing unit because what we care about in the end
// torch::Tensor streamed_sdpa(OffloadManager& manager,
//   const at::Tensor& query_, const at::Tensor& key, const at::Tensor& value,
//   const std::optional<at::Tensor>& attn_mask_, double dropout_p, bool is_causal,
//   const std::optional<at::Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa){

//     // First obtain the attention output of the prestored kv tensors that haven ofloaded and 
//     // are currently supported within the VRAM (GPU RAM).
//   auto [global_max, numerator, denominator] = chunked_sdpa(
//       query_, key, value, attn_mask_,  dropout_p,  is_causal, dropout_mask, scale,  enable_gqa
//   );

//   std::queue<std::reference_wrapper<OffloadedKVTensor>> loading_q;
//   std::queue<std::pair<torch::Tensor, torch::Tensor>> compute_q;
//   std::queue<std::reference_wrapper<OffloadedKVTensor>> offload_q;

//   std::mutex compute_mutex;
//   std::condition_variable compute_cv;

//   std::mutex offload_mutex;
//   std::condition_variable offload_cv;

//   std::atomic<bool> done_loading = false;



//   for (auto& offloaded_tensor: manager.get_reference_list()){
//     loading_q.push(std::ref(offloaded_tensor));
//   }
 
//   // Currently following a linear chunking method that does not follow any sort of threading, three stage
//   // loading or anything but it will be implemented once the base implementation gets done for sure.

//   // here we start the async process by starting the individual threads in async right 
//   // so do 

//   std::thread loader(
//       loader_thread,
//       std::ref(loading_q), 
//       std::ref(compute_q), 
//       std::ref(offload_q), 
//       std::ref(compute_mutex), 
//       std::ref(compute_cv),
//       std::ref(offload_mutex), 
//       std::ref(offload_cv),
//       std::ref(done_loading)
//     );

//   std::thread compute(
//       compute_thread, 
//       std::ref(compute_q),  
//       std::ref(compute_mutex),
//       std::ref(compute_cv),
     
//       std::ref(query_),
//       std::ref(attn_mask_),
//       dropout_p,
//       is_causal,
//       std::ref(dropout_mask),
//       scale,
//       enable_gqa,

//       std::ref(numerator),
//       std::ref(denominator),
//       std::ref(global_max),
//       std::ref(done_loading)

//     );
    
//   // std::thread offload(
//   //     offload_thread, 
//   //     std::ref(offload_q),
//   //     std::ref(offload_mutex),
//   //     std::ref(offload_cv),
//   //     std::ref(done_loading)
      

//   // );

//   loader.join();
//   compute.join();
//   // offload.join();
  

  
//   //-----------------------------------------------------------------------------
//   auto final_output = (numerator/denominator);


//   return final_output;  

// }

// Lets get this working for now with some tensors offloaded and the others not offloaded so that makes sense
// Then we move on to distributing the chunks even better

