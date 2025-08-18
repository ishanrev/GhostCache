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

#include "offload_manager.h"
#include <list>

#include "threads.h"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h> // gives you at::cuda::CUDAStream
#include <ATen/cuda/CUDAEvent.h>
#include <array>
#include <c10/cuda/CUDACachingAllocator.h>
// cuda_mem_util.h
#include <c10/cuda/CUDAFunctions.h>
#include <iostream>
#include <iomanip>
#include <string>
using namespace torch::indexing;

namespace py = pybind11;


OffloadManager::OffloadManager(int batch_size, int nh_q, int nh_k, int hd,  c10::ScalarType dtype, int chunk_size, int max_num_chunks) {
  reference_count_ = torch::zeros(1);
  max_chunk_size_ = chunk_size;
  chunk_size_ = std::vector<int>(max_num_chunks, 0);
  num_chunks_ = -1;

  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
  gpu_buf = torch::empty({2, 2, batch_size, nh_k, chunk_size, hd},options); // Double buffer x KV x B x Nh x T x hd
  
  local_max   = at::empty({batch_size, nh_q, 1}, options);
  local_sum   = at::empty({batch_size, nh_q, 1}, options);
  local_output   = at::empty({batch_size, nh_q, 1, hd}, options);

  global_max   = at::empty({batch_size, nh_q, 1}, options);
  denominator  = at::empty({batch_size, nh_q, 1}, options);
  numerator    = at::empty({batch_size, nh_q, 1, hd}, options);

  cudaStreamCreate(&streamLoad);
  cudaStreamCreate(&streamCompute); 
}
void OffloadManager::add_reference(const OffloadedKVTensor& ref) {
  offload_references_.push_back(ref);
  reference_count_ += 1;
}

torch::Tensor OffloadManager::get_reference_count() const {
  return reference_count_;
}

std::vector<OffloadedKVTensor>& OffloadManager::get_reference_list() {
  return offload_references_;
}








void offload(OffloadManager& offload_manager, torch::Tensor k, torch::Tensor v, torch::Tensor token_number){
    // Offloading logic here that will actally be based on a lot  of cahcing strategies and everything hopefully fingers crossed basically - essentially


    // similar to Paged Attention logic will be implemented here
    // First offload the new tensors
    k = k.cpu();
    v = v.cpu();

    if(offload_manager.num_chunks_!=-1 && offload_manager.chunk_size_[offload_manager.num_chunks_] < offload_manager.max_chunk_size_){

      auto& offload_tensor = offload_manager.get_reference_list()[offload_manager.num_chunks_];
      auto num_filled = offload_manager.chunk_size_[offload_manager.num_chunks_];
      if(!offload_tensor.ref.in_memory.has_value()){
        throw std::runtime_error("Attempted to access KV pair of an unintialized OffloadedKVTensor");
      }
      auto& kv_buffer = offload_tensor.ref.in_memory.value();
      kv_buffer[0].narrow(2, num_filled, 1).copy_(k);               
      kv_buffer[1].narrow(2, num_filled, 1).copy_(v);               
      

      offload_manager.chunk_size_[offload_manager.num_chunks_]++;
    }else{


   
    // Determine the location

        StorageType kv_location = StorageType::RAM;

        // Move the tensor to the specific location depending on the location chosen - 
        // further storages would require you to generate and create handshakes when you use the particular data transfer livraries so keep that in mind
        OffloadedKVTensor off_tensor;
        switch(kv_location){
          case StorageType::RAM: 
          
            std::vector<int64_t> buffer_dimension = {2, k.size(0), k.size(1), offload_manager.max_chunk_size_, k.size(3) };
            auto options = torch::TensorOptions().dtype(k.dtype()).pinned_memory(true).device(torch::kCPU);
            auto kv_buffer = torch::zeros(buffer_dimension, options);
            // auto v_buffer = torch::zeros(buffer_dimension, options);
            // std::vector<TensorIndex> index = {
            //   Slice(), Slice(), 0, Slice()
            // };
            // k_buffer.index_put_(index, k.cpu());
            // v_buffer.index_put_(index, v.cpu());

            kv_buffer[0].narrow(2, 0, 1).copy_(k);
            kv_buffer[1].narrow(2, 0, 1).copy_(v);

            LazyTensorReference reference = LazyTensorReference {
              .in_memory = kv_buffer,
              .dtype = k.scalar_type()
            };
            
            off_tensor = OffloadedKVTensor{
              
              .location = kv_location,
              .ref = reference,
              .token_num = token_number

            };
            break;
        }

        offload_manager.add_reference(off_tensor);
        offload_manager.num_chunks_ +=1;
        offload_manager.chunk_size_[offload_manager.num_chunks_] = 1;

    }

    return;

  }
// Binding function
void bind_offload_manager(pybind11::module_& m){


  py::class_<LazyTensorReference>(m, "LazyTensorReference")
        .def(py::init<>()).def_readwrite("in_memory", &LazyTensorReference::in_memory)
        .def_readwrite("dtype", &LazyTensorReference::dtype);
        

  py::class_<OffloadedKVTensor>(m, "OffloadedKVTensor")
        .def(py::init<>())  // default constructor
        .def_readwrite("location", &OffloadedKVTensor::location)
        .def_readwrite("ref", &OffloadedKVTensor::ref)
        .def_readwrite("token_num", &OffloadedKVTensor::token_num);

  py::class_<OffloadManager>(m, "OffloadManager")
      .def(py::init<int, int, int, int, c10::ScalarType, int, int>())
      .def("add_reference", &OffloadManager::add_reference, py::arg("ref"))
      .def_readwrite("chunk_size_", &OffloadManager::chunk_size_);

  m.def("offload", &offload, "Offload");
    
}

