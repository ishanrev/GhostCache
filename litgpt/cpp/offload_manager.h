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
#include <cuda_runtime.h>

using namespace torch::indexing;

namespace py = pybind11;

enum class StorageType { RAM, DISK, NVME_CACHE, REMOTE };

struct LazyTensorReference {
  std::optional<torch::Tensor> in_memory;
  at::ScalarType dtype;
  // int64_t last_access;
};

struct OffloadedKVTensor {
  StorageType location;

  LazyTensorReference ref;
  // size_t token_id;
  torch::Tensor token_num;

};
class OffloadManager{
  public:

  OffloadManager(int batch_size, int nh_q, int nh_k,  int hd,  c10::ScalarType dtype, int chunk_size = 300, int max_num_chunks = 30 );
  void add_reference(const OffloadedKVTensor& ref);
  torch::Tensor get_reference_count() const;
  std::vector<OffloadedKVTensor>& get_reference_list() ;
  int max_chunk_size_;
  int num_chunks_;
  std::vector<int> chunk_size_;
  torch::Tensor gpu_buf;
  torch::Tensor local_max;
  torch::Tensor local_output;
  torch::Tensor local_sum;
  torch::Tensor global_max;
  torch::Tensor numerator;
  torch::Tensor denominator;
  cudaStream_t streamLoad;
  cudaStream_t streamCompute;

  private:
  
  std::vector<OffloadedKVTensor> offload_references_;
  torch::Tensor reference_count_;
  
  
};
void offload(OffloadManager& offload_manager, torch::Tensor k, torch::Tensor v, torch::Tensor token_number);

void bind_offload_manager(pybind11::module_& m);




