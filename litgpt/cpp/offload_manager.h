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



namespace py = pybind11;

enum class StorageType { RAM, DISK, NVME_CACHE, REMOTE };

struct LazyTensorReference {
  std::optional<std::pair<torch::Tensor, torch::Tensor>> in_memory;
  at::ScalarType dtype;
  // int64_t last_access;
};

struct OffloadedKVTensor {
  StorageType location;
  LazyTensorReference ref;
  // size_t token_id;
  torch::Tensor token_num;

};

OffloadedKVTensor offload(torch::Tensor k, torch::Tensor v, torch::Tensor token_number);

class OffloadManager{
  public:

  OffloadManager();

  void add_reference(const OffloadedKVTensor& ref);
  torch::Tensor get_reference_count() const;
  std::list<OffloadedKVTensor> get_reference_list() const;
  private:
  
  std::list<OffloadedKVTensor> offload_references_;
  torch::Tensor reference_count_;


};

void bind_offload_manager(pybind11::module_& m);




