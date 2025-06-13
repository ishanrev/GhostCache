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

#ifdef offload_manager
#undef offload_manager
#endif

namespace py = pybind11;

enum class StorageType { RAM, DISK, NVME_CACHE, REMOTE };

struct LazyTensorReference {
  StorageType location;
  std::optional<std::pair<torch::Tensor, torch::Tensor>> in_memory;
  at::ScalarType dtype;
  // int64_t last_access;
};

struct OffloadedKVTensor {
  LazyTensorReference ref;
  // size_t token_id;
  torch::Tensor token_num;

};

OffloadedKVTensor offload(torch::Tensor k, torch::Tensor v, torch::Tensor token_number){
    // Offloading logic here that will actally be based on a lot  of cahcing strategies and everything hopefully fingers crossed basically - essentially

    // Determine the location

    StorageType kv_location = StorageType::RAM;

    // Move the tensor to the specific location depending on the location chosen - 
    // further storages would require you to generate and create handshakes when you use the particular data transfer livraries so keep that in mind
    OffloadedKVTensor off_tensor;
    switch(kv_location){
      case StorageType::RAM: 
        k = k.cpu();
        v = v.cpu();
        LazyTensorReference reference = LazyTensorReference {
          .location = kv_location,
          .in_memory = std::pair{
            k,v
          },
          .dtype = k.scalar_type()
        };
        
        off_tensor = OffloadedKVTensor{
          
          .ref = reference,
          .token_num = token_number

        };
    }

    return off_tensor;

}

class OffloadManager{
  public:

  OffloadManager(
     
  ){
    reference_count_ = torch::zeros(1);
  }

  void add_reference(LazyTensorReference ref){
    offload_references_.push_back(ref);
    reference_count_.add_(1);
  }
  
  private:
  
  std::list<LazyTensorReference> offload_references_;
  torch::Tensor reference_count_;


};




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){

  py::class_<LazyTensorReference>(m, "LazyTensorReference")
        .def(py::init<>())  // default constructor
        .def_readwrite("location", &LazyTensorReference::location)
        .def_readwrite("in_memory", &LazyTensorReference::in_memory)
        .def_readwrite("dtype", &LazyTensorReference::dtype);

  py::class_<OffloadManager>(m, "OffloadManager")
      .def(py::init<>())
      .def("add_reference", &OffloadManager::add_reference, py::arg("ref"));

  m.def("offload", &offload, "Offload");
    
}