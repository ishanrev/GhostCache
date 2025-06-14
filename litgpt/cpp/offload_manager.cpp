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


namespace py = pybind11;



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
          .in_memory = std::pair{
            k,v
          },
          .dtype = k.scalar_type()
        };
        
        off_tensor = OffloadedKVTensor{
          
          .location = kv_location,
          .ref = reference,
          .token_num = token_number

        };
        break;
    }

    return off_tensor;

}

// defining the OffloadManager class

OffloadManager::OffloadManager() {
    reference_count_ = torch::zeros(1);
}
void OffloadManager::add_reference(const OffloadedKVTensor& ref) {
    offload_references_.push_back(ref);
    reference_count_ += 1;
}

torch::Tensor OffloadManager::get_reference_count() const {
    return reference_count_;
}

std::list<OffloadedKVTensor> OffloadManager::get_reference_list() const {
    return offload_references_;
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
      .def(py::init<>())
      .def("add_reference", &OffloadManager::add_reference, py::arg("ref"));

  m.def("offload", &offload, "Offload");
    
}