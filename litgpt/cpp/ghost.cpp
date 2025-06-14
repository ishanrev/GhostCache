#include <pybind11/pybind11.h>
#include "offload_manager.h"
#include "async.h"


PYBIND11_MODULE(ghost, m) {
  bind_offload_manager(m);  // ✅ all modular bindings are called here
  // bind_async_operations(m);
  m.def("streamed_sdpa", &streamed_sdpa, "Streamed SDPA");

}
