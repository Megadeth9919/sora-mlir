#include "sora_mlir-c/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "sora_mlir-c/RegisterEverything.h"

PYBIND11_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "Sora-MLIR Dialects Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
  });
}