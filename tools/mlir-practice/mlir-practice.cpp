#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"

using namespace mlir;

int main(int argc, char ** argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  sora_mlir::registerAllPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "sorac-opt", registry));
}