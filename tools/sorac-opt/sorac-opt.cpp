#include "sora_mlir/Dialect/Sora/IR/SoraOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "sora_mlir/InitAll.h"

using namespace mlir;

int main(int argc, char ** argv) {
  DialectRegistry registry;
  sora_mlir::registerAllDialects(registry);
  sora_mlir::registerAllPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "sorac-opt", registry));
}