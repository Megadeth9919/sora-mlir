#include "sora_mlir/Dialect/Sora/IR/SoraOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char ** argv) {
  DialectRegistry registry;
  registry.insert<sora_mlir::sora::SoraDialect>();
  return asMainReturnCode(MlirOptMain(argc, argv, "sorac-opt", registry));
}