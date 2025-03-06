
#include "sora_mlir-c/RegisterEverything.h"
#include "sora_mlir/Dialect/Sora/IR/SoraOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"

void mlirRegisterAllDialects(MlirDialectRegistry registry) {
  static_cast<mlir::DialectRegistry *>(registry.ptr)
      ->insert<mlir::func::FuncDialect, sora_mlir::sora::SoraDialect>();
}
