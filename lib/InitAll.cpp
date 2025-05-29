#include "sora_mlir/InitAll.h"
#include "mlir/IR/Dialect.h"
#include "sora_mlir/Dialect/Sora/IR/SoraOps.h"
#include "sora_mlir/Dialect/Sora/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace sora_mlir {

void registerAllDialects(mlir::DialectRegistry &registry) {
  // Register all dialects here
  registry.insert<sora_mlir::sora::SoraDialect,
                  mlir::func::FuncDialect>();
}

void registerAllPasses() {
  // Register all passes here
  sora_mlir::sora::registerSoraPasses();
  
}


} // namespace sora_mlir
