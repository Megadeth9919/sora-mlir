#include "sora_mlir/Dialect/Sora/IR/SoraOps.h"

using namespace mlir;
using namespace sora_mlir::sora;


//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//

#include "sora_mlir/Dialect/Sora/IR/SoraOpsDialect.cpp.inc"
void SoraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sora_mlir/Dialect/Sora/IR/SoraOps.cpp.inc"
  >();
}


//===----------------------------------------------------------------------===//
// Sora Operator Definitions.
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "sora_mlir/Dialect/Sora/IR/SoraOps.cpp.inc"



