#pragma once
#include "mlir/Pass/Pass.h"

namespace sora_mlir {
namespace sora {

std::unique_ptr<mlir::Pass> createMergeHighDimPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "sora_mlir/Dialect/Sora/Transforms/Passes.h.inc"

} // sora
} // sora_mlir

