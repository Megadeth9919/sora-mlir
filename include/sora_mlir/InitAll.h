#pragma once

#include "mlir/IR/Dialect.h"

namespace sora_mlir {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // sora_mlir