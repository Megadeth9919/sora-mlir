#pragma once

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "sora_mlir/Dialect/Sora/IR/SoraOpsDialect.h.inc"
#include "sora_mlir/Interfaces/DimMergeInterface.h"


#define GET_OP_CLASSES
#include "sora_mlir/Dialect/Sora/IR/SoraOps.h.inc"
