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
// Sora Interfaces Definitions.
//===----------------------------------------------------------------------===//
#include "sora_mlir/Interfaces/DimMergeInterface.cpp.inc"


//===----------------------------------------------------------------------===//
// Sora Operator Definitions.
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "sora_mlir/Dialect/Sora/IR/SoraOps.cpp.inc"


//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//
LogicalResult SoftmaxOp::verify() {
  if (getInput().getType().getShape() != getOutput().getType().getShape())
    return emitOpError("input and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// GeluOp
//===----------------------------------------------------------------------===//
LogicalResult GeluOp::verify() {
  if (getInput().getType().getShape() != getOutput().getType().getShape())
    return emitOpError("input and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//
LogicalResult ConvertOp::verify() {
  if (getInput().getType().getShape() != getOutput().getType().getShape())
    return emitOpError("input and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// RmsnormOp
//===----------------------------------------------------------------------===//
LogicalResult RmsnormOp::verify() {
  if (getInput().getType().getShape() != getOutput().getType().getShape())
    return emitOpError("input and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// LayernormOp
//===----------------------------------------------------------------------===//
LogicalResult LayernormOp::verify() {
  if (getInput().getType().getShape() != getOutput().getType().getShape())
    return emitOpError("input and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// ElementwiseOp
//===----------------------------------------------------------------------===//
LogicalResult ElementwiseOp::verify() {
  if (getLhs().getType().getShape() != getOutput().getType().getShape() &&
      getRhs().getType().getShape() != getOutput().getType().getShape())
    return emitOpError("inputs and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// RopeOp
//===----------------------------------------------------------------------===//
LogicalResult RopeOp::verify() {
  if (getInput().getType().getShape() != getOutput().getType().getShape())
    return emitOpError("input and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// LinearW8Op
//===----------------------------------------------------------------------===//
LogicalResult LinearW8Op::verify() {
  // if (getInput().getType().getShape() != getOutput().getType().getShape())
  //   return emitOpError("input and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// MatmulW8Op
//===----------------------------------------------------------------------===//
LogicalResult MatmulW8Op::verify() {
  // if (getA().getType().getShape() != getOutput().getType().getShape() ||
  //     getB().getType().getShape() != getOutput().getType().getShape())
  //   return emitOpError("inputs and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
LogicalResult TransposeOp::verify() {
  // if (getInput().getType().getShape() != getOutput().getType().getShape())
  //   return emitOpError("input and output must have the same shape");
  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//
LogicalResult SplitOp::verify() {
  auto inputShape = getInput().getType().getShape();
  int64_t totalSplitSize = 0;
  for (auto output : getOutputs()) {
    auto outputShape = cast<RankedTensorType>(output.getType()).getShape();
    totalSplitSize += outputShape[getDim()];
  }
  if (inputShape[getDim()] != totalSplitSize)
    return emitOpError("input shape does not match the combined shape of outputs");
  return success();
}

//===----------------------------------------------------------------------===//
// ViewOp
//===----------------------------------------------------------------------===//
LogicalResult ViewOp::verify() {
  auto outputShape = getOutput().getType().getShape();
  auto shapeAttr = getShapeAttr();
  
  if (!shapeAttr) {
    return emitOpError("shape attribute is missing");
  }
  
  auto shapeValues = shapeAttr.getValue();
  if (outputShape.size() != shapeValues.size()) {
    return emitOpError("input and output must have the same shape");
  }
  
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (outputShape[i] != shapeValues[i].cast<mlir::IntegerAttr>().getInt()) {
      return emitOpError("input and output must have the same shape");
    }
  }
  
  return success();
}
