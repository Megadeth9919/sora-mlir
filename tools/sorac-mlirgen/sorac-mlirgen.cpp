#include "mlir/IR/TypeRange.h"
#include "sora_mlir/Dialect/Sora/IR/SoraOps.h"


#include "mlir/IR/Location.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"


#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace mlir;
using namespace sora_mlir;

class MLIRGen {
public:
  MLIRGen(MLIRContext &context) : builder(&context) {
    theModule = ModuleOp::create(unknowloc);
    builder.setInsertionPointToEnd(theModule.getBody());
  }

  func::FuncOp buildFuncOp(TypeRange inpuType, TypeRange outputType) {
    auto functionType = FunctionType::get(builder.getContext(), inpuType, outputType);
    auto op = builder.create<func::FuncOp>(unknowloc, "main", functionType);
    auto &entryBlock = op.front();
    builder.setInsertionPointToStart(&entryBlock);
    return op;
  }

  void softmax() {
    llvm::ArrayRef<int64_t> shape = {1, 284, 1152};
    llvm::ArrayRef<int64_t> scale_shape = {1, 284, 1};
    auto inOutType = RankedTensorType::get(shape, builder.getF16Type());
    auto outScaleType = RankedTensorType::get(scale_shape, builder.getF32Type());
    auto funcOp = buildFuncOp({inOutType},{ inOutType, outScaleType});
    auto softmaxOp = builder.create<sora::SoftmaxOp>(unknowloc, inOutType, outScaleType, funcOp.getArgument(0), -1, true);
    auto returnOp = builder.create<func::ReturnOp>(unknowloc, softmaxOp);
    // auto returnOp = dyn_cast<func::ReturnOp>(funcOp.back());
  }

  ModuleOp mlirGen() {
    auto softmaxOp = softmax();
    
    return theModule;
  }

private:
  ModuleOp theModule;

  OpBuilder builder;

  Location unknowloc = builder.getUnknownLoc();
};

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<sora::SoraDialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.loadDialect<sora::SoraDialect>();
  context.loadDialect<func::FuncDialect>();
  OwningOpRef<ModuleOp> module = MLIRGen(context).mlirGen();
  module->dump();
  return 0;
}