#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "sora_mlir/Dialect/Sora/Transforms/Passes.h"
#include "sora_mlir/Dialect/Sora/IR/SoraOps.h"
#include "sora_mlir/Interfaces/DimMergeInterface.h"


using namespace mlir;
using namespace sora_mlir;

namespace {

}

namespace sora_mlir {
namespace sora {

#define GEN_PASS_DEF_MERGEHIGHDIM
#include "sora_mlir/Dialect/Sora/Transforms/Passes.h.inc"


class MergeHighDimPass : public impl::MergeHighDimBase<MergeHighDimPass> {
  using impl::MergeHighDimBase<MergeHighDimPass>::MergeHighDimBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();
    for (auto op : func.getOps<DimMergeInterface>()) {
      int dim = op.getMergeDim();
      Value originalOperand = op->getOperand(0); // FIXME: this can from interface
      Value originalResult = op->getResult(0);
      auto originalType = originalOperand.getType().dyn_cast<RankedTensorType>();
      if (!originalType || originalType.getRank() < dim)
        return;

      int64_t prod = 1;
      for (int64_t i = 0; i < originalType.getRank() - dim; i++)
        prod *= originalType.getShape()[i];

      auto newShape = llvm::SmallVector<int64_t, 4>({prod});
      for (int64_t i = originalType.getRank() - dim; i < originalType.getRank(); i++) 
        newShape.emplace_back(originalType.getShape()[i]);
      auto newType = RankedTensorType::get(newShape, originalType.getElementType());

      OpBuilder builder(op);
      // Insert a view op before
      auto shapeAttr = builder.getI64ArrayAttr(newShape);
      auto ViewOp1 = builder.create<sora::ViewOp>(
          op->getLoc(), newType, originalOperand, shapeAttr);
      op->setOperand(0, ViewOp1.getResult());

      // Insert a view op after
      builder.setInsertionPoint(op->getNextNode());
      auto shapeAttr2 = builder.getI64ArrayAttr(originalType.getShape());
      originalResult.setType(newType);
      auto ViewOp2 = builder.create<sora::ViewOp>(
          op->getLoc(), originalType, originalResult, shapeAttr2);
      originalResult.replaceAllUsesWith(ViewOp2.getResult());
      ViewOp2->setOperand(0, originalResult);
    }
  }
};

std::unique_ptr<mlir::Pass> createMergeHighDimPass() {
  return std::make_unique<MergeHighDimPass>();
}

} // namespace sora
} // namespace sora_mlir