#ifndef SORAMLIR_SORA_PASSES_TD
#define SORAMLIR_SORA_PASSES_TD

include "mlir/Pass/PassBase.td"

// pass command line: --sora-merge-high-dimmension
def MergeHighDim : Pass<"sora-merge-high-dimmension", "func::FuncOp"> {
    let summary = "merge high dims";
    let constructor = "createMergeHighDimPass()"; // 自定义扩展创建pass的过程
    let dependentDialects = ["sora::SoraDialect"];
}

#endif // SORAMLIR_SORA_PASSES_TD