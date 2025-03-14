#ifndef DIM_MERGE_INTERFACE_TD
#define DIM_MERGE_INTERFACE_TD

include "mlir/IR/OpBase.td"

def DimMergeInterface :  OpInterface<"DimMergeInterface"> {
  let description = [{
  }];

  let cppNamespace = "::sora_mlir";

  let methods = [];
}

#endif // DIM_MERGE_INTERFACE_TD