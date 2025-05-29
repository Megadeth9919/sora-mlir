#ifndef DIM_MERGE_INTERFACE_TD
#define DIM_MERGE_INTERFACE_TD

include "mlir/IR/OpBase.td"

def DimMergeInterface :  OpInterface<"DimMergeInterface"> {
  let description = [{
  }];

  let cppNamespace = "::sora_mlir";

  let methods = [
    InterfaceMethod</*desc*/"get dim(reversed order), the dims before this dim will be merged",
                    /*retTy*/"int", 
                    /*methodName*/"getMergeDim",
                    /*args*/(ins),
                    /*methodBody*/"",
                    /*defaultImplementation*/"return 1;">
  ];
}

#endif // DIM_MERGE_INTERFACE_TD