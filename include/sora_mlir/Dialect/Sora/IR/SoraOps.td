#ifndef SORA_OPS
#define SORA_OPS

include "mlir/IR/OpBase.td"
// =============================================================================
//
// Defines Sora Dialect operations.
//
//===----------------------------------------------------------------------===//
def Sora_Dialect : Dialect {
  let name = "sora";
  let summary = "a top level dialect for sora compiler";
  let cppNamespace = "::sora_mlir::sora";
}


//===----------------------------------------------------------------------===//
// Sora Types.
//===----------------------------------------------------------------------===//

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// Sora Op Definition.
//===----------------------------------------------------------------------===//

// === BaseOp ===== //
class Sora_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sora_Dialect, mnemonic, traits> ;



def Sora_SoftmaxOp: Sora_Op<"Softmax"> {
  let summary = "Softmax operation";
  let description = [{
    # Math: y = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}
  }];
  let arguments = (ins
    AnyTensor:$input,
    SI32Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_scale
  );

  let results = (outs
    AnyTensor:$output,
    AnyTensorOrNone:$act_scale
  );

}
#endif  // SORA_OPS 