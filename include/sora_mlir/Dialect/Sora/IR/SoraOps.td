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
class AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;
def NamedTypeAttr: AnyStrAttrOf<["INT8", "INT16", "INT32", "FP16", "FP32", "FP64"]>;
def ElementWiseTypeAttr: AnyStrAttrOf<["add", "sub", "mul", "div"]>;

//===----------------------------------------------------------------------===//
// Sora Op Definition.
//===----------------------------------------------------------------------===//

// === BaseOp ===== //
class Sora_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sora_Dialect, mnemonic, traits> ;


def Sora_WeightOp : Sora_Op<"Weight"> {
  let summary = "weight operator";
  let description = [{

  }];
  let results = (outs AnyTensor:$output);
}

def Sora_NoneOp : Sora_Op<"None"> {
  let summary = "none operator";

  let description = [{
    A none Op to return a NoneType.
  }];
  let results = (outs NoneType);
}

def Sora_SoftmaxOp: Sora_Op<"Softmax", 
                            [DeclareOpInterfaceMethods<DimMergeInterface>]> {
  let summary = "Softmax operation";
  let description = [{
    # Math: y = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}
  }];
  let arguments = (ins
    AnyTensor:$input,
    SI32Attr:$dim,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_scale
  );

  let results = (outs
    AnyTensor:$output
  );
}

def Sora_GeluOp : Sora_Op<"Gelu", 
                          [DeclareOpInterfaceMethods<DimMergeInterface>]> {
  let summary = "GELU (Gaussian Error Linear Unit) operation";
  let description = [{
    # Math: y = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715 * x^3)))
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_scale
  );
  let results = (outs
    AnyTensor:$output
  );
}


def Sora_ConvertOp : Sora_Op<"Convert", 
                              [DeclareOpInterfaceMethods<DimMergeInterface>]> {
  let summary = "Convert operation to change the data type of a tensor";
  let description = [{
    # Converts the input tensor to the specified output type.
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_scale
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Sora_RmsnormOp : Sora_Op<"Rmsnorm", 
                              [DeclareOpInterfaceMethods<DimMergeInterface>]> {
  let summary = "Root Mean Square Layer Normalization operation";
  let description = [{
    # Math: y = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} * \gamma + \beta
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_scale
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Sora_LayernormOp : Sora_Op<"Layernorm", 
                              [DeclareOpInterfaceMethods<DimMergeInterface>]> {
  let summary = "Layer Normalization operation";
  let description = [{
    # Math: y = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} \gamma + \beta
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_scale
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Sora_ElementwiseOp : Sora_Op<"Elementwise"> {
  let summary = "Element-wise operation";
  let description = [{
    # Element-wise operations such as add, mul, sub, div.
  }];
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    ElementWiseTypeAttr:$op_type,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_scale
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Sora_RopeOp : Sora_Op<"Rope"> {
  let summary = "Rotary Position Embedding operation";
  let description = [{
    # Math: x_{\text{rotated}} = \begin{bmatrix} x_1 \cos(\theta_i \cdot pos) - x_2 \sin(\theta_i \cdot pos) \\ x_1 \sin(\theta_i \cdot pos) + x_2 \cos(\theta_i \cdot pos) \end{bmatrix}
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$cos_sin_table,
    SI32Attr:$dim,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_scale
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Sora_LinearW8Op : Sora_Op<"LinearW8", 
                              [DeclareOpInterfaceMethods<DimMergeInterface>]> {
  let summary = "Linear layer with 8-bit (int8)";
  let description = [{
    The operation performs: y = x * weight + bias (if bias is provided).
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    DefaultValuedAttr<BoolAttr, "false">:$do_bias,
    AnyTensorOrNone:$bias
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Sora_MatmulW8Op  : Sora_Op<"MatmulW8"> {
  let summary = "Matrix multiplication with 8-bit weights (int8)";
  let description = [{
    The operation performs: C = A * B, where A, B is quantized.
  }];
  let arguments = (ins
    AnyTensor:$A,
    AnyTensor:$B
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Sora_TransposeOp : Sora_Op<"Transpose"> {
  let summary = "Transpose operation to swap two dimensions of a tensor";
  let description = [{
    Transpose operation swaps two dimensions of the input tensor.
  }];
  let arguments = (ins
    AnyTensor:$input,
    SI32Attr:$dim_a,
    SI32Attr:$dim_b
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Sora_SplitOp : Sora_Op<"Split"> {
  let arguments = (ins
    AnyTensor:$input,
    SI32Attr:$split_size,
    SI32Attr:$dim
  );

  let results = (outs
    Variadic<AnyTensor>:$outputs
  );
}

def Sora_ViewOp : Sora_Op<"View"> {
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$shape
  );

  let results = (outs
    Variadic<AnyTensor>:$outputs
  );

  let results = (outs
    AnyTensor:$output
  );
}

#endif  // SORA_OPS 