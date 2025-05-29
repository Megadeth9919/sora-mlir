set(LLVM_TARGET_DEFINITIONS Passes.td)
# -name Sora 会生成 `void registerSoraPasses()`
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Sora)
# mlir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header)
# mlir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl)
add_public_tablegen_target(MLIRSoraPassIncGen)

add_mlir_doc(Passes MLIRSoraTransforms ./ -gen-pass-doc)