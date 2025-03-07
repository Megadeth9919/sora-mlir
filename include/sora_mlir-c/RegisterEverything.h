#ifndef SORAMLIR_C_REGISTER_EVERYTHING_H
#define SORAMLIR_C_REGISTER_EVERYTHING_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Appends all upstream dialects and extensions to the dialect registry.
MLIR_CAPI_EXPORTED void mlirRegisterAllDialects(MlirDialectRegistry registry);
MLIR_CAPI_EXPORTED void mlirRegisterAllLLVMTranslations(MlirContext context);
#ifdef __cplusplus
}
#endif

#endif // SORAMLIR_C_REGISTER_EVERYTHING_H