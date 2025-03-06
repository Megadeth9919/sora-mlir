#include "mlir/CAPI/Registration.h"
#include "sora_mlir/Dialect/Sora/IR/SoraOps.h"



MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Sora, sora, sora_mlir::sora::SoraDialect)

