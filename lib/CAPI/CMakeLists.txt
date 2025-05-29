add_mlir_public_c_api_library(SoraMLIRCAPI
  Dialects.cpp
  RegisterEverything.cpp

  LINK_COMPONENTS # from llvm
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  SoraMLIRSoraDialect
)
