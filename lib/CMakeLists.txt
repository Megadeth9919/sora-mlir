add_subdirectory(Dialect)
add_subdirectory(CAPI)

add_mlir_library(SoraMLIRInitAll
  InitAll.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  # Dialects (include pass)
  SoraMLIRSoraDialect
)
