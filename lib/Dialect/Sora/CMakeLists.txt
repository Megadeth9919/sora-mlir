file(GLOB _sources
        IR/*.cpp
        Transforms/*.cpp
    )

# add_mlir_library({lib_name} {src} DEPENDS ...)
add_mlir_library(SoraMLIRSoraDialect
  ${_sources} 
  
  DEPENDS
    # tableGen
    MLIRDimMergeInterfaceIncGen
    MLIRSoraOpsIncGen
    MLIRSoraPassIncGen

  LINK_LIBS PUBLIC
    MLIRFuncDialect
)