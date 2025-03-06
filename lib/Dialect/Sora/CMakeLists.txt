file(GLOB _sources
          IR/*.cpp
    )

# add_mlir_library({lib_name} {src} DEPENDS ...)
add_mlir_library(SoraDialect
 ${_sources} 
 DEPENDS
 MLIRSoraOpsIncGen
)