file(GLOB _sources
          IR/*.cpp
    )

# add_mlir_library({lib_name} {src} DEPENDS ...)
add_mlir_library(Sora
 ${_sources} 
 DEPENDS
 MLIRSoraOpsIncGen
)