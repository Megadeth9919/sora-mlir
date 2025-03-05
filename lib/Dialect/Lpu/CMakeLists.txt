file(GLOB _sources
          IR/*.cpp
          Interfaces/*.cpp
          Canonicalize/*.cpp
          Transforms/*.cpp
          Transforms/ProcessorOptimize/*.cpp
    )

# add_mlir_library({lib_name} {src} DEPENDS ...)
add_mlir_library(Lpu
 ${_sources} 
 DEPENDS 
 ...
)