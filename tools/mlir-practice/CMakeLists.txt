add_mlir_tool(sorac-mlirgen sorac-mlirgen.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)


target_link_libraries(sorac-mlirgen
PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIROptLib # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
)

# install(TARGETS sorac-mlirgen DESTINATION bin)