add_mlir_tool(sorac-mlirgen sorac-mlirgen.cpp)
target_link_libraries(sorac-mlirgen
  PRIVATE
  SoraDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIROptLib # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
)

install(TARGETS sorac-mlirgen DESTINATION bin)