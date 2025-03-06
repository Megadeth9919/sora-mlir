add_mlir_tool(sorac-opt sorac-opt.cpp)
target_link_libraries(sorac-opt
  PRIVATE SoraDialect
  MLIRIR MLIRParser MLIRSupport
  MLIROptLib # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
)