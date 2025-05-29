add_llvm_executable(sorac-opt sorac-opt.cpp)
target_link_libraries(sorac-opt
  PRIVATE 
  MLIRIR 
  MLIRParser 
  MLIRSupport
  MLIROptLib # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
  
  
  SoraMLIRInitAll
  SoraMLIRSoraDialect
  )
install(TARGETS sorac-opt DESTINATION bin)