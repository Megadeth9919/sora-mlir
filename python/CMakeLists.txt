# Refer to llvm-project/mlir/examples/standalone/python/CMakeLists.txt
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

include(AddMLIRPython)


################################################################################
# Structural groupings.
################################################################################
declare_mlir_python_sources(SoraMLIRPythonSources)
declare_mlir_python_sources(SoraMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sora_mlir"
  ADD_TO_PARENT SoraMLIRPythonSources)

################################################################################
# Dialect bindings
################################################################################
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SoraMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sora_mlir"
  TD_FILE dialects/SoraOps.td
  SOURCES
    dialects/SoraOps.py
  DIALECT_NAME sora
  )


declare_mlir_python_extension(SoraMLIRPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything # 与RegisterEverything.cpp中PYBIND11_MODULE(_mlirRegisterEverything, m)对应
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sora_mlir"
  SOURCES
    RegisterEverything.cpp
  EMBED_CAPI_LINK_LIBS
    SoraMLIRCAPI
    MLIRCAPITransforms
)

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
################################################################################
add_mlir_python_common_capi_library(SoraMLIRPythonCAPI
  INSTALL_COMPONENT SoraMLIRPythonModules
  INSTALL_DESTINATION python/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    SoraMLIRPythonSources
    # MLIRPythonExtension.RegisterEverything  # be replaced by SoraMLIRPythonExtention.RegisterEverything
    SoraMLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
  # EMBED_LIBS
  # ...
)

################################################################################
# The fully assembled package of modules.
################################################################################

add_mlir_python_modules(SoraMLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/mlir"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES
    SoraMLIRPythonSources
    SoraMLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.func
  COMMON_CAPI_LINK_LIBS
    SoraMLIRPythonCAPI
)