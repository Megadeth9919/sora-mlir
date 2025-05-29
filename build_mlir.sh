#!/bin/bash
set -e
# mkdir build
cd build`
cmake .. -GNinja \
  -DLLVM_DIR=externals/llvm-project/install/lib/cmake/llvm \
  -DMLIR_DIR=externals/llvm-project/install/lib/cmake/mlir \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH
    