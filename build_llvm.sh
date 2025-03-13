#!/bin/bash
set -e
cd externals/llvm-project/build
# rm -r *
cmake ../llvm -GNinja \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
ninja
ninja install