#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJECT_ROOT=$DIR
export INSTALL_PATH=${INSTALL_PATH:-$PROJECT_ROOT/install}
export LLVM_INSTALL_PATH=${LLVM_INSTALL_PATH:-$PROJECT_ROOT/externals/llvm-project/install}

export PYTHONPATH=$INSTALL_PATH/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/python/soracc_legacy:$PYTHONPATH

export PATH=$INSTALL_PATH/bin:$PATH
export PATH=$LLVM_INSTALL_PATH/bin:$PATH

