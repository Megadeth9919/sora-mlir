from transform.MLIRGenerator import MLIRGenerator
from mlir.ir import *
import mlir.dialects.SoraOps as sora

inputTypes = ['F32', 'F32']
outputTypes = ['F16']
mlir = MLIRGenerator([[1, 2, 3], [1, 2, 3]], [[1, 2, 3]], 'test', inputTypes, outputTypes)
sora.SoftmaxOp()
mlir.print_module()