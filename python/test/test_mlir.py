from MLIRGenerator import MLIRGenerator
from mlir.ir import *

inputTypes = ['F32', 'F32']
outputTypes = ['F16']
mlir = MLIRGenerator([[1, 2, 3], [1, 2, 3]], [[1, 2, 3]], 'test', inputTypes, outputTypes)
print(mlir.get_module_asm())