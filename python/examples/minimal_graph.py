from transform.MLIRAdaptor import *
from mlir.ir import *
import mlir.dialects.SoraOps as sora

input = TensorType((1, 2, 3), 'F32')
output = TensorType((1, 2, 3), 'F32')
output_scale = TensorType((1, 2, 1), 'INT8')
mlir = MLIRAdaptor('test', 
                   model_input_types=[input],
                   model_output_types=[output, output_scale])

sora.SoftmaxOp(output.to_mlir_type(), 
               output_scale.to_mlir_type(), 
               input=mlir.func_args[0], 
               dim=-1, 
               dynamic_scale=True,
               ip=mlir.insert_point)

mlir.print_module()