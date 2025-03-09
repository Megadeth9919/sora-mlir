from .MLIRAdaptor import *
from .BaseConverter import BaseConverter
from soracc_legacy.graph_ir import *
import mlir.dialects.SoraOps as sora
from tqdm import tqdm

class SoraLegacyConverter():
    def __init__(self,
                 graph: StaticGraph):
        super().__init__()
        
        self.graph = graph
        self.values = dict()
        self.convert_type = {
            'float16': 'F16',
            'float32': 'F32',
            'int8': 'INT8',
            'int16': 'INT16'
        }
        self.op_factory = {
            "softmax": lambda op: self.convert_softmax_op(op),
            "gelu": lambda op: self.convert_gelu_op(op),
            "convert": lambda op: self.convert_convert_op(op),
            "rmsnorm": lambda op: self.convert_rmsnorm_op(op),
            "layernorm": lambda op: self.convert_layernorm_op(op),
            "eltwise": lambda op: self.convert_elementwise_op(op),
            "div": lambda op: self.convert_div_op(op),
            "rotary": lambda op: self.convert_rope_op(op),
            "linear_w8": lambda op: self.convert_linearw8_op(op),
            "matmul": lambda op: self.convert_matmulw8_op(op),
            "transpose": lambda op: self.convert_transpose_op(op),
            "split": lambda op: self.convert_split_op(op),
            "view": lambda op: self.convert_view_op(op)
        }
        self.init_mlir()
        
    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None
    
    def get_mlir_element_type(self, t: Tensor):
        # DataType.float16  ->  float16  ->  F16 -> mlir.ir.F16Type.get()
        #               (soralegacy)       (Adaptor)     (MLIR)
        return self.mlir.get_element_type(self.convert_type[t.data_type.name])
    
    def get_mlir_tensor_type(self, t: Tensor):
        return self.mlir.get_tensor_type(t.shape, self.get_mlir_element_type(t))
    
    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))
        
    def add_value(self, tensor: Tensor, value: Value):
        if tensor._id in self.values:
            if self.values[tensor._id] != value:
                raise KeyError("Value {} conflict".format(tensor.name))
        self.values[tensor._id] = value

    def get_value(self, tensor: Tensor) -> Value:
        if tensor._id not in self.values:
            if tensor in self.graph.get_weights():
                return self.create_weight_op(tensor).output
            elif tensor in self.graph.get_const():
                return self.create_weight_op(tensor).output
            raise KeyError("Value {} not found".format(tensor.name))
        return self.values[tensor._id]
        
    def init_mlir(self):
        assert self.graph is not None
        self.mlir = MLIRAdaptor()
        model_inputs_types = []
        model_outputs_types = []
        for input in self.graph.get_inputs():
            model_inputs_types.append(self.get_mlir_tensor_type(input))
        for output in self.graph.get_outputs():
            model_outputs_types.append(self.get_mlir_tensor_type(output))
        self.mlir.init_module('Sora', model_inputs_types, model_outputs_types)
        for input, arg in zip(self.graph.get_inputs(), self.mlir.func_args):
            self.add_value(input, arg)
            
    def create_weight_op(self, tensor: Tensor) -> sora.WeightOp:
        weight_op = sora.WeightOp(output=self.get_mlir_tensor_type(tensor),
                                  loc=self.get_loc(tensor.name),
                                  ip=self.mlir.insert_point)
        return weight_op
    
    def convert_softmax_op(self, op: Softmax):
        input = self.get_value(op.get_input)
        dim = op.dim
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.SoftmaxOp(output=output_type,
                                input=input,
                                dim=dim,
                                dynamic_scale=op.act_scale_flag,
                                loc=self.get_loc(op.name),
                                ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)
    
    def convert_gelu_op(self, op: Gelu):
        input = self.get_value(op.get_input)
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.GeluOp(output=output_type,
                             input=input,
                             dynamic_scale=op.act_scale_flag,
                             loc=self.get_loc(op.name),
                             ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_convert_op(self, op: Convert):
        input = self.get_value(op.get_input)
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.ConvertOp(output=output_type,
                                input=input,
                                dynamic_scale=op.act_scale_flag,
                                loc=self.get_loc(op.name),
                                ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_rmsnorm_op(self, op: RMSnorm):
        input = self.get_value(op.get_input)
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.RmsnormOp(output=output_type,
                                input=input,
                                dynamic_scale=op.act_scale_flag,
                                loc=self.get_loc(op.name),
                                ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_layernorm_op(self, op: Layernorm):
        input = self.get_value(op.get_input)
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.LayernormOp(output=output_type,
                                input=input,
                                dynamic_scale=op.act_scale_flag,
                                loc=self.get_loc(op.name),
                                ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_elementwise_op(self, op: Eltwise):
        lhs = self.get_value(op.get_input_A)
        rhs = self.get_value(op.get_input_B)
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.ElementwiseOp(output=output_type,
                                    lhs=lhs,
                                    rhs=rhs,
                                    op_type=self.mlir.get_string_attr(op.type),
                                    dynamic_scale=op.act_scale_flag,
                                    loc=self.get_loc(op.name),
                                    ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_rope_op(self, op: RoPE):
        input = self.get_value(op.get_input)
        cos_sin_table = self.get_value(op.get_cos_sin_table())
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.RopeOp(output=output_type,
                            input=input,
                            cos_sin_table=cos_sin_table,
                            dim=op.dim,
                            dynamic_scale=op.act_scale_flag,
                            loc=self.get_loc(op.name),
                            ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_linearw8_op(self, op: LinearW8):
        input = self.get_value(op.get_feature)
        weight = self.get_value(op.get_weight[0])
        bias = self.get_value(op.get_bias) if op.bias_flag else self.mlir.none_op
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.LinearW8Op(output=output_type,
                                input=input,
                                weight=weight,
                                do_bias=op.bias_flag,
                                bias=bias,
                                loc=self.get_loc(op.name),
                                ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_matmulw8_op(self, op: Matmul):
        a = self.get_value(op.get_matrix_A)
        b = self.get_value(op.get_matrix_B)
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.MatmulW8Op(output=output_type,
                                A=a,
                                B=b,
                                loc=self.get_loc(op.name),
                                ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_transpose_op(self, op):
        # skip scale
        if op.get_input.get_def() and op.get_input.get_def().act_scale_flag and op.get_input == op.get_input.get_def().get_act_scale:
            return
        if op.get_input._id not in self.values:
            return
        
        input = self.get_value(op.get_input)
        output_type = self.get_mlir_tensor_type(op.get_output)
        new_op = sora.TransposeOp(output=output_type,
                                  input=input,
                                  dim_a=op.dim_a,
                                  dim_b=op.dim_b,
                                  loc=self.get_loc(op.name),
                                  ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def convert_split_op(self, op: Split):
        input = self.get_value(op.get_input)
        output_types = [self.get_mlir_tensor_type(t) for t in op.get_outputs()]
        new_op = sora.SplitOp(outputs=output_types,
                            input=input,
                            split_size=op.split_size,
                            dim=op.dim,
                            loc=self.get_loc(op.name),
                            ip=self.mlir.insert_point)
        for output_tensor, result in zip(op.get_outputs(), new_op.outputs):
            self.add_value(output_tensor, result)
            
    def convert_div_op(self, op: Div):
        op.type = 'div'
        self.convert_elementwise_op(op)

    def convert_view_op(self, op: View):
        # skip scale's view
        if op.get_input.get_def() and op.get_input.get_def().act_scale_flag and op.get_input == op.get_input.get_def().get_act_scale:
            return
        if op.get_input._id not in self.values:
            return
        
        input: OpResult = self.get_value(op.get_input)
        output_type = self.get_mlir_tensor_type(op.get_output)
        shape = self.mlir.get_array_attr(op.shape)
        new_op = sora.ViewOp(output=output_type,
                             input=input,
                             shape=shape,
                             loc=self.get_loc(op.name),
                             ip=self.mlir.insert_point)
        self.add_value(op.get_output, new_op.output)

    def generate_mlir(self, mlir_file=None):
        try:
            for op in tqdm(self.graph.get_ops(), desc="Convert Ops", unit="op"):
                self.op_factory[op.op_type](op)
        except Exception as e:
            print(f"An error occurred during MLIR generation: {e}")
            if mlir_file is not None:
                with open(mlir_file, 'w') as f:
                    f.write(self.mlir.get_module_asm())
            else:
                self.mlir.print_module()
            raise

        if mlir_file is not None:
            with open(mlir_file, 'w') as f:
                f.write(self.mlir.get_module_asm())
        else:
            self.mlir.print_module()
