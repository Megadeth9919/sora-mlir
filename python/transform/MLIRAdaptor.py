# import mlir.dialects.SoraOps as sora
from mlir.ir import *
from dataclasses import dataclass


@dataclass
class TensorType:
    shape: list
    element_type: str
    
    def to_mlir_type(self):
        return MLIRAdaptor.get_tensor_type(self.shape, self.element_type)

class MLIRAdaptor():
    def __init__(self,
                 model_name: str,
                 model_input_types: list[TensorType],
                 model_output_types: list[TensorType],
                 do_init_module: bool = True):
        assert (len(model_name) > 0)
        self.model_name = model_name
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        self.loc = Location.unknown(self.ctx)
        self.ctx.__enter__()
        self.loc.__enter__()
        self.model_input_types = model_input_types
        self.model_output_types = model_output_types
        self.num_input = len(model_input_types)
        self.num_output = len(model_output_types)
        self.input_mlir_types = [type.to_mlir_type() for type in self.model_input_types]
        self.output_mlir_types = [type.to_mlir_type() for type in self.model_output_types]
        # init module
        if do_init_module:
            self.declare_func()
  
    def __del__(self):
        try:
            self.loc.__exit__(None, None, None)
        except:
            pass
        try:
            self.ctx.__exit__(None, None, None)
        except:
            pass
    
    def get_module_asm(self, enable_debug_info=True):
        mlir_format = self.mlir_module.operation.get_asm(enable_debug_info)
        return mlir_format
    
    def print_module(self):
        self.mlir_module.operation.print()
        
    @staticmethod
    def get_element_type(element_type):
        mlir_type = {
            "INT8": IntegerType.get_signed(8),
            "UINT8": IntegerType.get_unsigned(8),
            "SINT8": IntegerType.get_signed(8),
            "INT16": IntegerType.get_signed(16),
            "UINT16": IntegerType.get_unsigned(16),
            "INT32": IntegerType.get_signed(32),
            "UINT32": IntegerType.get_unsigned(32),
            "INT64": IntegerType.get_signless(64),  # special
            "UINT64": IntegerType.get_unsigned(64),
            "BOOL": IntegerType.get_signless(1),
            "F64": F64Type.get(),
            "F32": F32Type.get(),
            "F16": F16Type.get(),
            "BF16": BF16Type.get(),
            "DICT": DictAttr.get(),
        }
        if element_type not in mlir_type:
            raise ValueError(f"Unsupported element type: {element_type}. "
                            f"Supported types are: {', '.join(mlir_type.keys())}")
        return mlir_type[element_type]
    
    @staticmethod
    def get_tensor_type(shape: list = None, element_type: str = None):
        assert (shape != None) and (element_type != None)
        return RankedTensorType.get(shape, MLIRAdaptor.get_element_type(element_type))
    
    @staticmethod
    def get_none_type():
        return NoneType.get()
    
    def declare_func(self):
        args_txt = str()
        for _idx, _type in enumerate(self.input_mlir_types):
            args_txt += "%args{}: {} loc(unknown)".format(_idx, _type.__str__())
            if (_idx + 1) < self.num_input:
                args_txt += ", "

        output_txt = str()
        for _idx, _type in enumerate(self.output_mlir_types):
            output_txt += _type.__str__()
            if (_idx + 1) < self.num_output:
                output_txt += ", "
        result_types = output_txt
        result_var_name = "%1"
        if self.num_output > 1:
            output_txt = "({})".format(output_txt)
            result_types = output_txt[1:-1]
            result_var_name = ",".join([f"%1#{var_id}" for var_id in range(self.num_output)])
        main_func = """
            module @\"{name}\" {{
                func.func @main({args}) -> ({result_types}) {{
                    %0 = \"sora.None\"() : () -> none loc(unknown)
                    %1:{last_output_num} = \"Placeholder.Op\"() : () -> {output}
                    return {result_var} : {result_types}
                }} loc(unknown)
            }} loc(unknown)
        """.format(name=self.model_name,
                   args=args_txt,
                   output=output_txt,
                   last_output_num=self.num_output,
                   result_var=result_var_name,
                   result_types=result_types)
        self.mlir_module = Module.parse(main_func, self.ctx)
        self.func = self.mlir_module.body.operations[0]
        self.entry_block = self.func.regions[0].blocks[0]
        self.insert_point = InsertionPoint(self.entry_block)
        self.none_op = self.entry_block.operations[0].operation.results[0]
        # remove Placeholder.Op and return Op.
        self.entry_block.operations[2].operation.erase()
        self.entry_block.operations[1].operation.erase()
        
        self.func_args = list()
        for i in self.entry_block.arguments:
            self.func_args.append(i)
