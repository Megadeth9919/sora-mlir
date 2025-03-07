# import mlir.dialects.SoraOps as sora
from mlir.ir import *
class Shape(tuple):
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return super().__new__(cls, args[0])
        else:
            return super().__new__(cls, args)
    def __init__(self, *args):
        self.dim = len(self)

    def prod(self, start=0) -> int:
        ret = 1
        for x in self[start:]:
            ret *= x
        return ret
class MLIRGenerator():
  
    def __init__(self,
                 input_shapes: list[Shape],
                 output_shapes: list[Shape],
                 model_name: str,
                 input_types: list = [],
                 output_types: list = [],
                 do_declare: bool = True):
        assert (len(model_name) > 0)
        self.model_name = model_name
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        self.loc = Location.unknown(self.ctx)
        self.ctx.__enter__()
        self.loc.__enter__()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.num_input = len(self.input_shapes)
        self.num_output = len(self.output_shapes)
        self.mlir_type = {
            "INT8": IntegerType.get_signed(8),
            "UINT8": IntegerType.get_unsigned(8),
            "SINT8": IntegerType.get_signed(8),
            "INT16": IntegerType.get_signed(16),
            "UINT16": IntegerType.get_unsigned(16),
            "INT32": IntegerType.get_signed(32),
            "UINT32": IntegerType.get_unsigned(32),
            "INT64": IntegerType.get_signless(64),
            "UINT64": IntegerType.get_unsigned(64),
            "BOOL": IntegerType.get_signless(1),
            "F64": F64Type.get(),
            "F32": F32Type.get(),
            "F16": F16Type.get(),
            "BF16": BF16Type.get(),
            "DICT": DictAttr.get(),
        }
        if do_declare:
            self.declare_func(input_types, output_types)
  
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
    
    def get_tensor_type(self, shapes:list|tuple, type=None):
        if type is None:
            type = self.mlir_type['F32']
        if shapes == []:
            return UnrankedTensorType.get(type)
        if shapes is None:
            return NoneType.get()
        if isinstance(shapes, tuple):
            shapes = list(shapes)
        assert (isinstance(shapes, list))
        assert (len(shapes) > 0)
        if not isinstance(shapes[0], list) and shapes[0] is not None:
            return RankedTensorType.get(tuple(shapes), type)
        # multi output
        out_types = []
        for s in shapes:
            if s == []:
                out_types.append(UnrankedTensorType.get(type))
            elif s is None:
                out_types.append(NoneType.get())
            else:
                out_types.append(RankedTensorType.get(tuple(s), type))
        return out_types
    
    def declare_func(self, input_types: list = [], output_types: list = []):
        if len(input_types) == 0:
            input_types = self.num_input * ['F32']
        if len(output_types) == 0:
            output_types = self.num_output * ['F32']

        self.input_types = list()
        self.input_op_types = list()
        self.output_types = list()
        for _shape, _type in zip(self.input_shapes, input_types):
            self.input_op_types.append(RankedTensorType.get(_shape, self.mlir_type['F32']))
            if isinstance(_type, str):
                self.input_types.append(RankedTensorType.get(_shape, self.mlir_type[_type]))
            else:
                self.input_types.append(RankedTensorType.get(_shape, _type))
        for _shape, _type in zip(self.output_shapes, output_types):
            t = _type
            if isinstance(_type, str):
                t = self.mlir_type[_type]
            self.output_types.append(self.get_tensor_type(_shape, t))
        args_txt = str()
        for _idx, _type in enumerate(self.input_types):
            args_txt += "%args{}: {} loc(unknown)".format(_idx, _type.__str__())
            if (_idx + 1) < self.num_input:
                args_txt += ", "

        output_txt = str()
        for _idx, _type in enumerate(self.output_types):
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
                func.func @main({args}) -> ({output}) {{
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
