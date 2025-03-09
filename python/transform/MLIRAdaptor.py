from mlir.ir import *
class MLIRAdaptor():
    def __init__(self):
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        self.loc = Location.unknown(self.ctx)
        self.ctx.__enter__()
        self.loc.__enter__()
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
        mlir_format = self.mlir_module.operation.get_asm(enable_debug_info=enable_debug_info)
        return mlir_format
    
    def print_module(self):
        self.mlir_module.operation.print(enable_debug_info=True)
        
    def get_element_type(self, element_type: str):
        if element_type not in self.mlir_type:
            raise ValueError(f"Unsupported element type: {element_type}. "
                            f"Supported types are: {', '.join(self.mlir_type.keys())}")
        return self.mlir_type[element_type]
    
    def get_tensor_type(self, shape: list = None, element_type: Type = None) -> ShapedType:
        assert (shape != None) and (element_type != None)
        return RankedTensorType.get(shape, element_type)
    
    def get_string_attr(self, str: str):
        return StringAttr.get(str, self.ctx)
    
    def get_array_attr(self, arr, data_type='INT64'):
        assert (data_type in self.mlir_type)
        if data_type.find("INT") >= 0:
            return ArrayAttr.get([IntegerAttr.get(self.mlir_type[data_type], x) for x in arr], self.ctx)
        else:
            raise NotImplementedError()
                
    
    def get_none_type(self):
        return NoneType.get()
    
    def init_module(self,
                    model_name: str,
                    model_input_types: list[Type],
                    model_output_types: list[Type]):
        
        assert (len(model_name) > 0)
        self.model_name = model_name
        self.model_input_types = model_input_types
        self.model_output_types = model_output_types
        self.num_input = len(model_input_types)
        self.num_output = len(model_output_types)
        
        args_txt = str()
        for _idx, _type in enumerate(self.model_input_types):
            args_txt += "%args{}: {} loc(unknown)".format(_idx, _type.__str__())
            if (_idx + 1) < self.num_input:
                args_txt += ", "

        output_txt = str()
        for _idx, _type in enumerate(self.model_output_types):
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
