from .graph_ir import *
import json
def cut_gelu_dscale(g: StaticGraph):
    for op in g.get_ops() :
        if op.op_type == "gelu" and op.act_scale_flag == True:
            input = op.get_input
            output = op.get_output
            act_scale = op.get_act_scale
            op.act_scale_flag = False
            new_out = Tensor(Shape(*output.shape), dtype=DataType.float16, name=f"""{output.name}_cut""")
            if act_scale.data is not None and input.data is not None:
                new_data = (input.data.astype("float16")*act_scale.data).astype("float16")
                new_out.set_data(new_data)
            op.clear_outputs()
            op.add_outputs((new_out,))
            new_op = Convert(f"""{op.name}_cvt""",DataType.int8)
            new_op.add_input_tensors((new_out,))
            new_op.act_scale_flag = True
            new_op.add_outputs((output,act_scale,))
            g.insert_after(op,new_op)
                    
