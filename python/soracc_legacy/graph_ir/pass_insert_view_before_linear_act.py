from .graph_ir import *
import math
def insert_view_before_linear_act(g: StaticGraph):
    """
    Fuse cvt + linear into cvt_linearw8
    """
    for op in g.get_ops()[:] :
        if op.op_type == "linearw8_act":
            input = op.get_feature
            output = op.get_output
            weight,scale = op.get_weight
            bias = op.get_bias
            act_scale = input.get_act_scale
            input.delete_user(op)
            act_scale.delete_user(op)
            
            N = math.prod(input.shape[:-1])
            new_input_shape = (N,input.shape[-1])
            new_input = Tensor(Shape(*new_input_shape), dtype=DataType.int8, name=f"""{op.name}_before_view""")
            if input.data is not None:
                new_input.set_data(input.get_data().reshape(*new_input_shape))
            new_act_view_op = View(f"""{op.name}_act_before_view""",new_input_shape)
            new_act_view_op.add_input_tensors((input,))
            new_act_view_op.add_outputs((new_input,))
            g.insert_before(op,new_act_view_op)
            
            new_sacle_shape = (N,1)
            new_act_scale = Tensor(Shape(*new_sacle_shape), dtype=DataType.float32, name=f"""{op.name}_before_view""")
            if act_scale.data is not None:
                new_act_scale.set_data(act_scale.get_data().reshape(*new_sacle_shape))
            new_act_scale_op = View(f"""{op.name}_act_scale_before_view""",new_sacle_shape)
            new_act_scale_op.add_input_tensors((act_scale,))
            new_act_scale_op.add_outputs((new_act_scale,))
            new_input.set_act_scale(new_act_scale)
            g.insert_before(op,new_act_scale_op)
            
            new_op = LinearW8Act(op.name,op.bias_flag,op.act_type)
            new_op.add_input_tensors((new_input,new_act_scale,))
            new_op.set_weight_scale(weight = weight, weight_scale = scale,bias = bias)
            N1 = math.prod(output.shape[:-1])
            new_ouput_shape = (N1,output.shape[-1])
            new_ouput = Tensor(Shape(*new_ouput_shape), dtype=DataType.float16, name=f"""{op.name}_after_view""")
            if output.data is not None:
                new_ouput.set_data(output.get_data().reshape(*new_ouput_shape))
            new_op.add_outputs((new_ouput,))
            g.insert_before(op,new_op)
            
            new_out_view_op = View(f"""{op.name}_out_after_view""",output.shape)
            new_out_view_op.add_input_tensors((new_ouput,))
            new_out_view_op.add_outputs((output,))
            g.insert_before(op,new_out_view_op)
            g.del_op(op)
            
                        
