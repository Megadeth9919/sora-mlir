from .graph_ir import *

def fuse_linearw8_transpose(g: StaticGraph):
    """
    Fuse linear + transpose into linearw8_transpose
    """
    for op in g.get_ops() :
        if op.op_type == "linear_w8" and len(op.get_users()) == 1:
            user = op.get_users()[0]
            if user.op_type == "transpose" and user.dim_a == 2 and user.dim_b == 3:
                trans_out_shape = user.get_output.shape
                new_op = LinearW8Transpose(f"""{op.name}_{user.name}""",trans_out_shape[1],trans_out_shape[3],op.bias_flag)
                op.get_input.clear_users()
                op.get_act_scale.clear_users()
                new_op.add_input_tensors((op.get_input, op.get_act_scale))
                weight,weight_scale = user.get_weight
                new_op.set_weight_scale(weight = weight , weight_scale = weight_scale,bias = user.get_bias)
                new_op.add_outputs((user.get_output,))
                print("fuse linear + transpose into linearw8_transpose")
                g.insert_after(op,new_op)
                g.insert_after(op,new_op)
                g.del_tensor(op.get_outputs()[0])
                g.del_op(user)
                g.del_op(op)
                        
