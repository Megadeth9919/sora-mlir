from .graph_ir import *

def fuse_cvt_linearw8(g: StaticGraph):
    """
    Fuse cvt + linear into cvt_linearw8
    """
    for op in g.get_ops() :
        if op.op_type == "convert" and len(op.get_users()) == 1:
            user = op.get_users()[0]
            if user.op_type == "linear_w8":
                new_op = CvtLinearW8(f"""{op.name}_{user.name}""",user.bias_flag)
                op.get_input.delete_user(op)
                new_op.add_input_tensors((op.get_input,))
                weight,weight_scale = user.get_weight
                new_op.set_weight_scale(weight = weight , weight_scale = weight_scale,bias = user.get_bias)
                new_op.add_outputs((user.get_output,))
                print("fuse cvt + linear into cvt_linearw8")
                g.insert_after(op,new_op)
                g.del_tensor(op.get_outputs()[0].get_act_scale)
                g.del_tensor(op.get_outputs()[0])
                g.del_op(user)
                g.del_op(op)

                        
