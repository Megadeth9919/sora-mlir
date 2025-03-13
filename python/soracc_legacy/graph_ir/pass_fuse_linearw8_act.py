from .graph_ir import *
import json
def fuse_linearw8_act(g: StaticGraph):
    """
    Fuse linear + act into linearw8_act
    """
    for op in g.get_ops() :
        if op.op_type == "linear_w8" and len(op.get_users()) == 1:
            user = op.get_users()[0]
            if user.op_type == "silu" or user.op_type == "gelu":
                new_op = LinearW8Act(f"""{op.name}_{user.name}""",op.bias_flag,user.op_type)
                op.get_feature.delete_user(op)
                op.get_act_scale.delete_user(op)
                new_op.add_input_tensors(op.get_inputs())
                weight,weight_scale = op.get_weight
                new_op.set_weight_scale(weight = weight , weight_scale = weight_scale,bias = op.get_bias)
                new_op.add_outputs(user.get_outputs())
                print("fuse linear + act into linearw8_act")
                g.insert_after(op,new_op)
                g.del_tensor(op.get_outputs()[0])
                g.del_op(user)
                g.del_op(op)


