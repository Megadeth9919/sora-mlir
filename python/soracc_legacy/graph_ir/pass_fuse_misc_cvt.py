from .graph_ir import *

def isMiscOp(op:Op):
    if op.op_type == "add":
        return True
    if op.op_type == "sub":
        return True
    if op.op_type == "mul":
        return True
    if op.op_type == "div":
        return True
    if op.op_type == "silu":
        return True
    if op.op_type == "gelu":
        return True
    if op.op_type == "softmax":
        return True
    if op.op_type == "layer_norm":
        return True
    if op.op_type == "rms_norm":
        return True 
    return False
def fuse_misc_cvt(g: StaticGraph):
    """
    Fuse misc + cvt into misc_cvt
    """
    for op in g.get_ops() :
        if isMiscOp(op) and len(op.get_users()) == 1:
            user = op.get_users()[0]
            if user.op_type == "convert":
                op.act_scale_flag = True
                op.clear_outputs()
                op.set_outputs(user.get_outputs())
                print("fuse misc + cvt into misc_cvt")
                g.del_tensor(op.get_outputs()[0])
                g.del_op(user)
                        
