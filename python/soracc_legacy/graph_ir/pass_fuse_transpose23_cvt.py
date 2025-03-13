from .graph_ir import *

def fuse_transpose23_cvt(g: StaticGraph):
    """
    Fuse transpose23 + cvt into TransposeCvt
    """
    for op in g.get_ops() :
        if op.op_type == "transpose" and ((op.dim_a == 3 and op.dim_b == 2) or (op.dim_a == 2 and op.dim_b == 3)  ) and len(op.get_users()) == 1:
            user = op.get_users()[0]
            if user.op_type == "convert" :
                new_op = TransposeCvt(f"""{op.name}_{user.name}""",2, 3 ,user.op_type)
                op.get_inputs()[0].clear_users()
                new_op.add_input_tensors(op.get_inputs())
                new_op.add_outputs(user.get_outputs())
                print("fuse transpose23 + cvt into TransposeCvt")
                g.insert_after(op,new_op)
                g.del_tensor(op.get_outputs()[0])
                g.del_op(user)
                g.del_op(op)
                        
