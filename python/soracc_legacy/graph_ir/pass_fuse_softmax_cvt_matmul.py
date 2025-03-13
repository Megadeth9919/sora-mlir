from .graph_ir import *
def fuse_softmax_cvt_matmul(g: StaticGraph):
    """
    Fuse softmax + cvt_matmul into softmax_cvt_matmul
    """
    for op in g.get_ops() :
        if op.op_type == "softmax" and len(op.get_users()) == 1:
            user = op.get_users()[0]
            if user.op_type == "matmul":
                # print(f"softmax users : {user.name} {user.name}") 
                name = f"""{op.name}_{user.name}"""
                type = op.get_output.data_type
                new_op = SoftmaxCvtMatmul(name, type , op.dim)
                op.get_input.clear_users()
                user.get_matrix_B.clear_users()
                user.get_act_scale_B.clear_users()
                user.get_act_scale_A.clear_users()
                new_op.add_input_tensors((op.get_input, user.get_matrix_B, user.get_act_scale_A, user.get_act_scale_B))
                new_op.add_outputs((user.get_output,))
                print("fuse softmax + cvt_matmul into softmax_cvt_matmul")
                g.insert_after(op,new_op)
                g.del_tensor(op.get_outputs()[0])
                g.del_op(op)
                g.del_op(user)
                    
