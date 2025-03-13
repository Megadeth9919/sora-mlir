from .graph_ir import *
def fuse_div_cvt_matmul(g: StaticGraph):
    """
    Fuse div + cvt_matmul into softmax_cvt_matmul
    """
    for op in g.get_ops() :
        if op.op_type == "div" and len(op.get_users()) == 1:
            user = op.get_users()[0]
            if user.op_type == "matmul":
                # print(f"div users : {user.name} {user.name}") 
                name = f"""{op.name}_{user.name}_{user.name}"""
                type = op.get_output.data_type
                new_op = DivCvtMatmul(name,op.get_input_B, type)
                op.get_input.clear_users()
                user.get_matrix_B.clear_users()
                op.get_divisor.clear_users()
                user.get_act_scale_B.clear_users()
                new_op.add_input_tensors((op.get_input, user.get_matrix_B, op.get_divisor, user.get_act_scale_B,))
                assert(user.get_matrix_B.data_type == DataType.int8)
                new_op.add_outputs((user.get_output,))
                print("fuse div + cvt_matmul into div_cvt_matmul")
                g.insert_after(op,new_op)
                g.del_tensor(op.get_outputs()[0])
                g.del_op(op)
                g.del_op(user)
                        
