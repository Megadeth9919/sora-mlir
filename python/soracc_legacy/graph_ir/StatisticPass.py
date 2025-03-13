from .graph_ir import *
import json
import yaml
def Statistic_Pass(g: StaticGraph):
    convert = 0
    linear = 0
    matmul = 0
    layernorm = 0
    rmsnorm = 0
    eltwise = 0
    div = 0
    silu = 0
    gelu = 0
    softmax = 0
    transpose = 0
    split = 0
    rope = 0
    cvt_linearW8 = 0
    linearW8_act = 0
    transpose_cvt = 0
    softmax_cvt_matmul = 0
    div_cvt_matmul = 0
    linearW8_transpose = 0
    op_info: dict[str, list[dict]] = {}

    for op in g.get_ops():
        if isinstance(op, LinearW8):
            shape_info = op.show_input_shape("Linear")
            add_shape_info(op_info, shape_info, "Linear")
            linear += 1
        elif isinstance(op, Matmul):
            shape_info = op.show_input_shape("Matmul")
            add_shape_info(op_info, shape_info, "Matmul")
            matmul += 1
        elif isinstance(op, Layernorm):
            shape_info = op.show_input_shape("Layernorm")
            add_shape_info(op_info, shape_info, "Layernorm")
            layernorm += 1 
        elif isinstance(op, RMSnorm):
            shape_info = op.show_input_shape("RMSnorm")
            add_shape_info(op_info, shape_info, "RMSnorm")
            rmsnorm += 1
        elif isinstance(op, Eltwise):
            shape_info = op.show_input_shape("Eltwise")
            add_shape_info(op_info, shape_info, "Eltwise")
            eltwise += 1
        elif isinstance(op, Div):
            shape_info = op.show_input_shape("Div")
            add_shape_info(op_info, shape_info, "Div")
            div += 1
        elif isinstance(op, Silu):
            shape_info = op.show_input_shape("Silu")
            add_shape_info(op_info, shape_info, "Silu")
            silu += 1
        elif isinstance(op, Gelu):
            shape_info = op.show_input_shape("Gelu")
            add_shape_info(op_info, shape_info, "Gelu")
            gelu += 1
        elif isinstance(op, Softmax):
            shape_info = op.show_input_shape("Softmax")
            add_shape_info(op_info, shape_info, "Softmax")
            softmax += 1
        elif isinstance(op, Transpose):
            shape_info = op.show_input_shape("Transpose")
            add_shape_info(op_info, shape_info, "Transpose")
            transpose += 1
        elif isinstance(op, View):
            pass
        elif isinstance(op, Split):
            shape_info = op.show_input_shape("Split")
            add_shape_info(op_info, shape_info, "Split")
            split += 1
        elif isinstance(op, RoPE):
            shape_info = op.show_input_shape("RoPE")
            add_shape_info(op_info, shape_info, "RoPE")
            rope += 1
        elif isinstance(op, Convert):
            shape_info = op.show_input_shape("Convert")
            add_shape_info(op_info, shape_info, "Convert")
            convert += 1
        elif isinstance(op, CvtLinearW8):
            shape_info = op.show_input_shape("CvtLinearW8")
            add_shape_info(op_info, shape_info, "CvtLinearW8")
            cvt_linearW8 += 1
        elif isinstance(op, LinearW8Act):
            shape_info = op.show_input_shape("LinearW8Act")
            add_shape_info(op_info, shape_info, "LinearW8Act")
            linearW8_act += 1
        elif isinstance(op, TransposeCvt):
            shape_info = op.show_input_shape("TransposeCvt")
            add_shape_info(op_info, shape_info, "TransposeCvt")
            transpose_cvt += 1
        elif isinstance(op, SoftmaxCvtMatmul):
            shape_info = op.show_input_shape("SoftmaxCvtMatmul")
            add_shape_info(op_info, shape_info, "SoftmaxCvtMatmul")
            softmax_cvt_matmul += 1
        elif isinstance(op, DivCvtMatmul):
            shape_info = op.show_input_shape("DivCvtMatmul")
            add_shape_info(op_info, shape_info, "DivCvtMatmul")
            div_cvt_matmul += 1
        elif isinstance(op, LinearW8Transpose):
            shape_info = op.show_input_shape("LinearW8Transpose")
            add_shape_info(op_info, shape_info, "LinearW8Transpose")
            linearW8_transpose += 1
        elif isinstance(op,LoadInst):
            pass
        else:
            raise NotImplementedError(f"not implemented op: {op}")

    for key in op_info.keys():
        tmp = list({json.dumps(i, sort_keys=True) for i in op_info[key]})
        op_info[key] = [json.loads(i) for i in tmp]

    print("linear:",linear)
    print("matmul:",matmul)
    print("layernorm:",layernorm)
    print("rmsnorm:",rmsnorm)
    print("eltwise:",eltwise)
    print("div:",div)
    print("silu:",silu)
    print("gelu:",gelu)
    print("softmax:",softmax)
    print("transpose:",transpose)
    print("split:",split)
    print("rope:",rope)
    print("convert:",convert)
    print("cvt_linearW8:",cvt_linearW8)
    print("linearW8_act:",linearW8_act)
    print("transpose_cvt:",transpose_cvt)
    print("softmax_cvt_matmul:",softmax_cvt_matmul)
    print("div_cvt_matmul:",div_cvt_matmul)
    print("linearW8_transpose:",linearW8_transpose)
    print("total:",linear+matmul+layernorm+rmsnorm+eltwise+div+silu+gelu+softmax+transpose+split+rope+convert+cvt_linearW8+linearW8_act+transpose_cvt+softmax_cvt_matmul+div_cvt_matmul+linearW8_transpose)
    write_op_info(op_info)

def add_shape_info(op_info: dict[str, list[dict]], shape_info: tuple, name: str):
    info = {}
    if name == "Transpose":
        info["input_shape"] = shape_info[0]
        info["output_shape"] = shape_info[2]
    if name == "Eltwise":
        info["input_shape"] = shape_info[0]
        info["type"] = shape_info[2]
    else:
        if len(shape_info[0]):
            info["input_shape"] = shape_info[0]
        if len(shape_info[1]):
            info["weight_shape"] = shape_info[1]
    if name in op_info:
        op_info[name].append(info)
    else:
        op_info[name] = [info]

def write_op_info(op_info: dict[str, list[dict]]):
    with open("case/op_info.yaml", "w") as f:
        yaml.dump(op_info, f, default_flow_style=None)