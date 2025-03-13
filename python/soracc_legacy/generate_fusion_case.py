from graph_ir import LinearW8Transpose, CompileFlags, DivCvtMatmul, SoftmaxCvtMatmul, CvtLinearW8, TransposeCvt, LinearW8Act
from model import *
from inst import *
import numpy as np
import sys
import pytest
from p_model import PModel
from utils import dump_information, gen_random_data
from model import model_base, sora_model
from profiler import InstProfiler

def compare_profiler_result(ori_g, opt_g, mode):
    return
    # profiler
    _, ori_inst = graph_ir.pass_impl.graph_compile(ori_g, mode)
    ori_profiler = InstProfiler(ori_inst)  
    
    _, opt_inst = graph_ir.pass_impl.graph_compile(opt_g, mode)
    opt_profiler = InstProfiler(opt_inst)  
    print("=================== Profiler ===========================")
    print("==============Origin graph=====================")
    ori_profiler.run()
    print("==============Opt graph=====================")
    opt_profiler.run()

def fuse_linearw8_transpose(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_linearw8_transpose"
    ori_name = "origin_linearw8_transpose"
    g = StaticGraph()
    in_shape = (1, 64, 32)
    act_scale_shape = (1, 64, 1)
    weight_shape = (1152, 32)
    weight_scale_shape = 1152
    bias_shape = 1152
    head_num = 16
    dim = weight_shape[-2] // head_num
    assert head_num * dim == weight_shape[-2]

    input_x = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="act_scale", const=True
    )

    
    # origin op
    graph = StaticGraph()
    qkv_shape = (input_x.shape[0:-2] + (input_x.shape[-2], head_num, dim))
    linear_q = Linear(graph, in_feature=weight_shape[-1], out_feature=weight_shape[-2], bias=False, name=ori_name + 'linear_q')
    q = linear_q(input_x, act_scale=act_scale, out_dtype=DataType.float16)
    q = View(graph, Shape(*qkv_shape), name=ori_name + 'view')(q)
    q = Transpose(graph, dim_a=1, dim_b=2, name=ori_name + 'permute')(q)
    
    # Op definition
    op = LinearW8Transpose(name, head_num, dim)
    op.add_input_tensors((input_x, act_scale))
    # weight = Weight(name=f"{name}.weight",
    #                 shape=Shape(weight_shape[-2], weight_shape[-1]),
    #                 data_type=DataType.int8)
    # weight_scale = Weight(name=f"{name}.scale",
    #                         shape=Shape(weight_shape[-2],),
    #                         data_type=DataType.float16, 
    #                         const=True)
    linear_param = graph.get_ops()[0].get_weights()
    op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1])
    new_shape = input_x.shape[0:-2] + (head_num, input_x.shape[-2], dim)
    op.add_outputs((q,))
    g.add(op)
    
    
    # gen_random_data([input_x, act_scale])
    # gen_random_data(graph.get_weights())
    wgt_data = np.ones(linear_param[0].shape, dtype=np.int8)
    linear_param[0].set_data(wgt_data)
    wgt_scale_data = np.ones(linear_param[1].shape, dtype=np.float16)
    linear_param[1].set_data(wgt_scale_data)
    input_x_data = np.ones(input_x.shape, dtype=np.int8)
    input_x_data[:, 32:, :] = 0
    input_x.set_data(input_x_data)
    act_scale_data = np.ones(act_scale.shape, dtype=np.float16)
    act_scale.set_data(act_scale_data)
    # gen_random_data([ret])

    p = PModel(graph=graph)
    output_list = p.run()

    dump_information(graph, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    # profiler
    compare_profiler_result(graph, g, mode)

    
def fuse_div_cvt_matmul(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_div_cvt_matmul"
    ori_name = "origin_div_cvt_matmul"
    g = StaticGraph()
    # embeding = 1152
    # head_num = 16
    embeding = 576
    head_num = 2
    head_dim = embeding // head_num
    assert head_num * head_dim == embeding
    in_shape = (1, head_num, 64, head_dim)
    act_scale_shape = (1, head_num, 64, 1)

    in_q = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input_q")
    k = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input_k")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="k_act_scale", const=True
    )
    k.set_act_scale(act_scale)

    
    # origin op
    graph = StaticGraph()
    attn_scale_data = head_dim ** 0.5
    attn_scale = Tensor(shape=in_q.shape, dtype=DataType.float16, name=ori_name + "_attn.q_scale", const=True)
    scale_data = np.zeros(in_q.shape)
    scale_data.fill(attn_scale_data)
    attn_scale.set_data(scale_data.astype(np.float16))
    q = Div(graph, name=ori_name + "q_div")(in_q, attn_scale)
    q = Convert(graph, DataType.int8, name=ori_name + "q_convert_to_int8")(q)
    # qkt = Matmul(graph, name=ori_name + "qkt_mutual")(q, k)
    
    # # fuse op
    # divisor = Tensor(shape=in_q.shape, dtype=DataType.float16, name=ori_name + "_attn.q_scale", const=True)
    # divisor.set_data((scale_data).astype(np.float16))
    # op = DivCvtMatmul(name, divisor, DataType.int8)
    # op.add_input_tensors((in_q, k, divisor, act_scale)) # 顺序固定
    # op.add_outputs((qkt,))
    # g.add(op)
    
    
    gen_random_data([in_q, k, act_scale])
    gen_random_data(graph.get_weights())

    p = PModel(graph=graph)
    output_list = p.run()

    dump_information(graph, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    # dump_information(g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(graph, g, mode)

def fuse_div_cvt_matmul_shape_28(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_div_cvt_matmul_shape_28"
    ori_name = "origin_div_cvt_matmul_shape_28"
    g = StaticGraph()
    # embeding = 1152
    # head_num = 16
    batch = 2
    # embeding = 576
    head_num = 2
    seq_len = 28
    head_dim = 72
    #embeding // head_num
    # assert head_num * head_dim == embeding
    in_shape = (batch, head_num, seq_len, head_dim)
    act_scale_shape = (batch, head_num, seq_len, 1)

    in_q = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input_q")
    k = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input_k")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="k_act_scale", const=True
    )
    k.set_act_scale(act_scale)

    
    # origin op
    graph = StaticGraph()
    attn_scale_data = head_dim ** 0.5
    attn_scale = Tensor(shape=in_q.shape, dtype=DataType.float16, name=ori_name + "_attn.q_scale", const=True)
    scale_data = np.zeros(in_q.shape)
    scale_data.fill(attn_scale_data)
    attn_scale.set_data(scale_data.astype(np.float16))
    q = Div(graph, name=ori_name + "q_div")(in_q, attn_scale)
    q = Convert(graph, DataType.int8, name=ori_name + "q_convert_to_int8")(q)
    qkt = Matmul(graph, name=ori_name + "qkt_mutual")(q, k)
    
    # fuse op
    divisor = Tensor(shape=in_q.shape, dtype=DataType.float16, name=ori_name + "_attn.q_scale", const=True)
    divisor.set_data((scale_data).astype(np.float16))
    op = DivCvtMatmul(name, divisor, DataType.int8)
    op.add_input_tensors((in_q, k, divisor, act_scale)) # 顺序固定
    op.add_outputs((qkt,))
    g.add(op)
    
    
    gen_random_data([in_q, k, act_scale])
    gen_random_data(graph.get_weights())

    p = PModel(graph=graph)
    output_list = p.run()

    dump_information(graph, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(graph, g, mode)

def fuse_div_cvt_matmul_shape_405(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_div_cvt_matmul_shape_405"
    ori_name = "origin_div_cvt_matmul_shape_405"
    g = StaticGraph()
    # embeding = 1152
    # head_num = 16
    batch = 2
    # embeding = 576
    head_num = 2
    seq_len = 405
    head_dim = 72
    #embeding // head_num
    # assert head_num * head_dim == embeding
    in_shape = (batch, head_num, seq_len, head_dim)
    act_scale_shape = (batch, head_num, seq_len, 1)

    in_q = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input_q")
    k = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input_k")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="k_act_scale", const=True
    )
    k.set_act_scale(act_scale)

    
    # origin op
    graph = StaticGraph()
    
    attn_scale_data = head_dim ** 0.5
    attn_scale = Tensor(shape=in_q.shape, dtype=DataType.float16, name=ori_name + "_attn.q_scale", const=True)
    scale_data = np.zeros(in_q.shape)
    scale_data.fill(attn_scale_data)
    attn_scale.set_data(scale_data.astype(np.float16))
    q = Div(graph, name=ori_name + "q_div")(in_q, attn_scale)
    q = Convert(graph, DataType.int8, name=ori_name + "q_convert_to_int8")(q)
    qkt = Matmul(graph, name=ori_name + "qkt_mutual")(q, k)
    
    # fuse op
    
    divisor = Tensor(shape=in_q.shape, dtype=DataType.float16, name=ori_name + "_attn.q_scale", const=True)
    divisor.set_data((scale_data).astype(np.float16))
    op = DivCvtMatmul(name, divisor, DataType.int8)
    op.add_input_tensors((in_q, k, divisor, act_scale)) # 顺序固定
    op.add_outputs((qkt,))
    g.add(op)
    
    
    gen_random_data([in_q, k, act_scale])
    gen_random_data(graph.get_weights())

    p = PModel(graph=graph)
    output_list = p.run()

    dump_information(graph, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(graph, g, mode)

def fuse_div_cvt_matmul_shape_M(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_div_cvt_matmul_shape_M"
    ori_name = "origin_div_cvt_matmul_shape_M"
    g = StaticGraph()
    # embeding = 1152
    # head_num = 16
    batch = 1
    # embeding = 576
    head_num = 2
    seq_len_q = 4000
    seq_len = 28
    head_dim = 72
    #embeding // head_num
    # assert head_num * head_dim == embeding
    q_in_shape = (batch, head_num, seq_len_q, head_dim)
    in_shape = (batch, head_num, seq_len, head_dim)
    act_scale_shape = (batch, head_num, seq_len, 1)

    in_q = Tensor(Shape(*q_in_shape), dtype=DataType.float16, name="input_q")
    k = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input_k")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="k_act_scale", const=True
    )
    k.set_act_scale(act_scale)

    
    # origin op
    graph = StaticGraph()
    
    attn_scale_data = head_dim ** 0.5
    attn_scale = Tensor(shape=in_q.shape, dtype=DataType.float16, name=ori_name + "_attn.q_scale", const=True)
    scale_data = np.zeros(in_q.shape)
    scale_data.fill(attn_scale_data)
    attn_scale.set_data(scale_data.astype(np.float16))
    q = Div(graph, name=ori_name + "q_div")(in_q, attn_scale)
    q = Convert(graph, DataType.int8, name=ori_name + "q_convert_to_int8")(q)
    qkt = Matmul(graph, name=ori_name + "qkt_mutual")(q, k)
    
    # fuse op
    
    divisor = Tensor(shape=in_q.shape, dtype=DataType.float16, name=ori_name + "_attn.q_scale", const=True)
    divisor.set_data((scale_data).astype(np.float16))
    op = DivCvtMatmul(name, divisor, DataType.int8)
    op.add_input_tensors((in_q, k, divisor, act_scale)) # 顺序固定
    op.add_outputs((qkt,))
    g.add(op)
    
    
    gen_random_data([in_q, k, act_scale])
    gen_random_data(graph.get_weights())

    p = PModel(graph=graph)
    output_list = p.run()

    dump_information(graph, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(graph, g, mode)

def fuse_softmax_cvt_matmul(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_softmax_cvt_matmul"
    ori_name = "origin_softmax_cvt_matmul"
    
    # embeding = 1152
    # head_num = 16
    embeding = 576
    head_num = 2
    seq_len = 64
    head_dim = embeding // head_num
    assert head_num * head_dim == embeding
    in_shape = (1, head_num, seq_len, seq_len)
    v_in_shape = (1, head_num, head_dim, seq_len)
    act_scale_shape = (1, head_num, head_dim, 1)

    qkt = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input_qkt")
    v_trans = Tensor(Shape(*v_in_shape), dtype=DataType.int8, name="input_v")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="v_act_scale", const=True
    )
    v_trans.set_act_scale(act_scale)

    
    # origin op
    ori_g = StaticGraph()
    atten = Softmax(ori_g, dim=-1, name=ori_name + "softmax")(qkt)
    atten = Convert(ori_g, DataType.int8, name=ori_name + "atten_convert_to_int8")(atten)
    attn_out = Matmul(ori_g, name=ori_name + "attn_matmul")(atten, v_trans)

    # fuse op
    opt_g = StaticGraph()
    op = SoftmaxCvtMatmul(name, DataType.int8, -1)
    op.add_input_tensors((qkt, v_trans, act_scale, act_scale)) # 顺序固定
    op.add_outputs((attn_out,))
    opt_g.add(op)
    
    
    gen_random_data([qkt, v_trans, act_scale])

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)

def fuse_softmax_cvt_matmul_shape_28_28(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_softmax_cvt_matmul_shape_28_28"
    ori_name = "origin_softmax_cvt_matmul_shape_28_28"
    
    # embeding = 1152
    # head_num = 16
    batch = 2
    embeding = 576
    head_num = 16
    seq_len = 28
    head_dim = embeding // head_num
    assert head_num * head_dim == embeding
    in_shape = (batch, head_num, seq_len, seq_len)
    v_in_shape = (batch, head_num, head_dim, seq_len)
    act_scale_shape = (batch, head_num, head_dim, 1)

    qkt = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input_qkt")
    v_trans = Tensor(Shape(*v_in_shape), dtype=DataType.int8, name="input_v")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="v_act_scale", const=True
    )
    v_trans.set_act_scale(act_scale)

    
    # origin op
    ori_g = StaticGraph()
    
    atten = Softmax(ori_g, dim=-1, name=ori_name + "softmax")(qkt)
    atten = Convert(ori_g, DataType.int8, name=ori_name + "atten_convert_to_int8")(atten)
    attn_out = Matmul(ori_g, name=ori_name + "attn_matmul")(atten, v_trans)

    # fuse op
    opt_g = StaticGraph()
    
    op = SoftmaxCvtMatmul(name, DataType.int8, -1)
    op.add_input_tensors((qkt, v_trans, act_scale, act_scale)) # 顺序固定
    op.add_outputs((attn_out,))
    opt_g.add(op)
    
    
    gen_random_data([qkt, v_trans, act_scale])

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)

def fuse_softmax_cvt_matmul_shape_405_405(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_softmax_cvt_matmul_shape_405_405"
    ori_name = "origin_softmax_cvt_matmul_shape_405_405"
    
    # embeding = 1152
    # head_num = 16
    batch = 1
    embeding = 576
    head_num = 2
    seq_len = 405
    head_dim = embeding // head_num
    assert head_num * head_dim == embeding
    in_shape = (batch, head_num, seq_len, seq_len)
    v_in_shape = (batch, head_num, head_dim, seq_len)
    act_scale_shape = (batch, head_num, head_dim, 1)

    qkt = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input_qkt")
    v_trans = Tensor(Shape(*v_in_shape), dtype=DataType.int8, name="input_v")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="v_act_scale", const=True
    )
    v_trans.set_act_scale(act_scale)

    
    # origin op
    ori_g = StaticGraph()
    
    atten = Softmax(ori_g, dim=-1, name=ori_name + "softmax")(qkt)
    atten = Convert(ori_g, DataType.int8, name=ori_name + "atten_convert_to_int8")(atten)
    attn_out = Matmul(ori_g, name=ori_name + "attn_matmul")(atten, v_trans)

    # fuse op
    opt_g = StaticGraph()
    
    op = SoftmaxCvtMatmul(name, DataType.int8, -1)
    op.add_input_tensors((qkt, v_trans, act_scale, act_scale)) # 顺序固定
    op.add_outputs((attn_out,))
    opt_g.add(op)
    
    
    gen_random_data([qkt, v_trans, act_scale])

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)
    
def fuse_softmax_cvt_matmul_shape_M_K(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_softmax_cvt_matmul_shape_M_K"
    ori_name = "origin_softmax_cvt_matmul_shape_M_K"
    
    # embeding = 1152
    # head_num = 16
    batch = 1
    embeding = 64
    head_num = 2
    seq_len = 1500
    head_dim = embeding // head_num
    assert head_num * head_dim == embeding
    in_shape = (batch, head_num, seq_len, 284)
    v_in_shape = (batch, head_num, head_dim, 284)
    act_scale_shape = (batch, head_num, head_dim, 1)

    qkt = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input_qkt")
    v_trans = Tensor(Shape(*v_in_shape), dtype=DataType.int8, name="input_v")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="v_act_scale", const=True
    )
    v_trans.set_act_scale(act_scale)

    
    # origin op
    ori_g = StaticGraph()
    
    atten = Softmax(ori_g, dim=-1, name=ori_name + "softmax")(qkt)
    atten = Convert(ori_g, DataType.int8, name=ori_name + "atten_convert_to_int8")(atten)
    attn_out = Matmul(ori_g, name=ori_name + "attn_matmul")(atten, v_trans)

    # fuse op
    opt_g = StaticGraph()
    
    op = SoftmaxCvtMatmul(name, DataType.int8, -1)
    op.add_input_tensors((qkt, v_trans, act_scale, act_scale)) # 顺序固定
    op.add_outputs((attn_out,))
    opt_g.add(op)
    
    
    gen_random_data([qkt, v_trans, act_scale])

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)

def fuse_cvt_linear(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_cvt_linearw8"
    ori_name = "origin_cvt_linearw8"
    
    embeding = 64
    head_num = 2
    seq_len = 64
    in_shape = (1, head_num, seq_len, embeding)
    bias = True
    input = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input")

    
    # origin op
    ori_g = StaticGraph()
    atten = Convert(ori_g, DataType.int8, name=ori_name + "convert_to_int8")(input)
    attn_out = Linear(ori_g, in_feature=embeding, out_feature=seq_len, bias=bias, name=ori_name + "proj")(atten, out_dtype=DataType.float16,  act_scale=atten.get_act_scale)

    # fuse op
    opt_g = StaticGraph()
    op = CvtLinearW8(name, bias)
    op.add_input_tensors((input,)) # 顺序固定
    op.add_outputs((attn_out,))
    linear_param = ori_g.get_ops()[1].get_weights()
    if bias:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1], bias=linear_param[2])
    else:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1])
    opt_g.add(op)
    
    
    gen_random_data([input,])
    gen_random_data(ori_g.get_weights())
    

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)
 
def fuse_cvt_linear_shape_28(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_cvt_linearw8_shape_28"
    ori_name = "origin_cvt_linearw8_shape_28"
    
    batch = 3
    head_num = 2
    seq_len = 28
    embeding = 72
    in_shape = (batch, head_num, seq_len, embeding)
    bias = True
    input = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input")

    
    # origin op
    ori_g = StaticGraph()
    
    atten = Convert(ori_g, DataType.int8, name=ori_name + "convert_to_int8")(input)
    attn_out = Linear(ori_g, in_feature=embeding, out_feature=seq_len, bias=bias, name=ori_name + "proj")(atten, out_dtype=DataType.float16,  act_scale=atten.get_act_scale)

    # fuse op
    opt_g = StaticGraph()
    
    op = CvtLinearW8(name, bias)
    op.add_input_tensors((input,)) # 顺序固定
    op.add_outputs((attn_out,))
    linear_param = ori_g.get_ops()[1].get_weights()
    if bias:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1], bias=linear_param[2])
    else:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1])
    opt_g.add(op)
    
    
    gen_random_data([input,])
    gen_random_data(ori_g.get_weights())
    

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)

def fuse_cvt_linear_shape_3000(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_cvt_linearw8_shape_3000"
    ori_name = "origin_cvt_linearw8_shape_3000"
    
    batch = 1
    head_num = 2
    seq_len = 3000
    embeding = 72
    in_shape = (batch, head_num, seq_len, embeding)
    bias = True
    input = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input")

    
    # origin op
    ori_g = StaticGraph()
    
    atten = Convert(ori_g, DataType.int8, name=ori_name + "convert_to_int8")(input)
    attn_out = Linear(ori_g, in_feature=embeding, out_feature=embeding, bias=bias, name=ori_name + "proj")(atten, out_dtype=DataType.float16,  act_scale=atten.get_act_scale)

    # fuse op
    opt_g = StaticGraph()
    
    op = CvtLinearW8(name, bias)
    op.add_input_tensors((input,)) # 顺序固定
    op.add_outputs((attn_out,))
    linear_param = ori_g.get_ops()[1].get_weights()
    if bias:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1], bias=linear_param[2])
    else:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1])
    opt_g.add(op)
    
    
    gen_random_data([input,])
    gen_random_data(ori_g.get_weights())
    

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)
    
def fuse_cvt_linear_shape_405(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_cvt_linearw8_shape_405"
    ori_name = "origin_cvt_linearw8_shape_405"
    
    batch = 1
    head_num = 2
    seq_len = 405
    embeding = 405
    in_shape = (batch, head_num, seq_len, embeding)
    bias = True
    input = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input")

    
    # origin op
    ori_g = StaticGraph()
    
    atten = Convert(ori_g, DataType.int8, name=ori_name + "convert_to_int8")(input)
    attn_out = Linear(ori_g, in_feature=embeding, out_feature=seq_len, bias=bias, name=ori_name + "proj")(atten, out_dtype=DataType.float16,  act_scale=atten.get_act_scale)

    # fuse op
    opt_g = StaticGraph()
    
    op = CvtLinearW8(name, bias)
    op.add_input_tensors((input,)) # 顺序固定
    op.add_outputs((attn_out,))
    linear_param = ori_g.get_ops()[1].get_weights()
    if bias:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1], bias=linear_param[2])
    else:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1])
    opt_g.add(op)
    
    
    gen_random_data([input,])
    gen_random_data(ori_g.get_weights())
    

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)

def fuse_transpose23_cvt(root_path, mode, dynamic_scale: bool = False):
    name = "fuse_transpose23_cvt"
    ori_name = "origin_transpose23_cvt"
    
    embeding = 135
    head_num = 16
    seq_len = 63
    in_shape = (1, head_num, seq_len, embeding)
    
    input = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input")

    
    # origin op
    ori_g = StaticGraph()
    trans = Transpose(ori_g, dim_a=3, dim_b=2, name=ori_name + "v_trans")(input)
    cvt_out = Convert(ori_g, DataType.int8, name=ori_name + "convert_to_int8")(trans)

    # fuse op
    opt_g = StaticGraph()
    op = TransposeCvt(name, dim_a=2, dim_b=3, out_type=DataType.int8)
    op.add_input_tensors((input,)) # 顺序固定
    op.add_outputs((cvt_out, cvt_out.get_act_scale))
    opt_g.add(op)
    
    
    gen_random_data([input,])

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    compare_profiler_result(ori_g, opt_g, mode)

def fuse_linearw8_act(root_path, mode, linear_bias=True, dynamic_scale: bool = False):
    name = "fuse_linearw8_act"
    ori_name = "origin_linearw8_act"
    
    embeding = 1152
    seq_len = 64
    in_shape = (1, seq_len, embeding)
    act_scale_shape = (1, seq_len, 1)
    
    input = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(Shape(*act_scale_shape), dtype=DataType.float32, const=True)
    input.set_act_scale(act_scale)
    
    # origin op
    ori_g = StaticGraph()
    linear = Linear(ori_g, name=ori_name + "fc", in_feature=embeding, out_feature=seq_len, bias=linear_bias)
    gelu = Gelu(ori_g, name=ori_name + "gelu")
    linear_out = linear(input, act_scale=act_scale, out_dtype=DataType.float16)
    act_out = gelu(linear_out, out_dtype=(DataType.float16))
    # linear_out.force_output = True

    # fuse op
    opt_g = StaticGraph()
    geluOp = ori_g.get_ops()[1]
    op = LinearW8Act(name, linear_bias, geluOp.op_type)
    op.add_input_tensors((input, act_scale))
    linear_param = ori_g.get_ops()[0].get_weights()
    if linear_bias:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1], bias=linear_param[2])
    else:
        op.set_weight_scale(weight=linear_param[0], weight_scale=linear_param[1])
        
    op.set_outputs((act_out, ))
    opt_g.add(op)
    
    
    gen_random_data([input, act_scale])
    gen_random_data(ori_g.get_weights())
    
    p = PModel(graph=ori_g)
    output_list = p.run()
    
    dump_information(ori_g, root_path, op_name=ori_name, mode=mode, call_pmode=False)
    dump_information(opt_g, root_path, op_name=name, mode=mode, call_pmode=False)
    
    # profiler
    compare_profiler_result(ori_g, opt_g, mode)



def test_op(root_path, mode, dynamic_scale: bool = False):
    name = "test_op"
    
    embeding = 33
    head_num = 3
    seq_len = 16
    in_shape = (1, head_num, seq_len, embeding)
    
    input = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input")
    input1 = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input1")
    input2 = Tensor(Shape(*in_shape), dtype=DataType.float16, name="input2")

    
    # origin op
    ori_g = StaticGraph()
    c = Eltwise(ori_g, t= "add" , name=name + "add")(input,input1)
    d = Eltwise(ori_g, t= "add" , name=name + "add1")(c,input2)
    
    
    gen_random_data([input,input1,input2])

    p = PModel(graph=ori_g)
    output_list = p.run()

    dump_information(ori_g, root_path, op_name=name, mode=mode, call_pmode=False)



functions = [
    # fuse_linearw8_transpose, 
    # fuse_softmax_cvt_matmul,
    # fuse_cvt_linear,
    fuse_div_cvt_matmul, 
    # fuse_div_cvt_matmul_shape_405,
    # fuse_div_cvt_matmul_shape_28,
    # fuse_div_cvt_matmul_shape_M,
    # fuse_softmax_cvt_matmul,
    # fuse_softmax_cvt_matmul_shape_M_K,
    # fuse_softmax_cvt_matmul_shape_28_28, 
    # fuse_softmax_cvt_matmul_shape_405_405, 
    # fuse_cvt_linear,
    # fuse_cvt_linear_shape_28, 
    # fuse_cvt_linear_shape_405,
    # fuse_cvt_linear_shape_3000, 
    # fuse_transpose23_cvt,
    # fuse_linearw8_act,
    # test_op
]


def main(debug: bool = False):
    root_path = "./to_rt"
    mode = CompileFlags()
    # mode.enable_slr_slice = True
    for idx in range(len(functions)):
        functions[idx](root_path,mode)


if __name__ == "__main__":
    main(True)