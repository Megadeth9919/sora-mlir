from graph_ir import *
from inst import *
import numpy as np
import sys
import pytest
from p_model import PModel
from utils import dump_information, gen_random_data
from model import model_base, sora_model



def mm_linearw8(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()
    in_shape = (1, 64, 32)
    act_scale_shape = (1, 64, 1)
    weight_shape = (64, 32)
    weight_scale_shape = 64
    bias_shape = 64

    input_x = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="act_scale", const=True
    )

    model = model_base.Linear(
        g,
        in_feature=weight_shape[-1],
        out_feature=weight_shape[0],
        bias=False,
        name="linearw8",
    )

    model(input_x, out_dtype=DataType.float16, act_scale=act_scale)
    gen_random_data([input_x, act_scale])
    gen_random_data(g.get_weights())

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="linearw8", mode=mode)


def mm_linearw8_bias(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()
    in_shape = (1, 64, 32)
    act_scale_shape = (1, 64, 1)
    weight_shape = (64, 32)
    weight_scale_shape = 64
    bias_shape = 64

    input_x = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="act_scale", const=True
    )

    model = model_base.Linear(
        g,
        in_feature=weight_shape[-1],
        out_feature=weight_shape[0],
        bias=True,
        name="linearw8",
    )

    model(input_x, out_dtype=DataType.float16, act_scale=act_scale)
    gen_random_data([input_x, act_scale])
    gen_random_data(g.get_weights())

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="linearw8_bias", mode=mode)


def mm_matmul_int(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(1, 1, 17, 16), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(1, 1, 17, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(1, 1, 21, 16), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(1, 1, 21, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int", mode=mode)

def mm_matmul_int_long(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(1, 1, 96, 129), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(1, 1, 96, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(1, 1, 1124, 129), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(1, 1, 1124, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int_long")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int_long", mode=mode)

def mm_matmul_int_k_64(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(1, 1, 77, 64), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(1, 1, 77, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(1, 1, 60, 64), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(1, 1, 60, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int_k_64")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int_k_64", mode=mode)

def mm_matmul_int_k_1024_parallel_n(root_path, mode, dynamic_scale: bool = False):
    name = "matmul_int_k_4096_parallel_n"
    g = StaticGraph()

    A = Tensor(Shape(1, 1, 270, 1024), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(1, 1, 270, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(1, 1, 260, 1024), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(1, 1, 260, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name=name)

    model(A, B)
    g.get_ops()[0].opt_func = "parallel_n"

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name=name, mode=mode)

def mm_linearw8_bias_parallel_n(root_path, mode, dynamic_scale: bool = False):
    name = "mm_linearw8_bias_parallel_n"
    
    g = StaticGraph()
    in_shape = (1, 64, 128)
    act_scale_shape = (1, 64, 1)
    weight_shape = (260, 128)
    weight_scale_shape = 260
    bias_shape = 260

    input_x = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float32, name="act_scale", const=True
    )

    model = model_base.Linear(
        g,
        in_feature=weight_shape[-1],
        out_feature=weight_shape[0],
        bias=True,
        name=name,
    )

    model(input_x, out_dtype=DataType.float16, act_scale=act_scale)
    gen_random_data([input_x, act_scale])
    gen_random_data(g.get_weights())
    
    g.get_ops()[0].opt_func = "parallel_n"

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name=name, mode=mode)

functions = [
    mm_linearw8,  # 14
    mm_linearw8_bias,  # 15
    mm_matmul_int, # 20
    mm_matmul_int_long,
    mm_matmul_int_k_64,
    mm_matmul_int_k_1024_parallel_n,
    mm_linearw8_bias_parallel_n,
]


def main(debug: bool = False):
    root_path = "./to_rt"
    mode = CompileFlags()
    mode.enable_slr_slice = True
    for idx in range(len(functions)):
        functions[idx](root_path,mode)


if __name__ == "__main__":
    main(True)
    # mm_matmul_int_long("./to_rt", CompileFlags())