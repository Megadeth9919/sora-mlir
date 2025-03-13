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
    in_shape = (56,405, 1152)
    act_scale_shape = (56, 405, 1)
    weight_shape = (1152, 1152)

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

    dump_information(g, root_path, op_name="linearw8_56_405_1152", mode=mode,dump_onnx=False)


def mm_linearw8_bias(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()
    in_shape = (2, 11340, 1152)
    act_scale_shape = (2, 11340, 1)
    weight_shape = (1152, 1152)

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

    dump_information(g, root_path, op_name="linearw8_bias_2_11340_1152", mode=mode,dump_onnx=False)

def mm_linearw8_1(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()
    in_shape = (2, 11340, 4608)
    act_scale_shape = (2, 11340, 1)
    weight_shape = (1152, 4608)

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

    dump_information(g, root_path, op_name="linearw8_2_11340_4608", mode=mode,dump_onnx=False)
def mm_linearw8_2(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()
    in_shape = (810, 28, 1152)
    act_scale_shape = (810, 28, 1)
    weight_shape = (1152, 1152)

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

    dump_information(g, root_path, op_name="linearw8_810_28_1152", mode=mode,dump_onnx=False)
def mm_linearw8_3(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()
    in_shape = (1, 284, 1152)
    act_scale_shape = (1, 284, 1)
    weight_shape = (1152, 1152)

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

    dump_information(g, root_path, op_name="linearw8_1_284_1152", mode=mode,dump_onnx=False)
def mm_matmul_int(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(56, 16, 405, 72), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(56, 16, 405, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(56, 16, 405, 72), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(56, 16, 405, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int_56_16_405_72", mode=mode,dump_onnx=False)

def mm_matmul_int_1(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(56, 16, 405, 405), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(56, 16, 405, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(56, 16, 72, 405), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(56, 16, 72, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int_56_16_405_405", mode=mode,dump_onnx=False)
def mm_matmul_int_2(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(1, 16, 22680, 72), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(1, 16, 22680, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(1, 16, 284, 72), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(1, 16, 284, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int_1_16_22680_72", mode=mode,dump_onnx=False)
def mm_matmul_int_3(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(1, 16, 22680, 284), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(1, 16, 22680, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(1, 16, 284, 284), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(1, 16, 284, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int_1_16_22680_284", mode=mode,dump_onnx=False)
def mm_matmul_int_4(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(810, 16, 28, 28), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(810, 16, 28, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(810, 16, 72, 28), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(810, 16, 72, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int_810_16_28_28", mode=mode,dump_onnx=False)
def mm_matmul_int_5(root_path, mode, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(810, 16, 28, 72), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(810, 16, 28, 1), dtype=DataType.float32, name="input0_scale", const=True)
    B = Tensor(Shape(810, 16, 28, 72), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(810, 16, 28, 1), dtype=DataType.float32, name="input1_scale", const=True)
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int")

    model(A, B)

    p = PModel(graph=g)
    output_list = p.run()

    dump_information(g, root_path, op_name="matmul_int_810_16_28_72", mode=mode,dump_onnx=False)
functions = [
    mm_linearw8,  # 14
    mm_linearw8_bias,  # 15
    mm_matmul_int, # 20
    mm_linearw8_1,  # 14
    mm_linearw8_2,  # 14
    mm_linearw8_3,  # 14
    mm_matmul_int_1, # 20
    mm_matmul_int_2, # 20
    mm_matmul_int_3, # 20
    mm_matmul_int_4, # 20
    mm_matmul_int_5, # 20
]


def main(debug: bool = False):
    root_path = "./to_rt"
    mode = CompileFlags()
    mode.enable_slr_slice = True
    for idx in range(len(functions)):
        functions[idx](root_path,mode)


if __name__ == "__main__":
    main(True)
    # mode = CompileFlags()
    # mode.enable_slr_slice = True
    # mm_linearw8_bias("./to_rt", mode)