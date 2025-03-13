import sys
from typing import Callable, List

import numpy as np
import pytest

from graph_ir import *
from inst import *
from model import model_base, sora_model
from p_model import PModel
from utils import (dump_information, gen_random_data)
import json

def misc_elementwise_add(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_add")

    input_x = Tensor(Shape(2, 11340, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(2, 11340, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_add"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode,dump_onnx=False)


def misc_elementwise_mul(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "mul", "elementwise_mul")

    input_x = Tensor(Shape(2, 11340, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(2, 1, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_mul_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_mul"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode,dump_onnx=False)


def misc_elementwise_broadcast0_add(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_broadcast0_add")

    input_x = Tensor(Shape(1,16, 22680, 284), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(1, 1,22680, 284), dtype=DataType.float16, name="input1", const=True)
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_broadcast0_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast0_add"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode,dump_onnx=False)

def misc_rmsnorm(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(56,16,405,72), dtype=DataType.float16, name="input")

    model = model_base.RMSNorm(g, weight_shape=Shape(1, 72), name="rmsnorm")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "rmsnorm_int8"
    else:
        output_dtype = DataType.float16
        op_name = "rmsnorm"

    model(input_x, out_dtype=output_dtype)

    gen_random_data([input_x])
    gen_random_data(g.get_weights())

    dump_information(g, root_path, op_name=op_name, mode=mode,dump_onnx=False)
def misc_rmsnorm_1(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(810,16,28,72), dtype=DataType.float16, name="input")

    model = model_base.RMSNorm(g, weight_shape=Shape(1, 72), name="rmsnorm")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "rmsnorm_1_int8"
    else:
        output_dtype = DataType.float16
        op_name = "rmsnorm_1"

    model(input_x, out_dtype=output_dtype)

    gen_random_data([input_x])
    gen_random_data(g.get_weights())

    dump_information(g, root_path, op_name=op_name, mode=mode,dump_onnx=False)


def misc_layernorm(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    model = model_base.LayerNorm(
        g, weight_shape=Shape(1, 1152), name="layernorm", affine=False
    )

    input_x = Tensor(Shape(2, 11340, 1152), dtype=DataType.float16, name="input")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "layernorm_int8"
    else:
        output_dtype = DataType.float16
        op_name = "layernorm"

    model(input_x, out_dtype=output_dtype)
    gen_random_data([input_x])
    gen_random_data(g.get_weights())

    dump_information(g, root_path, op_name=op_name, mode=mode,dump_onnx=False)


def misc_transpose01(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(56, 405, 16, 72), dtype=DataType.int8, name="input")
    gen_random_data([input_x])

    model = model_base.Transpose(g, 0, 1, name="transpose01")
    model(input_x)

    dump_information(g, root_path, op_name="transpose01", mode=mode,dump_onnx=False)

def misc_transpose12(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(56, 405, 16, 72), dtype=DataType.int8, name="input")
    gen_random_data([input_x])

    model = model_base.Transpose(g, 1, 2, name="transpose12")
    model(input_x)

    dump_information(g, root_path, op_name="transpose12", mode=mode,dump_onnx=False)

def misc_transpose23(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(1, 16, 284, 72), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Transpose(g, 2, 3, name="transpose23")
    model(input_x)

    dump_information(g, root_path, op_name="transpose23", mode=mode,dump_onnx=False)
    
def misc_transpose23_1(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(56, 16, 405, 72), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Transpose(g, 2, 3, name="transpose23_1")
    model(input_x)

    dump_information(g, root_path, op_name="transpose23_1", mode=mode,dump_onnx=False)


def misc_split(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(2, 6, 1152), dtype=DataType.float16, name="input",const=True)
    gen_random_data([input_x])

    model = model_base.Split(g, 6, 1, name="split")
    model(input_x)

    dump_information(g, root_path, op_name="split", mode=mode,dump_onnx=False)


def misc_softmax(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(810,16,28,28), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Softmax(g, name="softmax")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input_x, out_dtype=output_dtype)

    dump_information(g, root_path, op_name="softmax", mode=mode,dump_onnx=False)


def misc_softmax_1(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(56,16,405,405), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Softmax(g, name="softmax")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input_x, out_dtype=output_dtype)

    dump_information(g, root_path, op_name="softmax_1", mode=mode,dump_onnx=False)
def misc_softmax_2(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(1,16,22680,284), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Softmax(g, name="softmax")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input_x, out_dtype=output_dtype)

    dump_information(g, root_path, op_name="softmax_2", mode=mode,dump_onnx=False)
def misc_gelu(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(2, 11340, 4608), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Gelu(g, "gelu")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "gelu_int8"
    else:
        output_dtype = DataType.float16
        op_name = "gelu"

    model(input_x, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode,dump_onnx=False)


def misc_div(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    scale = Tensor(shape=Shape(56, 16, 405, 72), dtype=DataType.float16, name="input1")
    fake_data = np.ones(shape=(56, 16, 405, 72), dtype=np.float16)
    fake_data.fill(8.563)
    scale.set_data(fake_data)

    input_x = Tensor(Shape(56, 16, 405, 72), dtype=DataType.float16, name="input0")
    gen_random_data([input_x])

    model = model_base.Div(g, name="div")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input_x, divisor=scale, out_dtype=output_dtype)

    dump_information(g, root_path, op_name="div", mode=mode,dump_onnx=False)


def misc_convert(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(56, 16, 405, 72), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Convert(g, out_dtype=DataType.int8, name="convert")

    model(input_x)

    dump_information(g, root_path, op_name="convert", mode=mode,dump_onnx=False)


def misc_rope(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(810, 16, 28, 72), dtype=DataType.float16, name="input")

    dim = input_x.shape[-1] * input_x.shape[-2]
    model = model_base.RoPE(g, dim=dim, name="rope")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input_x, out_dtype=output_dtype)
    print(json.dumps(g.to_json(), indent=2))
    gen_random_data([input_x])
    for weight in g.get_weights():
        shape = weight.shape
        new_shape = Shape(shape[0], shape[1], shape[2], shape[3] // 2)
        fake_data = np.random.randint(low=-10, high=10, size=list(new_shape), dtype=np.int8)
        fake_data = fake_data.astype(np.float16)
        fake_data = np.repeat(fake_data, 2, axis=3)
        weight.set_data(fake_data)

    dump_information(g, root_path, op_name="rope", mode=mode,dump_onnx=False)

functions: List[Callable] = [
    misc_elementwise_add,   # 0
    misc_elementwise_mul,   # 1
    misc_elementwise_broadcast0_add,    # 2
    misc_rmsnorm,   # 6
    misc_layernorm, # 7
    misc_transpose01,   # 8   
    misc_transpose23,   # 9
    misc_split, # 10
    misc_softmax,   # 12
    misc_gelu,  # 13
    misc_convert,   # 16
    misc_div,   # 17
    misc_rope,  # 18
    misc_transpose12,   # 20
    misc_transpose23_1,   # 21
    misc_softmax_1,   # 22
    misc_softmax_2,   # 23
    misc_rmsnorm_1,   # 24
]


def main(debug: bool = False):
    root_path = "./case"
    mode = CompileFlags()
    mode.enable_slr_slice = True
    for idx in range(len(functions)):
        functions[idx](root_path,mode,dynamic_scale=False)


if __name__ == "__main__":
    main(False)
