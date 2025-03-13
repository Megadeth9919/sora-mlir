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

    input_x = Tensor(Shape(16, 63, 64), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(16, 63, 64), dtype=DataType.float16, name="input1")
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_add"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_mul(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "mul", "elementwise_mul")

    input_x = Tensor(Shape(16, 33, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(16, 33, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_mul_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_mul"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_broadcast0_add(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_broadcast0_add")

    input_x = Tensor(Shape(16, 6, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(1, 6, 1152), dtype=DataType.float16, name="input1", const=True)
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_broadcast0_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast0_add"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_broadcast1_add(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_broadcast1_add")

    input_x = Tensor(Shape(16, 63, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(16, 1, 1152), dtype=DataType.float16, name="input1", const=True)
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_broadcast1_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast1_add"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_broadcast1_mul(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "mul", "elementwise_broadcast1_mul")

    input_x = Tensor(Shape(16, 63, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(16, 1, 1152), dtype=DataType.float16, name="input1", const=True)
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_broadcast1_mul_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast1_mul"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_silu(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    model = model_base.Silu(g, "silu")

    input_x = Tensor(Shape(16, 2, 16), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "silu_int8"
    else:
        output_dtype = DataType.float16
        op_name = "silu"

    model(input_x, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_rmsnorm(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(17,8,63,72), dtype=DataType.float16, name="input")

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

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_layernorm(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    model = model_base.LayerNorm(
        g, weight_shape=Shape(1, 1152), name="layernorm", affine=False
    )

    input_x = Tensor(Shape(16, 63, 1152), dtype=DataType.float16, name="input")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "layernorm_int8"
    else:
        output_dtype = DataType.float16
        op_name = "layernorm"

    model(input_x, out_dtype=output_dtype)
    gen_random_data([input_x])
    gen_random_data(g.get_weights())

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_transpose01(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(63, 129, 2, 16), dtype=DataType.int8, name="input")
    gen_random_data([input_x])

    model = model_base.Transpose(g, 0, 1, name="transpose01")
    model(input_x)

    dump_information(g, root_path, op_name="transpose01", mode=mode)

def misc_transpose12(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(17, 33, 16, 72), dtype=DataType.int8, name="input")
    gen_random_data([input_x])

    model = model_base.Transpose(g, 1, 2, name="transpose12")
    model(input_x)

    dump_information(g, root_path, op_name="transpose12", mode=mode)

def misc_transpose23(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(1, 16, 284, 72), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Transpose(g, 2, 3, name="transpose23")
    model(input_x)

    dump_information(g, root_path, op_name="transpose23", mode=mode)


def misc_split(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(3, 6, 1152), dtype=DataType.float16, name="input",const=True)
    gen_random_data([input_x])

    model = model_base.Split(g, 6, 1, name="split")
    model(input_x)

    dump_information(g, root_path, op_name="split", mode=mode)


def misc_vector_add(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = np.arange(4800).astype(np.float16).reshape(16, 3, 2, 50)
    input_x_flip = np.flip(input_x, axis=3)

    input0 = Tensor(Shape(16, 3, 2, 50), dtype=DataType.float16, name="input0")
    input1 = Tensor(Shape(16, 3, 2, 50), dtype=DataType.float16, name="input1")
    input0.set_data(input_x)
    input1.set_data(input_x_flip)

    model = model_base.Eltwise(g, "add", "vectorAdd")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input0, input1, out_dtype=output_dtype)

    dump_information(g, root_path, op_name="vectorAdd", mode=mode)


def misc_softmax(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(33,8,28,28), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Softmax(g, name="softmax")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input_x, out_dtype=output_dtype)

    dump_information(g, root_path, op_name="softmax", mode=mode)


def misc_gelu(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(16, 63, 4608), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Gelu(g, "gelu")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "gelu_int8"
    else:
        output_dtype = DataType.float16
        op_name = "gelu"

    model(input_x, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def mm_linearw8(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    in_shape = (1, 284, 1152)
    act_scale_shape = (1, 284, 1)
    weight_shape = (1152, 1152)
    weight_scale_shape = 1152
    bias_shape = 1152

    input_x = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float16, name="input_scale"
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

    dump_information(g, root_path, op_name="linearw8", mode=mode)


def mm_linearw8_bias(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    in_shape = (1, 284, 1152)
    act_scale_shape = (1, 284, 1)
    weight_shape = (1152, 1152)
    weight_scale_shape = 1152
    bias_shape = 1152

    input_x = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float16, name="input_scale"
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

    dump_information(g, root_path, op_name="linearw8_bias", mode=mode)


def misc_div(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    scale = Tensor(shape=Shape(33, 8, 129, 72), dtype=DataType.float16, name="input1")
    fake_data = np.ones(shape=(33, 8, 129, 72), dtype=np.float16)
    fake_data.fill(8.563)
    scale.set_data(fake_data)

    input_x = Tensor(Shape(33, 8, 129, 72), dtype=DataType.float16, name="input0")
    gen_random_data([input_x])

    model = model_base.Div(g, name="div")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input_x, divisor=scale, out_dtype=output_dtype)

    dump_information(g, root_path, op_name="div", mode=mode)


def misc_convert(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(33, 8, 28, 72), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Convert(g, out_dtype=DataType.int8, name="convert")

    model(input_x)

    dump_information(g, root_path, op_name="convert", mode=mode)


def misc_rope(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(3, 8, 28, 72), dtype=DataType.float16, name="input")

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

    dump_information(g, root_path, op_name="rope", mode=mode)

def mm_matmul_int(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    A = Tensor(Shape(56, 16, 405, 72), dtype=DataType.int8, name="input0")
    A_act_scale = Tensor(Shape(56, 16, 405, 1), dtype=DataType.float16, name="input0_scale")
    B = Tensor(Shape(56, 16, 405, 72), dtype=DataType.int8, name="input1")
    B_act_scale = Tensor(Shape(56, 16, 405, 1), dtype=DataType.float16, name="input1_scale")
    A.set_act_scale(A_act_scale)
    B.set_act_scale(B_act_scale)
    gen_random_data([A, A_act_scale, B, B_act_scale])

    model = model_base.Matmul(g, name="matmul_int")

    model(A, B)

    dump_information(g, root_path, op_name="matmul_int", mode=mode)

def misc_elementwise_broadcast2_add(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_broadcast2_add")

    input_x = Tensor(Shape(16, 6, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(1, 6, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8 
        op_name = "elementwise_broadcast2_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast2_add"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_broadcast3_add(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_broadcast3_add")

    input_x = Tensor(Shape(16, 129, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(16, 1, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_broadcast3_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast3_add"

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)
def misc_split1(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(3, 6, 1152), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Split(g, 6, 1, name="split1")
    model(input_x)

    dump_information(g, root_path, op_name="split1", mode=mode)
functions: List[Callable] = [
    misc_elementwise_add,   # 0
    misc_elementwise_mul,   # 1
    misc_elementwise_broadcast0_add,    # 2
    misc_elementwise_broadcast1_add,    # 3
    misc_elementwise_broadcast1_mul,    # 4
    misc_silu,  # 5
    misc_rmsnorm,   # 6
    misc_layernorm, # 7
    misc_transpose01,   # 8   
    misc_transpose23,   # 9
    misc_split, # 10
    misc_vector_add,    # 11
    misc_softmax,   # 12
    misc_gelu,  # 13
    mm_linearw8,    # 14
    mm_linearw8_bias,   # 15
    misc_convert,   # 16
    misc_div,   # 17
    misc_rope,  # 18
    mm_matmul_int,  # 19
    misc_transpose12,   # 20
    misc_elementwise_broadcast2_add, #21
    misc_elementwise_broadcast3_add, #22
    misc_split1, #23
]


def main(debug: bool = False):
    if not debug:
        if len(sys.argv) < 2:
            print("Usage: python test_misc_lower.py <path> <indices...>")
            sys.exit(1)

        # Get the path and indices
        root_path = sys.argv[1]
        dynamic = sys.argv[2]
        indices = sys.argv[3:]
        dynamic_scale: bool = False
        if dynamic == "True":
            dynamic_scale = True
        elif dynamic == "False":
            dynamic_scale = False
        else:
            print("Dynamic scale must be either True or False")
            sys.exit(1)
        mode = CompileFlags()  # mode = 1 for separate dump file
        mode.target = CCTarget.verify
        # mode.enable_slr_slice = True
        for index in indices:
            func_index = int(index) - 1
            print(f"Running case: {functions[func_index].__name__}")
            if 0 <= func_index < len(functions):
                functions[func_index](root_path, mode=mode, dynamic_scale=dynamic_scale)
            else:
                print(f"Invalid index: {index}. Skipping.")
    else:
        idx = 18  # case
        root_path = "./cases"
        mode = CompileFlags()
        mode.target = CCTarget.verify
        functions[idx](root_path, mode)


if __name__ == "__main__":
    main(False)
