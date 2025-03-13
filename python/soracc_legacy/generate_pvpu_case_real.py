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

def misc_elementwise_add(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_add")

    input_x = Tensor(shape, dtype=DataType.float16, name="input0")
    input_y = Tensor(shape, dtype=DataType.float16, name="input1")
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_add"
    op_name += '_shape'
    for dim in shape:
        op_name += f'_{dim}'
    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_mul(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    assert 0

    model = model_base.Eltwise(g, "mul", "elementwise_mul")

    input_x = Tensor(Shape(1, 588, 1152), dtype=DataType.float16, name="input0")
    input_y = Tensor(Shape(1, 588, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_mul_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_mul"
    op_name += '_shape'
    for dim in shape:
        op_name += f'_{dim}'

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_broadcast0_add(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape = [Shape()]
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_broadcast0_add")

    input_x = Tensor(shape[0], dtype=DataType.float16, name="input0")
    input_y = Tensor(shape[1], dtype=DataType.float16, name="input1", const=True)
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_broadcast0_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast0_add"
    
    op_name += '_shape'
    for dim in shape[0]:
        op_name += f'_{dim}'

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_broadcast1_add(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape = [Shape()]
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "add", "elementwise_broadcast1_add")

    input_x = Tensor(shape[0], dtype=DataType.float16, name="input0")
    input_y = Tensor(shape[1], dtype=DataType.float16, name="input1", const=True)
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_broadcast1_add_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast1_add"
        
    op_name += '_shape'
    for dim in shape[0]:
        op_name += f'_{dim}'

    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_elementwise_broadcast1_mul(
    root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape = [Shape()]
):
    g = StaticGraph()
    

    model = model_base.Eltwise(g, "mul", "elementwise_broadcast1_mul")

    input_x = Tensor(shape[0], dtype=DataType.float16, name="input0")
    input_y = Tensor(shape[1], dtype=DataType.float16, name="input1", const=True)
    gen_random_data([input_x, input_y])

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "elementwise_broadcast1_mul_int8"
    else:
        output_dtype = DataType.float16
        op_name = "elementwise_broadcast1_mul"

    op_name += '_shape'
    for dim in shape[0]:
        op_name += f'_{dim}'
    model(input_x, input_y, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_silu(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    assert 0

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


def misc_rmsnorm(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    
# (56, 16, 405, 72) -> 384
# (810, 16, 28, 72) -> 384
    input_x = Tensor(shape, dtype=DataType.float16, name="input")

    model = model_base.RMSNorm(g, weight_shape=Shape(1, shape[-1]), name="rmsnorm")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "rmsnorm_int8"
    else:
        output_dtype = DataType.float16
        op_name = "rmsnorm"

    model(input_x, out_dtype=output_dtype)

    gen_random_data([input_x])
    gen_random_data(g.get_weights())
    op_name += '_shape'
    for dim in shape:
        op_name += f'_{dim}'

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_layernorm(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    
    model = model_base.LayerNorm(
        g, weight_shape=Shape(1, shape[-1]), name="layernorm", affine=False
    )

# (2, 11340, 1152)
    input_x = Tensor(shape, dtype=DataType.float16, name="input")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "layernorm_int8"
    else:
        output_dtype = DataType.float16
        op_name = "layernorm"

    model(input_x, out_dtype=output_dtype)
    gen_random_data([input_x])
    gen_random_data(g.get_weights())
    op_name += '_shape'
    for dim in shape:
        op_name += f'_{dim}'

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


def misc_softmax(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    

    input_x = Tensor(shape, dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Softmax(g, name="softmax")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "softmax_int8"
    else:
        output_dtype = DataType.float16
        op_name = "softmax"
    op_name += '_shape'
    for dim in shape:
        op_name += f'_{dim}'

    model(input_x, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_gelu(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    

    input_x = Tensor(shape, dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Gelu(g, "gelu")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name = "gelu_int8"
    else:
        output_dtype = DataType.float16
        op_name = "gelu"
    op_name += '_shape'
    for dim in shape:
        op_name += f'_{dim}'

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


def misc_div(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    

    scale = Tensor(shape, dtype=DataType.float16, name="input1")
    fake_data = np.ones(shape=shape, dtype=np.float16)
    fake_data.fill(8.563)
    scale.set_data(fake_data)

    input_x = Tensor(shape=shape, dtype=DataType.float16, name="input0")
    gen_random_data([input_x])

    model = model_base.Div(g, name="div")

    if dynamic_scale:
        output_dtype = DataType.int8
        op_name="div_int8"
    else:
        output_dtype = DataType.float16
        op_name="div"
    op_name += '_shape'
    for dim in shape:
        op_name += f'_{dim}'

    model(input_x, divisor=scale, out_dtype=output_dtype)

    dump_information(g, root_path, op_name=op_name, mode=mode)


def misc_convert(root_path: str, mode: CompileFlags, dynamic_scale: bool = False, shape: Shape =()):
    g = StaticGraph()
    

    input_x = Tensor(shape, dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Convert(g, out_dtype=DataType.int8, name="convert")

    model(input_x)
    op_name="convert"
    op_name += '_shape'
    for dim in shape:
        op_name += f'_{dim}'

    dump_information(g, root_path, op_name, mode=mode)


def misc_rope(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(33, 8, 28, 72), dtype=DataType.float16, name="input")

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
        # 1 3 4 5 7 8 13 14 17 18
        for index in indices:
            func_index = int(index) - 1
            if func_index == 0: # misc_elementwise_add
                shape_list = [
                    # real
                    # Shape(2, 1, 1152),
                    # Shape(2, 11340, 1152),
                    # simplifield
                    Shape(1, 588, 1152),
                ]
            elif func_index == 2: # misc_elementwise_broadcast0_add
                shape_list = [
                    [Shape(2, 6, 1152), Shape(1, 6, 1152)],
                ]
            elif func_index == 3: # misc_elementwise_broadcast1_add
                shape_list = [
                    # real
                    # [Shape(2, 11340, 1152), Shape(2, 1, 1152)],
                    # [Shape(1, 16, 22680, 284), Shape(1, 1, 22680, 284)],
                    # simplifield
                    [Shape(2, 588, 1152), Shape(2, 1, 1152)],
                    [Shape(1, 2, 1176, 284), Shape(1, 1, 1176, 284)],
                ]
            elif func_index == 4: # misc_elementwise_broadcast1_mul
                shape_list = [
                    # real
                    # [Shape(2, 11340, 1152), Shape(2, 1, 1152)],
                    # simplifield
                    [Shape(2, 588, 1152), Shape(2, 1, 1152)],
                ]
            elif func_index == 6: # misc_rmsnorm
                shape_list = [
                    # real
                    # Shape(56, 16, 405, 72),
                    # Shape(810, 16, 28, 72),
                    # simplifield
                    Shape(2, 704, 72),
                ]
            elif func_index == 7: # misc_layernorm
                shape_list = [
                    # real
                    # Shape(2, 11340, 1152),
                    # simplifield
                    Shape(1, 728, 1152),
                ]
            elif func_index == 12: # misc_softmax
                shape_list = [
                    # real
                    # Shape(810, 16, 28, 28),
                    # Shape(56, 16, 405, 405),
                    # Shape(1, 16, 22680, 284),
                    # simplifield
                    Shape(2, 704, 28),
                    Shape(2, 704, 405),
                    Shape(2, 704, 284),
                ]
            elif func_index == 13: # misc_gelu
                shape_list = [
                    # real
                    # Shape(2, 11340, 4608),
                    # simplified
                    Shape(1, 280, 4608),
                ]
            elif func_index == 16: # misc_convert
                shape_list = [
                    # real
                    # Shape(810, 16, 28, 72),
                    # Shape(810, 16, 28, 28),
                    # Shape(810, 28, 1152),
                    # Shape(810, 16, 72, 28),
                    # Shape(2, 11340, 1152),
                    # Shape(1, 16, 22680, 284),
                    # Shape(1, 16, 72, 284),
                    # Shape(1, 284, 1152),
                    # Shape(56, 16, 405, 72),
                    # Shape(56, 16, 405, 405),
                    # Shape(56, 16, 72, 405),
                    # Shape(56, 405, 1152),
                    # simplified
                    Shape(1, 1408, 72),
                    Shape(2, 704, 28),
                    Shape(1, 1176, 1152),
                    Shape(2, 704, 284),
                    Shape(2, 704, 405),
                ]
            elif func_index == 17: # misc_div
                shape_list = [
                    # real
                    # Shape(56, 16, 405, 72),
                    # Shape(1, 16, 22680, 72),
                    # Shape(810, 16, 28, 72),
                    # simplified
                    Shape(1, 1408, 72),
                ]
            print(f"Running case: {functions[func_index].__name__}")
            for shape in shape_list:
                if 0 <= func_index < len(functions):
                    functions[func_index](root_path, mode=mode, dynamic_scale=dynamic_scale, shape=shape)
                else:
                    print(f"Invalid index: {index}. Skipping.")
    else:
        idx = 18  # case
        root_path = "./cases"
        mode = CompileFlags()
        mode.target = CCTarget.verify
        mode.enable_slr_slice = True
        functions[idx](root_path, mode)


if __name__ == "__main__":
    main(False)
