import sys
from typing import Callable, List

import numpy as np
import pytest

from graph_ir import *
from inst import *
from model import model_base, sora_model
from p_model import PModel
from utils import (dump_information, gen_random_data, dump_graph_to_json)

from transform.SoraLegacyConverter import SoraLegacyConverter

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
    
    g.complete()
    # dump_information(g, root_path, op_name="softmax", mode=mode)

    dump_graph_to_json(g, 'softmax', '/home/qya/AiCompiler/sorac/tmp')
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()

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

    # dump_information(g, root_path, op_name=op_name, mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()


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

    # dump_information(g, root_path, op_name=op_name, mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()

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

    # dump_information(g, root_path, op_name=op_name, mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()


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

    # dump_information(g, root_path, op_name=op_name, mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()


def misc_transpose01(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(63, 129, 2, 16), dtype=DataType.int8, name="input")
    gen_random_data([input_x])

    model = model_base.Transpose(g, 0, 1, name="transpose01")
    model(input_x)

    # dump_information(g, root_path, op_name="transpose01", mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()


def misc_split(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()

    input_x = Tensor(Shape(3, 6, 1152), dtype=DataType.float16, name="input",const=True)
    gen_random_data([input_x])

    model = model_base.Split(g, 6, 1, name="split")
    model(input_x)

    # dump_information(g, root_path, op_name="split", mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()





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

    # dump_information(g, root_path, op_name=op_name, mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()


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

    # dump_information(g, root_path, op_name="linearw8", mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()


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

    # dump_information(g, root_path, op_name="linearw8_bias", mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()



def misc_convert(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(33, 8, 28, 72), dtype=DataType.float16, name="input")
    gen_random_data([input_x])

    model = model_base.Convert(g, out_dtype=DataType.int8, name="convert")

    model(input_x)

    # dump_information(g, root_path, op_name="convert", mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()


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
    gen_random_data([input_x])
    for weight in g.get_weights():
        shape = weight.shape
        new_shape = Shape(shape[0], shape[1], shape[2], shape[3] // 2)
        fake_data = np.random.randint(low=-10, high=10, size=list(new_shape), dtype=np.int8)
        fake_data = fake_data.astype(np.float16)
        fake_data = np.repeat(fake_data, 2, axis=3)
        weight.set_data(fake_data)

    # dump_information(g, root_path, op_name="rope", mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()

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

    # dump_information(g, root_path, op_name="matmul_int", mode=mode)
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir()


mode = CompileFlags()  # mode = 1 for separate dump file
mode.target = CCTarget.verify
misc_softmax('case', mode, True)
print('---------------------------------')
misc_elementwise_add('case', mode, True)
print('---------------------------------')
misc_elementwise_broadcast0_add('case', mode, True)
print('---------------------------------')
misc_rmsnorm('case', mode, True)
print('---------------------------------')
misc_layernorm('case', mode, True)
print('---------------------------------')
misc_transpose01('case', mode, True)
print('---------------------------------')
misc_split('case', mode, True)
print('---------------------------------')
misc_gelu('case', mode, True)
print('---------------------------------')
mm_linearw8('case', mode, True)
print('---------------------------------')
mm_linearw8_bias('case', mode, True)
print('---------------------------------')
misc_convert('case', mode, True)
print('---------------------------------')
misc_rope('case', mode, True)
print('---------------------------------')
mm_matmul_int('case', mode, True)