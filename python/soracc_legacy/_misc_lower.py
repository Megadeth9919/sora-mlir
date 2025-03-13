from graph_ir import *
from inst import *
import numpy as np 
import yaml
import sys
import json
import pytest
import os
from p_model import PModel
from utils import dump_information, gen_random_data



def layer_norm(x: ndarray, gamma: ndarray, eps=1e-5, use_beta: bool = False, beta=None):
    # 检查gamma和beta的形状是否与x的特征维度匹配
    if gamma.shape != (x.shape[-1],):
        raise ValueError("gamma shape must match the number of features in x")
    if use_beta and beta is not None and beta.shape != gamma.shape:
        raise ValueError("beta shape must match gamma shape")
    
    B = x.shape[0]  
    M = x.shape[1]  
    N = x.shape[2]
    
    # split the input tensor into three slr
    split_tensor_for_multi_slir = np.split(x.astype(np.float32), 3, axis=-1)
    intput_sum = np.zeros((B, M), dtype=np.float32)
    input_square_sum = np.zeros((B, M), dtype=np.float32)
    for slr_id in range(3):
        intput_sum += np.sum(split_tensor_for_multi_slir[slr_id], axis=-1)
        input_square_sum += np.sum(np.square(split_tensor_for_multi_slir[slr_id]), axis=-1)
        
    mean = np.float32(intput_sum) / np.float32(N)
    square_mean = np.float32(input_square_sum) / np.float32(N)
    variance = np.float32(square_mean) - np.float32(np.square(np.float32(mean)))
    variance_reciprocal = 1 / np.float32(np.sqrt(np.float32(variance) + np.float32(eps)))
    
    mean_real = np.repeat(np.expand_dims(mean, axis=-1), N, axis=-1)
    variance_real = np.repeat(np.expand_dims(variance_reciprocal, axis=-1), N, axis=-1)
    output = (x - np.float32(mean_real)) * np.float32(variance_real).astype(np.float32)

    output = output * np.float32(gamma)
    # 添加偏移项（如果使用）
    if use_beta:
        if beta is None:
            beta = np.zeros_like(gamma)  # 如果未提供beta，则默认为全零
        output += beta
    
    return output.astype(np.float16)


def rms_norm(x: ndarray, gamma: ndarray, eps=1e-5, use_beta=False, beta=None):
    # 检查gamma和beta的形状是否与x的特征维度匹配
    if gamma.shape != (x.shape[-1],):
        raise ValueError("gamma shape must match the number of features in x")
    if use_beta and beta is not None and beta.shape != gamma.shape:
        raise ValueError("beta shape must match gamma shape")
    
    B = x.shape[0]
    M = x.shape[1]
    N = x.shape[2]

    split_tensor_for_multi_slir = np.split(x.astype(np.float32), 3, axis=-1)

    input_square_sum = np.zeros((B, M), dtype=np.float32)
    for slr_id in range(3):
        input_square_sum += np.sum(np.square(split_tensor_for_multi_slir[slr_id]), axis=-1)

    rms = np.sqrt(np.float32(input_square_sum) / np.float32(N))

    normed_x = 1 / np.float32(rms + np.float32(eps))
    normed_x_real = np.repeat(np.expand_dims(normed_x, axis=-1), N, axis=-1)

    output = x * np.float32(normed_x_real)
    # 缩放
    output = output * gamma.astype(np.float32)
    
    # 添加偏移项（如果使用）
    if use_beta:
        if beta is None:
            beta = np.zeros_like(gamma)  # 如果未提供beta，则默认为全零
        output += beta
    
    return output.astype(np.float16)

def silu(x: ndarray):
    # Sigmoid 函数
    sigmoid = 1 / (1 + np.exp(-x))
    
    # SiLU 计算
    return x * sigmoid 

def gelu_approx(x: ndarray):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def test_lower_misc_elementwise_add(root_path):
    g = StaticGraph()

    x = Tensor(Shape(1, 2, 16), dtype=DataType.float16, name="input0")
    y = Tensor(Shape(1, 2, 16), dtype=DataType.float16, name="input1")
    gen_random_data([x, y])

    output_data = np.add(x.get_data(), y.get_data())
    print("op out:",output_data)

    op = Eltwise("elementwise_add", "add")
    op.add_input_tensors([x, y])

    out = Tensor(Shape(1, 2, 16), dtype=DataType.float16, name="output")
    out.set_data(output_data)
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    dump_information(g, root_path)
    pass


@pytest.mark.skip(reason="fail")
def test_lower_misc_elementwise_broadcast1_add(root_path):
    g = StaticGraph()

    x = Tensor(Shape(3, 576, 1152), dtype=DataType.float16, name="input0")
    y = Tensor(Shape(3, 1, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([x, y])

    output_data = np.add(x.get_data(), y.get_data())
    print("op out:",output_data)

    op = Eltwise("elementwise_broadcast1_add", "add")
    op.add_input_tensors([x, y])

    out = Tensor(Shape(3, 576, 1152), dtype=DataType.float16, name="output")
    out.set_data(output_data)
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    dump_information(g, root_path)
    pass


@pytest.mark.skip(reason="fail")
def test_lower_misc_elementwise_broadcast0_add(root_path):
    g = StaticGraph()

    x = Tensor(Shape(3, 6, 1152), dtype=DataType.float16, name="input0")
    y = Tensor(Shape(1, 6, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([x, y])

    output_data = np.add(x.get_data(), y.get_data())
    print("op out:",output_data)

    op = Eltwise("elementwise_broadcast0_add", "add")
    op.add_input_tensors([x, y])

    out = Tensor(Shape(3, 6, 1152), dtype=DataType.float16, name="output")
    out.set_data(output_data)
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    dump_information(g, root_path)
    pass


@pytest.mark.skip(reason="fail")
def test_lower_misc_silu(root_path):
    g = StaticGraph()

    x = Tensor(Shape(1, 2, 16), dtype=DataType.float16, name="input")
    gen_random_data([x])

    output_data = silu(x.get_data())

    op = Silu("silu")
    op.add_input_tensors([x])

    out = Tensor(Shape(1, 2, 16), dtype=DataType.float16, name="output")
    out.set_data(output_data)
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)

    dump_information(g, root_path)
    pass


@pytest.mark.skip(reason="fail")
def test_lower_misc_rmsnorm(root_path):
    g = StaticGraph()

    x = Tensor(Shape(432, 4, 16, 72), dtype=DataType.float16, name="input")
    gamma = Weight("weight", Shape(1, 72), data_type=DataType.float16)
    gen_random_data([x, gamma])

    # TODO There is a bug in the implementation of rms_norm
    # output_data = rms_norm(x.get_data(), gamma.get_data().flatten())
    # print("op out:",output_data)

    op = RMSnorm("rmsnorm")
    op.add_input_tensors([x])

    op.set_weight(weight=gamma)

    out = Tensor(Shape(432, 4, 16, 72), dtype=DataType.float16, name="output")
    # out.set_data(output_data)
    gen_random_data([out])
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    dump_information(g, root_path)  
    pass


@pytest.mark.skip(reason="fail")
def test_lower_misc_layernorm(root_path):
    g = StaticGraph()

    x = Tensor(Shape(3, 576, 1152), dtype=DataType.float16, name="input")
    gamma = Weight("weight", Shape(1152), data_type=DataType.float16)
    gen_random_data([x, gamma])

    output_data = layer_norm(x.get_data(), gamma.get_data())
    print("op out:",output_data)

    op = Layernorm("layernorm", affine=False)
    op.add_input_tensors([x])

    op.set_weight_bias(gamma=gamma)

    out = Tensor(Shape(3, 576, 1152), dtype=DataType.float16, name="output")
    out.set_data(output_data)
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    dump_information(g, root_path)
    pass


@pytest.mark.skip(reason="fail")
def test_lower_misc_transpose01(root_path):
    g = StaticGraph()
    # 2,3,5
    input_x = np.random.randint(0, 127, (2, 3, 2, 2), dtype=np.int8)
    output_x = np.transpose(input_x, (1, 0, 2, 3))
    print("np out:",output_x)
    x = Tensor(Shape(2, 3, 2, 2), dtype=DataType.int8, name="input")
    
    op = Transpose("transpose01", 0, 1)
    x.set_data(input_x)
    x.add_user(op)

    op.set_inputs(
        [
            x,
        ]
    )
    
    out = Tensor(Shape(3, 2, 2, 2), dtype=DataType.int8, name="output")
    out.set_data(output_x)

    op.set_outputs([out])
    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)

    dump_information(g, root_path)



@pytest.mark.skip(reason="fail")
def test_lower_misc_transpose23(root_path):
    g = StaticGraph()

    x = Tensor(Shape(432, 4, 16, 72), dtype=DataType.int8, name="input")
    gen_random_data([x])

    output_x = np.transpose(x.get_data(), (0, 1, 3, 2))
    print("op out:",output_x)

    op = Transpose("transpose23", 2, 3)

    op.add_input_tensors([x])

    out = Tensor(Shape(432, 4, 72, 16), dtype=DataType.int8, name="output")
    out.set_data(output_x)
    op.set_outputs([out])

    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    dump_information(g, root_path)


@pytest.mark.skip(reason="fail")
def test_lower_misc_split(root_path):
    g = StaticGraph()

    x = Tensor(Shape(3, 6, 1152), dtype=DataType.float16, name="input")
    gen_random_data([x])

    output_data = np.split(x.get_data(), 6, axis=1)
    print("op out:",output_data)

    op = Split("split", 6, 1)
    op.add_input_tensors([x])

    out_1 = Tensor(Shape(3, 1, 1152), dtype=DataType.float16, name="output0")
    out_1.set_data(output_data[0])
    out_2 = Tensor(Shape(3, 1, 1152), dtype=DataType.float16, name="output1")
    out_2.set_data(output_data[1])    
    out_3 = Tensor(Shape(3, 1, 1152), dtype=DataType.float16, name="output2")
    out_3.set_data(output_data[2])
    out_4 = Tensor(Shape(3, 1, 1152), dtype=DataType.float16, name="output3")
    out_4.set_data(output_data[3])
    out_5 = Tensor(Shape(3, 1, 1152), dtype=DataType.float16, name="output4")
    out_5.set_data(output_data[4])    
    out_6 = Tensor(Shape(3, 1, 1152), dtype=DataType.float16, name="output5")
    out_6.set_data(output_data[5])
    op.set_outputs([out_1, out_2, out_3, out_4, out_5, out_6])

    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    print("pmodel out:",output_list[1].data)
    print("pmodel out:",output_list[2].data)

    dump_information(g, root_path)
    pass

def test_lower_vector_add(root_path):
    g = StaticGraph()
    # 2,3,5
    input_x = np.arange(100).astype(np.float16).reshape(1, 1, 2, 50)
    input_x_flip = np.flip(input_x,axis=3)
    print("input0:",input_x)
    print("input1:",input_x_flip)
    output_x = np.add(input_x, input_x_flip)
    print("op out:",output_x)
    input0 = Tensor(Shape(1, 1, 2, 50), dtype=DataType.float16, name="input0")
    input1 = Tensor(Shape(1, 1, 2, 50), dtype=DataType.float16, name="input1")
    
    op = Eltwise("vectorAdd", type="add")

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    # input0.set_data(input_x, op.name + '/' + input0.name + ".bin")
    input0.set_data(input_x)
    input0.add_user(op)

    # input1.set_data(input_x, op.name + '/' + input1.name + ".bin")
    
    input1.set_data(input_x_flip)
    input1.add_user(op)

    op.set_inputs(
        [
            input0, input1
        ]
    )

    out = Tensor(Shape(1, 1, 2, 50), dtype=DataType.float16, name="output")
    out.set_data(output_x)

    op.set_outputs([out])
    g.add(load_inst)
    g.add(op)

    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)

    dump_information(g, root_path)

def test_lower_softmax(root_path):
    g = StaticGraph()

    x = Tensor(Shape(432, 4, 16, 16), dtype=DataType.float16, name="input")
    gen_random_data([x])

    output_data = np.exp(x.get_data()) / np.sum(np.exp(x.get_data()), axis=-1, keepdims=True)
    print("op out:",output_data)

    op = Softmax("softmax")
    op.add_input_tensors([x])

    out = Tensor(Shape(432, 4, 16, 16), dtype=DataType.float16, name="output")
    out.set_data(output_data)
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)

    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    dump_information(g, root_path)

    pass

def test_lower_misc_gelu(root_path):
    g = StaticGraph()


    x = Tensor(Shape(3, 576, 4608), dtype=DataType.float16, name="input")
    gen_random_data([x])


    output_data = gelu_approx(x.get_data())
    print("op out:",output_data)

    op = Gelu("gelu")
    op.add_input_tensors([x])

    out = Tensor(Shape(3, 576, 4608), dtype=DataType.float16, name="output")
    out.set_data(output_data)
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data,dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)

    dump_information(g, root_path)
    pass


def test_lower_to_linearw8(root_path):
    g = StaticGraph()

    # input_x_data = np.random.randint(0, 127, (2, 3, 4), dtype=np.int8)
    # input_w_data = np.random.randint(0, 127, (4, 5), dtype=np.int8)
    # input_scale_data = np.random.randn(5).astype(np.float16)

    x = Tensor(Shape(432,4,1152), dtype=DataType.int8, name="input")
    w = Weight("weight", Shape(1152,1152), data_type=DataType.int8)
    scale = Weight("scale", Shape(1152), data_type=DataType.float16)
    output = Tensor(Shape(432,4,1152), dtype=DataType.int8, name="output")

    gen_random_data([x, w, scale, output])

    op = LinearW8(name="linearw8", bias_flag = False)

    op.add_input_tensors([x])
    op.set_weight_scale(weight = w, weight_scale = scale)
    op.set_outputs([output])

    g.add(op)

    dump_information(g, root_path)  

def test_lower_to_matmul(root_path):
    g = StaticGraph()

    A = Tensor(Shape(432, 4, 16, 72), dtype=DataType.int8, name="input0")
    B = Tensor(Shape(432, 4, 16, 72), dtype=DataType.int8, name="input1")
    gen_random_data([A, B])

    output = np.multiply(A.get_data(), B.get_data())

    op = Matmul(name="Matmul")

    op.add_input_tensors([A, B])
    output = Tensor(Shape(432,4,1152), dtype=DataType.int8, name="output")
    op.set_outputs([output])

    g.add(op)

    dump_information(g, root_path)  

@pytest.mark.skip(reason="fail")
def test_lower_misc_elementwise_broadcast1_mul(root_path):
    g = StaticGraph()

    x = Tensor(Shape(3, 576, 1152), dtype=DataType.float16, name="input0")
    y = Tensor(Shape(3, 1, 1152), dtype=DataType.float16, name="input1")
    gen_random_data([x, y])

    output_data = np.multiply(x.get_data(), y.get_data())
    print("op out:",output_data)

    op = Eltwise("elementwise_broadcast1_mul", "mul")
    op.add_input_tensors([x, y])

    out = Tensor(Shape(3, 576, 1152), dtype=DataType.float16, name="output")
    out.set_data(output_data)
    op.set_outputs([out])

    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    f = open("pvpu_ins_7_23.bin","rb")
    inst_data = f.read()
    inst_array = np.frombuffer(inst_data, dtype=np.int8)
    inst_array.reshape(16384)
    inst_tensor.set_data(inst_array)
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])

    g.add(load_inst)
    g.add(op)
    p = PModel(graph=g)
    output_list = p.run()
    print("pmodel out:",output_list[0].data)
    dump_information(g, root_path)
    pass

functions = [
    test_lower_misc_elementwise_add,
    test_lower_misc_elementwise_broadcast0_add,
    test_lower_misc_elementwise_broadcast1_add,
    test_lower_misc_layernorm,
    test_lower_misc_rmsnorm,
    test_lower_misc_silu,
    test_lower_softmax,
    test_lower_misc_split,
    test_lower_misc_transpose01,
    test_lower_misc_transpose23,
    test_lower_vector_add,
    test_lower_misc_gelu,
    test_lower_to_linearw8,
    test_lower_misc_elementwise_broadcast1_mul,
    test_lower_to_matmul,
]

def main(debug: bool = False):
    if not debug:
        if len(sys.argv) < 2:
            print("Usage: python test_misc_lower.py <path> <indices...>")
            sys.exit(1)

        # Get the path and indices
        root_path = sys.argv[1]
        indices = sys.argv[2:]

        for index in indices:
            func_index = int(index) - 1
            if 0 <= func_index < len(functions):
                functions[func_index](root_path)
            else:
                print(f"Invalid index: {index}. Skipping.")
    else:
        idx = 3 # case 
        root_path = "/home/wangjiaqi/workspace/sora_cc/build"
        functions[idx](root_path)     

if __name__ == "__main__":
    main(False)
