import json
import model.model_base as mb
import numpy as np
from model.sora_model import get_single_STDiT3Block, get_STDiT3, get_mlp, STDiT3BlockOnly
from utils import  gen_random_data
from .graph_ir import StaticGraph, Tensor, LinearW8, Shape, Weight, DataType
from .pass_impl import ddr_address_alloc, CCTarget
from .pass_core_slice import *

def test_n_part_division():
    ret = n_part_division(9, 3)
    assert ret == [3, 3, 3]

    ret = n_part_division(10, 3)
    assert ret == [4, 4, 2]

    ret = n_part_division(11, 3)
    assert ret == [4, 4, 3]

    ret = n_part_division(12, 3)
    assert ret == [4, 4, 4]
    
    ret = n_part_division(3, 3)
    assert ret == [1, 1, 1]

    ret = n_part_division(2, 3)
    assert ret == [1, 1, 0]
    
    ret = n_part_division(1, 3)
    assert ret == [1, 0, 0]

def test_first_none_trivial_dim_not_last():
    ret = first_none_trivial_dim_not_last(Shape(1, 2, 3, 4), last_limit=0)
    assert ret == (1, 2)
    
    ret = first_none_trivial_dim_not_last(Shape(1, 2, 3, 4), last_limit=1)
    assert ret == (1, 2)
    
    ret = first_none_trivial_dim_not_last(Shape(1, 2, 3, 4), last_limit=2)
    assert ret == (1, 2)
    
    ret = first_none_trivial_dim_not_last(Shape(1, 2, 3, 4), last_limit=3)
    assert ret == (0, 1)
    
    ret = first_none_trivial_dim_not_last(Shape(1, 1, 1, 4), last_limit=0)
    assert ret == (3, 4)
    
    ret = first_none_trivial_dim_not_last(Shape(1, 1, 1, 4), last_limit=1)
    assert ret == (2, 1)
    
    ret = first_none_trivial_dim_not_last(Shape(1, 1, 1, 4), last_limit=2)
    assert ret == (1, 1)
    
    ret = first_none_trivial_dim_not_last(Shape(1, 2, 3, 4), last_limit=3)
    assert ret == (0, 1)
    
    ret = first_none_trivial_dim_not_last(Shape(1, 2, 3, 4), last_limit=4)
    assert ret == (0, 1)
    
def test_clone():
    for mm in (mb.Linear, mb.Matmul):
        g = StaticGraph()
        if mm == mb.Linear:
            op_wrapper = mm(g, in_feature=128, out_feature=40, bias=True, name=f'{str(mm)}')
            x = Tensor(Shape(100, 3, 128), dtype=DataType.int8)
            scale = Tensor(Shape(100, 3, 1), dtype=DataType.int8)
            y = op_wrapper(x, DataType.float16, act_scale=scale)
        else:
            op_wrapper = mm(g, name=f'{str(mm)}_dynamic')
            x = Tensor(Shape(100, 3, 128), dtype=DataType.float16)
            x2 = Tensor(Shape(100, 40, 128), dtype=DataType.float16)
            y = op_wrapper(x, x2, DataType.float16)

        clone_op = op_clone(g.get_ops()[0])
        print(clone_op)
        if mm == mb.Linear:
            assert clone_op.get_act_scale.shape == g.get_ops()[0].get_act_scale.shape
            assert clone_op.get_act_scale.addr == g.get_ops()[0].get_act_scale.addr
        assert clone_op.get_inputs()[0].shape == x.shape
        assert clone_op.get_inputs()[0].addr == x.addr
        if mm == mb.Matmul:
            assert clone_op.get_inputs()[1].shape == x2.shape
            assert clone_op.get_inputs()[1].addr == x2.addr
        assert clone_op.get_output.shape == y.shape
        assert clone_op.get_output.addr == y.addr
        for tid in range(len(clone_op.get_weights())):
            assert clone_op.get_weights()[tid].shape == g.get_ops()[0].get_weights()[tid].shape
            assert clone_op.get_weights()[tid].addr == g.get_ops()[0].get_weights()[tid].addr

def test_matmul_w8a8_clone():
   
    g = StaticGraph()
    op_wrapper = mb.Matmul(g, name=f'matmul_w8a8')
    x = Tensor(Shape(100, 3, 128), dtype=DataType.int8)
    x_scale = Tensor(Shape(100, 3, 1), dtype=DataType.float16)
    x2 = Tensor(Shape(100, 40, 128), dtype=DataType.int8)
    x2_scale = Tensor(Shape(100, 40, 1), dtype=DataType.float16)
    
    x.set_act_scale(x_scale)
    x2.set_act_scale(x2_scale)
    y = op_wrapper(x, x2, DataType.float16)

    clone_op = op_clone(g.get_ops()[0])
    print(clone_op)
    assert id(clone_op.get_inputs()[0].act_scale) == id(clone_op.get_inputs()[-2])
    assert id(clone_op.get_inputs()[1].act_scale) == id(clone_op.get_inputs()[-1])
    assert clone_op.get_inputs()[0].shape == x.shape
    assert clone_op.get_inputs()[0].addr == x.addr
    assert clone_op.get_inputs()[1].shape == x2.shape
    assert clone_op.get_inputs()[1].addr == x2.addr
    assert clone_op.get_output.shape == y.shape
    assert clone_op.get_output.addr == y.addr


def test_eltwise_clone():
    for eltwise in (mb.Eltwise, mb.Div):
        g = StaticGraph()
        if eltwise == mb.Eltwise:
            op_wrapper = eltwise(g, t="add", name=f'{str(eltwise)}')
        else:
            op_wrapper = eltwise(g, name=f'{str(eltwise)}')
            
        x1 = Tensor(Shape(100, 3, 4), dtype=DataType.float16)
        x2 = Tensor(Shape(1, 3, 4), dtype=DataType.float16)
        y = op_wrapper(x1, x2)
        g.complete()
        ddr_address_alloc(g)
        
        clone_op = op_clone(g.get_ops()[0])
        print(clone_op)
        assert clone_op.get_inputs()[0].addr == x1.addr
        assert clone_op.get_inputs()[0].shape == x1.shape
        assert clone_op.get_inputs()[1].addr == x2.addr
        assert clone_op.get_inputs()[1].shape == x2.shape
        assert clone_op.get_output.addr == y.addr
        assert clone_op.get_output.shape == y.shape

def test_activation_clone():
    for act in (mb.Gelu, mb.Silu, mb.Softmax):
        g = StaticGraph()
        op_wrapper = act(g, name=f'{str(act)}_dynamic')
        x = Tensor(Shape(100, 3, 4), dtype=DataType.float16)
        y = op_wrapper(x, DataType.int8)
        ddr_address_alloc(g)
        
        clone_op = op_clone(g.get_ops()[0])
        print(clone_op)
        assert id(clone_op.get_output.act_scale) == id(clone_op.get_act_scale)
        assert clone_op.get_input.shape == x.shape
        assert clone_op.get_output.shape == y.shape
        assert clone_op.get_output.get_act_scale.shape == y.get_act_scale.shape
        assert clone_op.get_input.addr == x.addr
        assert clone_op.get_output.addr == y.addr
        assert clone_op.get_output.get_act_scale.addr == y.get_act_scale.addr

def test_normalization_clone():
    for norm in (mb.LayerNorm, mb.RMSNorm):
        g = StaticGraph()
        op_wrapper = norm(g, Shape(1, 4), f'{str(norm)}_dynamic')
        x = Tensor(Shape(100, 3, 4), dtype=DataType.float16)
        y = op_wrapper(x, DataType.int8)

        clone_op = op_clone(g.get_ops()[0])
        print(clone_op)
        assert id(clone_op.get_output.act_scale) == id(clone_op.get_act_scale)
        assert clone_op.get_input.shape == x.shape
        assert clone_op.get_output.shape == y.shape
        assert clone_op.get_output.get_act_scale.shape == y.get_act_scale.shape
        assert clone_op.get_input.addr == x.addr
        assert clone_op.get_output.addr == y.addr
        assert clone_op.get_output.get_act_scale.addr == y.get_act_scale.addr
        for tid in range(len(clone_op.get_weights())):
            assert clone_op.get_weights()[tid].shape == g.get_ops()[0].get_weights()[tid].shape
            assert clone_op.get_weights()[tid].addr == g.get_ops()[0].get_weights()[tid].addr

def test_rope_clone():
    g = StaticGraph()
    op_wrapper = mb.RoPE(g, dim=128, name=f'Rope')
    x = Tensor(Shape(100, 3, 128), dtype=DataType.float16)
    y = op_wrapper(x)

    clone_op = op_clone(g.get_ops()[0])
    print(clone_op)
    assert clone_op.get_input.shape == x.shape
    assert clone_op.get_output.shape == y.shape
    assert clone_op.get_input.addr == x.addr
    assert clone_op.get_output.addr == y.addr
    for tid in range(len(clone_op.get_weights())):
        assert clone_op.get_weights()[tid].shape == g.get_ops()[0].get_weights()[tid].shape
        assert clone_op.get_weights()[tid].addr == g.get_ops()[0].get_weights()[tid].addr

def test_convert_clone():
    g = StaticGraph()
    op_wrapper = mb.Convert(g, out_dtype=DataType.int8, name=f'convert')
    x = Tensor(Shape(100, 3, 128), dtype=DataType.float16)
    y = op_wrapper(x)

    clone_op = op_clone(g.get_ops()[0])
    print(clone_op)
    assert id(clone_op.get_output.act_scale) == id(clone_op.get_act_scale)
    assert clone_op.get_input.shape == x.shape
    assert clone_op.get_output.shape == y.shape
    assert clone_op.get_output.get_act_scale.shape == y.get_act_scale.shape
    assert clone_op.get_input.addr == x.addr
    assert clone_op.get_output.addr == y.addr
    assert clone_op.get_output.get_act_scale.addr == y.get_act_scale.addr

def test_transpose_clone():
    g = StaticGraph()
    op_wrapper = mb.Transpose(g, dim_a=-2, dim_b=-1, name=f'transpose')
    x = Tensor(Shape(100, 3, 128), dtype=DataType.float16)
    y = op_wrapper(x)

    clone_op = op_clone(g.get_ops()[0])
    print(clone_op)
    assert clone_op.get_input.shape == x.shape
    assert clone_op.get_output.shape == y.shape
    assert clone_op.get_input.addr == x.addr
    assert clone_op.get_output.addr == y.addr

def test_loadinst_clone():
    g = StaticGraph()
    x = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([x])
    g.add(load_inst)

    clone_op = op_clone(g.get_ops()[0])
    print(clone_op)
    assert clone_op.get_input.shape == x.shape
    assert clone_op.get_input.addr == x.addr
    

def test_core_slice():
    g = StaticGraph()
    in_shape = Shape(100, 3, 128)
    out_shape = Shape(100, 3, 5)
    act_scale_shape = Shape(100, 3, 1)
    op = LinearW8('linear_w8')
    x = Tensor(in_shape, dtype=DataType.int8)
    act_scale = Tensor(act_scale_shape, dtype=DataType.float16)
    op.add_input_tensors((x, act_scale))
    weight = Weight(name=f".weight",
                    shape=Shape(5, 128),
                    data_type=DataType.int8)
    scale = Weight(name=f".scale",
                    shape=Shape(5),
                    data_type=DataType.float16)
    op.set_weight_scale(weight=weight, weight_scale=scale)
    new_shape = out_shape
    y = op.create_tensor(new_shape, dtype=DataType.float16)
    g.add(op)
    g.complete()

    ddr_address_alloc(g)
    print(g.get_ops()[0])
    core_num = 3
    ret = core_slice_tensor(g, core_num)
    idx, dim = first_none_trivial_dim_not_last(in_shape, last_limit=1)
    for i in range(core_num):
        for op in ret[i]:
            print(op)

def test_matmul_w8a8_slice():
    g = StaticGraph()
    op_wrapper = mb.Matmul(g, name=f'matmul_w8a8')
    x = Tensor(Shape(100, 3, 128), dtype=DataType.int8)
    x_scale = Tensor(Shape(100, 3, 1), dtype=DataType.float16)
    x2 = Tensor(Shape(100, 40, 128), dtype=DataType.int8)
    x2_scale = Tensor(Shape(100, 40, 1), dtype=DataType.float16)
    
    x.set_act_scale(x_scale)
    x2.set_act_scale(x2_scale)
    y = op_wrapper(x, x2, DataType.float16)
    g.complete()

    ddr_address_alloc(g)
    print(g.get_ops()[0])
    core_num = 3
    ret = core_slice_tensor(g, core_num)
    for i in range(core_num):
        for op in ret[i]:
            print(op)

def test_eltwise_slice():
    for eltwise in (mb.Div, mb.Eltwise):
        for shape in ((1, 70, 3, 4), (100, 3, 4)):
            g = StaticGraph()
            if eltwise == mb.Eltwise:
                op_wrapper = eltwise(g, t="add", name=f'{str(eltwise)}')
            else:
                op_wrapper = eltwise(g, name=f'{str(eltwise)}')
                
            x1 = Tensor(Shape(*shape), dtype=DataType.float16)
            x2 = Tensor(Shape(*shape), dtype=DataType.float16)
            y = op_wrapper(x1, x2)
            g.complete()
            ddr_address_alloc(g)
            
            core_num = 3
            ret = core_slice_tensor(g, core_num)
            for i in range(core_num):
                for op in ret[i]:
                    print(op)

def test_loadinst_softmax_slice():
    g = StaticGraph()
    inst_tensor = Tensor(Shape(16384), dtype=DataType.int8, name="pvpu_ins")
    load_inst = LoadInst("LoadInst")
    load_inst.add_input_tensors([inst_tensor])
    g.add(load_inst)
    
    input_x = Tensor(Shape(432, 4, 16, 16), dtype=DataType.float16, name="input")
    model = mb.Softmax(g, name="softmax")
    output_dtype = DataType.int8
    model(input_x, out_dtype=output_dtype)
    g.complete()
    
    ddr_address_alloc(g)
    
    core_num = 3
    ret = core_slice_tensor(g, core_num)
    for i in range(core_num):
        for op in ret[i]:
            print(op)
    

def test_linear_slice():
    op = LinearW8('linear_w8')
    x = Tensor(Shape(2, 3, 4), dtype=DataType.int8)
    x.addr = 0
    act_scale = Tensor(Shape(2, 3, 1), dtype=DataType.float16)
    act_scale.addr = 0
    op.add_input_tensors((x, act_scale))
    weight = Weight(name=f".weight",
                    shape=Shape(5, 4),
                    data_type=DataType.int8)
    weight.addr = 0
    scale = Weight(name=f".scale",
                    shape=Shape(5),
                    data_type=DataType.float16)
    scale.addr = 0
    op.set_weight_scale(weight=weight, weight_scale=scale)
    new_shape = Shape(2, 3, 5)
    y = op.create_tensor(new_shape, dtype=DataType.float16)
    y.addr = 0

    op1 = op_clone(op)
    op2 = op_clone(op)
    op = core_slice_linear(op, 0, 3)
    print(op)
    op1 = core_slice_linear(op1, 1, 3)
    print(op1)
    op2 = core_slice_linear(op2, 2, 3)
    print(op2)

def test_activation_slice():
    for act in (Gelu, Silu):
        g = StaticGraph()
        op = act(f'{str(act)}')
        x = Tensor(Shape(2, 3, 4), dtype=DataType.float16)
        op.add_input_tensors((x, ))
        new_shape = Shape(2, 3, 4)
        y = op.create_tensor(new_shape, dtype=DataType.float16)
        g.add(op)
        g.complete()

        ddr_address_alloc(g)

        core_num = 3
        ret = core_slice_tensor(g, core_num)
        for i in range(core_num):
            for op in ret[i]:
                print(op)

def test_layernorm_slice():
    for norm in (mb.LayerNorm, mb.RMSNorm):
        g = StaticGraph()
        op_wrapper = norm(g, Shape(1, 4),f'{str(norm)}_dynamic')
        x = Tensor(Shape(2, 3, 4), dtype=DataType.float16)
        y = op_wrapper(x, DataType.int8)
        g.complete()
        ddr_address_alloc(g)
        core_num = 3
        ret = core_slice_tensor(g, core_num)
        for i in range(core_num):
            for op in ret[i]:
                print(op)

def test_rope_slice():
    g = StaticGraph()
    op_wrapper = mb.RoPE(g, dim=128, name=f'Rope')
    x = Tensor(Shape(100, 3, 128), dtype=DataType.float16)
    y = op_wrapper(x)
    g.complete()
    ddr_address_alloc(g)
    core_num = 3
    ret = core_slice_tensor(g, core_num)
    for i in range(core_num):
        for op in ret[i]:
            print(op)

def test_convert_slice():
    g = StaticGraph()
    op_wrapper = mb.Convert(g, out_dtype=DataType.int8, name=f'convert')
    x = Tensor(Shape(100, 3, 128), dtype=DataType.float16)
    y = op_wrapper(x)
    g.complete()
    ddr_address_alloc(g)
    core_num = 3
    ret = core_slice_tensor(g, core_num)
    for i in range(core_num):
        for op in ret[i]:
            print(op)

def test_transpose_slice():
    g = StaticGraph()
    op_wrapper = mb.Transpose(g, dim_a=-2, dim_b=-1, name=f'transpose')
    x = Tensor(Shape(100, 3, 128), dtype=DataType.float16)
    y = op_wrapper(x)
    
    g.complete()
    ddr_address_alloc(g)
    core_num = 3
    ret = core_slice_tensor(g, core_num)
    for i in range(core_num):
        for op in ret[i]:
            print(op)

def test_mlp_slice():
    x = Tensor(Shape(2, 3, 1152), dtype=DataType.int8)
    scale = Tensor(Shape(2, 3, 1), dtype=DataType.float16)
    x.set_act_scale(scale)
    g = get_mlp(x)
    
    g.complete()

    ddr_address_alloc(g)

    core_num = 3
    ret = core_slice_tensor(g, core_num)
    for i in range(core_num):
        subgraph = dict()
        for idx, op in enumerate(ret[i]):
            subgraph[idx] = str(op)
        with open(f"mlp_subgraph_{i}.json", "w", encoding="utf-8") as f:
            json.dump(subgraph, f, indent=4)
    # for i in range(core_num):
    #     _g = StaticGraph()
    #     for op in ret[i]:
    #         _g.add(op)
    #     with open(f"mlp_subgraph_{i}.json", "w") as f:
    #         json.dump(_g.to_json(), f, indent=4)

def test_STDiT3BlockOnly_slice():
    g = StaticGraph()

    T, S = 4, 144
    B = 2
    cond_size = 52
    hidden_size = 1152
    block = STDiT3BlockOnly(graph=g, hidden_size=hidden_size, num_heads=16)
    
    x_tensor = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16)
    y_tensor = Tensor(Shape(B, cond_size, hidden_size), dtype=DataType.float16)
    t_tensor = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16)
    t_mlp_tensor = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16)
    y_scale = Tensor(Shape(B, cond_size, 1), dtype=DataType.int8)

    x_tensor_data = np.random.randn(B, T * S, hidden_size).astype(np.float16)
    y_tensor_data = np.random.randn(B, cond_size, hidden_size).astype(np.int8)
    t_tensor_data = np.random.randn(B, 6, hidden_size).astype(np.float16)
    t_mlp_tensor_data = np.random.randn(B, 6, hidden_size).astype(np.float16)
    y_scale_data = np.random.randn(B, cond_size, 1).astype(np.int8)

    x_tensor.set_data(x_tensor_data)
    y_tensor.set_data(y_tensor_data)
    t_tensor.set_data(t_tensor_data)
    t_mlp_tensor.set_data(t_mlp_tensor_data)
    y_scale.set_data(y_scale_data)
    y_tensor.set_act_scale(y_scale)

    out_tensor = block(x_tensor, y_tensor, t_tensor, t_mlp_tensor, T=T, S=S)
    g.complete()
    
    ddr_address_alloc(g, CCTarget.runtime)

    graph = dict()
    for idx, op in enumerate(g.get_ops()):
        graph[idx] = str(op).replace('\n', '  ')
    # with open(f"stdit_graph.json", "w") as f:
    #     json.dump(graph, f, indent=4)
    
    core_num = 3
    ret = core_slice_tensor(g, core_num)
    for i in range(core_num):
        subgraph = dict()
        for idx, op in enumerate(ret[i]):
            subgraph[idx] = str(op).replace('\n', '  ')
        # with open(f"stdit_subgraph_{i}.json", "w") as f:
        #     json.dump(subgraph, f, indent=4)
        
    assert out_tensor.shape[:-1] == x_tensor.shape[:-1] and out_tensor.shape[-1] == hidden_size
