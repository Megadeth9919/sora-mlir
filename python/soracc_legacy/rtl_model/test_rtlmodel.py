from rtl_model.rtl_model import *
from graph_ir.graph_ir import StaticGraph, Tensor, Weight, Shape, DataType
from p_model import PModel
from model.model_base import Transpose, Eltwise, Linear, Silu, Matmul
from graph_ir.pass_impl import lower, ddr_address_alloc, pack_numpy_array
from inst.sora_inst import *
import numpy as np
import pytest

# TODO: Base address to config
HBM_ADDR_BASE = HW_Info.get_hbm_start()


@pytest.mark.skip(reason="fail")
def test_transpose():
    # TODO: input/output tensor config automatically generated
    data_in = np.random.randint(low=-128, high=127, size=[1, 2, 3, 4], dtype=np.int8)

    # Graph Definition
    g = StaticGraph()
    t_in = Tensor(Shape(1, 2, 3, 4), dtype=DataType.int8)
    t_in.set_data(data_in)

    v_trans = Transpose(g, dim_a=0, dim_b=1, name='test_transpose')(t_in)
    g.complete()

    # Compilation
    inst_collect = InstCollector()
    rt_info = ddr_address_alloc(g)
    in_data = pack_numpy_array([t.get_data() for t in g.get_inputs() if t.get_data() is not None])

    lower(g, inst_collect)

    inst_list = [inst_collect.get_insts()]

    # PModel
    p_model = PModel(g)

    data_out = p_model.run()[0].data

    # RtlModel
    rtl = RTLModel(debug=True, model_inst_list=inst_list)

    check = rtl.run(input_data=in_data, input_addr=rt_info.inputs.start, output_data=data_out, output_addr=rt_info.outputs.start)

    assert check


@pytest.mark.skip(reason="fail")
def test_eltwise():
    data_in1 = np.random.rand(2, 3, 4).astype(np.float16)
    data_in2 = np.random.rand(2, 3, 4).astype(np.float16)

    # Graph Definition
    g = StaticGraph()
    t_in1 = Tensor(Shape(2, 3, 4), dtype=DataType.float16)
    t_in1.set_data(data_in1)
    t_in2 = Tensor(Shape(2, 3, 4), dtype=DataType.float16)
    t_in2.set_data(data_in2)

    _ = Eltwise(g, t='add', name='test_eltwise')(t_in1, t_in2, out_dtype=DataType.float16)

    # Compilation
    g.complete()
    inst_collect = InstCollector()
    rt_info = ddr_address_alloc(g)
    in_data = pack_numpy_array([t.get_data() for t in g.get_inputs() if t.get_data() is not None])

    lower(g, inst_collect)

    inst_list = [inst_collect.get_insts()]
    inst_list[0][1].release = [PUType.MISC]

    inst_list[0][2].wait = [PUType.LD]
    inst_list[0][2].output_mode = MiscOutput.fp16
    inst_list[0][2].release = [PUType.ST]

    inst_list[0][3].wait = [PUType.MISC]
    inst_list[0][3].release = [PUType.LD]

    inst_list[0][4].wait = [PUType.ST]

    inst_list[0][5].release = [PUType.MISC]

    inst_list[0][6].wait = [PUType.LD]
    inst_list[0][6].output_mode = MiscOutput.fp16
    inst_list[0][6].release = [PUType.ST]

    inst_list[0][7].wait = [PUType.MISC]

    # PModel
    p_model = PModel(g)

    data_out = p_model.run()[0].data

    # RtlModel
    rtl = RTLModel(debug=True, model_inst_list=inst_list)

    check = rtl.run(input_data=in_data, input_addr=rt_info.inputs.start,
                    output_data=data_out, output_addr=rt_info.outputs.start)

    assert check


@pytest.mark.skip(reason="fail")
def test_activation():
    data_in = np.random.rand(2, 3, 4).astype(np.float16)

    # Graph Definition
    g = StaticGraph()
    t_in = Tensor(Shape(2, 3, 4), dtype=DataType.float16)
    t_in.set_data(data_in)

    _ = Silu(g, name='test_silu')(t_in, out_dtype=DataType.float16)

    g.complete()
    # Compilation
    inst_collect = InstCollector()
    rt_info = ddr_address_alloc(g)
    in_data = pack_numpy_array([t.get_data() for t in  g.get_inputs() if t.get_data() is not None])

    lower(g, inst_collect)
    inst_list = [inst_collect.get_insts()]
    inst_list[0][0].release = [PUType.MISC]

    inst_list[0][1].wait = [PUType.LD]
    inst_list[0][1].output_mode = MiscOutput.fp16
    inst_list[0][1].release = [PUType.ST]

    inst_list[0][2].wait = [PUType.MISC]
    inst_list[0][2].release = [PUType.LD]

    inst_list[0][3].wait = [PUType.ST]
    inst_list[0][3].release = [PUType.MISC]

    inst_list[0][4].wait = [PUType.LD]
    inst_list[0][4].output_mode = MiscOutput.fp16
    inst_list[0][4].release = [PUType.ST]

    inst_list[0][5].wait = [PUType.MISC]
    inst_list[0][5].release = [PUType.SYS]

    inst_list[0][6].wait = [PUType.ST]

    # PModel
    p_model = PModel(g)

    data_out = p_model.run()[0].data

    # RtlModel
    rtl = RTLModel(debug=True, model_inst_list=inst_list)

    check = rtl.run(input_data=in_data, input_addr=rt_info.inputs.start,
                    output_data=data_out, output_addr=rt_info.outputs.start)

    assert check


@pytest.mark.skip(reason="Not implemented yet")
def test_softmax():
    data_in = np.random.rand(2, 3, 4).astype(np.float16)
    data_out = np.exp(data_in) / np.sum(np.exp(data_in), axis=-1, keepdims=True)

    g = StaticGraph()


@pytest.mark.skip(reason="Not implemented yet")
def test_normalize():
    data_in = np.random.rand(2, 3, 4).astype(np.float16)
    data_out = data_in / np.sqrt(np.sum(np.square(data_in), axis=-1, keepdims=True))

    g = StaticGraph()


@pytest.mark.skip(reason="Not implemented yet")
def test_split():
    data_in = np.random.rand(2, 3, 4).astype(np.float16)
    data_out1 = data_in[:, :, :2]
    data_out2 = data_in[:, :, 2:]

    g = StaticGraph()


@pytest.mark.skip(reason="Not implemented yet")
def test_linear():
    data_in = np.random.randint(low=-128, high=127, size=[2, 3, 4], dtype=np.int8)
    weight = np.random.randint(low=-128, high=127, size=[4, 5], dtype=np.int8)
    data_out = np.matmul(data_in, weight)

    g = StaticGraph()
    t_in = Tensor(Shape(2, 3, 4), dtype=DataType.int8)
    t_in.set_data(data_in)

    v_mm = Linear(g, in_feature=4, out_feature=5, bias=False, name='test_mm')
    v_mm(t_in)

    g.complete()
    inst_collect = InstCollector()
    rt_info = ddr_address_alloc(g)
    in_data = pack_numpy_array([t.get_data() for t in  g.get_inputs() if t.get_data() is not None])

    lower(g, inst_collect)

    # import json
    # print(json.dumps(inst_collect.to_json(), indent=2))

    inst_list = [inst_collect.get_insts()]
    inst_list[0][-1].wait = [PUType.ST]

    # RtlModel
    rtl = RTLModel(debug=True, model_inst_list=inst_list)

    check = rtl.run(input_data=in_data, input_addr=rt_info.inputs.start, output_data=data_out, output_addr=rt_info.outputs.start)

    assert check

@pytest.mark.skip(reason="Not implemented yet")
def test_mm():
    data_in = np.random.randint(low=-128, high=127, size=[3, 4], dtype=np.int8)
    data_in = data_in.astype(np.float16)
    weight = np.random.randint(low=-128, high=127, size=[4, 5], dtype=np.int8)
    weight = weight.astype(np.float16)
    data_out = np.matmul(data_in, weight)
    data_in = data_in.reshape(3, 4)
    weight = weight.reshape(5, 4)

    g = StaticGraph()
    t_in = Tensor(Shape(3, 4), dtype=DataType.float16)
    t_in.set_data(data_in)

    w = Tensor(Shape(5, 4), dtype=DataType.float16)
    w.set_data(weight)

    v_mm = Matmul(g, name='test_mm')
    v_mm(t_in, w)

    g.complete()
    inst_collect = InstCollector()
    rt_info = ddr_address_alloc(g)
    in_data = pack_numpy_array([t.get_data() for t in  g.get_inputs() if t.get_data() is not None])

    lower(g, inst_collect)

    # import json
    # print(json.dumps(inst_collect.to_json(), indent=2))

    inst_list = [inst_collect.get_insts()]
    inst_list[0][-1].wait = [PUType.ST]

    # RtlModel
    rtl = RTLModel(debug=True, model_inst_list=inst_list)

    check = rtl.run(input_data=in_data, input_addr=rt_info.inputs.start, output_data=data_out, output_addr=rt_info.outputs.start)

    assert check
