from .graph_ir import StaticGraph, Tensor, LinearW8, Shape
from .pass_impl import ddr_address_alloc, pack_numpy_array
from .pass_mem_allocate import graph_mem_allocate ,ddr_address_alloc
from .pass_mem_allocate import RtInfo
import numpy as np
from dataclasses import asdict
import json
from utils import hw_info

def test_addr_assign():
    g = StaticGraph()
    op = LinearW8('linear_w8')
    x = Tensor(Shape(2, 3, 4), name='x')
    op.add_input_tensors((x, ))
    new_shape = Shape(3, 4, 5)
    y = op.create_tensor(new_shape, name='y')
    g.add(op)

    g.complete()
    ddr_address_alloc(g)

    assert x.addr == 0
    assert y.addr == hw_info.high_align(48, 64)


def test_pack_numpy_array():
    a = np.zeros([2, 3], dtype=np.float16)
    a.fill(1)

    b= np.zeros([2, 4], dtype=np.int8)
    b.fill(2)

    ret = pack_numpy_array([a, b])

    print(ret)


def test_rt_info():
    rt_info = RtInfo()
    # print(rt_info.serilize())
