from model import sora_model
from .pass_mem_allocate import DynamicMemoryPool, StaticMemoryPool, graph_mem_allocate, getAddrTable, recoverTensorAddr
from graph_ir import *
from inst import *
from utils import *
import pytest


def test_memory_pool():
    pool = DynamicMemoryPool(0, 4096000)
    a = pool.allocate(10)
    print("a", a)
    b = pool.allocate(16)
    print("b", b)
    pool.free(a)
    c = pool.allocate(32)
    print("c", c)
    pool.free(b)
    d = pool.allocate(64)
    print("d", d)
    pool.free(c)
    pool.free(d)
    e = pool.allocate(100)
    print("e", e)


def test_sddit_mem_allocate(mode: int = 0):
    g = StaticGraph()
    
    hidden_size = 1152
    num_heads = 16
    mlp_ration = 4
    depth = 28

    model = sora_model.STDiT3BlockOnly(
        graph=g,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ration,
        depth=depth,
    )
    # One
    B, T, S = 2, 28, 405
    cond_size = 284

    x = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16, name="x_in")
    y = Tensor(Shape(1, cond_size, hidden_size), dtype=DataType.float16, name="y")
    t = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16, name="t")
    t_mlp = Tensor(Shape(B, 6 * hidden_size), dtype=DataType.float16, name="t_mlp")
    cross_attn_mask = Tensor(Shape(1, 1, B * T * S, cond_size), dtype=DataType.float16)

    model(x=x, y=y, t=t, t_mlp=t_mlp, T=T, S=S, cross_attn_mask=cross_attn_mask)
    g.complete()

    # file_path = "/home/fpga5/wangjiaqi/sora/golden_outputs/params_timestep_28.safetensors"
    # load_inputs_data(g.get_inputs(), file_path)

    graph_mem_allocate(g, 0)
    
    for input in g.get_inputs():
        assert input.addr is not None
    for output in g.get_outputs():
        assert output.addr is not None
    for weight in g.get_weights():
        assert weight.addr is not None
    for cst in g.get_const():
        assert cst.addr is not None
    for cached in g.get_cached():
        assert cached.addr is not None


def test_dynamic_memory_pool_allocate():
    pool = DynamicMemoryPool(0, 1024)
    addr1 = pool.allocate(256)
    assert addr1 == 0
    assert pool.Free[0].addr == 256
    assert pool.Free[0].size == 768

    addr2 = pool.allocate(512)
    assert addr2 == 256
    assert pool.Free[0].addr == 768
    assert pool.Free[0].size == 256

    addr3 = pool.allocate(256)
    assert addr3 == 768
    assert pool.Free[0].size == 0

    addr4 = pool.allocate(256)
    assert addr4 == None


def test_dynamic_memory_pool_allocate1():
    pool = DynamicMemoryPool(0, 1024)
    addr1 = pool.allocate(1234)

    assert addr1 == None


def test_dynamic_memory_pool_free():
    pool = DynamicMemoryPool(0, 1024)
    addr1 = pool.allocate(256)
    addr2 = pool.allocate(512)
    pool.free(addr1)

    assert len(pool.Free) == 2
    assert pool.Free[0].addr == 0
    assert pool.Free[0].size == 256
    assert pool.Free[1].addr == 768
    assert pool.Free[1].size == 256


def test_dynamic_memory_pool_free1():
    pool = DynamicMemoryPool(0, 1024)
    addr1 = pool.allocate(256)
    addr2 = pool.allocate(512)
    pool.free(addr2)

    assert len(pool.Free) == 1
    assert pool.Free[0].addr == 256
    assert pool.Free[0].size == 768


def test_dynamic_memory_pool_free2():
    pool = DynamicMemoryPool(0, 1024)
    addr1 = pool.allocate(256)
    addr2 = pool.allocate(512)

    with pytest.raises(ValueError, match="free 1024 not found"):
        pool.free(1024)


def test_dynamic_memory_pool_get_max_size():
    pool = DynamicMemoryPool(0, 1024)
    pool.allocate(256)
    pool.allocate(512)
    assert pool.getMaxSize() == 768
    pool.allocate(256)
    assert pool.getMaxSize() == 1024


def test_dynamic_memory_pool_merge():
    pool = DynamicMemoryPool(0, 1024)
    addr1 = pool.allocate(256)
    addr2 = pool.allocate(256)
    addr3 = pool.allocate(256)
    pool.free(addr2)
    pool.free(addr1)

    assert len(pool.Free) == 2
    pool.merge()
    assert len(pool.Free) == 2
    assert pool.Free[0].addr == 0
    assert pool.Free[0].size == 512
    assert pool.Free[1].addr == 768
    assert pool.Free[1].size == 256

    pool.free(addr3)
    assert len(pool.Free) == 1
    assert pool.Free[0].addr == 0
    assert pool.Free[0].size == 1024


def test_static_memory_pool_allocate():
    pool = StaticMemoryPool(0, 1024)
    addr1 = pool.allocate(256)
    assert addr1 == 0
    addr2 = pool.allocate(512)
    assert addr2 == 256
    addr3 = pool.allocate(256)
    assert addr3 == 768
    with pytest.raises(AssertionError):
        pool.allocate(256)


def test_static_memory_pool_get_size():
    pool = StaticMemoryPool(0, 1024)
    pool.allocate(256)
    pool.allocate(512)
    assert pool.get_size() == 768
    pool.allocate(256)
    assert pool.get_size() == 1024

# @pytest.mark.skip(reason="no permission to access the file")
# def test_add_load():
#     file_path = '/home/fpga5/sora-case/addr_table.txt'
#     log_data = loadTenorAddr(file_path)
#     assert len(log_data) == 5331
#     print(log_data)

def test_sddit_mem_recover(mode:int =0):
    g = StaticGraph()
    
    hidden_size = 1152
    num_heads = 16
    mlp_ration = 4
    depth = 28

    model = sora_model.STDiT3BlockOnly(
        graph=g,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ration,
        depth=depth,
    )
    # OLD
    # B, T, S = 2, 16, 144
    # cond_size = 52


    # One
    B, T, S = 2, 28, 405
    cond_size = 284


    # Two
    # B, T, S = 2, 28, 405
    # cond_size = 224

    # Three
    # B, T, S = 2, 28, 405
    # cond_size = 274

    x = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16, name="x_in")
    y = Tensor(Shape(1, cond_size, hidden_size), dtype=DataType.float16, name="y")
    t = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16, name="t")
    t_mlp = Tensor(Shape(B, 6 * hidden_size), dtype=DataType.float16, name="t_mlp")
    cross_attn_mask = Tensor(Shape(1, 1, B * T * S, cond_size), dtype=DataType.float16)

    model(x=x, y=y, t=t, t_mlp=t_mlp, T=T, S=S, cross_attn_mask=cross_attn_mask)
    g.complete()
    # file_path = "/home/fpga5/wangjiaqi/sora/golden_outputs/params_timestep_28.safetensors"
    # load_inputs_data(g.get_inputs(), file_path)
    # file_path = '/home/fpga5/sora-case/addr_table.txt'
    addr_tabel = getAddrTable().table
    recoverTensorAddr(g,addr_tabel)

    for input in g.get_inputs():
        assert input.addr is not None
    for output in g.get_outputs():
        assert output.addr is not None
    for weight in g.get_weights():
        assert weight.addr is not None
    for cst in g.get_const():
        assert cst.addr is not None
    for cached in g.get_cached():
        assert cached.addr is not None