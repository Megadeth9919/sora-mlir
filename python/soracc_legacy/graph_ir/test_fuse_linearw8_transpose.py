from model import sora_model
from .pass_mem_allocate import DynamicMemoryPool, StaticMemoryPool, graph_mem_allocate, getAddrTable, recoverTensorAddr
from .pass_fuse_linearw8_transpose import fuse_linearw8_transpose
from .StatisticPass import Statistic_Pass
from graph_ir import *
from inst import *
from utils import *
import pytest
def test_fuse_misc_cvt():
    g = StaticGraph()
    hidden_size = 1152
    num_heads = 16
    mlp_ration = 4
    depth = 2

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
    cross_attn_mask = Tensor(Shape(1, 1, B * T * S, cond_size), dtype=DataType.float16, name="cross_attn_mask")
    gen_random_data([cross_attn_mask])

    model(x=x, y=y, t=t, t_mlp=t_mlp, T=T, S=S, cross_attn_mask=cross_attn_mask)
    g.complete()
    with open("before_fuse.json", "w") as f:
        json.dump(g.to_json(), f, indent=4)
    fuse_linearw8_transpose(g)
    with open("after_fuse.json", "w") as f:
        json.dump(g.to_json(), f, indent=4)