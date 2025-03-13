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



def sora_only_block(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    
    g = StaticGraph()
    
    hidden_size = 1152
    num_heads = 16
    mlp_ration = 4
    depth = 1

    model = sora_model.STDiT3BlockOnly(
        graph=g,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ration,
        depth=depth,
        re_compute=True,
        do_cache=True
    )
    # OLD
    # B, T, S = 2, 16, 144
    # cond_size = 52


    # One
    # B, T, S = 2, 28, 405
    # cond_size = 284


    # Two
    B, T, S = 2, 28, 405
    cond_size = 224

    # Three
    # B, T, S = 2, 28, 405
    # cond_size = 274
    

    x = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16, name="x_in")
    y = Tensor(Shape(1, cond_size, hidden_size), dtype=DataType.float16, name="y")
    t = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16, name="t")
    t_mlp = Tensor(Shape(B, 6 * hidden_size), dtype=DataType.float16, name="t_mlp")
    model(x=x, y=y, t=t, t_mlp=t_mlp, T=T, S=S)
    
    g.complete()
    converter = SoraLegacyConverter(g)
    converter.generate_mlir(root_path + '/output.mlir')
    
mode = CompileFlags()  # mode = 1 for separate dump file
mode.target = CCTarget.verify
sora_only_block(root_path='case', mode=mode, dynamic_scale=True)