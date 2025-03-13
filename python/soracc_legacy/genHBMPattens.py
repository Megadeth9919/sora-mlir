from graph_ir import *
from inst import *
import numpy as np
import sys
import pytest
from p_model import PModel
from utils import dump_information, gen_random_data
from model import model_base, sora_model


def gen_load_patten(root_path, batch, k,stride, mode: CompileFlags = CompileFlags(), dynamic_scale: bool = False):
    g = StaticGraph()
    model = model_base.FakeLoad(g, "fakeload",stride)

    input_x = Tensor(Shape( batch, k), dtype=DataType.int8, name="input")
    gen_random_data([input_x])

    output_dtype = DataType.int8
    op_name = f"fakeload_{batch}_{k}_{stride}"
    
    model(input_x, out_dtype=output_dtype)
    g.complete()
    output = g.get_outputs()[0]
    
    gen_random_data([output])

    dump_information(g, root_path, call_pmode= False,op_name=op_name, mode=mode)
batches=[16,128]
ks=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,256<<10,512<<10,1<<20,2<<20,4<<20]
strides =[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,256<<10,512<<10,1<<20,2<<20,4<<20,8<<20]
for batch in batches:
    for idx,k in enumerate(ks):
        gen_load_patten("cases",batch,k,strides[idx])

fixks=[4<<10,64<<10]
fixstrides=[0,1,2,4,8,16,32,64,128,256,512,1<<10,2<<10,4<<10,8<<10,16<<10,32<<10,64<<10,128<<10,256<<10,512<<10,1<<20,2<<20,4<<20,8<<20,16<<20,32<<20,64<<20,128<<20]
for k in fixks:
    for stride in fixstrides:
        gen_load_patten("cases",64,k,stride)

# gen_load_patten("case",1,7<<20,7<<20)