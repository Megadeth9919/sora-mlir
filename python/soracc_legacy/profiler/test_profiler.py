from graph_ir.graph_ir import DataType, Shape, Tensor
from model import sora_model
from graph_ir.pass_impl import CompileFlags, graph_compile, CCTarget
from model.model_base import *
# from model.sora_model import get_mlp, get_single_STDiT3Block, get_STDiT3, get_single_OnlySTDiT3Block


from .profiler import InstProfiler


def profiler_op_demo():
    # step1: create a static graph
    x_tensor = Tensor(shape=Shape(31, 19, 1152), dtype=DataType.float16)  
    
    g = StaticGraph()  # 1 create a static graph

    # step2: create an op
    op = Convert(graph = g, out_dtype=DataType.int8)
    op(x_tensor)

#     # step3: complete the graph
#     g.complete()
#     
    
#     # step4: compile the graph
#     f = CompileFlags()
#     f.enable_gen_golden = False
#     f.enable_slr_slice = False
#     _, inst = graph_compile(g, f)

#     # step5: profile the instructions
#     profiler = InstProfiler(inst)  
#     profiler.run()


def test_profiler_softmax(dynamic_scale: bool = False):
    g = StaticGraph()
    

    input_x = Tensor(Shape(1,24,15120,28), dtype=DataType.float16, name="input")

    model = Softmax(g, name="softmax")

    if dynamic_scale:
        output_dtype = DataType.int8
    else:
        output_dtype = DataType.float16

    model(input_x, out_dtype=output_dtype)
    g.complete()
    
    
    # step4: compile the graph
    f = CompileFlags()
    f.enable_gen_golden = False
    f.enable_slr_slice = False
    _, inst = graph_compile(g, f)

    # step5: profile the instructions
    profiler = InstProfiler(inst)  
    profiler.run()


def profiler_OnlySTDiT3Block_demo():
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
        do_cache=True,
        re_compute=True,
    )

    g1 = StaticGraph()
    model1 = sora_model.STDiT3BlockOnly(
        graph=g1,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ration,
        depth=depth,
        do_cache=False,
        re_compute=False,
    )
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
    model1(x=x, y=y, t=t, t_mlp=t_mlp, T=T, S=S)

    f = CompileFlags()
    f.target = CCTarget.runtime
    f.optimization = 1
    g.complete()
    _, inst, _ = graph_compile(g, f, debug=False)

    profiler = InstProfiler(inst, task_name="STDiT3BlockOnly_B-{}_T-{}_S-{}_cond-{}".format(B, T, S, cond_size))
    profiler.run(dump_breakdown=False, dump_txt=True)
    
    g1.complete()
    _, inst, _ = graph_compile(g1, f, sparsification=True)

    profiler = InstProfiler(inst)
    profiler.run(dump_breakdown=False, dump_txt=True)


def profiler_softmax_demo():
    g = StaticGraph()
    
    input_x = Tensor(Shape(56, 16, 405, 405), dtype=DataType.float16, name="input")

    model = Softmax(g, name="softmax")

    model(input_x, out_dtype=DataType.float16)
    
    f = CompileFlags()
    f.enable_gen_golden = False
    f.enable_slr_slice = False
    _, inst = graph_compile(g, f)

    profiler = InstProfiler(inst)
    profiler.run(dump_breakdown=False)

def profiler_matmul_softmax_demo():
    g = StaticGraph()
    
    hidden_size = 1152
    num_heads = 16

    matmul = Matmul(g, name="qk_matmul")
    softmax = Softmax(g, name="softmax")
    
    # One
    B, T, S = 2, 28, 405
    cond_size = 284

    # Two
    # B, T, S = 2, 28, 405
    # cond_size = 224

    # Three
    # B, T, S = 2, 28, 405
    # cond_size = 274

    x = Tensor(Shape(B * T, num_heads, S, hidden_size // num_heads), dtype=DataType.float16, name="x")
    y = Tensor(Shape(B * T, num_heads, S, hidden_size // num_heads), dtype=DataType.float16, name="y")
    
    x_int = Convert(graph=g, out_dtype=DataType.int8)(x)
    y_int = Convert(graph=g, out_dtype=DataType.int8)(y)
    tem = matmul(x_int, y_int, out_dtype=DataType.float16)
    softmax(tem, out_dtype=DataType.int8)

    f = CompileFlags()
    f.enable_gen_golden = False
    f.enable_slr_slice = False
    _, inst = graph_compile(g, f)

    profiler = InstProfiler(inst, task_name="Matmul_Softmax_B-{}_T-{}_S-{}_cond-{}".format(B, T, S, cond_size))
    profiler.run(dump_breakdown=True, dump_txt=True)