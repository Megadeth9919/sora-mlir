from model import sora_model
from graph_ir import StaticGraph, Tensor, Shape, DataType, graph_compile, pack_numpy_array, CompileFlags, CCTarget
import argparse
from inst import *
from utils import *
from p_model import PModel
import tqdm
parser = argparse.ArgumentParser("sora compiler")
parser.add_argument('--output', type=str, default='to_rt')
args = parser.parse_args()

def getNPtype(t: DataType):
    match t:
        case DataType.float32:
            return np.float32
        case DataType.float16:
            return np.float16
        case DataType.int8:
            return np.int8
        case DataType.int16:
            return np.int16
        case _ :
            assert False

def mlp(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    
    input_shape = (2, 64, 1152)
    act_scale_shape = (2, 64, 1)
    mlp_ratio = 4

    input_x = Tensor(Shape(*input_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float16, name="act_scale"
    )
    input_x.set_act_scale(act_scale)
    gen_random_data([input_x, act_scale])
    

    model = sora_model.MLP(
        g,
        in_features=input_shape[-1],
        hidden_features=input_shape[-1] * mlp_ratio,
        stage="spatial",
        layer_idx=0,
        name="spatial_blocks.0.mlp",
    )

    model(input_x, out_dtype=DataType.float16)
    g.complete()
    gen_random_data(g.get_weights())

    dump_information(g, root_path, op_name="mlp", mode=mode)

def attention(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    
    input_shape = (2, 64, 1152)
    act_scale_shape = (2, 64, 1)


    input_x = Tensor(Shape(*input_shape), dtype=DataType.int8, name="input")
    act_scale = Tensor(
        Shape(*act_scale_shape), dtype=DataType.float16, name="act_scale"
    )
    input_x.set_act_scale(act_scale)
    gen_random_data([input_x, act_scale])

    hidden_size = 1152
    num_heads = 16
    qkv_bias = True
    qkv_norm = True
    norm_layer = "rms"
    rope = False
    stage = "spatial"
    layer_idx = 0
    
    model = sora_model.Attention(
        g,
        hidden_size,
        num_heads,
        qkv_bias,
        qkv_norm,
        norm_layer,
        rope,
        stage = stage,
        layer_idx = layer_idx,
        matmul_int=True
    )

    model(input_x, out_dtype=DataType.float16)
    g.complete()
    gen_random_data(g.get_weights())

    dump_information(g, root_path, op_name="attention", mode=mode)


def dit3_block(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    g = StaticGraph()
    
    hidden_size = 1152
    num_heads = 16
    mlp_ratio = 4
    rope = False
    temporal = False

    model = sora_model.STDiT3Block(
        graph=g,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        rope=rope,
        temporal=temporal,
        matmul_int=True
    )
    # one
    B, T, S = 2, 28, 405
    cond_size = 284
    
    # B, T, S = 2, 16, 144
    # cond_size = 52

    x = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16, name="x")
    y = Tensor(Shape(1, cond_size, hidden_size), dtype=DataType.float16, name="y")
    t = Tensor(Shape(B, hidden_size * 6), dtype=DataType.float16, name="t")
    cross_attn_mask = Tensor(Shape(1, 1, B * T * S, cond_size), dtype=DataType.float16)

    gen_random_data([x, y, t, cross_attn_mask])

    model(x=x, y=y, t=t, T=T, S=S, cross_attn_mask=cross_attn_mask)
    g.complete()
    gen_random_data(g.get_weights())
    gen_random_data(g.get_const())

    dump_information(g, root_path, op_name="dit3_block_real", mode=mode)

def load_data(path: str, g: StaticGraph):
    g.complete()
    for out  in g.get_outputs():
        print("output",out.to_json())
    for inp in g.get_inputs():
        print("input",inp.to_json())
    for cached in g.get_cached():
        print("cached",cached.to_json())
    for cached in g.get_cached():
        if cached in g.get_intermediate():
            print("duplicate tensor",cached.to_json())
    for intermidiate in g.get_intermediate():
        print("intermediate",intermidiate.to_json())
    for cts in g.get_const():
        print("cts",cts.to_json())
    with tqdm.tqdm(total=len(g.get_outputs()), desc='Load output') as pbar:
        for output in g.get_outputs():
            data_path = path + "/output/" + output.name + ".bin"
            data = np.fromfile(data_path, dtype=getNPtype(output.data_type))
            data = data.reshape(*output.shape)
            output.set_data(data)
            pbar.update(1)
    with tqdm.tqdm(total=len(g.get_intermediate()), desc='Load intermediate') as pbar1:        
        for inter in g.get_intermediate():
            data_path = path + "/inter_tensor/" + inter.name + ".bin"
            data = np.fromfile(data_path, dtype=getNPtype(inter.data_type))
            data = data.reshape(*inter.shape)
            inter.set_data(data)
            pbar1.update(1)
    with tqdm.tqdm(total=len(g.get_cached()), desc='Load cached') as pbar2:             
        for cached in g.get_cached():
            data_path = path + "/cache/" + cached.name + ".bin"
            data = np.fromfile(data_path, dtype=getNPtype(cached.data_type))
            data = data.reshape(*cached.shape)
            cached.set_data(data)
            pbar2.update(1)
# TODO: cond_size = 284/224/274
# TODO: T = 96/4+math.ceil(96/24)=48
def sora_only_block(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    mode.target = CCTarget.runtime
    
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
    # input_x = np.fromfile("/data1/shared/OpenSora/x0.bin", dtype=np.float16)
    input_x = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/x0.bin", dtype=np.float16)
    input_x = input_x.reshape(2, 28*405, 1152)
    # input_y = np.fromfile("/data1/shared/OpenSora/y0.bin", dtype=np.float16)
    input_y = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/y0.bin", dtype=np.float16)
    input_y = input_y.reshape(1, 224, 1152)
    # input_tmlp = np.fromfile("/data1/shared/OpenSora/t_mlp0.bin", dtype=np.float16)
    input_tmlp = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/t_mlp0.bin", dtype=np.float16)
    input_tmlp = input_tmlp.reshape(2, 6*1152)
    x.set_data(input_x)
    y.set_data(input_y)
    t_mlp.set_data(input_tmlp)
    model(x=x, y=y, t=t, t_mlp=t_mlp, T=T, S=S)
    # golden_path = "/data1/shared/OpenSora/layerGolden/"
    golden_path = "/home/fpga5/layerGolden/"
    load_data(golden_path, g)
    # dump_information(g, root_path, op_name="single_layer_random",call_pmode=True, mode=mode,dump=True, dump_json=False, dump_onnx=False,debug=False,fake_pmodel=True)

def sora_only_block_sparsification(root_path: str, mode: CompileFlags, dynamic_scale: bool = False):
    mode.target = CCTarget.runtime
    
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
    cond_size = 96

    # Three
    # B, T, S = 2, 28, 405
    # cond_size = 274
    

    x = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16, name="x_in")
    y = Tensor(Shape(1, cond_size, hidden_size), dtype=DataType.float16, name="y")
    t = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16, name="t")
    t_mlp = Tensor(Shape(B, 6 * hidden_size), dtype=DataType.float16, name="t_mlp")
    if not mode.compile_only:
        # input_x = np.fromfile("/data1/shared/OpenSora/x0.bin", dtype=np.float16)
        input_x = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/x0.bin", dtype=np.float16)
        input_x = input_x.reshape(2, 28*405, 1152)
        # input_y = np.fromfile("/data1/shared/OpenSora/y0.bin", dtype=np.float16)
        input_y = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/y0.bin", dtype=np.float16)
        input_y = input_y.reshape(1, 224, 1152)
        # input_tmlp = np.fromfile("/data1/shared/OpenSora/t_mlp0.bin", dtype=np.float16)
        input_tmlp = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/t_mlp0.bin", dtype=np.float16)
        input_tmlp = input_tmlp.reshape(2, 6*1152)
        x.set_data(input_x)
        y.set_data(input_y)
        t_mlp.set_data(input_tmlp)
    model(x=x, y=y, t=t, t_mlp=t_mlp, T=T, S=S)
    if not mode.compile_only:
        golden_path = "/data1/shared/OpenSora/pGolden/"
        load_data(golden_path, g)
        
    dump_information(g, root_path, op_name="single_layer_random",call_pmode=True, mode=mode,dump=not mode.compile_only, dump_json=False, dump_onnx=False,debug=False,fake_pmodel=True)
    g1 = StaticGraph()
    model1 = sora_model.STDiT3BlockOnly(
        graph=g1,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ration,
        depth=depth,
        re_compute=False,
        do_cache=False
    )
    if not mode.compile_only:
        # input_x = np.fromfile("/data1/shared/OpenSora/x0.bin", dtype=np.float16)
        input_x = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/x0.bin", dtype=np.float16)
        input_x = input_x.reshape(2, 28*405, 1152)
        # input_y = np.fromfile("/data1/shared/OpenSora/y0.bin", dtype=np.float16)
        input_y = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/y0.bin", dtype=np.float16)
        input_y = input_y.reshape(1, 224, 1152)
        # input_tmlp = np.fromfile("/data1/shared/OpenSora/t_mlp0.bin", dtype=np.float16)
        input_tmlp = np.fromfile("/home/fpga5/wangjiaqi/sora/01-22/input6/t_mlp0.bin", dtype=np.float16)
        input_tmlp = input_tmlp.reshape(2, 6*1152)
        x.set_data(input_x)
        y.set_data(input_y)
        t_mlp.set_data(input_tmlp)
    
    model1(x=x, y=y, t=t, t_mlp=t_mlp, T=T, S=S)
    g1.complete()
    if not mode.compile_only:
        for cached1 in g1.get_cached():
            for cached in g.get_cached():
                cached1.set_data(cached.get_data()) 
    dump_information(g1, root_path, op_name="single_layer_random_spars", call_pmode=True, sparsification=True, mode=mode,dump=not mode.compile_only, dump_json=False,dump_onnx=False,debug=False,complete=False,fake_pmodel=mode.compile_only)
    
if __name__ == '__main__':
    mode = CompileFlags()
    mode.target == CCTarget.verify
    mode.compile_only = True
    mode.enable_slr_slice = True
    # mode.optimization = 1
    # mlp(args.output, mode)
    # dit3_block(args.output, mode)
    # attention(args.output, mode)
    sora_only_block(args.output, mode)
    # sora_only_block_sparsification(args.output, mode)
