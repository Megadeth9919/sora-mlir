from model import *
import json
from p_model import PModel
# from rtl_model import RTLModel
from utils import *
import numpy as np

def getNPtype(w: Weight):
    match w.data_type:
        case DataType.float16:
            return np.float16
        case DataType.int8:
            return np.int8
        case DataType.int16:
            return np.int16
        case _ :
            assert False

def main():
    g = StaticGraph()

    T, S = 28, 405
    B = 2
    cond_size = 284
    hidden_size = 1152

    x_tensor = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16)
    y_tensor = Tensor(Shape(1, cond_size, hidden_size), dtype=DataType.float16)
    t_tensor = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16)
    t_mlp_tensor = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16)
    cross_attn_mask = Tensor(Shape(1, 1, B * T * S, cond_size), dtype=DataType.float16)
    # y_scale = Tensor(Shape(B, cond_size, 1), dtype=DataType.int8)

    
    x_tensor_data = np.fromfile("sora_pmodel_check/bin_int8/x_in.bin", dtype=np.float16).reshape(x_tensor.shape)
    y_tensor_data = np.fromfile("sora_pmodel_check/bin_int8/y.bin", dtype=np.float16).reshape(y_tensor.shape)
    t_mlp_tensor_data = np.fromfile("sora_pmodel_check/bin_int8/t_mlp.bin", dtype=np.float16).reshape(t_mlp_tensor.shape)
    t_tensor_data = np.fromfile("sora_pmodel_check/bin_int8/t.bin", dtype=np.float16).reshape(t_tensor.shape)
    cross_attn_mask_data = np.fromfile("sora_pmodel_check/cross_attn_mask.bin", dtype=np.float16).reshape(cross_attn_mask.shape)
    
    # x_norm1_data = np.fromfile("sora_pmodel_check/bin_int8/x_norm_1.bin", dtype=np.float16).reshape(x_tensor.shape)
    
    # y_tensor_data = np.random.randn(B, cond_size, hidden_size).astype(np.float16)
    # t_tensor_data = np.random.randn(B, 6, hidden_size).astype(np.float16) # unused now
    # t_mlp_tensor_data = np.random.randn(B, 6, hidden_size).astype(np.float16)
    # y_scale_data = np.random.randn(B, cond_size, 1).astype(np.int8)

    x_tensor.set_data(x_tensor_data)
    y_tensor.set_data(y_tensor_data)
    t_tensor.set_data(t_tensor_data)
    t_mlp_tensor.set_data(t_mlp_tensor_data)
    cross_attn_mask.set_data(cross_attn_mask_data)
    # y_scale.set_data(y_scale_data)
    # y_tensor.set_act_scale(y_scale)

    block = STDiT3BlockOnly(graph=g, hidden_size=hidden_size, num_heads=16, depth=1, matmul_int=True)

    out_tensor = block(x_tensor, y_tensor, t_tensor, t_mlp_tensor, T=T, S=S, cross_attn_mask=cross_attn_mask)
    g.complete()
    # with open("stdit3.json", "w") as fff:
    #     fff.write(json.dumps(g.to_json(), indent=2))
    #     print("dump graph")
    inputs = [x_tensor, y_tensor, t_tensor, t_mlp_tensor,]
    index = 0
    for t in g.get_inputs() :
        if t not in inputs:
            if len(t.shape) == 3:
                t_data = np.random.randn(t.shape[0],t.shape[1],t.shape[2]).astype(getNPtype(t))
                t.set_data(t_data)
            elif len(t.shape) == 4:
                t_data = np.random.randn(t.shape[0],t.shape[1],t.shape[2],t.shape[3]).astype(getNPtype(t))
                t.set_data(t_data)
            t.name = f'temp{index}'
            index += 1
    index = 0
    for output in g.get_outputs():
        output.name = "output{index}"
        if len(output.shape) == 3:
            o_data = np.random.randn(output.shape[0],output.shape[1],output.shape[2]).astype(getNPtype(output))
            output.set_data(o_data)
        elif len(output.shape) == 4:
            o_data = np.random.randn(output.shape[0],output.shape[1],output.shape[2],output.shape[3]).astype(getNPtype(output))
            output.set_data(o_data)
        index += 1
    index = 0

    for weight in g.get_weights():
        weight.name = "weight{index}"
        if len(weight.shape) == 1:
            w_data = np.random.randn(weight.shape[0]).astype(getNPtype(weight))
            weight.set_data(w_data)
        elif len(weight.shape) == 2:
            w_data = np.random.randn(weight.shape[0],weight.shape[1]).astype(getNPtype(weight))
            weight.set_data(w_data)
        elif len(weight.shape) == 3:
            w_data = np.random.randn(weight.shape[0],weight.shape[1],weight.shape[2]).astype(getNPtype(weight))
            weight.set_data(w_data)
        elif len(weight.shape) == 4:
            w_data = np.random.randn(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]).astype(getNPtype(weight))
            weight.set_data(w_data)
        index += 1
    # print(json.dumps(g.to_json(), indent=2))

    p = PModel(graph=g,param_path="/data1/shared/OpenSora/model_quant.safetensors")
    # c: RTLModel = RTLModel(cfg=cfg, graph=g)
    
    #input_list = [x_tensor_data, y_tensor_data, t_tensor_data, ]
    output_list = p.run()
    output = g.get_outputs()[-1]
    output.set_data(output_list[0].data)
    
    dump_information(g, "cases", op_name="stdit3")


def show_op_info():
    g = get_STDiT3()
    for op in g.get_ops():
        if isinstance(op, Matmul):
            print(op)

if __name__ == '__main__':
    main()
    # show_op_info()
