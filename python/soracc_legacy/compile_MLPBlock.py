from model import *
import json
from p_model import PModel
# from rtl_model import RTLModel
from utils import *
import numpy as np
import yaml
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
    x_tensor = Tensor(name="x_tensor", shape=Shape(2, 7695, 1152), dtype=DataType.int8)
    x_tensor_act_scale = Tensor(name="y_tensor.act_scale", shape=Shape(2, 7695, 1), dtype=DataType.float16)

    x_tensor_act_scale_data = np.ones([2, 7695, 1]).astype(np.float16)
    x_tensor_data = np.random.randn(2, 7695, 1152).astype(np.int8)


    x_tensor.set_data(x_tensor_data)
    x_tensor_act_scale.set_data(x_tensor_act_scale_data)
    x_tensor.set_act_scale(x_tensor_act_scale)
    g = get_mlp(x_tensor=x_tensor)
    g.complete()
    print(json.dumps(g.to_json(), indent=2))
    inputs = [x_tensor]
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
    output = g.get_outputs()[0]
    output.set_data(output_list[0].data)
    
    dump_information(g, "cases/mlp", op_name="mlp")


def show_op_info():
    g = get_STDiT3()
    for op in g.get_ops():
        if isinstance(op, Matmul):
            print(op)

if __name__ == '__main__':
    main()
    # show_op_info()
