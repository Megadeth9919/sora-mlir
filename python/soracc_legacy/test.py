from profiler import test_profiler
from p_model import PModel
from model import model_base, sora_model, StaticGraph, Tensor, Shape, DataType
from utils import gen_random_data, dump_information
from inst import *
from graph_ir import *
import yaml

if __name__ == '__main__':
    # path = "ddr_output2.yaml"
    # with open(path, "r") as f:
    #     data = yaml.load(f, Loader=yaml.FullLoader)
    
    # for _i in data:
    #     name = _i["name"]
    #     name = name.split("/")[-1].replace("fakeload_", "").replace("_02-08", "")
    #     cycle = _i["load_wgt_ch0_cycle"]
    #     print(f"{name} {cycle}")

    test_profiler.profiler_OnlySTDiT3Block_demo()
    # test_profiler.profiler_matmul_softmax_demo()
    # test_profiler.test_profiler_softmax()
    # g = StaticGraph()
    # in_shape = (1, 64, 4096)
    # act_scale_shape = (1, 64, 1)
    # weight_shape = (260, 4096)
    # weight_scale_shape = 260
    # bias_shape = 260

    # input_x = Tensor(Shape(*in_shape), dtype=DataType.int8, name="input")
    # act_scale = Tensor(
    #     Shape(*act_scale_shape), dtype=DataType.float16, name="act_scale", const=True
    # )

    # model = model_base.Linear(
    #     g,
    #     in_feature=weight_shape[-1],
    #     out_feature=weight_shape[0],
    #     bias=True,
    #     name="linearw8",
    # )

    # model(input_x, out_dtype=DataType.float16, act_scale=act_scale)
    # gen_random_data([input_x, act_scale])
    # gen_random_data(g.get_weights())

    # p = PModel(graph=g)
    # output_list = p.run()

    # dump_information(g, "./cases", op_name="linearw8", mode=CompileFlags())