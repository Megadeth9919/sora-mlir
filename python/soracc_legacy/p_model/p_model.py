# -*-coding:utf-8-*-
import datetime
import os
import numpy as np

from typing import List, Tuple, Dict
import logging
import math

from graph_ir import *
from utils import tools
import time
import shutil
import tqdm

logger = logging.getLogger(__name__)


def check_subnormal_fp_value(data, data_name, data_type):
    assert data_type in ("float16", "float32")
    if data.shape == ():
        if np.abs(data) >= np.finfo(data_type).smallest_normal or data == np.array(0, dtype=data_type):
            pass
        else:
            logger.warning(f"Subnormal number in {data_name}!")
    else:
        subnormal_num = np.sum(np.abs(data[np.nonzero(data)]) < np.finfo(data_type).smallest_normal)
        if subnormal_num > 0:
            logger.warning(f"{subnormal_num} subnormal {data_type} number in {data_name}!")


def set_subnormal_fp_value_to_zero(data, dtype):
    assert data.dtype == np.dtype(dtype)
    if data.shape == ():
        if np.abs(data) < np.finfo(dtype).smallest_normal:
            data = np.array(0, dtype=dtype)
    else:
        data[np.abs(data) < np.finfo(dtype).smallest_normal] = 0
    return data

def set_subnormal_fp_value_to_zero_convert(data, dtype):
    assert data.dtype == np.dtype(dtype)
    if data.shape == ():
        if np.abs(data) < np.finfo(dtype).smallest_normal:
            data = np.array(0, dtype=dtype)
    else:
        data[np.abs(data) < np.finfo(dtype).smallest_normal] = np.finfo(dtype).smallest_normal.astype(dtype)
    return data

def assert_no_subnormal_fp_value(data, dtype):
    if data.shape == ():
        assert np.abs(data) >= np.finfo(dtype).smallest_normal or data == np.float16(0)
    else:
        pass


def quant_output_to_int(output_tensor, out_int_scaling_factor, output_dtype):
    # 输出fp16量化成int8
    assert output_tensor.dtype == np.float32
    assert out_int_scaling_factor.dtype == np.float32
    assert output_dtype in ("int8", "int16")
    check_subnormal_fp_value(out_int_scaling_factor, "out_int_scaling_factor", "float32")
    out_int_scaling_factor = set_subnormal_fp_value_to_zero_convert(out_int_scaling_factor, "float32")
    assert_no_subnormal_fp_value(out_int_scaling_factor, "float32")
    output_tensor = set_subnormal_fp_value_to_zero_convert(output_tensor, "float32")
    output_tensor = output_tensor * out_int_scaling_factor
    output_tensor = set_subnormal_fp_value_to_zero_convert(output_tensor, "float32")
    round_tensor = np.round(output_tensor)
    if output_dtype == "int8":
        output_tensor = np.clip(round_tensor,
                                -127,  # np.iinfo(np.int8).min,
                                np.iinfo(np.int8).max).astype("int8")
    elif output_dtype == "int16":
        output_tensor = np.clip(round_tensor,
                                -32767,
                                np.iinfo(np.int16).max).astype("int16")
    else:
        raise ValueError
    return output_tensor


def quantized_matmul(activation, weight, activation_scale, weight_scale, matmul_dtype):
    if matmul_dtype == "w8a8":
        assert activation.dtype == np.int8
        assert weight.dtype == np.int8
        # assert np.min(activation) >= -127
    elif matmul_dtype == "w8a8_dequant":
        assert activation.dtype == np.int8
        assert weight.dtype == np.int8
    elif matmul_dtype == "w8a8_attn":
        assert activation.dtype == np.int8
        assert weight.dtype == np.int8
    elif matmul_dtype == "w8a8_attn_dequant":
        assert activation.dtype == np.int8
        assert weight.dtype == np.int8
    elif matmul_dtype == "w8a16":
        assert activation.dtype == np.int16
        assert weight.dtype == np.int8
    elif matmul_dtype == "w16a16":
        assert activation.dtype == np.int16
        assert weight.dtype == np.int16
    else:
        raise ValueError
    input_tensor = activation.astype("int32")
    weight = weight.astype("int32")
    assert_no_subnormal_fp_value(activation_scale, "float16")
    assert_no_subnormal_fp_value(weight_scale, "float16")
    
    if matmul_dtype == "w8a8_attn":
        output_tensor = np.matmul(input_tensor, weight).astype("float32")
        output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float32")
        scale = weight_scale.transpose(0, 1, 3, 2).astype("float32") * activation_scale.astype("float32")
        assert_no_subnormal_fp_value(scale, "float32")
        output_tensor = output_tensor.astype("float32") * scale.astype("float32")
        output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float32")
        output_tensor = output_tensor.astype("float16")
        return output_tensor
    
    elif matmul_dtype == "w8a8_dequant":
        input_tensor = activation.astype("float16") * activation_scale.astype("float16")
        weight = weight.astype("float16") * weight_scale.astype("float16")
        output_tensor = np.matmul(input_tensor, weight.swapaxes(-1,-2)).astype("float32")
        output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float32")
        output_tensor = output_tensor.astype("float16")
        return output_tensor

    elif matmul_dtype == "w8a8_attn_dequant":
        input_tensor = activation.astype("float16") * activation_scale.astype("float16")
        weight = weight.astype("float16") * weight_scale.astype("float16").transpose(0, 1, 3, 2)
        output_tensor = np.matmul(input_tensor, weight).astype("float32")
        return output_tensor
    
    else:
        output_tensor = np.matmul(input_tensor, weight.swapaxes(-1,-2)).astype("float32")
        output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float32")
        scale = np.matmul(activation_scale.astype("float32"), weight_scale.reshape(1, -1).astype("float32"))
        assert_no_subnormal_fp_value(scale, "float32")
        output_tensor = output_tensor.astype("float32") * scale.squeeze().astype("float32")
        output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float32")
        output_tensor = output_tensor.astype("float16")
        return output_tensor
      

def hardware_float_eltwise_add_mul(tensor_a, tensor_b, eltwise_type):
    assert eltwise_type in ("add", "mul")
    assert tensor_a.shape[-1] == tensor_b.shape[-1]
    assert tensor_a.dtype == np.float16
    assert tensor_b.dtype == np.float16
    tensor_a = set_subnormal_fp_value_to_zero(tensor_a, "float16")
    tensor_b = set_subnormal_fp_value_to_zero(tensor_b, "float16")
    if eltwise_type == "add":
        output_tensor = tensor_a.astype("float32") + tensor_b.astype("float32")
    elif eltwise_type == "mul":
        output_tensor = tensor_a.astype("float32") * tensor_b.astype("float32")
    else:
        raise ValueError
    output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float32")
    output_tensor = output_tensor.astype("float16")
    output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float16")
    return output_tensor


class PModel:
    def __init__(
            self,
            graph: Optional[StaticGraph] = None,
            ir_path: Optional[str] = None,
            param_path: Optional[str] = None,
    ):
        # Check input
        if graph is None:
            raise ValueError("Graph should be provided. Have not supported loading from file yet.")
        if graph is None and ir_path is None:
            raise ValueError("Either graph or ir_path should be provided")
        if ir_path is not None:
            assert os.path.isfile(ir_path), "IR file not found"

        # Path config
        # self.base_dir = "./.output"
        # if not os.path.exists(self.base_dir):
        #     os.mkdir(self.base_dir)
        #
        # self.output_dir = os.path.join(self.base_dir, "pmodel_output")
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)
        # os.mkdir(self.output_dir)
        #
        # os.mkdir(os.path.join(self.output_dir, "output"))
        # os.mkdir(os.path.join(self.output_dir, "param"))
        #
        # input_data_dir = os.path.join(self.base_dir, "ir_output", "input")
        # input_param_dir = os.path.join(self.base_dir, "ir_output", "param")
        # self.input_param_dir = input_param_dir
        # self.input_data_dir = input_data_dir

        # IR config
        self.ir: StaticGraph = graph
        self.dynamic_scale_reg = None

        # Param config
        self.param_path = param_path
        self.params: Dict[str, Tensor] = {}
        
        self.pmodel_log_dir = "pmodel.log"
        if os.path.exists(self.pmodel_log_dir):
            os.remove(self.pmodel_log_dir)

    def dump_inter_tensor(self, output_dir: str):
        inter_tensor_dir = os.path.join(output_dir, "inter_tensor")
        os.mkdir(inter_tensor_dir)
        for _, tensor in enumerate(self.ir.get_intermediate()):
            tensor_name = tensor.name
            tensor_data = tensor.data
            tensor_data.tofile(os.path.join(inter_tensor_dir, tensor_name+".bin"))

    def dump_cached_tensor(self, output_dir: str):
        cache_dir = os.path.join(output_dir, "cache")
        os.mkdir(cache_dir)
        for _, tensor in enumerate(self.ir.get_cached()):
            tensor_name = tensor.name
            tensor_data = tensor.data
            tensor_data.tofile(os.path.join(cache_dir, tensor_name+".bin"))
    
    def dump_output_tensor(self, output_dir: str):
        output_tensor_dir = os.path.join(output_dir, "output")
        os.mkdir(output_tensor_dir)
        for _, tensor in enumerate(self.ir.get_outputs()):
            tensor_name = tensor.name
            tensor_data = tensor.data
            tensor_data.tofile(os.path.join(output_tensor_dir, tensor_name+".bin"))

    def param_init(self):
        # Check input
        if self.param_path is None:
            return
        else:
            assert os.path.isfile(self.param_path), "Param file not found"
            assert self.param_path.endswith(".safetensors"), "Param file should be a safetensors file"
            self.params = tools.safetensors_metadata_parser(self.param_path)

        # Load param and scale to IR
        for _, const_tensor in enumerate(self.ir.get_const()):
            if const_tensor.data is None and isinstance(const_tensor, Tensor):
                const_name = const_tensor.name
                if const_name in self.params:
                    target_shape = tuple(i for i in const_tensor.shape if i != 1)
                    real_shape = np.squeeze(self.params[const_name]).shape
                    assert target_shape == real_shape, f"Shape of {const_name} in graph is {target_shape}, but shape of {const_name} in param file is {real_shape}"
                    
                    const_tensor.set_data(self.params[const_name].reshape(const_tensor.shape))
                else:
                    raise ValueError(f"Param {const_name} not found in param file")

        for _, layer_define in enumerate(self.ir):
            if isinstance(layer_define, LinearW8):
                weight_name = layer_define.name + ".weight"
                weight_scale_name = layer_define.name + ".weight_quantizer"
                
                weight, weight_scale = layer_define.get_weight
                weight.set_data(self.params[weight_name])

                target_shape = tuple(i for i in weight_scale.shape if i != 1)
                real_shape = np.squeeze(self.params[weight_scale_name]).shape
                assert target_shape == real_shape, f"Shape of {const_name} in graph is {target_shape}, but shape of {const_name} in param file is {real_shape}"
                
                weight_scale.set_data(self.params[weight_scale_name].reshape(weight_scale.shape))

                if layer_define.bias_flag:
                    bias_name = layer_define.name + ".bias"
                    bias = layer_define.get_bias
                    bias_data = self.params[bias_name].copy()
                    bias.set_data(bias_data)

            elif isinstance(layer_define, Layernorm):
                if layer_define.affine:
                    gamma_name = layer_define.name + ".weight"
                    beta_name = layer_define.name + ".bias"

                    gamma, beta = layer_define.get_gamma, layer_define.get_beta
                    gamma.set_data(self.params[gamma_name])
                    beta.set_data(self.params[beta_name])
                
                else:
                    gamma_data = np.ones(shape=(1, layer_define._inputs[0].shape[-1],), dtype=np.float16)

                    gamma = layer_define.get_gamma
                    gamma.set_data(gamma_data)

            elif isinstance(layer_define, RMSnorm):
                weight_name = layer_define.name + ".weight"

                weight = layer_define.get_weight
                weight.set_data(self.params[weight_name].reshape(1, -1))
            elif isinstance(layer_define, RoPE):
                cos_sin_table = layer_define.get_cos_sin_table()
                cos_sin_table_data = self.params["rope.cos_sin_table"]
                assert cos_sin_table_data.shape[-1] == cos_sin_table.shape[-1]
                seq_len = cos_sin_table.shape[-2]
                cos_sin_table_data = cos_sin_table_data[None, None, :seq_len, :]
                cos_sin_table.set_data(cos_sin_table_data)
            else:
                continue

    def run(self,
            input_data: Optional[List[np.ndarray]] = None,
            cache_data: Optional[List[np.ndarray]] = None,
            dump_path: Optional[str] = None,
            dump_inter_flag: bool = True,
            dump_cached_flag: bool = True,
            fake_pmodel:bool = False
            ) -> Tuple[Tensor]:
        """
            根据网路结构和网络参数计算每一层的输出，输入一般可以设置为随机的
        """
        self.param_init()
        if fake_pmodel :
            return 
        if input_data is not None:
            for idx, input_tensor in enumerate(self.ir.get_inputs()):
                input_tensor.set_data(input_data[idx])
        else:
            for _, input_tensor in enumerate(self.ir.get_inputs()):
                assert input_tensor.data is not None, "Input data should be provided"
        
        if cache_data is not None:
            for idx, cached_tensor in enumerate(self.ir.get_cached()):
                cached_tensor.set_data(cache_data[idx])

        with tqdm.tqdm(total=len(self.ir), desc='PModel') as pbar:
            for idx, layer_define in enumerate(self.ir):
                logger.info("Start to run p model...")
                bar = [i for i in range(0, len(self.ir), max(len(self.ir) // 10 , 1))]
                if idx in bar:
                    finished_ratio = math.floor(idx / len(self.ir) * 20)
                    logger.info("=" * finished_ratio + ">" + " " * (20 - finished_ratio) + f" {finished_ratio // 20}%")

                # Set input tensor
                # if idx == 0:
                #     layer_define.set_inputs(input_data)

                # Forward to each layer
                # Step 0(Optional): Load weights
                # Step 1: Forward to each layer
                # Step 2: Refresh/Save output tensor
                #  ——————      ——————             ——————
                # | op_1 | -> | op_2 | -> ... -> | op_n |
                #  ——————      ——————             ——————
                #     ↑   ....    ↓   Refresh previous output tensor
                print(f"Pmodel for layer type {layer_define.op_type} {layer_define.name}")
                if isinstance(layer_define, LinearW8):
                    output_tensor_list = self.linear_forward(layer_define)
                elif isinstance(layer_define, Matmul):
                    output_tensor_list = self.matmul_forward(layer_define)
                elif isinstance(layer_define, (Conv2D, Conv3D)):
                    output_tensor_list = self.conv_forward(layer_define)
                elif isinstance(layer_define, RoPE):
                    output_tensor_list = self.rotary_forward(layer_define)
                elif isinstance(layer_define, Softmax):
                    output_tensor_list = self.softmax_forward(layer_define)
                elif isinstance(layer_define, Layernorm):
                    output_tensor_list = self.layernorm_forward(layer_define)
                elif isinstance(layer_define, RMSnorm):
                    output_tensor_list = self.layernorm_forward(layer_define)
                elif isinstance(layer_define, Eltwise):
                    output_tensor_list = self.eltwise_forward(layer_define)
                elif isinstance(layer_define, Div):
                    output_tensor_list = self.div_forward(layer_define)
                # elif isinstance(layer_define, Op):
                #     output_tensor_list = self.concat_forward(layer_define)
                elif isinstance(layer_define, Silu):
                    output_tensor_list = self.silu_forward(layer_define)
                elif isinstance(layer_define, Gelu):
                    output_tensor_list = self.gelu_forward(layer_define)
                elif isinstance(layer_define, Transpose):
                    output_tensor_list = self.transpose_forward(layer_define)
                elif isinstance(layer_define, View):
                    output_tensor_list = self.view_forward(layer_define)
                elif isinstance(layer_define, Split):
                    output_tensor_list = self.split_forward(layer_define)
                elif isinstance(layer_define, Convert):
                    output_tensor_list = self.convert_forward(layer_define)
                elif isinstance(layer_define, LoadInst):
                    pass
                elif isinstance(layer_define, Copy):
                    output_tensor_list = self.copy_forward(layer_define)
                elif isinstance(layer_define, FakeLoad):
                    pass
                else:
                    raise ValueError(f'unsupport layer: {layer_define}')

                # Update output tensor
                for i, output_tensor in enumerate(layer_define.get_outputs()):
                    output_tensor.set_data(output_tensor_list[i])
                
                # with open("a.log", "a") as a_log:
                #     for i, output_tensor in enumerate(layer_define.get_outputs()):
                        
                #         a_log.write(f"{output_tensor.name}\n")
                #         a_log.write(str(output_tensor) + "\n")
                #         a_log.write(str(output_tensor.data) + "\n\n")

                # Update pbar
                pbar.update(1)
            if dump_path is None:
                dump_path = "./.output"
                if not os.path.exists(dump_path):
                    os.mkdir(dump_path)
                
            assert os.path.exists(dump_path), f"Output dir {dump_path} not found"
            case_dir = os.path.join(dump_path, "pmodel_output_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"{datetime.datetime.now().microsecond // 1000:03d}"))
            os.mkdir(case_dir)
            
            if dump_inter_flag:
                self.dump_inter_tensor(case_dir)
            
            if dump_cached_flag:
                self.dump_cached_tensor(case_dir)
            
            self.dump_output_tensor(case_dir)

        return

    @staticmethod
    def copy_forward(layer_define: Copy):
        input_tensor = layer_define.get_inputs()[0]
        # output_tensor = layer_define.get_outputs()[0]
        # output_tensor.set_data(input_tensor.data)
        return [input_tensor.data, ]
   
    @staticmethod
    def conv_forward(layer_define: Union[Conv2D, Conv3D]):
        input_tensor = layer_define.get_inputs()
        weight = layer_define.get_weights()
        bias = layer_define.get_weights()
        # TODO add stride attribute
        # stride = layer_define["structure"]["stride"]

        return [input_tensor, ]

    def mask_forward(self, layer_define: Op):
        raise NotImplementedError

    @staticmethod
    def linear_forward(layer_define: LinearW8):
        activation_dtype = layer_define.get_feature.data_type
        weight_dtype = layer_define.get_weight[0].data_type
        bias_flag = layer_define.get_bias is not None

        if weight_dtype == DataType.unit4:  # w4a8 linear
            raise NotImplementedError

        elif weight_dtype in (DataType.int8, DataType.int16):  # w8a8/w8a16 linear or w8a8 attention
            if activation_dtype == DataType.int8 and weight_dtype == DataType.int8:
                # matmul_dtype = "w8a8_dequant"
                matmul_dtype = "w8a8"
            elif activation_dtype == DataType.int16 and weight_dtype == DataType.int8:
                matmul_dtype = "w8a16"
            elif activation_dtype == DataType.int16 and weight_dtype == DataType.int16:
                matmul_dtype = "w16a16"
            else:
                raise ValueError
            # weight_scaling_factor_duplicate = layer_define._weights["weight_scale"]

            weight_scaling_factor = layer_define.get_weight[1].data
            assert_no_subnormal_fp_value(weight_scaling_factor, "float16")

            # assert (weight_scaling_factor == weight_scaling_factor_duplicate).all()

            if layer_define.get_act_scale.data is not None:
                activation_scaling_factor = layer_define.get_act_scale.data.astype("float16")
            else:
                raise ValueError("Activation scale should be provided")
            # mypy check
            assert layer_define.get_feature.data is not None
            input_tensor = layer_define.get_feature.data.astype(np.int8)
            weight = layer_define.get_weight[0].data
            tensor = quantized_matmul(input_tensor, weight, activation_scaling_factor, weight_scaling_factor, matmul_dtype)

        elif weight_dtype in (DataType.float16, "bfloat16"):
            raise NotImplementedError

        else:
            raise ValueError

        if bias_flag:
            assert layer_define.get_bias is not None
            bias = layer_define.get_bias.data
            set_subnormal_fp_value_to_zero(bias, "float16")
            
            tensor = tensor + bias

        tensor = set_subnormal_fp_value_to_zero(tensor, "float16")
        return [tensor, ]

    @staticmethod
    def matmul_forward_fp_version(layer_define: Matmul):
        # 输入检查

        # mypy check
        assert layer_define.get_matrix_A.data is not None
        assert layer_define.get_matrix_B.data is not None
        input_tensor_a = layer_define.get_matrix_A.data.astype(np.int64)
        input_tensor_b = np.moveaxis(layer_define.get_matrix_B.data.astype(np.int64), -1, -2)
        tensor = np.matmul(input_tensor_a, input_tensor_b)
        tensor = set_subnormal_fp_value_to_zero(tensor, "float16")

        ret = [tensor, ]
        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(tensor)
            ret.append(act_scale)
        return ret


    @staticmethod
    def matmul_forward(layer_define: Matmul):
        # 输入检查

        # mypy check
        assert layer_define.get_matrix_A.data is not None
        assert layer_define.get_matrix_B.data is not None
        input_tensor_a = layer_define.get_matrix_A.data
        input_tensor_a_scale = layer_define.get_matrix_A.get_act_scale.data
        input_tensor_b = np.moveaxis(layer_define.get_matrix_B.data, -1, -2)
        input_tensor_b_scale = layer_define.get_matrix_B.get_act_scale.data
        
        tensor = quantized_matmul(input_tensor_a, input_tensor_b, input_tensor_a_scale, input_tensor_b_scale, "w8a8_attn")

        ret = [tensor, ]
        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(tensor)
            ret.append(act_scale)
        return ret

    @staticmethod
    def layernorm_forward(layer_define: Union[Layernorm, RMSnorm]):
        variance_epsilon = layer_define.var_epsilon

        # Norm层格式检查
        # assert len(input_tensor_list) == 1

        # 生成输入的相关信息(计算使用FP32)

        # mypy check
        assert layer_define.get_input.data is not None
        input_tensor = layer_define.get_input.data.astype("float32")

        # weight_index += 1
        # bias_flag = layer_define["structure"]["bias_flag"]
        rms_flag = layer_define.op_type == "rmsnorm"
        # 由于layernorm涉及co_misc的操作，并且是fp16计算，因此计算顺序会影响最终的结果
        # 因此必须按照实际的多slr分组，对tensor进行分slr计算，每个slr计算出自己的求和，再做累加
        # MISC的其他计算不涉及这个问题，因为不涉及co_misc
        B = layer_define.get_input.shape[-3]
        M = layer_define.get_input.shape[-2]
        N = layer_define.get_input.shape[-1]

        split_slr_dim = -1  # 切N方向
        split_tensor_for_multiple_slr = np.array_split(input_tensor, 3, axis=split_slr_dim)

        # 计算使用FP32
        if rms_flag:
            assert layer_define.get_weights()[0].data is not None
            weight = layer_define.get_weights()[0].data.astype("float16")

            input_square_sum = np.float32(0)
            for slr_id in range(3):
                input_square_sum += np.sum(np.square(split_tensor_for_multiple_slr[slr_id]), axis=-1).astype("float32")
            variance = np.float32(input_square_sum) / np.float32(N)
            rms = 1 / np.sqrt(np.float32(variance) + np.float32(variance_epsilon))
            rms_real = np.repeat(np.expand_dims(rms, axis=-1), N, axis=-1)
            tensor = input_tensor * np.float32(rms_real)
            tensor = weight.astype(np.float32) * tensor.astype(np.float32)
        else:
            assert layer_define.get_weights()[0].data is not None
            weight = layer_define.get_weights()[0].data.astype("float16")
            affine = False
            if len(layer_define.get_weights()) > 1:
                assert layer_define.get_weights()[1].data is not None
                bias = layer_define.get_weights()[1].data.astype("float16")
                affine = True

            input_sum = np.zeros((B, M), dtype=np.float32)
            input_square_sum = np.float32(0)
            for slr_id in range(3):
                input_sum += np.sum(split_tensor_for_multiple_slr[slr_id], axis=-1).astype("float32")
                input_square_sum += np.sum(np.square(split_tensor_for_multiple_slr[slr_id].astype("float32")), axis=-1)

            mean = np.float32(input_sum) / np.float32(N)
            square_mean = np.float32(input_square_sum) / np.float32(N)
            variance = np.float32(square_mean) - np.float32(np.square(np.float32(mean)))
            variance_reciprocal = 1 / np.float32(np.sqrt(np.float32(variance) + np.float32(variance_epsilon)))

            mean_real = np.repeat(np.expand_dims(mean, axis=-1), N, axis=-1)
            variance_real = np.repeat(np.expand_dims(variance_reciprocal, axis=-1), N, axis=-1)
            tensor = (input_tensor - np.float32(mean_real)) * np.float32(variance_real)
            tensor = tensor.astype(np.float32) * weight.astype(np.float32)

            tensor = tensor.astype(np.float16)  # 模拟使用两条MISC计算带bias的layernorm

            # bias运算
            tensor = tensor.astype(np.float32) + (0 if not affine else bias.astype(np.float32))

        # 转换为FP16
        tensor = tensor.astype("float16")

        # assert not dynamic_scale_flag

        tensor = set_subnormal_fp_value_to_zero(tensor, "float16")
        ret = [tensor, ]

        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(tensor)
            tensor_int = tools.quant_output_to_int(tensor, 1 / act_scale, "int8")
            ret = [tensor_int, act_scale, ]

        return ret

    @staticmethod
    def rotary_forward(layer_define: RoPE):
        assert layer_define.get_input.data is not None
        input_tensor = layer_define.get_input.data
        cos_sin_table = layer_define.get_cos_sin_table().data

        assert cos_sin_table is not None
        cos_cache = cos_sin_table[..., : cos_sin_table.shape[-1] // 2]
        sin_cache = cos_sin_table[..., cos_sin_table.shape[-1] // 2: ]

        # sin_cache = cos_sin_table[0::2]
        # cos_cache = cos_sin_table[1::2]

        # cos_cache = cos_cache[..., : cos_cache.shape[-1] // 2].repeat(2, axis=-1)
        # sin_cache = sin_cache[..., : sin_cache.shape[-1] // 2].repeat(2, axis=-1)

        # out_data = input_tensor * cos_cache + tools.rotate_adjacent(input_tensor) * (sin_cache)

        output1 = input_tensor * cos_cache
        output2 = input_tensor * sin_cache
        output_tem = np.zeros(input_tensor.shape).astype("float16")
        output_tem[..., 0::2] = output2[..., 1::2]
        output_tem[..., 1::2] = output2[..., 0::2]        

        out_data = output1 + output_tem

        out_data = out_data.astype("float16")
        tensor = set_subnormal_fp_value_to_zero(out_data, "float16")
        ret = [tensor, ]

        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(tensor)
            tensor_int = tools.quant_output_to_int(tensor, 1 / act_scale, "int8")
            ret = [tensor_int, act_scale, ]
        return ret

    @staticmethod
    def softmax_forward(layer_define: Softmax):
        # 计算使用FP32

        # mypy check
        assert layer_define.get_input.data is not None
        input_tensor = layer_define.get_input.data.astype("float32")
        exp = np.exp((input_tensor - np.max(input_tensor, axis=-1, keepdims=True)))
        tensor = exp / np.sum(exp, axis=-1, keepdims=True)

        # 转换为FP16
        tensor = tensor.astype("float16")

        tensor = set_subnormal_fp_value_to_zero(tensor, "float16")
        ret = [tensor, ]

        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(tensor)
            tensor_int = tools.quant_output_to_int(tensor, 1 / act_scale, "int8")
            ret = [tensor_int, act_scale, ]
        return ret

    @staticmethod
    def silu_forward(layer_define: Silu):
        # 只需要进行IR操作就可以
        # mypy check
        assert layer_define.get_input.data is not None
        input_tensor = layer_define.get_input.data.astype("float32")
        # 输出生成
        reciprocal = 1 / (1 + np.exp(-input_tensor))
        tensor = input_tensor * reciprocal.astype("float32")
        tensor = tensor.astype("float16")
        tensor = set_subnormal_fp_value_to_zero(tensor, "float16")
        ret = [tensor, ]

        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(tensor)
            tensor_int = tools.quant_output_to_int(tensor, 1 / act_scale, "int8")
            ret = [tensor_int, act_scale, ]
        return ret

    @staticmethod
    def gelu_forward(layer_define: Gelu):
        # 只需要进行IR操作就可以
        # mypy check
        assert layer_define.get_input.data is not None
        # assert layer_define.get_output.data is not None
        input_tensor = layer_define.get_input.data.astype("float32")
        input_tensor = tools.set_subnormal_fp_value_to_zero(input_tensor, "float32")
        # dynamic_scale_flag = layer_define["structure"]["dynamic_scale_flag"]

        # 输出生成
        # 使用numpy的gelu实现
        output_tensor = input_tensor * tools.erf(input_tensor)
        output_tensor = output_tensor.astype("float16")

        # assert not dynamic_scale_flag
        output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float16")
        ret = [output_tensor, ]

        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(output_tensor)
            tensor_int = tools.quant_output_to_int(output_tensor, 1 / act_scale, "int8")
            ret = [tensor_int, act_scale, ]
        return ret

    @staticmethod
    def eltwise_forward(layer_define: Eltwise):
        # mypy check
        assert layer_define.get_input_A.data is not None
        assert layer_define.get_input_B.data is not None
        input_tensor_0 = layer_define.get_input_A.data.astype("float16")
        input_tensor_1 = layer_define.get_input_B.data.astype("float16")

        tensor = hardware_float_eltwise_add_mul(input_tensor_0, input_tensor_1, layer_define.type)
        tensor = set_subnormal_fp_value_to_zero(tensor, "float16")
        ret = [tensor, ]

        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(tensor)
            tensor_int = tools.quant_output_to_int(tensor, 1 / act_scale, "int8")
            ret = [tensor_int, act_scale, ]
        return ret

    @staticmethod
    def div_forward(layer_define: Div):
        # mypy check
        assert layer_define.get_input.data is not None
        assert layer_define.get_divisor != 0, "Divisor should not be zero"
        input_tensor = layer_define.get_input.data.astype("float16")
        divisor = layer_define.get_divisor.data.astype("float16")
        tensor = input_tensor / divisor
        tensor = set_subnormal_fp_value_to_zero(tensor, "float16")
        ret = [tensor, ]

        if layer_define.act_scale_flag:
            act_scale = tools.dynamic_activation_scale(tensor)
            tensor_int = tools.quant_output_to_int(tensor, 1 / act_scale, "int8")
            ret = [tensor_int, act_scale, ]
        return ret

    @staticmethod
    def concat_forward(layer_define: Op):
        raise NotImplementedError

    @staticmethod
    def transpose_forward(layer_define: Transpose):
        # mypy check
        assert layer_define.get_input.data is not None
        assert np.abs(np.array(layer_define.get_trans_dims[0]) - np.array(layer_define.get_trans_dims[1])) == 1
        target_shape = [i for i in range(len(layer_define.get_input.data.shape))]
        target_shape[layer_define.get_trans_dims[0]], target_shape[layer_define.get_trans_dims[1]] =\
            target_shape[layer_define.get_trans_dims[1]], target_shape[layer_define.get_trans_dims[0]]

        tensor = np.transpose(layer_define.get_input.data, target_shape)

        return [tensor, ]

    @staticmethod
    def split_forward(layer_define: Split):

        # mypy check
        assert layer_define.get_input.data is not None
        split_dim = layer_define.dim
        split_size = layer_define.split_size
        split_all_size = layer_define.get_input.data.shape[split_dim]

        assert (split_all_size % split_size) == 0

        tensor = np.split(layer_define.get_input.data, split_size, axis=split_dim)

        return tensor

    @staticmethod
    def view_forward(layer_define: View):
        # mypy check
        assert layer_define.get_input.data is not None
        target_shape = layer_define.shape
        tensor = np.reshape(layer_define.get_input.data, target_shape)

        return [tensor, ]

    @staticmethod
    def convert_forward(layer_define: Convert):
        assert layer_define.get_input.data is not None
        input_tensor = layer_define.get_input.data.astype("float32")

        input_tensor = set_subnormal_fp_value_to_zero_convert(input_tensor, "float32")

        # eps = 1.e-4
        # activation_scale = (np.float16(np.max(np.abs(activation), axis=-1, keepdims=True)) / 127).astype(np.float16)
        activation_scale = (np.float32(np.max(np.abs(input_tensor), axis=-1, keepdims=True)) / 127).astype(np.float32)
        # activation_scale[np.abs(activation_scale) < eps] = eps

        activation_scale = set_subnormal_fp_value_to_zero_convert(activation_scale, "float32")

        output_tensor_int = quant_output_to_int(input_tensor, np.float32(1 / activation_scale), "int8")

        activation_scale = np.float32(activation_scale)
        dynamic_scale = set_subnormal_fp_value_to_zero_convert(activation_scale, "float32")

        return [output_tensor_int, dynamic_scale, ]
