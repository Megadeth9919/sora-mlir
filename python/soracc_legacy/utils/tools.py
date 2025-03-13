import os
import json
import struct
import numpy as np
import logging
from utils.hw_info import *
from typing import List
from inst import *

logger = logging.getLogger(__name__)

nan_fp16_hex = 0xFE00

dtype_dict = {"F32": np.float32, "F16": np.float16, "I32": np.int32, "I8": np.int8, "U8": np.uint8}

def divide_exactly(a, b):
    result = a // b
    assert result * b == a
    return result


def generate_random(data_shape, data_dtype):
    if data_dtype == "int8":
        return np.random.randint(-(2 ** (3)), 2**3 - 1, data_shape, dtype=np.int8)
    elif data_dtype == "uint4":
        return np.random.randint(0, 2**2 - 1, data_shape, dtype=np.int8)
    elif data_dtype == "uint16":
        return np.random.randint(0, 2**2 - 1, data_shape, dtype=np.uint16)
    elif data_dtype == "int16":
        return np.random.randint(-(2 ** (3)), 2**3 - 1, data_shape, dtype=np.int16)
    elif data_dtype == "float16":
        assert len(data_shape) == 3
        return np.random.uniform(low=-0.5, high=0.5, size=data_shape).astype("float16")
        # return np.random.randn(*data_shape).astype("float16")
    elif data_dtype == "bfloat16":
        logger.warning("bfloat16 not support, use float16 instead!")
        return np.random.randn(*data_shape).astype("float16")
    elif data_dtype == "float32":
        return np.random.randn(*data_shape).astype("float32")
    else:
        raise ValueError


from functools import reduce


def generate_seq(data_shape, data_dtype):
    size = reduce(lambda x, y: x * y, data_shape, 1)
    if data_dtype == "int8":
        return np.array(range(size), np.int8).reshape(data_shape)
    elif data_dtype == "uint16":
        return np.array(range(size), np.uint16).reshape(data_shape)
    elif data_dtype == "int16":
        return np.array(range(size), np.int16).reshape(data_shape)
    elif data_dtype == "float16":
        return np.array(range(size), np.float16).reshape(data_shape)
    elif data_dtype == "float32":
        return np.array(range(size), np.float32).reshape(data_shape)
    else:
        raise ValueError


def get_byte_by_dtype(data_dtype):
    if data_dtype == "uint2":
        return 0.25
    elif data_dtype == "uint4":
        return 0.5
    elif data_dtype == "int8":
        return 1
    elif data_dtype == "uint16":
        return 2
    elif data_dtype == "int16":
        return 2
    elif data_dtype == "float16":
        return 2
    elif data_dtype == "bfloat16":
        return 2
    elif data_dtype == "float32":
        return 4
    else:
        raise ValueError


def get_bytes_by_dtype(data_size, data_dtype):
    if data_dtype == "uint2":
        return math.ceil(data_size / 4)
    elif data_dtype == "uint4":
        return math.ceil(data_size / 2)
    elif data_dtype == "int8":
        return int(data_size)
    elif data_dtype == "int16":
        return int(data_size * 2)
    elif data_dtype == "uint16":
        return int(data_size * 2)
    elif data_dtype == "float16":
        return int(data_size * 2)
    elif data_dtype == "bfloat16":
        return int(data_size * 2)
    elif data_dtype == "float32":
        return int(data_size * 4)
    else:
        raise ValueError


def encode_fp16_to_int(dec_fp16_value):
    # 十进制单精度浮点转IEEE-754 float16编码
    encode_fp16 = struct.unpack("H", struct.pack("e", dec_fp16_value))[0]
    if math.isnan(dec_fp16_value):
        assert encode_fp16 == nan_fp16_hex
    return int(encode_fp16)


def set_subnormal_fp_value_to_zero(data, dtype):
    assert data.dtype == np.dtype(dtype)
    if data.shape == ():
        if np.abs(data) < np.finfo(dtype).smallest_normal:
            data = np.array(0, dtype=dtype)
    else:
        data[np.abs(data) < np.finfo(dtype).smallest_normal] = 0
    return data


def assert_no_subnormal_fp_value(data, dtype):
    if data.shape == ():
        assert np.abs(data) >= np.finfo(dtype).smallest_normal or data == np.float16(0)
    else:
        pass
        # assert np.min(np.abs(data[np.nonzero(data)])) >= np.finfo(dtype).smallest_normal


def check_subnormal_fp_value(data, data_name, data_type):
    # check is there any subnormal floating point number
    assert data_type in ("float16", "float32")
    if data.shape == ():
        if np.abs(data) >= np.finfo(data_type).smallest_normal or data == np.array(
            0, dtype=data_type
        ):
            pass
        else:
            logger.warning(f"Subnormal number in {data_name}!")
    else:
        subnormal_num = np.sum(
            np.abs(data[np.nonzero(data)]) < np.finfo(data_type).smallest_normal
        )
        if subnormal_num > 0:
            logger.warning(
                f"{subnormal_num} subnormal {data_type} number in {data_name}!"
            )


def align_ceil(value, alignment):
    return int(math.ceil(value / alignment)) * alignment


def get_buffer_stride_by_bytes(data_bytes, buffer_name):
    if buffer_name == "int weight buffer":
        buffer_width = WeightBuffer.get_bytewidth() * WeightBuffer.sub_bank_num_int()
        # buffer_min_write_addr_jump = WeightBuffer.get_min_stride_num()
    elif buffer_name == "fp weight buffer":
        buffer_width = WeightBuffer.get_bytewidth() * WeightBuffer.sub_bank_num_float()
        # buffer_min_write_addr_jump = WeightBuffer.get_min_stride_num()
    elif buffer_name == "weight buffer":
        # buffer_width = cfg.WEIGHT_BUFFER_INT_PER_CHANNEL_WIDTH_B
        buffer_width = WeightBuffer.get_bytewidth()
        # buffer_min_write_addr_jump = WeightBuffer.get_min_stride_num()
    elif buffer_name == "meta buffer":
        buffer_width = MetaBuffer.get_bytewidth()
        buffer_min_write_addr_jump = MetaBuffer.get_min_stride_num()
    elif buffer_name == "global buffer":
        buffer_width = GlobalBuffer.get_bytewidth()
        buffer_min_write_addr_jump = GlobalBuffer.get_min_stride_num()
    elif buffer_name == "register file":
        buffer_width = GlobalBuffer.get_bytewidth()
        buffer_min_write_addr_jump = GlobalBuffer.get_min_stride_num()
    else:
        raise ValueError
    buffer_strides = math.ceil(data_bytes / buffer_width) * buffer_width
    return buffer_strides


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


def tiling_to_list(start_id, end_id, split_unit):
    # return a list of tiling tuple: [(tiling_start_id, tiling_end_id)]
    # example: start_id = 1, end_id = 11, split_unit = 3
    # return [(1, 4), (4, 7), (7, 10), (10, 11)]
    assert end_id > start_id
    assert start_id >= 0
    assert split_unit >= 0

    tiling_start_end_id_list = list()
    tiling_start_id = None
    tiling_end_id = start_id
    for i in range(start_id, end_id + split_unit, split_unit):
        tiling_start_id = tiling_end_id
        tiling_end_id = min(i, end_id)
        if i > start_id:
            tiling_start_end_id_list.append(
                (
                    tiling_start_id,
                    tiling_end_id,
                )
            )
    return tiling_start_end_id_list


def pack_4bit_data(numpy_int8_data_to_pack):
    assert numpy_int8_data_to_pack.dtype == np.int8
    data_shape = list(numpy_int8_data_to_pack.shape)
    data_shape[-1] = divide_exactly(data_shape[-1], 2)
    numpy_int8_data_to_pack = numpy_int8_data_to_pack.flatten()
    packed_uint4_data = numpy_int8_data_to_pack[1::2] << 4
    packed_uint4_data += numpy_int8_data_to_pack[0::2]
    packed_uint4_data = packed_uint4_data.reshape(data_shape)
    return packed_uint4_data


def pack_2bit_data(numpy_int8_data_to_pack):
    assert numpy_int8_data_to_pack.dtype == np.int8
    data_shape = list(numpy_int8_data_to_pack.shape)
    data_shape[-1] = divide_exactly(data_shape[-1], 4)
    numpy_int8_data_to_pack = numpy_int8_data_to_pack.flatten()
    packed_uint2_data = numpy_int8_data_to_pack[3::4] << 6
    packed_uint2_data += numpy_int8_data_to_pack[2::4] << 4
    packed_uint2_data += numpy_int8_data_to_pack[1::4] << 2
    packed_uint2_data += numpy_int8_data_to_pack[0::4]
    packed_uint2_data = packed_uint2_data.reshape(data_shape)
    return packed_uint2_data


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate((-x2, x1), axis=-1)


def rotate_adjacent(x):
    """Rotates adjacent the hidden dims of the input."""
    x_tem = np.zeros(x.shape).astype("float16")
    x_tem[..., 0::2] = -x[..., 1::2]
    x_tem[..., 1::2] = x[..., 0::2]
    return x_tem


def round_shift_cut(value, shift_cut):
    assert shift_cut >= 0
    if shift_cut >= 1:
        return (value + (1 << (shift_cut - 1))) >> shift_cut  # 四舍五入
    return value


def load_4bit_tensor(data_bin_name, unsigned=False):
    if unsigned:
        _8bit_data = np.fromfile(data_bin_name, dtype=np.uint8)
        tensor = np.zeros(2 * len(_8bit_data)).astype(np.uint8)
    else:
        _8bit_data = np.fromfile(data_bin_name, dtype=np.int8)
        tensor = np.zeros(2 * len(_8bit_data)).astype(np.int8)
    high_8bit_data = _8bit_data >> 4
    low_8bit_data = (_8bit_data << 4) >> 4

    tensor[0::2] = low_8bit_data
    tensor[1::2] = high_8bit_data
    return tensor


def int4_to_int8_tensor(input_tensor, unsigned=False):
    input_tensor = input_tensor.flatten()

    high_4bit_data = input_tensor[0::2]
    low_4bit_data = input_tensor[1::2]

    if unsigned:
        high_4bit_data = high_4bit_data.astype(np.uint8)
        low_4bit_data = low_4bit_data.astype(np.uint8)
    else:
        high_4bit_data = high_4bit_data.astype(np.int8)
        low_4bit_data = low_4bit_data.astype(np.int8)

    _8bit_data = (high_4bit_data << 4) ^ (low_4bit_data & 15)
    if unsigned:
        return _8bit_data.astype(np.uint8)
    else:
        return _8bit_data.astype(np.int8)


def int8_to_int4_tensor(input_tensor, unsigned=False):
    assert input_tensor.dtype in ("int8", "uint8")
    if unsigned:
        _8bit_data = input_tensor.astype("uint8")
        tensor = np.zeros(2 * len(_8bit_data)).astype(np.uint8)
    else:
        _8bit_data = input_tensor.astype("int8")
        tensor = np.zeros(2 * len(_8bit_data)).astype(np.int8)
    high_8bit_data = _8bit_data >> 4
    low_8bit_data = (_8bit_data << 4) >> 4

    tensor[0::2] = low_8bit_data
    tensor[1::2] = high_8bit_data
    return tensor


def int4_to_int32_tensor(input_tensor, unsigned=False):
    _4bit_data_0 = input_tensor[:, 0]
    _4bit_data_1 = input_tensor[:, 1]
    _4bit_data_2 = input_tensor[:, 2]
    _4bit_data_3 = input_tensor[:, 3]
    _4bit_data_4 = input_tensor[:, 4]
    _4bit_data_5 = input_tensor[:, 5]
    _4bit_data_6 = input_tensor[:, 6]
    _4bit_data_7 = input_tensor[:, 7]

    _32bit_data = (
        (_4bit_data_7 << 28)
        ^ (_4bit_data_6 << 24)
        ^ (_4bit_data_5 << 20)
        ^ (_4bit_data_4 << 16)
        ^ (_4bit_data_3 << 12)
        ^ (_4bit_data_2 << 8)
        ^ (_4bit_data_1 << 4)
        ^ _4bit_data_0
    )
    if unsigned:
        return _32bit_data.astype(np.uint32)
    else:
        return _32bit_data.astype(np.int32)


def float16_to_int8_tensor(input_tensor):
    assert input_tensor.dtype == np.float16
    input_data = list(input_tensor.flatten().tobytes())
    input_data_8bit = np.array(input_data).astype("uint8")

    return input_data_8bit


def int8_to_uint16_tensor(input_tensor):
    assert input_tensor.dtype in (np.int8, np.uint8)
    assert input_tensor.size % 2 == 0
    output_tensor = input_tensor.astype("uint8").flatten().view(dtype="uint16")
    return output_tensor


def int8_to_float16_tensor(input_tensor):
    assert input_tensor.dtype in (np.int8, np.uint8)
    assert input_tensor.size % 2 == 0
    output_tensor = input_tensor.astype("uint8").flatten().view(dtype="float16")
    return output_tensor


def int16_to_float16_tensor(input_tensor):
    assert input_tensor.dtype in (np.uint16, np.int16)
    output_tensor = input_tensor.astype("uint16").flatten().view(dtype="float16")
    return output_tensor


def div_mod(dividend, divisor):
    return int(dividend / divisor), int(dividend % divisor)


def store_tensor(
    input_tensor, input_shape, data_type, output_name_prefix, mode="bin", order="C"
):
    """
    save tensor to txt and bin
    """
    assert input_tensor.shape == tuple(input_shape)
    # assert len(input_shape) == 3
    assert data_type in (
        "uint2",
        "uint4",
        "int8",
        "int16",
        "uint16",
        "int32",
        "float16",
        "bfloat16",
        "float32",
    )
    # logger.debug("Save to %s" % output_name_prefix)
    if data_type in ("uint2", "uint4"):
        input_tensor = input_tensor.astype(np.uint8)
    elif data_type == "int8":
        input_tensor = input_tensor.astype(np.int8)
    elif data_type == "int16":
        input_tensor = input_tensor.astype(np.int16)
    elif data_type == "uint16":
        input_tensor = input_tensor.astype(np.uint16)
    elif data_type == "int32":
        input_tensor = input_tensor.astype(np.int32)
    elif data_type == "float16":
        input_tensor = input_tensor.astype(np.float16)
        check_subnormal_fp_value(input_tensor, output_name_prefix, data_type)
    elif data_type == "bfloat16":
        input_tensor = input_tensor.astype(np.float16)
        # check_subnormal_fp_value(input_tensor, output_name_prefix, data_type)
    elif data_type == "float32":
        input_tensor = input_tensor.astype(np.float32)
        check_subnormal_fp_value(input_tensor, output_name_prefix, data_type)
    else:
        raise NotImplementedError

    if mode == "bin":
        input_tensor.tofile(output_name_prefix + ".bin")
    elif mode == "npy":
        np.save(output_name_prefix + ".npy", input_tensor)
    elif mode == "txt":
        save_rtl_tensor(input_tensor, output_name_prefix + ".txt", data_type, order)
    elif mode == "full":
        save_rtl_tensor(input_tensor, output_name_prefix + ".txt", data_type, order)
        input_tensor.tofile(output_name_prefix + ".bin")
    else:
        raise NotImplementedError


# 在这里我们定义一个将权重文件，中间变量文件，输入图片文件存储到指定位置的函数
def save_rtl_tensor(fix_tensor, output_name, data_type, order: str = "C"):
    """
    save rtl tensor to file
    """
    NG = 1
    assert order in ["C", "F"]  # 行排和列排
    # num of group
    # 这一部分是为了和硬件RTL给出的值做对比用来调试，标准的格式为0x16进制，NG个一组，大端模式，不齐补0
    if data_type == "uint4" and order == "F":
        comp_rtl_tensor = fix_tensor.flatten(order)
        comp_rtl_tensor = np.concatenate(
            [
                comp_rtl_tensor,
                np.zeros(
                    (math.ceil(comp_rtl_tensor.size / 8) * 8 - comp_rtl_tensor.size)
                ),
            ]
        )
        comp_rtl_tensor = comp_rtl_tensor.reshape((-1, 8))
        comp_rtl_tensor = int4_to_int32_tensor(
            comp_rtl_tensor.astype("uint32"), unsigned=True
        )
        np.savetxt(output_name, comp_rtl_tensor, fmt=NG * "%08x", delimiter="\n")
    else:
        comp_rtl_tensor = fix_tensor.flatten(order=order).copy()
        comp_rtl_tensor = np.concatenate(
            [
                comp_rtl_tensor,
                np.zeros(
                    (math.ceil(comp_rtl_tensor.size / NG) * NG - comp_rtl_tensor.size)
                ),
            ]
        )
        comp_rtl_tensor = comp_rtl_tensor.reshape((-1, NG))
        if data_type in ("uint2", "uint4", "int8"):
            comp_rtl_tensor = comp_rtl_tensor[:, ::-1].astype(np.uint8)
            np.savetxt(output_name, comp_rtl_tensor, fmt=NG * "%08x", delimiter="\n")
        elif data_type == "uint16":
            comp_rtl_tensor = comp_rtl_tensor[:, ::-1].astype(np.uint16)
            np.savetxt(output_name, comp_rtl_tensor, fmt=NG * "%08x", delimiter="\n")
        elif data_type == "int32":
            comp_rtl_tensor = comp_rtl_tensor[:, ::-1].astype(np.uint32)
            np.savetxt(output_name, comp_rtl_tensor, fmt=NG * "%08x", delimiter="\n")
        elif data_type == "float16":
            comp_rtl_tensor = comp_rtl_tensor[:, ::-1].astype(np.float16)
            text_list = []
            for i in comp_rtl_tensor:
                y = struct.unpack(
                    "H", struct.pack("e", i)
                )  # 将浮点数按照2byte float转换
                z = str(hex(y[0]).zfill(4))[-4:].replace("x", "0")
                text_list.append("0000" + z + "\n")
            file = open(output_name, "w")
            file.write("".join(text_list))
            file.close()
        elif data_type == "float32":
            comp_rtl_tensor = comp_rtl_tensor[:, ::-1].astype(np.float32)
            text_list = []
            for i in comp_rtl_tensor:
                y = struct.unpack(
                    "I", struct.pack("f", i)
                )  # 将浮点数按照4byte float转换
                z = str(hex(y[0]).zfill(8))[-8:].replace("x", "0")
                text_list.append(z + "\n")
            file = open(output_name, "w")
            file.write("".join(text_list))
            file.close()
        else:
            raise NotImplementedError


def param_info_parser(param_info: dict):
    return dtype_dict[param_info["dtype"]], param_info["shape"], param_info["data_offsets"]


def safetensors_metadata_parser(file_path):
    header_size = 8
    meta_data = {}
    if os.stat(file_path).st_size > header_size:
        with open(file_path, "rb") as f:
            b8 = f.read(header_size)
            if len(b8) == header_size:
                header_len = int.from_bytes(b8, 'little', signed=False)
                headers = f.read(header_len)
                if len(headers) == header_len:
                    meta_data = sorted(json.loads(headers.decode("utf-8")).items())
            
            param_dict = {}
            if len(meta_data) == 0:
                raise ValueError("Error: Invalid safetensors metadata file")
            else:
                for _, (name, value) in enumerate(meta_data):
                    if name == "__metadata__" or name == "__version__":
                        continue
                    dtype, shape, data_offsets = param_info_parser(value)
                    f.seek(data_offsets[0] + header_len + header_size)
                    raw_data = f.read(data_offsets[1] - data_offsets[0])
                    param_dict[name] = np.frombuffer(raw_data, dtype=dtype).reshape(shape)
    return param_dict


def load_tensor_from_safetensors(ckpt_path):
    """
    load tensor from safetensors
    """
    assert os.path.exists(ckpt_path), ckpt_path
    tensor_dict = {}
    with open(ckpt_path, "rb") as f:
        while True:
            try:
                tensor_name = f.readline().decode().strip()
                tensor_shape = tuple(map(int, f.readline().decode().strip().split()))
                tensor_dtype = f.readline().decode().strip()
                tensor_data = np.frombuffer(
                    f.read(np.prod(tensor_shape) * 4), dtype=np.float32
                )
                tensor_data = tensor_data.reshape(tensor_shape)
                tensor_dict[tensor_name] = tensor_data
            except Exception as e:
                break
    return tensor_dict


def load_tensor(data_bin_name, data_bin_shape, data_type):
    """
    load fix type tensor
    """
    # logger.debug("Load from %s" % data_bin_name)
    assert os.path.exists(data_bin_name), data_bin_name
    assert data_type in (
        "uint2",
        "uint4",
        "int8",
        "int16",
        "uint16",
        "float16",
        "bfloat16",
        "float32",
    )

    # 按照int8的形式将数据读取出来
    if data_type in ("uint2", "uint4"):
        tensor = np.fromfile(data_bin_name, dtype=np.uint8)
    elif data_type == "int8":
        tensor = np.fromfile(data_bin_name, dtype=np.int8)
        tensor = np.clip(tensor, -127, np.iinfo(np.int8).max)
    elif data_type == "int16":
        tensor = np.fromfile(data_bin_name, dtype=np.int16)
    elif data_type == "uint16":
        tensor = np.fromfile(data_bin_name, dtype=np.uint16)
    elif data_type == "float16":
        tensor = np.fromfile(data_bin_name, dtype=np.float16)
        check_subnormal_fp_value(tensor, data_bin_name, data_type)
    elif data_type == "bfloat16":
        tensor = np.fromfile(data_bin_name, dtype=np.float16)
        # check_subnormal_fp_value(tensor, data_bin_name, data_type)
    elif data_type == "float32":
        tensor = np.fromfile(data_bin_name, dtype=np.float32)
        check_subnormal_fp_value(tensor, data_bin_name, data_type)
    else:
        raise NotImplementedError
    tensor = tensor.reshape(data_bin_shape)
    return tensor


def fp16_matmul(activation, weight):
    assert activation.dtype == np.float16
    assert weight.dtype == np.float16
    B, N, K = weight.shape
    assert activation.shape == (B, 1, K)
    activation = set_subnormal_fp_value_to_zero(activation, "float16")
    weight = set_subnormal_fp_value_to_zero(weight, "float16")

    tmp_mul = activation.astype(np.float64) * weight.astype(
        np.float64
    )  # 由于硬件是采用融合乘加，因此乘完的数没有精度损失，这里用FP64模拟
    tmp_result_shape = (B, N)
    k_group_num = math.ceil(
        K / MMCore.get_parallel_dim(MM_ParallelType.float)[-1]
    )  # 按K方向并行度分组，与硬件采用相同累加顺序

    output_tensor = np.zeros(tmp_result_shape, dtype=np.float16)
    for i in range(
        k_group_num
    ):  # 累加维度，注意这里不能用np.sum，否则会因为累加顺序不同而导致数字对不上
        tmp_sum = np.zeros(
            tmp_result_shape, dtype=np.float16
        )  # 硬件每次融合乘加之间，采用fp16传递
        for j in range(
            min(
                MMCore.get_parallel_dim(MM_ParallelType.float)[-1],
                K - i * MMCore.get_parallel_dim(MM_ParallelType.float)[-1],
            )
        ):
            tmp_sum = tmp_sum.astype(np.float64) + tmp_mul[
                :, :, i * MMCore.get_parallel_dim(MM_ParallelType.float)[-1] + j
            ].astype(np.float64)
            tmp_sum = tmp_sum.astype(np.float16)
            tmp_sum = set_subnormal_fp_value_to_zero(tmp_sum, "float16")
        output_tensor = output_tensor.astype(np.float16) + tmp_sum.astype(np.float16)
    output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float16")
    return output_tensor


def quantized_matmul(activation, weight, activation_scale, weight_scale, matmul_dtype):
    if matmul_dtype == "w8a8":
        assert activation.dtype == np.int8
        assert weight.dtype == np.int8
        # assert np.min(activation) >= -127
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
    output_tensor = np.matmul(input_tensor, weight.swapaxes(-1, -2)).astype("float32")
    output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float32")
    scale = weight_scale.astype("float32") * activation_scale.astype("float32")
    assert_no_subnormal_fp_value(scale, "float32")
    output_tensor = output_tensor.astype("float32") * scale.squeeze().astype("float32")
    output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float32")
    output_tensor = output_tensor.astype("float16")
    return output_tensor


def erf(x):
    x = x.astype("float32")
    ret = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.047715 * np.power(x, 3))))
    ret = set_subnormal_fp_value_to_zero(ret, "float32")
    return ret


def dynamic_activation_scale(activation):
    assert activation.dtype == np.float16
    activation = set_subnormal_fp_value_to_zero(activation, "float16")

    eps = 1.e-4
    activation_scale = (np.float32(np.max(np.abs(activation), axis=-1, keepdims=True)) / 127).astype(np.float32)
    # activation_scale[np.abs(activation_scale) < eps] = eps
    activation_scale = set_subnormal_fp_value_to_zero(activation_scale, "float32")

    return activation_scale


def quant_output_to_int(output_tensor, out_int_scaling_factor, output_dtype):
    # 输出fp16量化成int8
    assert output_tensor.dtype == np.float16
    assert out_int_scaling_factor.dtype == np.float32
    assert output_dtype in ("int8", "int16")
    check_subnormal_fp_value(
        out_int_scaling_factor, "out_int_scaling_factor", "float32"
    )
    out_int_scaling_factor = set_subnormal_fp_value_to_zero(
        out_int_scaling_factor, "float32"
    )
    assert_no_subnormal_fp_value(out_int_scaling_factor, "float32")
    output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float16")
    output_tensor = np.float16(output_tensor * out_int_scaling_factor)
    output_tensor = set_subnormal_fp_value_to_zero(output_tensor, "float16")
    round_tensor = np.round(output_tensor)
    if output_dtype == "int8":
        output_tensor = np.clip(
            round_tensor, -127, np.iinfo(np.int8).max  # np.iinfo(np.int8).min,
        ).astype("int8")
    elif output_dtype == "int16":
        output_tensor = np.clip(round_tensor, -32767, np.iinfo(np.int16).max).astype(
            "int16"
        )
    else:
        raise ValueError
    return output_tensor


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_yaml(yaml_path):
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def parse_inst_rely_to_PUType(inst_rely: dict) -> tuple[List[PUType], List[PUType]]:
    waits: List[PUType] = list()
    releases: List[PUType] = list()
    for wait in inst_rely["wait"]:
        waits.append(getattr(PUType, wait))
    for release in inst_rely["release"]:
        releases.append(getattr(PUType, release))
    return waits, releases


def parse_json_to_inst(json_inst: List[List[dict]]) -> List[List[Inst]]:
    model_inst_list: List[List[Inst]] = list()

    for slr_inst in json_inst:
        slr_inst_list: List[Inst] = list()
        for inst in slr_inst:
            if inst["inst_type"] == "load":
                wait, release = parse_inst_rely_to_PUType(inst)
                ld_inst = LDInst()
                ld_inst.wait = wait
                ld_inst.release = release
                ld_inst.length_1d = inst["length_1d"]
                ld_inst.loop_1d = inst["loop_1d"]
                ld_inst.src_1d_stride = inst["src_1d_stride"]
                ld_inst.src_addr = inst["src_addr"]
                ld_inst.dst_addr = inst["dst_addr"]
                ld_inst.mode = getattr(LDMode, inst["mode"])
                ld_inst.loop_direction = getattr(LoopDir, inst["loop_direction"])
                ld_inst.src_bank_id = inst["src_bank_id"]
                ld_inst.dst_group_id = inst["dst_group_id"]
                ld_inst.dst_bank_id = inst["dst_bank_id"]
                ld_inst.bank_addr_stride = inst["bank_addr_stride"]

                slr_inst_list.append(ld_inst)
            elif inst["inst_type"] == "store":
                wait, release = parse_inst_rely_to_PUType(inst)
                st_inst = STInst()
                st_inst.wait = wait
                st_inst.release = release
                st_inst.length_1d = inst["length_1d"]
                st_inst.loop_1d = inst["loop_1d"]
                st_inst.dst_1d_stride = inst["dst_1d_stride"]
                st_inst.dst_addr = inst["dst_addr"]
                st_inst.src_addr = inst["src_addr"]
                st_inst.src_bank_id = inst["src_bank_id"]
                st_inst.src_group_id = inst["src_group_id"]
                st_inst.loop_direction = getattr(LoopDir, inst["loop_direction"])
                st_inst.bank_addr_stride = inst["bank_addr_stride"]
                st_inst.mode = getattr(STMode, inst["mode"])
                
                slr_inst_list.append(st_inst)
            elif inst["inst_type"] == "mm_inst":
                wait, release = parse_inst_rely_to_PUType(inst)
                mm_inst = MMInst()
                mm_inst.wait = wait
                mm_inst.release = release
                mm_inst.input_mode = getattr(MMInputMode, inst["input_mode"])
                mm_inst.output_mode = getattr(MMOutputMode, inst["output_mode"])
                mm_inst.act_start_addr = inst["act_start_addr"]
                mm_inst.act_bank_group_id = inst["act_bank_group_id"]
                mm_inst.out_bank_group_id = inst["out_bank_group_id"]
                mm_inst.bias_flag = inst["bias_flag"]
                mm_inst.output_flag = inst["output_flag"]
                mm_inst.weights_start_addr = inst["weights_start_addr"]
                mm_inst.M = inst["M"]
                mm_inst.relu = inst["relu"]
                mm_inst.bias_start_addr = inst["bias_start_addr"]
                mm_inst.out_start_addr = inst["out_start_addr"]
                mm_inst.N = inst["N"]
                mm_inst.K = inst["K"]
                mm_inst.act_scale_start_addr = inst["act_scale_start_addr"]
                mm_inst.out_scale_start_addr = inst["out_scale_start_addr"]
                mm_inst.weights_scale_start_addr = inst["weights_scale_start_addr"]

                slr_inst_list.append(mm_inst)
            elif inst["inst_type"] == "misc_inst":
                wait, release = parse_inst_rely_to_PUType(inst)
                misc_inst = MISCInst()
                misc_inst.wait = wait
                misc_inst.release = release
                misc_inst.op = getattr(MiscOp, inst["op"])
                misc_inst.input_a_mode = getattr(MiscMode, inst["input_a_mode"])
                misc_inst.input_b_mode = getattr(MiscMode, inst["input_b_mode"])
                misc_inst.output_mode = getattr(MiscMode, inst["output_mode"])
                misc_inst.in_a_start_addr = inst["in_a_start_addr"]
                misc_inst.in_a_bank_id = inst["in_a_bank_id"]
                misc_inst.in_a_bank_group = inst["in_a_bank_group"]
                misc_inst.in_b_start_addr = inst["in_b_start_addr"]
                misc_inst.in_b_bank_id = inst["in_b_bank_id"]
                misc_inst.in_b_bank_group = inst["in_b_bank_group"]
                misc_inst.out_start_addr = inst["out_start_addr"]
                misc_inst.out_bank_id = inst["out_bank_id"]
                misc_inst.out_bank_group = inst["out_bank_group"]
                misc_inst.meta_addr = inst["meta_addr"]
                misc_inst.K = inst["K"]
                misc_inst.batch_flag = inst["batch_flag"]
                misc_inst.dynamic_scale = inst["dynamic_scale"]
                misc_inst.batch = inst["batch"]
                misc_inst.sp_table_idx = inst["sp_table_idx"]
                misc_inst.out_swap = inst["out_swap"]
                misc_inst.reg_index = inst["reg_index"]
    
                slr_inst_list.append(misc_inst)
            elif inst["inst_type"] == "sys":
                wait, release = parse_inst_rely_to_PUType(inst)
                sys_inst = SYSCInst()
                sys_inst.wait = wait
                sys_inst.release = release
                sys_inst.op = getattr(SysOp, inst["op"])
                
                slr_inst_list.append(sys_inst)
            elif inst["inst_type"] == "rs_inst":
                wait, release = parse_inst_rely_to_PUType(inst)
                rs_inst = RSInst()
                rs_inst.wait = wait
                rs_inst.release = release
                rs_inst.src_bank_id = inst["src_bank_id"]
                rs_inst.dst_bank_id = inst["dst_bank_id"]
                rs_inst.src_bank_group_id = inst["src_bank_group_id"]
                rs_inst.dst_bank_group_id = inst["dst_bank_group_id"]
                rs_inst.src_addr = inst["src_addr"]
                rs_inst.dst_addr = inst["dst_addr"]
                rs_inst.M = inst["M"]
                rs_inst.K = inst["K"]
                rs_inst.data_type = getattr(RSDataType, inst["data_type"])

                slr_inst_list.append(rs_inst)
        model_inst_list.append(slr_inst_list)
    return model_inst_list


# def parse_json_to_inst(json_inst: List[List[dict]]) -> List[List[Inst]]:
#     model_inst_list: List[List[Inst]] = list()

#     for slr_inst in json_inst:
#         slr_inst_list: List[Inst] = list()
#         for inst in slr_inst:
#             if inst["inst_type"] == "load":
#                 wait, release = parse_inst_rely_to_PUType(inst)
#                 ld_inst = LDInst()
#                 ld_inst.wait = wait
#                 ld_inst.release = release
#                 ld_inst.length_1d = inst["length_1d"]
#                 ld_inst.loop_1d = inst["loop_1d"]
#                 ld_inst.src_1d_stride = inst["src_1d_stride"]
#                 ld_inst.src_addr = inst["src_addr"]
#                 ld_inst.dst_addr = inst["dst_addr"]
#                 ld_inst.mode = getattr(LDMode, inst["mode"])
#                 ld_inst.loop_direction = getattr(LoopDir, inst["loop_direction"])
#                 ld_inst.src_bank_id = inst["src_bank_id"]
#                 ld_inst.dst_group_id = inst["dst_group_id"]
#                 ld_inst.dst_bank_id = inst["dst_bank_id"]
#                 ld_inst.bank_addr_stride = inst["bank_addr_stride"]

#                 slr_inst_list.append(ld_inst)
#             elif inst["inst_type"] == "store":
#                 wait, release = parse_inst_rely_to_PUType(inst)
#                 st_inst = STInst()
#                 st_inst.wait = wait
#                 st_inst.release = release
#                 st_inst.length_1d = inst["length_1d"]
#                 st_inst.loop_1d = inst["loop_1d"]
#                 st_inst.dst_1d_stride = inst["dst_1d_stride"]
#                 st_inst.dst_addr = inst["dst_addr"]
#                 st_inst.src_addr = inst["src_addr"]
#                 st_inst.src_bank_id = inst["src_bank_id"]
#                 st_inst.src_group_id = inst["src_group_id"]
#                 st_inst.loop_direction = getattr(LoopDir, inst["loop_direction"])
#                 st_inst.bank_addr_stride = inst["bank_addr_stride"]
#                 st_inst.mode = getattr(STMode, inst["mode"])
                
#                 slr_inst_list.append(st_inst)
#             elif inst["inst_type"] == "mm":
#                 pass
#             elif inst["inst_type"] == "misc":
#                 pass
#             elif inst["inst_type"] == "sys":
#                 wait, release = parse_inst_rely_to_PUType(inst)
#                 sys_inst = SYSCInst()
#                 sys_inst.wait = wait
#                 sys_inst.release = release
#                 sys_inst.op = getattr(SysOp, inst["op"])
                
#                 slr_inst_list.append(sys_inst)
#         model_inst_list.append(slr_inst_list)
#     return model_inst_list
