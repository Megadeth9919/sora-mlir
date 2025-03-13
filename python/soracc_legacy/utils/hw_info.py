from enum import Enum
from collections import namedtuple
from inst.sora_inst import MiscOp
import math

class HW_Info:
    @staticmethod
    def get_slr_count() -> int:
        return 3

    @staticmethod
    def get_hbm_start() -> int:
        return 0x4000000000

    # HBM size in bytes
    @staticmethod
    def get_hbm_size() -> int:
        # FIXME: this is just a pseudo value, need to be updated
        return 32 * 1024 * 1024 * 1024

    @staticmethod
    def get_hbm_channels() -> int:
        return 8
    
    @staticmethod
    def get_hbm_channel_size() -> int:
        # FIXME: this is just a pseudo value, need to be updated
        return 1024 * 1024 * 1024
    
    @staticmethod
    def get_channel_stride(bytes: int) -> int:
        return HW_Info.get_hbm_channel_size() // bytes
   
    @staticmethod
    def get_quant_group() -> int:  # 量化组数
        return 128
    
    @staticmethod
    def get_hbm_freq() -> int:  # HBM频率
        return 200 # 250MHz

    @staticmethod
    def get_hbm_axi_num() -> int:  # HBM AXI总线数
        return 8 # 8

    @staticmethod
    def get_hbm_axi_width() -> int:  # HBM AXI数据宽度
        return 256  # 256 bits

    @staticmethod
    def get_hbm_bandwidth() -> int:  # HBM带宽
        return HW_Info.get_hbm_axi_num() * HW_Info.get_hbm_axi_width() * HW_Info.get_hbm_freq() * 1000 * 1000  # bits/s
    
    @staticmethod
    def get_hbm_axi_bits() -> int:
        return HW_Info.get_hbm_axi_num() * HW_Info.get_hbm_axi_width()

    @staticmethod
    def get_hbm_utilization() -> float:  # HBM利用率
        return 0.90
    
    @staticmethod
    def get_ddr_axi_bits() -> int:
        return 1024  # 64 bits
    
    @staticmethod
    def get_ddr_freq() -> int:  # DDR频率
        return 200  # 150MHz
    
    @staticmethod
    def get_ddr_bandwidth() -> int:  # DDR带宽
        return HW_Info.get_ddr_axi_bits() * HW_Info.get_ddr_freq() * 1000 * 1000  # bits/s
    
    @staticmethod
    def get_ddr_utilization() -> float:  # DDR利用率
        return 0.70

    @staticmethod
    def get_main_freq() -> int:  # 频率
        return 400  # 300MHz -> 330/350MHz
    

BufferType = Enum('BufferType', ('Global', 'Weight', 'Meta'))

def high_align(n: int, m: int) -> int:
    return math.ceil(n / m) * m

def low_align(n: int, m: int) -> int:
    return math.floor(n / m) * m

class GlobalBuffer:
    # 每个bank能放多少行的数据
    @staticmethod
    def get_chunk_num_in_bank(bytes_num: int) -> int:
        if bytes_num >= GlobalBuffer.get_bank_bytes_num():
            raise ValueError(f'a line of data({bytes_num}) OOM with global buffer')
        return GlobalBuffer.get_bank_bytes_num() // high_align(bytes_num, GlobalBuffer.get_phy_bytewidth())

    # 单个bank能放多少bytes的数据
    @staticmethod
    def get_bank_bytes_num() -> int:
        return GlobalBuffer.get_depth() * GlobalBuffer.get_bytewidth()

    # LD/ST指令中跨越一个bank的stride值
    @staticmethod
    def get_a_bank_stride() -> int:
        return GlobalBuffer.get_depth()

    @staticmethod
    def get_bank_num_in_BGroup() -> int:
        return 16

    @staticmethod
    def get_Group_num() -> int:
        return 3

    @staticmethod
    def get_depth() -> int:
        return 4096

    @staticmethod
    def get_bitwidth() -> int:
        return 256

    # 最小寻址的地址跳变
    @staticmethod
    def get_bytewidth() -> int:
        return GlobalBuffer.get_bitwidth() // 8

    @staticmethod
    def get_min_stride_num() -> int:
        return 1
    
    @staticmethod
    def get_phy_bytewidth() -> int:
        return GlobalBuffer.get_bitwidth() // 8
    
    @staticmethod
    def get_addr_len_in_bank(bytes_num: int) -> int:
        return high_align(bytes_num, GlobalBuffer.get_phy_bytewidth()) // GlobalBuffer.get_bytewidth()
    
    @staticmethod
    def get_connect_bank_num() -> int:
        return 2

class WeightBuffer:
    @staticmethod
    def get_chunk_num_in_bank(bytes_num: int) -> int:
        return WeightBuffer.get_bank_bytes_num() // high_align(bytes_num, WeightBuffer.get_phy_bytewidth())

    # 单个bank能放多少bytes的数据
    @staticmethod
    def get_bank_bytes_num() -> int:
        return WeightBuffer.get_depth() * WeightBuffer.get_bytewidth()

    # LD/ST指令中跨越一个bank的stride值
    @staticmethod
    def get_a_bank_stride() -> int:
        return WeightBuffer.get_depth()

    @staticmethod
    def get_bank_num() -> int:
        return 16

    @staticmethod
    def get_depth() -> int:
        return 1024

    @staticmethod
    def get_bitwidth() -> int:
        return 256

    # 最小寻址的地址跳变
    @staticmethod
    def get_bytewidth() -> int:
        return WeightBuffer.get_bitwidth() // 8

    @staticmethod
    def sub_bank_num_int() -> int:
        return 4

    @staticmethod
    def sub_bank_num_float() -> int:
        return 2

    @staticmethod
    def get_phy_bytewidth() -> int:
        return WeightBuffer.get_bitwidth()  // 8
    
    @staticmethod
    def get_addr_len_in_bank(bytes_num: int) -> int:
        return high_align(bytes_num, WeightBuffer.get_phy_bytewidth()) // GlobalBuffer.get_bytewidth()
    
    @staticmethod
    def get_connect_bank_num() -> int:
        return 2
        
MetaBankType = Enum('MetaBankType', ('ActScale', 'WeightScale', 'OutScale', 'Bias'))

class MetaBuffer:
    @staticmethod
    def get_chunk_num_in_bank(bytes_num: int) -> int:
        return MetaBuffer.get_bank_bytes_num() // high_align(bytes_num, MetaBuffer.get_phy_bytewidth())

    # 单个bank能放多少bytes的数据
    @staticmethod
    def get_bank_bytes_num() -> int:
        return MetaBuffer.get_depth() * MetaBuffer.get_bytewidth()

    @staticmethod
    def get_a_bank_stride() -> int:
        return MetaBuffer.get_depth()

    @staticmethod
    def get_bank_num() -> int:
        return 4

    @staticmethod
    def get_bank_id(t: MetaBankType) -> int:
        if t == MetaBankType.ActScale:
            return 0
        if t == MetaBankType.WeightScale:
            return 1
        if t == MetaBankType.OutScale:
            return 2
        if t == MetaBankType.Bias:
            return 3

    @staticmethod
    def get_depth() -> int:
        return 512

    @staticmethod
    def get_bitwidth() -> int:
        return 256

    # 最小寻址的地址跳变
    @staticmethod
    def get_bytewidth() -> int:
        return MetaBuffer.get_bitwidth() // 8

    @staticmethod
    def get_min_stride_num() -> int:
        return 2

    @staticmethod
    def get_phy_bytewidth() -> int:
        return GlobalBuffer.get_bitwidth() // 8

    @staticmethod
    def get_addr_len_in_bank(bytes_num: int) -> int:
        return high_align(bytes_num, MetaBuffer.get_phy_bytewidth()) // MetaBuffer.get_bytewidth()

MM_ParallelType = Enum('MM_ParallelType', ('int', 'float'))
MM_ParallelInfo = namedtuple("MM_ParallelInfo", "M K N")

class MMCore:
    @staticmethod
    def get_parallel_dim(t: MM_ParallelType) -> MM_ParallelInfo:
        if t == MM_ParallelType.int:
            return MM_ParallelInfo(16, 32, 16)
        if t == MM_ParallelType.float:
            return MM_ParallelInfo(2, 16, 8)
        raise ValueError(f'unknow type: {t}')
    
    @staticmethod
    def get_weight_bank_num(t: MM_ParallelType) -> int:
        if t == MM_ParallelType.int:
            return 16
        if t == MM_ParallelType.float:
            return 8
        raise ValueError(f'unknow type: {t}')
    
    @staticmethod
    def get_act_bank_num(t: MM_ParallelType) -> int:
        if t == MM_ParallelType.int:
            return 16
        if t == MM_ParallelType.float:
            return 2
        raise ValueError(f'unknow type: {t}')
    
    @staticmethod
    def get_out_bank_num(t: MM_ParallelType) -> int:
        if t == MM_ParallelType.int:
            return 16
        if t == MM_ParallelType.float:
            return 2
        raise ValueError(f'unknow type: {t}')
    
    @staticmethod
    def get_dsp_freq() -> int:  # 频率
        return 200  # 300MHz

    @staticmethod
    def get_cycle_data() -> dict:
        return {
            'START_UP_DELAY': 6,
            'INT_LATENCY': 4,
            'INT_ACCUMULATOR_LATENCY': 1,
            'INT32_TO_FP32_LATENCY': 4,
            'SCALE_MUL_LATENCY': 3 + 4,

            'FP_LATENCY': 13,
            'FP_ACCUMULATOR_LATENCY': 2,
            'FP16_TO_FP32_LATENCY': 1,

            'FP32_TO_FP16_LATENCY': 3,
            'BIAS_LATENCY': 8,
            'RELU_LATENCY': 1,

            'MULTI_QUANT_SCALE_LATENCY': 5 + 5,
            'FP_OUTPUT_LATENCY': 1,

            'INIT_LATENCY': 32
        }

class PVPUCore:
    @staticmethod
    def get_parallel_dim() -> int:
        return 16 * 4

    @staticmethod
    def get_slr_location() -> list[int]:  # PVPUs所在的SLR编号
        return [0, 1, 2]
    
    @staticmethod
    def get_pvpu_freq(div: int = 1) -> int:  # 频率
        return MMCore.get_dsp_freq() // div
    
    @staticmethod
    def get_cycle_data() -> dict:
        return {
            MiscOp.lut: 1,  # To be updated
            MiscOp.data_convert: 47 + 85,
            MiscOp.dynamic_int: 85,
            MiscOp.abs_max: 96,
            MiscOp.elt_add: 17,
            MiscOp.elt_sub: 17,  # To be updated
            MiscOp.elt_mul: 17,
            MiscOp.elt_div: 40,
            MiscOp.exp: 8,  # To be updated
            MiscOp.tanh: 9,  # To be updated
            MiscOp.silu: 75,  # To be updated
            MiscOp.gelu0: 75,
            MiscOp.gelu1: 0,
            MiscOp.softmax_pp: 70,
            MiscOp.softmax: 96,
            MiscOp.layernorm_pp: 104,
            MiscOp.layernorm: 49,
            MiscOp.rmsnorm_pp: 104,
            MiscOp.rmsnorm: 49,
            MiscOp.set_k: 10,  # To be updated
            MiscOp.load_ins: 12  # To be updated
        }

RSDType = Enum('RSDType', ('int8', 'int16'))

class RSModule:
    @staticmethod
    def get_parallel_dim(t: RSDType) -> int:
        if t == RSDType.int8:
            return 32 * 2
        elif t == RSDType.int16:
            return 16 * 2
        else:
            raise ValueError(f'unknow type: {t}')
    @staticmethod
    def get_start_latency() -> int:
        return 75
