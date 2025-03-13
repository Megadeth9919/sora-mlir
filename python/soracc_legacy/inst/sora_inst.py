from collections import namedtuple
from enum import Enum, IntEnum, unique
from typing import Union, Optional, Any, Sequence
import json
import math
import time
from functools import cache
import bitarray
import tqdm
from dataclasses import dataclass
from bitarray import util

@dataclass
class FieldsInfo:
    commons_name: str = ''
    length: int = 0
    field: str = ''
    default: int = 0

@dataclass
class W_R_Info:
    commons_name: str = ''
    length: int = 0
    field: str = ''
    default: int = 0

@dataclass
class Reserve:
    length: int = 0

@dataclass
class OpField:
    length: int = 0
    value: int = 0

@unique
class PUType(IntEnum):
    No   = 0
    ST   = 1
    LD   = 2
    MM   = 3
    MISC = 4
    SYS  = 5
    RS   = 6


@unique
class LDMode(IntEnum):
    HBM2Bbuffer = 0
    DDR2Bbuffer = 1
    DDR2global = 2
    DDR2Meta = 3
    HBM2Global = 4
    HBM2BbufferNStride = 5


class Inst:
    config: Optional["InstConfig"] = None

    def to_bin(self):
        return type(self).config.to_bin(self)

    def to_json(self):
        my_info = dict(vars(self))
        return member_to_json(my_info)

    def from_json(self, js: dict):
        assign_member_by_json(self, js)

    def __setattr__(self, name, value):
        if __debug__:
            type(self).config.check(name, value)
        object.__setattr__(self, name, value)

class InstConfig:
    def __init__(self,
                 fields_map: tuple[Union[FieldsInfo, Reserve, OpField, W_R_Info], ...],
                 w_r_info_map:tuple[PUType, ...]=tuple()) -> None:
        self.fields_map = fields_map
        self.w_r_info_map = w_r_info_map
        self.name_field_map = {}
        for info in fields_map:
            if not isinstance(info, FieldsInfo):
                continue
            if not info.field:
                continue
            self.name_field_map[info.field] = info

    def check(self, name, value):
        if not isinstance(value, int) :
            return
        if value < 0:
            raise ValueError(f'assign {name}:{value} error')         
        if name not in self.name_field_map:
            return
        info = self.name_field_map[name]
        if value > 2 ** info.length:
            raise ValueError(f'assign {name}:{value} out of range: {2 ** info.length - 1}')

    @staticmethod
    def reverse_each_8_bits(input_bits: bitarray.bitarray) -> bitarray.bitarray:
        # 检查输入的长度是否是8的倍数
        if len(input_bits) % 8 != 0:
            raise ValueError("bitarray 的长度必须是 8 的倍数")

        # 翻转每 8 位块
        reversed_bits = util.zeros(len(input_bits))
        for i in range(0, len(input_bits), 8):
            block = input_bits[i:i+8]  # 提取每 8 位块
            reversed_block = block[::-1]  # 翻转块
            reversed_bits[i:i+8] = reversed_block  # 添加到结果

        return reversed_bits

    def to_bin(self, inst) -> bitarray.bitarray:
        inst_values = vars(inst)

        total_len = 0
        for info in self.fields_map:
            total_len += info.length
        ret = util.zeros(total_len)
        pos = 0
        for info in self.fields_map:
            if isinstance(info, FieldsInfo):
                value = info.default
                if info.field:
                    value = inst_values.get(info.field, info.default)
                else:
                    print(f'no match field: {info}')
                # hotspot
                bin_str = f'{value:0>{info.length}b}'[::-1]
                bin_val = bitarray.bitarray(bin_str)
                ret[pos : pos + info.length] = bin_val
            elif isinstance(info, OpField):
                bin_str = f'{info.value:0>{info.length}b}'[::-1]
                bin_val = bitarray.bitarray(bin_str)
                ret[pos : pos + info.length] = bin_val
            elif isinstance(info, W_R_Info):
                value = 0
                if info.field:
                    for x in inst_values.get(info.field):
                        if x in self.w_r_info_map:
                            value += 2 ** self.w_r_info_map.index(x)
                        else: 
                            continue
                else:
                    raise ValueError(f'unmatched field: {info.field}')
                # no reverse in w_r_field
                bin_str = f'{value:0>{info.length}b}'
                bin_val = bitarray.bitarray(bin_str)
                ret[pos: pos + info.length] = bin_val
            pos += info.length
        return self.reverse_each_8_bits(ret)


class InstCollector:
    registered_inst: dict[str, type[Inst]] = dict()

    def __init__(self):
        self._insts = []

    def add(self, inst):
        if  isinstance(inst,LDInst):
            assert(type(inst.dst_addr) == int)
        self._insts.append(inst)

    def get_insts(self):
        return self._insts

    def to_json(self):
        type2str = {}
        for k, v in InstCollector.registered_inst.items():
            type2str[v] = k

        ret = []
        for inst in self._insts:
            js = inst.to_json()
            js["inst_type"] = type2str[type(inst)]
            ret.append(js)
        return ret

    def from_json(self, js: list[dict]):
        assert not self._insts
        for inst_dict in js:
            t = InstCollector.registered_inst[inst_dict["inst_type"]]
            dict_copy = {k: v for k, v in inst_dict.items() if k != "inst_type"}
            inst = t()
            inst.from_json(dict_copy)
            self._insts.append(inst)

    def __len__(self):
        return len(self._insts)

    def __getitem__(self, key):
        return self._insts[key]

    def __setitem__(self, key, val):
        self._insts[key] = val

    def to_bin(self) -> bytes:
        return insts_to_bin(self._insts)

def insts_to_bin(insts: Sequence[Inst],info) -> bytes:
    with tqdm.tqdm(total=len(insts), desc='Inst2Bin') as pbar:
        ret = bitarray.bitarray()
        offset, size, i = 0, 0, 0
        for index, inst in enumerate(insts):
            ret += inst.to_bin()
            if isinstance(inst, SYSCInst) and inst.op == SysOp.interrupt:
                size = int(len(ret) // 8) - offset
                info[i].offset = offset
                info[i].size = size
                i+=1
                offset = offset + size
            pbar.update(1)
        assert i == len(info)
    return ret.tobytes(), info

def register_inst(name: str, config: InstConfig):
    def wrapper(cls: type[Inst]):
        InstCollector.registered_inst[name] = cls
        cls.config = config
        return cls

    return wrapper


def member_to_json(d: dict)->dict:
    ret: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, IntEnum):
            ret[str(k)] = v.name
        elif isinstance(v, list):
            wait_release_list: list[str] = []
            for e in v:
                wait_release_list.append(e.name)
            ret[str(k)] = wait_release_list
        elif type(v) in (int, float, bool, str):
            ret[str(k)] = v
    return ret


def assign_member_by_json(instance, d: dict):
    for k, v in d.items():
        old = getattr(instance, k)
        if isinstance(old, IntEnum):
            setattr(instance, k, type(old)[v])
        elif isinstance(old, list):
            wait_release_list = []
            for e in v:
                wait_release_list.append(PUType[e])
            setattr(instance, k, wait_release_list)
        elif type(v) in (int, float, bool, str):
            setattr(instance, k, v)

@unique
class LoopDir(IntEnum):
    inter_bank = 0
    in_bank = 1

LD_config_fields_map = (
    FieldsInfo("dst_addr", length=12, field="dst_addr", default=0),
    Reserve(length=3),
    FieldsInfo("mode", length=3, field="mode", default=0),
    W_R_Info("release", length=5, field="release", default=0),
    W_R_Info("wait", length=5, field="wait", default=0),
    OpField(length=4, value=0b0001),
    FieldsInfo("src_1d_stride", length=34, field="src_1d_stride", default=0),
    FieldsInfo("src_addr", length=36, field="src_addr", default=0),
    FieldsInfo("1d_length", length=23, field="length_1d", default=0),
    FieldsInfo("dst_group_id", length=2, field="dst_group_id", default=0),
    FieldsInfo("loop_dir", length=1, field="loop_direction", default=0),
    FieldsInfo("bank_addr_stride", length=12, field="bank_addr_stride", default=0),
    FieldsInfo("dst_bank_id", length=5, field="dst_bank_id", default=0),
    FieldsInfo("1d_loop", length=15, field="loop_1d", default=0)
)
LD_config_w_r_info_map = (PUType.ST, PUType.MM, PUType.MISC, PUType.SYS, PUType.RS)
LD_config = InstConfig(LD_config_fields_map, LD_config_w_r_info_map)

@register_inst("load", LD_config)
class LDInst(Inst):
    config = LD_config
    def __init__(self) -> None:
        self.wait: list[PUType] = []
        self.release: list[PUType] = []
        self.length_1d: int = 0
        self.loop_1d: int = 0
        self.src_1d_stride: int = 0
        self.src_addr: int = 0x0
        self.dst_addr: int = 0x0
        self.mode: LDMode = LDMode.HBM2Bbuffer
        self.loop_direction: LoopDir = LoopDir.inter_bank
        self.src_bank_id: int = 0
        self.dst_group_id: int = 0
        self.dst_bank_id: int = 0
        self.bank_addr_stride: int = 0

ST_config_fields_map = (
    FieldsInfo("src_bank_id", length=4, field="src_bank_id", default=0),
    FieldsInfo("mode", length=2, field="mode", default=0),
    FieldsInfo("src_addr", length=12, field="src_addr", default=0),
    W_R_Info("release", length=5, field="release", default=0),
    W_R_Info("wait", length=5, field="wait", default=0),
    OpField(length=4, value=0b0010),
    FieldsInfo("dst_addr", length=36, field="dst_addr", default=0),
    FieldsInfo("1d_length", length=23, field="length_1d", default=0),
    FieldsInfo("loop_dir", length=1, field="loop_direction", default=0),
    FieldsInfo("src_group_id", length=2, field="src_group_id", default=0),
    FieldsInfo("dst_1d_stride", length=34, field="dst_1d_stride", default=0),
    FieldsInfo("bank_addr_stride", length=12, field="bank_addr_stride", default=0),
    FieldsInfo("1d_loop", length=15, field="loop_1d", default=0),
    Reserve(length=5),
)
ST_config_w_r_info_map = (PUType.LD, PUType.MM, PUType.MISC, PUType.SYS, PUType.RS)
ST_config = InstConfig(ST_config_fields_map, ST_config_w_r_info_map)

@unique
class STMode(IntEnum):
    Global2HBM = 0
    Global2DDR = 1
    Meta2DDR = 2
    


@register_inst("store", ST_config)
class STInst(Inst):
    config = ST_config

    def __init__(self) -> None:
        self.wait: list[PUType] = []
        self.release: list[PUType] = []
        self.length_1d: int = 0
        self.loop_1d: int = 0
        self.dst_1d_stride: int = 0
        self.dst_addr: int = 0x0
        self.src_addr: int = 0x0
        self.src_bank_id: int = 0
        self.src_group_id: int = 0
        self.loop_direction: LoopDir = LoopDir.inter_bank
        self.bank_addr_stride: int = 0
        self.mode: STMode = STMode.Global2DDR

    def to_json(self):
        my_info = dict(vars(self))
        return member_to_json(my_info)

    def from_json(self, js: dict):
        assign_member_by_json(self, js)


@unique
class MMInputMode(IntEnum):
    w4a8 = 0
    w8a8 = 1
    fp16 = 2
    w8a16 = 3


@unique
class MMOutputMode(IntEnum):
    int8 = 0
    fp16 = 1


@unique
class MMAccMode(IntEnum):
    int32 = 0
    int32_fp32 = 1


MM_config_fields_map = (
    FieldsInfo("act_start_addr", length=12, field="act_start_addr", default=0),
    FieldsInfo("input_mode", length=2, field="input_mode", default=0),
    FieldsInfo("act_bank_group_id", length=2, field="act_bank_group_id", default=0),
    FieldsInfo("out_bank_group_id", length=2, field="out_bank_group_id", default=0),
    W_R_Info("release", length=5, field="release", default=0),
    W_R_Info("wait", length=5, field="wait", default=0),
    OpField(length=4, value=0b0011),
    FieldsInfo("weights_start_addr", length=11, field="weights_start_addr", default=0),
    Reserve(length=1),
    FieldsInfo("weights_scale_start_addr", length=10, field="weights_scale_start_addr", default=0),
    Reserve(length=18),
    FieldsInfo("N", length=12, field="N", default=0),
    FieldsInfo("out_start_addr", length=12, field="out_start_addr", default=0),
    FieldsInfo("M", length=12, field="M", default=0),
    FieldsInfo("bias_flag", length=1, field="bias_flag", default=0),
    FieldsInfo("output_mode", length=1, field="output_mode", default=0),
    Reserve(length=2),
    FieldsInfo("K", length=15, field="K", default=0),
    Reserve(length=1),
    FieldsInfo("act_scale_start_addr", length=10, field="act_scale_start_addr", default=0),
    Reserve(length=2),
    FieldsInfo("out_scale_start_addr", length=9, field="out_scale_start_addr", default=0),
    Reserve(length=2),
    FieldsInfo("bias_start_addr", length=9, field="bias_start_addr", default=0),
)
MM_config_w_r_info_map = (PUType.LD, PUType.ST, PUType.MISC, PUType.SYS, PUType.RS)
MM_config = InstConfig(MM_config_fields_map, MM_config_w_r_info_map)


@register_inst("mm_inst", MM_config)
class MMInst(Inst):
    config = MM_config

    def __init__(self) -> None:
        self.wait: list[PUType] = []
        self.release: list[PUType] = []
        self.input_mode: MMInputMode = MMInputMode.w4a8
        self.output_mode: MMOutputMode = MMOutputMode.int8
        self.act_start_addr: int = 0x0
        self.act_bank_group_id: int = 0
        self.out_bank_group_id: int = 0
        self.bias_flag: bool = False
        self.output_flag: bool = False
        self.weights_start_addr: int = 0x0
        self.M: int = 0
        self.relu: bool = False
        self.bias_start_addr: int = 0x0
        self.out_start_addr: int = 0x0
        self.N: int = 0
        self.K: int = 0
        self.act_scale_start_addr: int = 0
        self.out_scale_start_addr: int = 0
        self.weights_scale_start_addr: int = 0x0

    def to_json(self):
        my_info = dict(vars(self))
        return member_to_json(my_info)

    def from_json(self, js: dict):
        assign_member_by_json(self, js)


@unique
class MiscOp(IntEnum):
    lut = 0
    data_convert = 1
    dynamic_int = 2
    abs_max = 3
    elt_add = 4
    elt_sub = 5
    elt_mul = 6
    elt_div = 7
    exp = 8
    tanh = 9
    silu = 10
    gelu0 = 11
    gelu1 = 12
    softmax_pp = 13
    softmax = 14
    layernorm_pp = 15
    layernorm = 16
    rmsnorm_pp = 17
    rmsnorm = 18
    set_k = 125
    load_ins = 127



@unique
class MiscOutput(IntEnum):
    int8 = 0
    fp16 = 1
    int16 = 2

@unique
class MiscMode(IntEnum):
    int8 = 0
    int16 = 1
    int32 = 2
    fp8 = 3
    fp16 = 4
    bf16 = 5
    fp32 = 6
    tf32 = 7


@unique
class MiscDynScale(IntEnum):
    off = 0
    cal_max = 1
    # FIXME 修改命名
    scale_quant = 2


MISC_config_fields_map = (
    FieldsInfo(commons_name="input_a_mode", length=3, field="input_a_mode", default=0),
    FieldsInfo(commons_name="input_b_mode", length=3, field="input_b_mode", default=0),
    FieldsInfo(commons_name="output_mode", length=3, field="output_mode", default=0),
    FieldsInfo(commons_name="op", length=7, field="op", default=0),
    Reserve(length=2),
    W_R_Info(commons_name="release", length=5, field="release", default=0),
    W_R_Info(commons_name="wait", length=5, field="wait", default=0),
    OpField(length=4, value=0b0100),
    FieldsInfo(commons_name="in_a_start_addr", length=12, field="in_a_start_addr", default=0),
    FieldsInfo(commons_name="in_a_bank_id", length=4, field="in_a_bank_id", default=0),   
    FieldsInfo(commons_name="in_b_start_addr", length=12, field="in_b_start_addr", default=0),
    FieldsInfo(commons_name="in_b_bank_id", length=4, field="in_b_bank_id", default=0),
    FieldsInfo(commons_name="out_swap", length=1, field="out_swap", default=0),
    FieldsInfo(commons_name="reg_index", length=12, field="reg_index", default=0),
    Reserve(length=3),
    FieldsInfo(commons_name="out_start_addr", length=12, field="out_start_addr", default=0),
    FieldsInfo(commons_name="out_bank_id", length=4, field="out_bank_id", default=0), 
    Reserve(length=3),
    FieldsInfo(commons_name="meta_addr", length=13, field="meta_addr", default=0),  
    FieldsInfo(commons_name="K", length=16, field="K", default=0),
    FieldsInfo(commons_name="batch", length=14, field="batch", default=0),
    FieldsInfo(commons_name="sp_table_idx", length=10, field="sp_table_idx", default=0),
    FieldsInfo(commons_name="dynamic_scale", length=1, field="dynamic_scale", default=0),
    FieldsInfo(commons_name="batch_flag", length=1, field="batch_flag", default=0),
    FieldsInfo(commons_name="out_bank_group", length=2, field="out_bank_group", default=0),
    FieldsInfo(commons_name="in_b_bank_group", length=2, field="in_b_bank_group", default=0),
    FieldsInfo(commons_name="in_a_bank_group", length=2, field="in_a_bank_group", default=0),  
    FieldsInfo(commons_name="fK", length=32, field="fK", default=0),  
)

MISC_config_w_r_info_map = (PUType.LD, PUType.ST, PUType.MM, PUType.SYS, PUType.RS)
MISC_config = InstConfig(MISC_config_fields_map, MISC_config_w_r_info_map)


@register_inst("misc_inst", MISC_config)
class MISCInst(Inst):
    config = MISC_config

    def __init__(self) -> None:
        self.wait: list[PUType] = []
        self.release: list[PUType] = []
        self.op: MiscOp = MiscOp.elt_add
        self.input_a_mode: MiscMode = MiscMode.int8
        self.input_b_mode: MiscMode = MiscMode.int8
        self.output_mode: MiscMode = MiscMode.int8
        self.in_a_start_addr: int = 0x0
        self.in_a_bank_id: int = 0x0
        self.in_a_bank_group: int = 0x0
        self.in_b_start_addr: int = 0x0
        self.in_b_bank_id: int = 0x0
        self.in_b_bank_group: int = 0x0
        self.out_start_addr: int = 0x0
        self.out_bank_id: int = 0x0
        self.out_bank_group: int = 0x0 
        self.meta_addr: int = 0x0
        self.K: int = 0x0
        self.batch_flag: bool = False
        self.dynamic_scale: bool = False
        self.batch: int = 0x0
        self.sp_table_idx: int = 0x0
        self.out_swap:int =0x0
        self.reg_index:int =0x0
        self.fK:int = 0x0

    def to_json(self):
        my_info = dict(vars(self))
        return member_to_json(my_info)

    def from_json(self, js: dict):
        assign_member_by_json(self, js)


@unique
class SysOp(IntEnum):
    sync = 0
    interrupt = 1


SYS_config_fields_map = (
    Reserve(length=15) ,
    FieldsInfo("interrupt", length=1, field="op", default=0),
    FieldsInfo("optype", length=2, field="", default=0),
    W_R_Info("release", length=5, field="release", default=0),
    W_R_Info("wait", length=5, field="wait", default=0),
    OpField(length=4, value=0b1111),
    Reserve(length=64) ,
)
SYS_config_w_r_info_map = (PUType.LD, PUType.ST, PUType.MM, PUType.MISC, PUType.RS)
SYS_config = InstConfig(SYS_config_fields_map, SYS_config_w_r_info_map)


@register_inst("sys", SYS_config)
class SYSCInst(Inst):
    config = SYS_config

    def __init__(self) -> None:
        self.wait: list[PUType] = []
        self.release: list[PUType] = []
        self.op: SysOp = SysOp.interrupt

    def to_json(self):
        my_info = dict(vars(self))
        return member_to_json(my_info)

    def from_json(self, js: dict):
        assign_member_by_json(self, js)

SPGen_config_fields_map = (
    FieldsInfo("src_addr", length=12, field="src_addr", default=0),
    FieldsInfo("src_group", length=2, field="src_group", default=0),
    Reserve(length=1) ,
    FieldsInfo("table_sel", length=1, field="table_sel", default=0),
    FieldsInfo("optype", length=2, field="", default=1),
    W_R_Info("release", length=5, field="release", default=0),
    W_R_Info("wait", length=5, field="wait", default=0),
    OpField(length=4, value=0b1111),
    FieldsInfo("frame_group_size", length=4, field="frame_group_size", default=0),
    FieldsInfo("torken_group_size", length=8, field="torken_group_size", default=0),
    FieldsInfo("table_offset", length=8, field="table_offset", default=0),
    FieldsInfo("threshold", length=16, field="threshold", default=0),
    Reserve(length=28) ,
)
# FIXME why there is not w_r_info_map
SPGen_config = InstConfig(SPGen_config_fields_map)


@register_inst("sp_gen", SPGen_config)
class SPGenInst(Inst):
    config = SPGen_config

    def __init__(self) -> None:
        self.wait: list[PUType] = []
        self.release: list[PUType] = []
        self.table_sel: int  = 0x0
        self.src_addr: int  = 0x0
        self.src_group: int  = 0x0
        self.frame_group_size: int  = 0x0
        self.torken_group_size: int  = 0x0
        self.table_offset: int  = 0x0
        self.threshold: int  = 0x0

    def to_json(self):
        my_info = dict(vars(self))
        return member_to_json(my_info)

    def from_json(self, js: dict):
        assign_member_by_json(self, js)


SPRecover_config_fields_map = (
    FieldsInfo("src_addr", length=12, field="src_addr", default=0),
    Reserve(length=3) ,
    FieldsInfo("mode", length=1, field="mode", default=0),
    FieldsInfo("optype", length=2, field="", default=2),
    W_R_Info("release", length=5, field="release", default=0),
    W_R_Info("wait", length=5, field="wait", default=0),
    OpField(length=4, value=0b1111),
    FieldsInfo("frame_group_size", length=4, field="frame_group_size", default=0),
    FieldsInfo("torken_group_size", length=8, field="torken_group_size", default=0),
    FieldsInfo("table_offset", length=8, field="table_offset", default=0),
    FieldsInfo("dest_addr", length=10, field="dest_addr", default=0),
    FieldsInfo("dims", length=16, field="dims", default=0),
    Reserve(length=16) ,
)
SPRecover_config = InstConfig(SPRecover_config_fields_map)


@register_inst("sp_recover", SPRecover_config)
class SPRecoverInst(Inst):
    config = SPRecover_config

    def __init__(self) -> None:
        self.wait: list[PUType] = []
        self.release: list[PUType] = []
        self.table_sel: int  = 0x0
        self.src_addr: int  = 0x0
        self.src_group: int  = 0x0
        self.frame_group_size: int  = 0x0
        self.torken_group_size: int  = 0x0
        self.table_offset: int  = 0x0
        self.dest_addr: int  = 0x0
        self.dims: int  = 0x0

    def to_json(self):
        my_info = dict(vars(self))
        return member_to_json(my_info)

    def from_json(self, js: dict):
        assign_member_by_json(self, js)

RS_config_fields_map = (
    FieldsInfo("M", length=14, field="M", default=0),
    FieldsInfo("dst_bank_id", length=4, field="dst_bank_id", default=0),
    W_R_Info("release", length=5, field="release", default=0),
    W_R_Info("wait", length=5, field="wait", default=0),
    OpField(length=4, value=0b0101),
    FieldsInfo("K", length=14, field="K", default=0),
    FieldsInfo("src_addr", length=12, field="src_addr", default=0),
    FieldsInfo("data_type", length=1, field="data_type", default=0),
    Reserve(length=1),
    FieldsInfo("src_bank_id", length=4, field="src_bank_id", default=0),
    FieldsInfo("dst_addr", length=12, field="dst_addr", default=0),
    FieldsInfo("src_bank_group_id", length=2, field="src_bank_group_id", default=0),
    FieldsInfo("dst_bank_group_id", length=2, field="dst_bank_group_id", default=0),
    Reserve(length=16),
)
RS_config_w_r_info_map = (PUType.LD, PUType.ST, PUType.MM, PUType.MISC, PUType.SYS)
RS_config = InstConfig(RS_config_fields_map, RS_config_w_r_info_map)


class RSDataType(IntEnum):
    int8 = 0
    int16 = 1


@register_inst("rs_inst", RS_config)
class RSInst(Inst):
    config = RS_config

    def __init__(self) -> None:
        self.wait: list[PUType] = []
        self.release: list[PUType] = []
        self.src_bank_id: int = 0
        self.dst_bank_id: int = 0
        self.src_bank_group_id: int = 0
        self.dst_bank_group_id: int = 0
        self.src_addr: int = 0x0
        self.dst_addr: int = 0x0
        self.M: int = 0
        self.K: int = 0
        self.data_type: RSDataType = RSDataType.int8

    def to_json(self):
        my_info = dict(vars(self))
        return member_to_json(my_info)

    def from_json(self, js: dict):
        assign_member_by_json(self, js)
