# -*-coding:utf-8-*-
# mypy: ignore-errors
# FIXME too many error
# TODO: New SYSInst(Sparse Instruction)
import os
import filecmp
from typing import List, Dict

import numpy as np
import logging

from utils.hw_info import *
from inst.sora_inst import *
from utils import tools
import yaml

NORM_VARIANCE_EPS = 1e-6

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


@unique
class Opcode(IntEnum):
    LD   = 0b0001
    ST   = 0b0010
    MM   = 0b0011
    MISC = 0b0100
    SYS  = 0b1111
    RS   = 0b0101


INST_TYPE = ("LD", "ST", "MM", "MISC", "SYS", "RS")


def get_inst_type(inst: Inst) -> Opcode:
    if isinstance(inst, LDInst):
        return Opcode.LD
    elif isinstance(inst, STInst):
        return Opcode.ST
    elif isinstance(inst, MMInst):
        return Opcode.MM
    elif isinstance(inst, MISCInst):
        return Opcode.MISC
    elif isinstance(inst, RSInst):
        return Opcode.RS
    elif isinstance(inst, SYSCInst):
        return Opcode.SYS
    else:
        raise ValueError("Undefined instruction type!")


class Scheduler:  # 协调指令执行顺序，确保正确的执行逻辑和资源分配。
    def __init__(self, slr_id: int, inst_list: list[Inst]):
        self.slr_id = slr_id
        self.inst_list = inst_list  # 存储每条指令的class
        self.inst_type_num: int = len(INST_TYPE)

        self.inst_type_list: Dict[Opcode, List[Inst]] = {
            Opcode[INST_TYPE[i]]: [] for i in range(self.inst_type_num)
        }  # 存储每种指令的实例列表
        self.depend_reg: Dict[Opcode, Dict[Opcode, int]] = {
            Opcode[INST_TYPE[i]]: {
                Opcode[INST_TYPE[j]]: 0 for j in range(self.inst_type_num)
            }
            for i in range(self.inst_type_num)
        }
        self.total_inst_num = len(inst_list)

        # Inst ID: LD -> 0, ST -> 1, MM -> 2, MISC -> 3, RS -> 4, SYS -> 5
        for _inst in inst_list:
            if isinstance(_inst, LDInst):
                self.inst_type_list[Opcode.LD].append(_inst)
            elif isinstance(_inst, STInst):
                self.inst_type_list[Opcode.ST].append(_inst)
            elif isinstance(_inst, MMInst):
                self.inst_type_list[Opcode.MM].append(_inst)
            elif isinstance(_inst, MISCInst):
                self.inst_type_list[Opcode.MISC].append(_inst)
            elif isinstance(_inst, RSInst):
                self.inst_type_list[Opcode.RS].append(_inst)
            elif isinstance(_inst, SYSCInst):
                self.inst_type_list[Opcode.SYS].append(_inst)
            else:
                logger.error("Undefined instruction type!")

        self.inst_type_cnt: Dict[Opcode, int] = {
            Opcode[INST_TYPE[i]]: 0 for i in range(self.inst_type_num)
        }

    def check_depend_reg(self, module_id, wait_list):
        flag = True
        for _type, depend_r in self.depend_reg[module_id].items():
            if (_type in wait_list) and (depend_r == 0):
                flag = False
                break
        return flag

    def write_depend_reg(self, module_id, wait_list, release_list):
        this_inst_type = Opcode[INST_TYPE[module_id]]
        for wait in wait_list:
            assert isinstance(wait, PUType)
            wait_inst_type = self._pu_type_to_opcode(wait)
            assert wait_inst_type != this_inst_type
            self.depend_reg[this_inst_type][wait_inst_type] -= 1
        for release in release_list:
            assert isinstance(release, PUType)
            release_inst_type = self._pu_type_to_opcode(release)
            assert release_inst_type != this_inst_type
            self.depend_reg[release_inst_type][this_inst_type] += 1

    def print_depend_reg(self, debug=False):
        total_depend_reg = 0

        for module_id in range(self.inst_type_num):
            inst_type = INST_TYPE[module_id]
            depend_reg = self.depend_reg[Opcode[inst_type]]
            total_depend_reg += np.sum([v for _, v in depend_reg.items()])

        if debug:
            for module_id in range(self.inst_type_num):
                inst_type = INST_TYPE[module_id]
                logger.debug(
                    "depend_reg %4s: %s, exec inst num: %d"
                    % (
                        INST_TYPE[module_id],
                        str(self.depend_reg[Opcode[inst_type]]),
                        self.inst_type_cnt[Opcode[inst_type]],
                    )
                )
        if total_depend_reg == 0:
            logger.info("SLR %d depend reg check pass!" % self.slr_id)
        else:
            logger.error("SLR %d depend reg check fail" % self.slr_id)
            for module_id in range(self.inst_type_num):
                inst_type = INST_TYPE[module_id]
                logger.error(
                    "depend_reg %4s: %s, exec inst num: %d"
                    % (
                        INST_TYPE[module_id],
                        str(self.depend_reg[Opcode[inst_type]]),
                        self.inst_type_cnt[Opcode[inst_type]],
                    )
                )
            raise Exception("SLR %d dependency not clear!")

    def depend_reg_warning(self):
        self.print_depend_reg(debug=True)

        for i in range(self.inst_type_num):
            inst_type = INST_TYPE[i]
            logger.warning("======= " + inst_type + " next inst =======")
            if len(self.inst_type_list[Opcode[inst_type]]) > 0:
                logger.warning(self.inst_type_list[Opcode[inst_type]][0])
            else:
                logger.warning("None")

    @staticmethod
    def _pu_type_to_opcode(pu_type: PUType):
        if pu_type == PUType.LD:
            return Opcode.LD
        elif pu_type == PUType.ST:
            return Opcode.ST
        elif pu_type == PUType.MM:
            return Opcode.MM
        elif pu_type == PUType.MISC:
            return Opcode.MISC
        elif pu_type == PUType.RS:
            return Opcode.RS
        elif pu_type == PUType.SYS:
            return Opcode.SYS
        else:
            raise ValueError("Undefined PU type!")

    def parse_wait_release_list(self, wait_release_tag: List[PUType]):
        wait_release_list = []
        for _tag in wait_release_tag:
            wait_release_list.append(self._pu_type_to_opcode(_tag))
        return wait_release_list

    def send_module_inst(self, module_id):
        if len(self.inst_type_list[module_id]) != 0:
            wait_list = self.parse_wait_release_list(
                self.inst_type_list[module_id][0].wait
            )
            # wait_list = self.inst_type_list[module_id][0].wait
            if self.check_depend_reg(module_id, wait_list):
                return self.inst_type_list[module_id].pop(0)
            else:
                return None
        else:
            return None


class PVPUModules:
    def __init__(self, pvpu_id: int):
        self._id = pvpu_id

        self._acc_register: Dict[str, np.ndarray] = dict()
        self._max_register: Optional[np.ndarray] = None
        self._dynamic_scale_register: Optional[np.ndarray] = None
        self._ready_data: List[np.ndarray] = list()
        self._wait_data: Optional[np.ndarray] = None

        self.dynamic_scale_after_sync = None

        self.communication_bus: Optional[List[Dict]] = None
        self.sync_flag = False
        self.init_flag = False

    @property
    def id(self):
        return self._id

    def write_data(self, input_data):
        self._ready_data = input_data

    def read_data(self):
        return self._wait_data

    def init_register(self):
        self._acc_register = dict()
        self._max_register = None

    def communication(self):
        assert (
            self.communication_bus is not None
        ), "Have not initialized MISC communication bus!!!"

        self.communication_bus[self._id] = {
            "max_register": self._max_register,
            "acc_register": self._acc_register,
            "_dynamic_scale_register": self._dynamic_scale_register,
        }

    def sync_acc_reg(self, acc_register, *args):
        if len(args) == 1:
            self._max_register = args[0]
        self._acc_register = acc_register

    def sync_mm_dynamic_scale(self, max_dynamic_scale):
        self.dynamic_scale_after_sync = max_dynamic_scale

    def exec(self, inst_param: MISCInst):
        misc_op = inst_param.op

        if misc_op in (MiscOp.elt_add, MiscOp.elt_mul):
            assert len(self._ready_data) == 2
            self._PVPU_eltwise_exec(inst_param)
        elif misc_op == MiscOp.softmax:
            self._PVPU_softmax_exec(inst_param, preprocess=False)
        elif misc_op == MiscOp.softmax_pp:
            self._PVPU_softmax_exec(inst_param, preprocess=True)
        elif misc_op in (MiscOp.layernorm, MiscOp.rms_layernorm):
            self._PVPU_layernorm_exec(inst_param, preprocess=False)
        elif misc_op in (MiscOp.layernorm_pp, MiscOp.rms_layernorm_pp):
            self._PVPU_layernorm_exec(inst_param, preprocess=True)
        elif misc_op == MiscOp.silu:
            self._PVPU_silu_exec(inst_param)
        elif misc_op == MiscOp.gelu:
            self._PVPU_gelu_exec(inst_param)
        elif misc_op == MiscOp.rope:
            self._PVPU_RoPE_exec(inst_param)
        else:
            logger.error("Undefined MISC operand!")
            raise NotImplementedError
        self.communication()

    def _PVPU_softmax_exec(self, inst_param: MISCInst, preprocess=False):
        K = inst_param.K

        if preprocess:
            if inst_param.last_flag == 1:
                self.init_flag = True

            data = self._ready_data[0].astype("float32")
            assert data.size == K, "MISC input data size does not match K !!!"
            if self._max_register is None:
                self._max_register = np.max(data).astype("float32")
                max_diff = self._max_register
            else:
                max_diff = self._max_register
                self._max_register = np.max([self._max_register, np.max(data)]).astype(
                    "float32"
                )
                max_diff = self._max_register - max_diff

            if len(self._acc_register.keys()) == 0:
                self._acc_register = {
                    "exp_mean": np.sum(np.exp((data - self._max_register)))
                }
            else:
                self._acc_register["exp_mean"] = self._acc_register[
                    "exp_mean"
                ] * np.exp(-max_diff)
                self._acc_register["exp_mean"] = self._acc_register[
                    "exp_mean"
                ] + np.sum(np.exp((data - self._max_register)))

            self._wait_data = None

        else:
            data = self._ready_data[0].astype("float32")

            exp_acc = self._acc_register["exp_mean"]

            out_data = (np.exp(data - self._max_register) / exp_acc).astype("float16")

            out_mode = inst_param.output_mode
            if out_mode in (0, 2):  # int8/int16
                out_int_scale = np.array(inst_param.out_int_scale).astype("uint16")
                out_int_scale = tools.int16_to_float16_tensor(out_int_scale)
                if out_mode == 0:
                    out_dtype = "int8"
                elif out_mode == 2:
                    out_dtype = "int16"
                else:
                    raise ValueError
                out_data = tools.quant_output_to_int(out_data, out_int_scale, out_dtype)
                out_data = out_data.view(dtype="int8")
            elif out_mode == 1:  # fp16
                out_data = out_data.astype("float16")
                out_data = tools.set_subnormal_fp_value_to_zero(out_data, "float16")
                out_data = tools.float16_to_int8_tensor(out_data)
            else:
                logger.error("Undefined output mode!")
                raise NotImplementedError

            self._wait_data = out_data

            if self.init_flag or self.sync_flag:
                self.sync_flag = self.init_flag = False
                self.init_register()

        self._ready_data = list()

    def _PVPU_layernorm_exec(self, inst_param: MISCInst, preprocess=False):
        norm_type = inst_param.op
        K = inst_param.K
        out_data: np.ndarray = np.zeros(K, dtype="float32")

        if norm_type in (
            MiscOp.rms_layernorm_pp,
            MiscOp.rms_layernorm,
        ):  # RMSNorm (self._acc_register = [K, square_sum])
            if preprocess:
                if inst_param.last_flag == 1:
                    self.init_flag = True
                data = self._ready_data[0].astype("float32")
                assert K == data.size, "MISC input data size does not match K !!!"
                _K = np.array(K, dtype="int32")
                if len(self._acc_register.keys()) == 0:
                    square_sum = np.sum(np.square(data)).astype("float32")
                    self._acc_register = {"count": _K, "square_sum": square_sum}
                else:
                    old_num = self._acc_register["count"]
                    old_acc = self._acc_register["square_sum"]
                    new_num = np.array(K, dtype="int32")
                    new_acc = np.sum(np.square(data)).astype("float32")

                    self._acc_register["count"] = old_num + new_num
                    self._acc_register["square_sum"] = old_acc + new_acc
            else:
                assert (
                    len(self._ready_data) == 2
                ), "MISC(rms_layernorm) input data size does not match !!!"
                data = self._ready_data[0].astype("float32")
                weights = self._ready_data[1].astype("float32")

                square_sum = np.float32(self._acc_register["square_sum"]) / np.float32(
                    self._acc_register["count"]
                )
                rms = 1 / np.sqrt(
                    np.float32(square_sum) + np.float32(NORM_VARIANCE_EPS)
                ).astype("float32")

                out_data = data * rms
                out_data = weights * out_data.astype("float32")
                out_data = out_data.astype("float16")

        else:  # Layer Norm (self._acc_register = [K, mean, square_sum])
            if preprocess:
                if inst_param.last_flag == 1:
                    self.init_flag = True
                data = self._ready_data[0].astype("float32")
                assert K == data.size, "MISC input data size does not match K !!!"
                _K = np.array(K, dtype="float32")
                if len(self._acc_register.keys()) == 0:
                    data_sum = np.sum(data).astype("float32")
                    square_sum = np.sum(np.square(data)).astype("float32")
                    self._acc_register = {
                        "count": _K,
                        "sum": data_sum,
                        "square_sum": square_sum,
                    }
                else:
                    old_num = self._acc_register["count"]
                    old_sum = self._acc_register["sum"]
                    old_square_sum = self._acc_register["square_sum"]
                    new_num = np.array(K, dtype="int32")
                    new_sum = np.sum(data).astype("float32")
                    new_square_sum = np.sum(np.square(data)).astype("float32")

                    self._acc_register["count"] = old_num + new_num
                    self._acc_register["sum"] = old_sum + new_sum
                    self._acc_register["square_sum"] = old_square_sum + new_square_sum
            else:
                assert (
                    len(self._ready_data) == 2
                ), "MISC(layernorm) input data size does not match !!!"
                data = self._ready_data[0].astype("float32")
                weights = self._ready_data[1].astype("float32")

                mean = self._acc_register["sum"].astype("float32") / np.float32(
                    self._acc_register["count"]
                )
                square_mean = self._acc_register["square_sum"].astype(
                    "float32"
                ) / np.float32(self._acc_register["count"])

                variance = np.float32(square_mean) - np.square(np.float32(mean))
                variance_reciprocal = 1 / np.sqrt(
                    np.float32(variance) + np.float32(NORM_VARIANCE_EPS)
                )
                out_data = (np.float32(data) - np.float32(mean)) * np.float32(
                    variance_reciprocal
                ).astype("float32")
                out_data = out_data.astype(np.float32) * weights.astype(np.float32)
                out_data = out_data.astype("float16")

        if preprocess:
            self._wait_data = None

        else:
            out_mode = inst_param.output_mode
            if out_mode == 0:  # int8
                out_int_scale = np.array(inst_param.out_int_scale).astype("uint16")
                out_int_scale = tools.int16_to_float16_tensor(out_int_scale)
                out_data = tools.quant_output_to_int(out_data, out_int_scale, "int8")
            elif out_mode == 1:  # fp16
                out_data = out_data.astype("float16")
                out_data = tools.set_subnormal_fp_value_to_zero(out_data, "float16")
                out_data = tools.float16_to_int8_tensor(out_data)
            else:
                logger.error("Undefined output mode!")
                raise NotImplementedError
            self._wait_data = out_data

            if self.init_flag or self.sync_flag:
                self.sync_flag = self.init_flag = False
                self.init_register()
        self._ready_data = list()

    def _PVPU_silu_exec(self, inst_param: MISCInst):
        data = self._ready_data[0].astype("float32")

        reciprocal = 1 / (1 + np.exp(-data))
        out_data = data * reciprocal.astype("float32")
        out_data = out_data.astype("float16")

        out_mode = inst_param.output_mode
        if out_mode == 0:  # int8
            out_int_scale = np.array(inst_param.out_int_scale).astype("uint16")
            out_int_scale = tools.int16_to_float16_tensor(out_int_scale)
            out_data = tools.quant_output_to_int(out_data, out_int_scale, "int8")
        elif out_mode == 1:  # fp16
            out_data = out_data.astype("float16")
            out_data = tools.set_subnormal_fp_value_to_zero(out_data, "float16")
            out_data = tools.float16_to_int8_tensor(out_data)
        else:
            logger.error("Undefined output mode!")
            raise NotImplementedError

        self._wait_data = out_data
        self._ready_data = list()

    def _PVPU_gelu_exec(self, inst_param: MISCInst):
        data = self._ready_data[0].astype("float32")

        out_data = data * tools.erf(data)
        out_data = out_data.astype("float16")

        out_mode = inst_param.output_mode
        if out_mode == 0:  # int8
            out_int_scale = np.array(inst_param.out_int_scale).astype("uint16")
            out_int_scale = tools.int16_to_float16_tensor(out_int_scale)
            out_data = tools.quant_output_to_int(out_data, out_int_scale, "int8")
        elif out_mode == 1:  # fp16
            out_data = out_data.astype("float16")
            out_data = tools.set_subnormal_fp_value_to_zero(out_data, "float16")
            out_data = tools.float16_to_int8_tensor(out_data)
        else:
            logger.error("Undefined output mode!")
            raise NotImplementedError

        self._wait_data = out_data
        self._ready_data = list()

    def _PVPU_eltwise_exec(self, inst_param: MISCInst):
        operate = inst_param.op

        A_data = self._ready_data[0].astype("float16")
        B_data = self._ready_data[1].astype("float16")

        if operate == MiscOp.elt_add:  # add
            out_data = tools.hardware_float_eltwise_add_mul(A_data, B_data, "add")
        elif operate == MiscOp.elt_mul:  # mul
            out_data = tools.hardware_float_eltwise_add_mul(A_data, B_data, "mul")
        elif operate == "eltwise_mul_dynamic_scale_preprocess":
            out_data = tools.hardware_float_eltwise_add_mul(
                A_data, B_data, "eltwise_mul"
            )
            self._dynamic_scale_register = (
                np.float16(np.max(np.abs(out_data))) / 127
            ).astype("float16")
            return  # eltwise_mul_dynamic_scale_preprocess do not write results back to global buffer
        else:
            logger.error("Undefined operand!")
            raise NotImplementedError

        out_data = tools.set_subnormal_fp_value_to_zero(out_data, "float16")
        out_mode = inst_param.output_mode
        if out_mode == 0:  # int8
            out_int_scale = np.array(inst_param.out_int_scale).astype("uint16")
            out_int_scale = tools.int16_to_float16_tensor(out_int_scale)
            if np.isnan(out_int_scale):
                assert inst_param.out_int_scale == tools.nan_fp16_hex  # NaN
                assert self.dynamic_scale_after_sync is not None
                out_int_scale = (1 / self.dynamic_scale_after_sync).astype("float16")
            out_data = tools.quant_output_to_int(out_data, out_int_scale, "int8")
        elif out_mode == 1:  # fp16
            out_data = out_data.astype("float16")
            out_data = tools.set_subnormal_fp_value_to_zero(out_data, "float16")
            out_data = tools.float16_to_int8_tensor(out_data)
        else:
            logger.error("Undefined output mode!")
            raise NotImplementedError

        self._wait_data = out_data
        self._ready_data = list()

    def _PVPU_RoPE_exec(self, inst_param: MISCInst):
        data = self._ready_data[0]

        """  LLaMA-2 Half Rotary
        cos_cache = self._ready_data[1]
        sin_cache = self._ready_data[2]

        out_data = data * cos_cache + tools.rotate_half(data) * sin_cache
        """

        cos_cache = self._ready_data[1]
        sin_cache = self._ready_data[2]

        cos_cache = cos_cache[..., : cos_cache.shape[-1] // 2].repeat(2, axis=-1)
        sin_cache = sin_cache[..., : sin_cache.shape[-1] // 2].repeat(2, axis=-1)

        out_data = data * cos_cache + tools.rotate_adjacent(data) * sin_cache

        out_mode = inst_param.output_mode
        if out_mode == 0:  # int8
            out_int_scale = np.array(inst_param.out_int_scale).astype("uint16")
            out_int_scale = tools.int16_to_float16_tensor(out_int_scale)
            out_data = tools.quant_output_to_int(out_data, out_int_scale, "int8")
        elif out_mode == 1:  # fp16
            out_data = out_data.astype("float16")
            out_data = tools.set_subnormal_fp_value_to_zero(out_data, "float16")
            out_data = tools.float16_to_int8_tensor(out_data)
        else:
            logger.error("Undefined output mode!")
            raise NotImplementedError

        self._wait_data = out_data


class SLRModule:
    def __init__(
        self,
        SLR_id: int,
    ):
        self.pvpu = PVPUModules(SLR_id)

        self.hbm: Optional[np.ndarray] = None  # waiting for connect

        self.weight_buffer = np.zeros(
            shape=(
                WeightBuffer.get_bank_num(),
                WeightBuffer.get_depth(),
                WeightBuffer.get_bytewidth(),
            ),
            dtype=np.int8,
        )  # 仿真时每个channel的bank合并为1个

        self.meta_buffer = np.zeros(
            shape=(
                MetaBuffer.get_bank_num(),
                MetaBuffer.get_depth(),
                MetaBuffer.get_bytewidth(),
            ),
            dtype=np.int8,
        )
        # 用于存储scale/zero_point等信息

        self.global_buffer = np.zeros(
            shape=(
                GlobalBuffer.get_Group_num() * GlobalBuffer.get_bank_num_in_BGroup(),
                GlobalBuffer.get_depth(),
                GlobalBuffer.get_bytewidth(),
            ),
            dtype=np.int8,
        )

        self.weight_bank_num = WeightBuffer.get_bank_num()
        self.weight_buffer_depth = WeightBuffer.get_depth()
        self.weight_buffer_bank_width = WeightBuffer.get_bytewidth()
        self.weight_buffer_size = (
            WeightBuffer.get_depth() * WeightBuffer.get_bytewidth()
        )

        self.meta_buffer_depth = MetaBuffer.get_depth()
        self.meta_buffer_bank_width = MetaBuffer.get_bytewidth()
        self.meta_buffer_total_width = (
            MetaBuffer.get_bytewidth() * MetaBuffer.get_bank_num()
        )
        self.meta_buffer_size = MetaBuffer.get_depth() * MetaBuffer.get_bytewidth()

        self.global_buffer_depth = GlobalBuffer.get_depth()
        self.global_buffer_bank_width = GlobalBuffer.get_bytewidth()
        self.global_buffer_size = (
            GlobalBuffer.get_depth() * GlobalBuffer.get_bytewidth()
        )

        self.MM_INT_PARALLELISM = MMCore.get_parallel_dim(MM_ParallelType.int)
        self.MM_FP_PARALLELISM = MMCore.get_parallel_dim(MM_ParallelType.float)

        self.SLR_id = SLR_id
        self.hbm_channel_base = SLR_id * HW_Info.get_hbm_channels()

    # def _buffer_read(
    #     self,
    #     buffer_type: BufferType,
    #     bank_id: int,
    #     start_addr: int,
    #     addr_stride: int,
    #     data_size: int,
    #     dtype: str = "int8",
    #     loop: int = 0,
    # ):
    #     # TODO: to support global buffer new feature
    #     data_byte = tools.get_byte_by_dtype(dtype)

    #     if buffer_type == BufferType.Global:
    #         target_buffer = self.global_buffer
    #         target_buffer_width = self.global_buffer_bank_width
    #         target_buffer_depth = self.global_buffer_depth
    #     elif buffer_type == BufferType.Meta:
    #         target_buffer = self.meta_buffer
    #         target_buffer_width = self.meta_buffer_bank_width
    #         target_buffer_depth = self.meta_buffer_depth
    #     elif buffer_type == BufferType.Weight:
    #         target_buffer = self.weight_buffer
    #         target_buffer_width = self.weight_buffer_bank_width
    #         target_buffer_depth = self.weight_buffer_depth
    #     else:
    #         raise ValueError

    #     length, residue = tools.div_mod(int(data_size * data_byte), target_buffer_width)
    #     overflow_flag = start_addr + length + loop * addr_stride > target_buffer_depth
    #     assert length + int(overflow_flag) <= target_buffer_depth
    #     if overflow_flag:
    #         end_addr = (start_addr + length) % target_buffer_depth
    #     else:
    #         end_addr = (
    #             start_addr + length
    #         )  # differ when end_addr == target_buffer_depth
    #     assert not length == residue == 0

    #     load_data = np.array([])
    #     if buffer_type == BufferType.Global:
    #         if loop > 1:
    #             loop_data_size = tools.divide_exactly(data_size, loop)
    #             for i in range(loop):
    #                 # Confirm data length and residue for each loop
    #                 loop_length, loop_residue = tools.div_mod(
    #                     int(loop_data_size * data_byte), target_buffer_width
    #                 )
    #                 loop_overflow_flag = start_addr + loop_length > target_buffer_depth
    #                 if loop_overflow_flag:
    #                     end_addr = (start_addr + loop_length) % target_buffer_depth
    #                 else:
    #                     end_addr = (
    #                         start_addr + loop_length
    #                     )  # differ when end_addr == target_buffer_depth
    #                 assert not loop_length == loop_residue == 0

    #                 if loop_overflow_flag:
    #                     load_data_frt = target_buffer[bank_id, start_addr:, :].flatten()
    #                     load_data_lst = target_buffer[bank_id, 0:end_addr, :].flatten()

    #                     load_data = np.concatenate(
    #                         [load_data, load_data_frt, load_data_lst]
    #                     )

    #                 else:
    #                     load_data_tem = target_buffer[
    #                         bank_id, start_addr:end_addr, :
    #                     ].flatten()

    #                     load_data = np.concatenate([load_data, load_data_tem])

    #                 if loop_residue > 0:
    #                     load_data = np.concatenate(
    #                         [load_data, target_buffer[bank_id, end_addr, :loop_residue]]
    #                     )

    #                 # Update start address for each loop
    #                 start_addr = (
    #                     end_addr + int(loop_residue > 0) + addr_stride
    #                 ) % target_buffer_depth

    #         else:
    #             if overflow_flag:
    #                 load_data_frt = target_buffer[bank_id, start_addr:, :].flatten()
    #                 load_data_lst = target_buffer[bank_id, 0:end_addr, :].flatten()

    #                 load_data = np.concatenate([load_data_frt, load_data_lst])

    #             else:
    #                 load_data = target_buffer[bank_id, start_addr:end_addr, :].flatten()

    #             if residue > 0:
    #                 load_data = np.concatenate(
    #                     [load_data, target_buffer[bank_id, end_addr, :residue]]
    #                 )

    #     elif buffer_type == BufferType.Meta:
    #         if overflow_flag:
    #             load_data_frt = target_buffer[:, start_addr:, :]
    #             load_data_lst = target_buffer[:, 0:end_addr, :]

    #             load_data = np.concatenate([load_data_frt, load_data_lst], axis=1)

    #         else:
    #             load_data = target_buffer[:, start_addr:end_addr, :]

    #         if residue > 0:
    #             load_data = np.concatenate(
    #                 [load_data, target_buffer[:, end_addr : (end_addr + 1), :]], axis=1
    #             )

    #     elif buffer_type == BufferType.Weight:
    #         if overflow_flag:
    #             load_data_frt = target_buffer[:, start_addr:, :]
    #             load_data_lst = target_buffer[:, 0:end_addr, :]

    #             load_data = np.concatenate([load_data_frt, load_data_lst], axis=1)

    #         else:
    #             load_data = target_buffer[:, start_addr:end_addr, :]

    #         if residue > 0:
    #             load_data = np.concatenate(
    #                 [load_data, target_buffer[:, end_addr : (end_addr + 1), :]], axis=1
    #             )

    #     return load_data.astype("int8")

    def _buffer_read(
        self,
        buffer_type: BufferType,
        bank_id: int = 0,
        start_addr: int = 0,
        bank_addr_stride: int = 0,
        data_size: int = 0,
        dtype: str = "int8",
        loop: int = 1,
        loop_dir: int = 0,
    ):
        data_type = tools.get_byte_by_dtype(dtype)

        if buffer_type == BufferType.Global:
            target_buffer = self.global_buffer
            target_buffer_width = self.global_buffer_bank_width
            target_buffer_depth = self.global_buffer_depth
            bank_num = GlobalBuffer.get_bank_num_in_BGroup()

        elif buffer_type == BufferType.Meta:
            target_buffer = self.meta_buffer
            target_buffer_width = self.meta_buffer_bank_width
            target_buffer_depth = self.meta_buffer_depth

        elif buffer_type == BufferType.Weight:
            pass

        data = np.array([])
        loop_data_size = data_size // loop

        if buffer_type == BufferType.Global:
            for i in range(loop):
                loop_length, loop_residue = tools.div_mod(
                    int(loop_data_size * data_type), target_buffer_width
                )

                if loop_dir == 0: # inter_bank
                    assert (
                        loop_data_size * data_type
                        < target_buffer_width * target_buffer_depth
                    ), "Global buffer read overflow!"
                    index = i % bank_num
                    load_data = target_buffer[
                        bank_id + index, start_addr : start_addr + loop_length, :
                    ].flatten()

                    data = np.concatenate([data, load_data])

                    if loop_residue > 0:
                        data = np.concatenate(
                            [
                                data,
                                target_buffer[
                                    bank_id + index,
                                    start_addr + loop_length,
                                    :loop_residue,
                                ],
                            ]
                        )

                    start_addr = start_addr + bank_addr_stride * math.floor(
                        i / bank_num
                    )
                else: # intra_bank
                    assert (
                        data_size * data_type
                        < target_buffer_width * target_buffer_depth
                    ), "Global buffer read overflow!"
                    load_data = target_buffer[
                        bank_id, start_addr : start_addr + loop_length, :
                    ].flatten()

                    data = np.concatenate([data, load_data])

                    if loop_residue > 0:
                        data = np.concatenate(
                            [
                                data,
                                target_buffer[
                                    bank_id, start_addr + loop_length, :loop_residue
                                ],
                            ]
                        )
                    start_addr = start_addr + bank_addr_stride

        elif buffer_type == BufferType.Meta:
            for i in range(loop):
                loop_length, loop_residue = tools.div_mod(
                    int(loop_data_size * data_type), target_buffer_width
                )

                assert (
                    loop_data_size * data_type
                    < target_buffer_width * target_buffer_depth
                ), "Meta buffer read overflow!"

                load_data = target_buffer[
                    bank_id, start_addr : start_addr + loop_length, :
                ].flatten()

                data = np.concatenate([data, load_data])

                if loop_residue > 0:
                    data = np.concatenate(
                        [
                            data,
                            target_buffer[
                                bank_id, start_addr + loop_length, :loop_residue
                            ],
                        ]
                    )

                start_addr = start_addr + bank_addr_stride
        elif buffer_type == BufferType.Weight:
            pass
        else:
            raise ValueError

        return data.astype(np.int8)

    # def _buffer_write(self,
    #                   buffer_type: BufferType,
    #                   bank_id: int,
    #                   start_addr: int,
    #                   addr_stride: int,
    #                   data: np.ndarray,
    #                   dtype: str = "int8",
    #                   loop: int = 1
    #                   ):
    #     # TODO: to support global buffer new feature
    #     assert len(data.shape) == 1
    #     data_byte = tools.get_byte_by_dtype(dtype)

    #     length, residue, end_addr = 0, 0, 0
    #     data_size = data.size

    #     if buffer_type == BufferType.Global:
    #         assert addr_stride % 2 == 0
    #         target_buffer = self.global_buffer
    #         target_buffer_width = self.global_buffer_bank_width
    #         target_buffer_depth = self.global_buffer_depth
    #         # target_buffer_min_write_addr_jump = GlobalBuffer.get_min_stride_num()  # 代表1d_loop每次跳地址的最小间隔

    #     elif buffer_type == BufferType.Meta:
    #         assert addr_stride == 0
    #         target_buffer = self.meta_buffer
    #         target_buffer_width = self.meta_buffer_bank_width
    #         target_buffer_depth = self.meta_buffer_depth
    #         # target_buffer_min_write_addr_jump = MetaBuffer.get_min_stride_num()
    #         # write meta buffer may have residue

    #     elif buffer_type == BufferType.Weight:
    #         assert addr_stride % 2 == 0
    #         target_buffer = self.weight_buffer  # 需要注意 float16模式下，只写每个channel中的前一半bank
    #         target_buffer_width = self.weight_buffer_bank_width
    #         target_buffer_depth = self.weight_buffer_depth
    #         # target_buffer_min_write_addr_jump = 2  # WeightBuffer.get_min_stride_num()
    #         assert residue == 0

    #     else:
    #         raise ValueError

    #     # Load weight buffer 和 meta buffer 的地址必须是2的整数倍
    #     # 同时一定要注意，weight buffer和meta buffer，如果1d_length对应的bank内地址是单数，下一次1d_loop跳地址会按照双数去跳
    #     # 例如1d_length = 1, 1d_loop = 2, 会写地址0的一个数，地址2的一个数
    #     # assert start_addr % target_buffer_min_write_addr_jump == 0
    #     buffer_width = target_buffer.shape[-1]

    #     if buffer_type in (BufferType.Global, BufferType.Meta, BufferType.Weight):
    #         loop_data_size = data_size // loop
    #         for i in range(loop):
    #             # Load data for each loop
    #             loop_load_data = data[i * loop_data_size: (i + 1) * loop_data_size]

    #             # Confirm data length and residue for each loop
    #             loop_length, loop_residue = tools.div_mod(int(loop_data_size * data_byte), target_buffer_width)
    #             loop_overflow_flag = (start_addr + loop_length > target_buffer_depth)
    #             if loop_overflow_flag:
    #                 end_addr = (start_addr + loop_length) % target_buffer_depth
    #             else:
    #                 end_addr = start_addr + loop_length  # differ when end_addr == target_buffer_depth
    #             assert not loop_length == loop_residue == 0

    #             if loop_overflow_flag:
    #                 length_tem = loop_length - end_addr
    #                 load_data_frt = loop_load_data[:length_tem * buffer_width].reshape([-1, buffer_width])
    #                 load_data_lst = loop_load_data[length_tem * buffer_width:loop_length * buffer_width].reshape([-1, buffer_width])

    #                 target_buffer[bank_id, start_addr:, :] = load_data_frt
    #                 target_buffer[bank_id, 0: end_addr, :] = load_data_lst
    #             else:
    #                 load_data_ord = loop_load_data[:loop_length * buffer_width].reshape([-1, buffer_width])

    #                 target_buffer[bank_id, start_addr: end_addr, :] = load_data_ord

    #             if loop_residue > 0:
    #                 target_buffer[bank_id, end_addr, :loop_residue] = loop_load_data[loop_length * buffer_width:]

    #             # Update start address for each loop, addr_stride must be a multiple of 2
    #             start_addr = (end_addr + int(loop_residue > 0) + addr_stride) % target_buffer_depth
    # start_addr = tools.align_ceil(start_addr, target_buffer_min_write_addr_jump)

    def _buffer_write(
        self,
        buffer_type: BufferType,
        bank_id: int = 0,
        start_addr: int = 0,
        bank_addr_stride: int = 0,
        data: np.ndarray = 0,
        dtype: str = "int8",
        loop: int = 1,
        loop_dir: int = 0,
    ):
        assert len(data.shape) == 1, "Data shape must be 1D!"
        assert data.size != 0, "Data size must be greater than 0!"

        data_byte = tools.get_byte_by_dtype(dtype)

        if buffer_type == BufferType.Global:
            assert (
                bank_addr_stride % 2 == 0
            ), "Global buffer address stride must be a multiple of 2!"
            target_buffer = self.global_buffer
            target_buffer_width = self.global_buffer_bank_width
            target_buffer_depth = self.global_buffer_depth
            bank_num = GlobalBuffer.get_bank_num_in_BGroup()

        elif buffer_type == BufferType.Meta:
            assert bank_addr_stride == 0, "Meta buffer address stride must be 0!"
            target_buffer = self.meta_buffer
            target_buffer_width = self.meta_buffer_bank_width
            target_buffer_depth = self.meta_buffer_depth
            bank_num = MetaBuffer.get_bank_num()

        elif buffer_type == BufferType.Weight:
            assert (
                bank_addr_stride % 2 == 0
            ), "Weight buffer address stride must be a multiple of 2!"
            target_buffer = self.weight_buffer
            target_buffer_width = self.weight_buffer_bank_width
            target_buffer_depth = self.weight_buffer_depth
            bank_num = WeightBuffer.get_bank_num()

        loop_length = data.size // loop
        for i in range(loop):
            loop_data = data[i * loop_length : (i + 1) * loop_length]

            loop_depth, loop_residue = tools.div_mod(
                int(loop_length * data_byte), target_buffer_width
            )
            # TODO:
            if loop_dir == 0:  # inter-bank
                assert (
                    loop_data.size * data_byte
                    <= target_buffer_width * target_buffer_depth
                ), "Data size exceeds buffer size!"

                loop_data_reorder = loop_data[
                    : loop_depth * target_buffer_width
                ].reshape([-1, target_buffer_width])

                index = i % bank_num

                target_buffer[
                    bank_id + index, start_addr : start_addr + loop_depth, :
                ] = loop_data_reorder

                if loop_residue > 0:
                    target_buffer[
                        bank_id + index, start_addr + loop_depth, :loop_residue
                    ] = loop_data[loop_depth * target_buffer_width :]
                start_addr = start_addr + bank_addr_stride * math.floor(i / bank_num)
            else:  # intra-bank
                assert (
                    data.size * data_byte <= target_buffer_width * target_buffer_depth
                ), "Data size exceeds buffer size!"
                end_addr = start_addr + loop_depth
                loop_data_reorder = loop_data[
                    : loop_depth * target_buffer_width
                ].reshape([-1, target_buffer_width])
                target_buffer[bank_id, start_addr:end_addr, :] = loop_data_reorder
                if loop_residue > 0:
                    target_buffer[bank_id, end_addr, :loop_residue] = loop_data[
                        loop_depth * target_buffer_width :
                    ]
                start_addr = start_addr + bank_addr_stride
        pass

    def exec_inst(self, inst: Inst):
        assert self.hbm is not None, "HBM data has not been inited!"

        if isinstance(inst, LDInst):
            self.LD_exec(inst)
        elif isinstance(inst, STInst):
            self.ST_exec(inst)
        elif isinstance(inst, MMInst):
            self.MM_exec(inst)
        elif isinstance(inst, MISCInst):
            self.PVPU_exec(inst)
        elif isinstance(inst, RSInst):
            self.RS_exec(inst)
        else:
            logger.error("Unknown inst type: %s" % str(type(inst)))
            raise ValueError

    def LD_exec(self, ld_inst: LDInst):

        total_data_size = ld_inst.length_1d * ld_inst.loop_1d
        hbm_start_addr = ld_inst.src_addr

        # read data from main memory
        load_data = np.zeros(total_data_size, dtype=np.int8)
        for loop_id in range(ld_inst.loop_1d):
            real_length = ld_inst.length_1d
            stride_length = ld_inst.src_1d_stride
            load_data[loop_id * real_length : (loop_id + 1) * real_length] = self.hbm[
                hbm_start_addr
                + loop_id * stride_length : hbm_start_addr
                + loop_id * stride_length
                + real_length
            ]

        # load buffer start address
        buffer_bank_id = ld_inst.dst_bank_id
        buffer_start_addr = ld_inst.dst_addr
        buffer_group_id = ld_inst.dst_group_id
        bank_addr_stride = ld_inst.bank_addr_stride  # must be a multiple of 2
        loop_direction = ld_inst.loop_direction

        if ld_inst.mode == LDMode.HBM2Bbuffer:  # weight buffer
            self._buffer_write(
                buffer_type=BufferType.Weight,
                bank_id=buffer_bank_id,
                start_addr=buffer_start_addr,
                addr_stride=bank_addr_stride,
                data=load_data,
                dtype="int8",
                loop=ld_inst.loop_1d,
            )

        elif ld_inst.mode == LDMode.HBM2global:  # global buffer
            self._buffer_write(
                buffer_type=BufferType.Global,
                bank_id=buffer_group_id * GlobalBuffer.get_bank_num_in_BGroup()
                + buffer_bank_id,
                start_addr=buffer_start_addr,
                bank_addr_stride=bank_addr_stride,
                data=load_data,
                dtype="int8",
                loop=ld_inst.loop_1d,
                loop_dir=loop_direction,
            )

        elif ld_inst.mode == LDMode.HBM2Meta:  # meta buffer
            self._buffer_write(
                buffer_type=BufferType.Meta,
                bank_id=buffer_bank_id,
                start_addr=buffer_start_addr,
                addr_stride=0,
                data=load_data,
                dtype="int8",
                loop=ld_inst.loop_1d,
            )

        else:
            logger.error("Undefined data load mode!")
            raise NotImplementedError

    def ST_exec(self, st_inst: STInst):
        # buffer data address
        assert self.hbm is not None
        total_data_size = st_inst.length_1d * st_inst.loop_1d
        buffer_bank_id = st_inst.src_bank_id
        buffer_group_id = st_inst.src_group_id

        if st_inst.mode == STMode.Global2HBM:
            load_data = self._buffer_read(
                buffer_type=BufferType.Global,
                bank_id=buffer_group_id * GlobalBuffer.get_bank_num_in_BGroup()
                + buffer_bank_id,
                start_addr=st_inst.src_addr,
                bank_addr_stride=st_inst.bank_addr_stride,
                data_size=total_data_size,
                dtype="int8",
                loop=st_inst.loop_1d,
                loop_dir=st_inst.loop_direction,
            )
        elif st_inst.mode == STMode.Meta2HBM:
            load_data = self._buffer_read(
                buffer_type=BufferType.Meta,
                bank_id=0,
                start_addr=st_inst.src_addr,
                bank_addr_stride=st_inst.bank_addr_stride,
                data_size=total_data_size,
                dtype="int8",
                loop=st_inst.loop_1d,
            )
            pass

        # write main memory start address
        dst_addr = st_inst.dst_addr

        # data to main memory
        for loop_id in range(st_inst.loop_1d):
            real_length = st_inst.length_1d
            stride_length = st_inst.dst_1d_stride
            self.hbm[
                dst_addr
                + loop_id * stride_length : dst_addr
                + loop_id * stride_length
                + real_length
            ] = load_data[loop_id * real_length : (loop_id + 1) * real_length]

    def MM_exec(self, inst_param: MMInst):
        # MM mode parsing
        input_mode = inst_param.input_mode
        output_mode = inst_param.output_mode

        # buffer bank ID
        act_bank_group_id = inst_param.act_bank_group_id
        out_bank_group_id = inst_param.out_bank_group_id

        # specific operation flag
        bias_flag = inst_param.bias_flag
        output_flag = inst_param.output_flag
        relu_flag = inst_param.relu

        # memory start address
        act_start_addr = inst_param.act_start_addr
        act_scale_start_addr = inst_param.act_scale_start_addr

        weights_start_addr = inst_param.weights_start_addr
        weights_scale_start_addr = inst_param.weights_scale_start_addr
        bias_start_addr = inst_param.bias_start_addr

        out_start_addr = inst_param.out_start_addr
        out_scale_start_addr = inst_param.out_scale_start_addr

        # parameter
        M = inst_param.M + 1
        N = inst_param.N + 1
        K = inst_param.K + 1

        if input_mode in (0, 1, 3):  # W4A8 / W8A8 / W8A16
            # load act data
            if input_mode == 0:  # w4a8
                weight_dtype = "uint4"
                act_dtype = "int8"
                raise NotImplementedError
            elif input_mode == 1:  # w8a8
                weight_dtype = "int8"
                act_dtype = "int8"
            elif input_mode == 3:  # w8a16
                weight_dtype = "int8"
                act_dtype = "int16"
                raise NotImplementedError
            else:
                raise ValueError

            act_data = self._buffer_read(
                buffer_type=BufferType.Global,
                bank_id=act_bank_group_id,
                start_addr=act_start_addr,
                addr_stride=0,
                data_size=M * N * K,
                dtype=act_dtype,
            )
            if act_dtype == "int16":
                act_data = act_data.view(dtype="int16")
            elif act_dtype == "int8":
                pass
            else:
                raise ValueError
            act_data = act_data.reshape([1, -1]).astype("int32")
            # load weight data
            weights_data = self._buffer_read(
                buffer_type=BufferType.Weight,
                bank_id=-1,
                start_addr=weights_start_addr,
                addr_stride=0,
                data_size=N * K,
                dtype=weight_dtype,
            )
        elif input_mode == 2:  # fp16
            # load act data
            act_data = self._buffer_read(
                buffer_type=BufferType.Global,
                bank_id=act_bank_group_id,
                start_addr=act_start_addr,
                addr_stride=0,
                data_size=M * N * K,
                dtype="float16",
            )
            act_data = tools.int8_to_float16_tensor(act_data.reshape([1, -1])).reshape(
                (K, 1)
            )
            act_data = tools.set_subnormal_fp_value_to_zero(act_data, "float16")
            # load weight data
            weights_data = self._buffer_read(
                buffer_type=BufferType.Weight,
                bank_id=-1,
                start_addr=weights_start_addr,
                addr_stride=0,
                data_size=N * K,
                dtype="float16",
            )
        else:
            logger.error("Undefined weight data!")
            raise NotImplementedError

        weights_data_banks = np.split(weights_data, weights_data.shape[0], axis=0)
        weights_data_grp = list()
        for bank in weights_data_banks:
            if input_mode == 0:  # W4A8
                bank = tools.int8_to_int4_tensor(bank.flatten(), unsigned=True).reshape(
                    [-1, 1]
                )[:K]
            elif input_mode in (1, 3):  # W8A8 / W8A16
                bank = bank.reshape([-1, 1])[:K]
            elif input_mode == 2:  # fp16
                bank = tools.int8_to_float16_tensor(bank.flatten()).reshape([-1, 1])[:K]
            else:
                raise NotImplementedError
            weights_data_grp.append(bank)
        weights_data = np.concatenate(weights_data_grp, axis=1)

        if input_mode == 1:  # W8A8
            weights_data = weights_data.astype("int32")
            # load weight scale
            weights_scale = self._buffer_read(
                buffer_type=BufferType.Meta,
                bank_id=-1,
                start_addr=weights_scale_start_addr,
                addr_stride=0,
                data_size=MMCore.get_parallel_dim(MM_ParallelType.int)[1],
                dtype="float16",
            )

            weights_scale_banks = np.split(
                weights_scale, weights_scale.shape[0], axis=0
            )
            weights_scale_channel = list()
            for bank in weights_scale_banks:
                channel = tools.int8_to_float16_tensor(bank).reshape(
                    [-1, WeightBuffer.get_bank_bytes_num()]
                )
                weights_scale_channel.append(channel)
            weights_scale = np.concatenate(weights_scale_channel, axis=1).astype(
                "float16"
            )
            weights_scale = tools.set_subnormal_fp_value_to_zero(
                weights_scale, "float16"
            )

            # load act scale
            act_scale = self._buffer_read(
                buffer_type=BufferType.Meta,
                bank_id=-1,
                start_addr=act_scale_start_addr,
                addr_stride=0,
                data_size=MMCore.get_parallel_dim(MM_ParallelType.float)[1],
                dtype="float16",
            )

            act_scale_banks = np.split(act_scale, act_scale.shape[0], axis=0)
            act_scale_channel = list()
            for bank in act_scale_banks:
                channel = tools.int8_to_uint16_tensor(bank).reshape(
                    [-1, WeightBuffer.get_bank_bytes_num()]
                )
                act_scale_channel.append(channel)
            act_scale = np.concatenate(act_scale_channel, axis=1).astype("int32")

            assert weights_data.shape[0] == K

            # calculate
            output_data = np.zeros(
                (1, MMCore.get_parallel_dim(MM_ParallelType.float)[1])
            ).astype("float32")
            for i in range(1):
                this_group_origin_weights = weights_data[
                    i * HW_Info.get_quant_group() : (i + 1) * HW_Info.get_quant_group(),
                    :,
                ]
                this_group_act_scale = act_scale[i, :]

                if input_mode == 0:  # W4A8
                    weights_data_tem = this_group_origin_weights - this_group_act_scale
                else:  # W8A8
                    weights_data_tem = this_group_origin_weights

                act_data_tem = act_data[
                    :,
                    i * HW_Info.get_quant_group() : (i + 1) * HW_Info.get_quant_group(),
                ]
                matmul = np.matmul(act_data_tem, weights_data_tem)

                # if acc_mode == 1:
                if True:
                    matmul = matmul.astype("float32")
                    matmul = tools.set_subnormal_fp_value_to_zero(matmul, "float32")
                    scale = act_scale.astype("float32") * weights_scale[i, :].astype(
                        "float32"
                    )
                    scale = tools.set_subnormal_fp_value_to_zero(scale, "float32")
                    acc_data_fp32_tem = (scale * matmul).astype("float32")
                    acc_data_fp32_tem = tools.set_subnormal_fp_value_to_zero(
                        acc_data_fp32_tem, "float32"
                    )
                    output_data += acc_data_fp32_tem
                # else:
                #     output_data += matmul

            # if acc_mode == 0:
            #     scale = (act_scale.astype("float32") * weights_scale[:, :].astype("float32"))
            #     scale = tools.set_subnormal_fp_value_to_zero(scale, "float32")
            #     output_data = tools.set_subnormal_fp_value_to_zero(output_data, "float32")
            #     output_data = (scale * output_data).astype("float32")
            #     output_data = tools.set_subnormal_fp_value_to_zero(output_data, "float32")

        elif input_mode == 2:
            weights_data = weights_data.transpose((1, 0)).astype("float16")
            weights_data = tools.set_subnormal_fp_value_to_zero(weights_data, "float16")
            act_data = tools.set_subnormal_fp_value_to_zero(act_data, "float16")
            act_data = act_data.reshape((1, 1, K))
            weights_data = weights_data.reshape(
                (1, MMCore.get_parallel_dim(MM_ParallelType.float)[2], K)
            )
            # calculate
            output_data = tools.fp16_matmul(act_data, weights_data)

        else:
            logger.error("Undefined input mode!")
            raise NotImplementedError

        output_data = output_data.astype("float16")

        # load bias (flag)
        if bias_flag:
            bias = self._buffer_read(
                buffer_type=BufferType.Meta,
                bank_id=-1,
                start_addr=bias_start_addr,
                addr_stride=0,
                data_size=MMCore.get_parallel_dim(MM_ParallelType.float)[1],
                dtype="float16",
            )

            bias_banks = np.split(bias, bias.shape[0], axis=0)
            bias_channel = list()
            for bank in bias_banks:
                channel = tools.int8_to_float16_tensor(bank).reshape(
                    [-1, WeightBuffer.get_bank_bytes_num()]
                )
                bias_channel.append(channel[0])

            bias = (
                np.concatenate(bias_channel, axis=-1).reshape(1, -1).astype("float16")
            )
            assert output_data.shape[1] == bias.shape[1]

            output_data = output_data + bias

        # relu (flag)
        if relu_flag:
            output_data[output_data < 0] = 0

        output_data = output_data[:N]  # 限制输出范围
        # quant factor
        if output_mode == 0:  # int8
            # TODO: 考虑output_scale
            output_data = tools.quant_output_to_int(output_data, 1.0, "int8")
            output_data = output_data.flatten()

        elif output_mode == 1:  # fp16
            output_data = tools.set_subnormal_fp_value_to_zero(output_data, "float16")
            output_data = tools.float16_to_int8_tensor(output_data)

        else:
            logger.error("Undefined output mode!")
            raise NotImplementedError

        # output
        assert output_flag == 1
        self._buffer_write(
            buffer_type=BufferType.Global,
            bank_id=out_bank_group_id,
            start_addr=out_start_addr,
            addr_stride=0,
            data=output_data,
            dtype="int8",
        )

    def PVPU_exec(self, inst_param: MISCInst):
        # load input_data
        input_data_grp = list()

        batch_size = inst_param.batch

        in_a_group_id = inst_param.in_a_bank_group
        in_a_bank_id = inst_param.in_a_bank_id
        in_a_start_addr = inst_param.in_a_start_addr

        in_b_group_id = inst_param.in_b_bank_group
        in_b_bank_id = inst_param.in_b_bank_id
        in_b_start_addr = inst_param.in_b_start_addr

        dynamic_scale_flag = inst_param.dynamic_scale_flag

        K = inst_param.K
        misc_op = inst_param.op

        in_a_bank_group_id = (
            in_a_group_id * GlobalBuffer.get_bank_num_in_BGroup() + in_a_bank_id
        )
        in_b_bank_group_id = (
            in_b_group_id * GlobalBuffer.get_bank_num_in_BGroup() + in_b_bank_id
        )

        input_data = self._buffer_read(
            buffer_type=BufferType.Global,
            bank_id=in_a_bank_group_id,
            start_addr=in_a_start_addr,
            addr_stride=0,
            data_size=K * batch_size,
            loop=batch_size,
            dtype="float16",
        )
        input_data = tools.int8_to_float16_tensor(input_data)

        input_data_grp.append(input_data)

        if misc_op in (
            MiscOp.elt_add,
            MiscOp.elt_mul,
            MiscOp.layernorm,
            MiscOp.rms_layernorm,
        ):
            input_data = self._buffer_read(
                buffer_type=BufferType.Global,
                bank_id=in_b_bank_group_id,
                start_addr=in_b_start_addr,
                addr_stride=0,
                data_size=K * batch_size,
                loop=batch_size,
                dtype="float16",
            )
            input_data = tools.int8_to_float16_tensor(input_data)

            input_data_grp.append(input_data)
        elif misc_op == MiscOp.rope:
            input_data = self._buffer_read(
                buffer_type=BufferType.Global,
                bank_id=in_b_bank_group_id,
                start_addr=in_b_start_addr,
                addr_stride=0,
                data_size=K * 2,
                dtype="float16",
            )
            input_data = tools.int8_to_float16_tensor(input_data)

            sin_cache = input_data[0::2]
            cos_cache = input_data[1::2]
            input_data_grp.append(cos_cache)
            input_data_grp.append(sin_cache)

        # passing in data
        self.pvpu.write_data(input_data_grp)

        # PVPU_exec
        self.pvpu.exec(inst_param)

        # write back to global buffer
        output_data = self.pvpu.read_data()
        if output_data is None:
            pass
        else:
            out_start_adr = inst_param.out_start_addr
            out_group_id = inst_param.out_bank_group
            out_bank_id = inst_param.out_bank_id

            out_bank_group_id = (
                out_group_id * GlobalBuffer.get_bank_num_in_BGroup() + out_bank_id
            )
            self._buffer_write(
                buffer_type=BufferType.Global,
                bank_id=out_bank_group_id,
                start_addr=out_start_adr,
                addr_stride=0,
                data=output_data,
                loop=batch_size,
                dtype="int8",
            )  # all data is encoded to int8

    def RS_exec(self, inst_param: RSInst):
        # operation direct with hbm
        assert self.hbm is not None
        total_data_size = inst_param.M * inst_param.K  # int8 data
        hbm_src_adr = inst_param.src_addr
        hbm_dst_adr = inst_param.dst_addr

        # read data from main memory
        matrix = self.hbm[hbm_src_adr : hbm_src_adr + total_data_size]
        matrix = matrix.astype("int8").reshape([inst_param.M, inst_param.K])
        matrix = matrix.transpose((1, 0))
        self.hbm[hbm_dst_adr : hbm_dst_adr + total_data_size] = matrix.flatten()


class HardwareModule:
    def __init__(self, slr_num: int = 3, multi_process: bool = False):
        self.slr_num = slr_num
        self.multi_process = multi_process

        self.PVPU_bus: List[Dict] = [dict() for _ in range(slr_num)]
        """
            PVPU_bus =[
                {   # PVPU_0
                    "max_register": np.nparray, # size == 1
                    "acc_register": Dict[str, np.ndarray]  # size in (1, 2)
                },
                ...
            ]
        """
        self.hbm = np.zeros(HW_Info.get_hbm_size() // 4, dtype=np.int8)

        self.slr = list()
        self.slr_sync_flag = [0 for _ in range(slr_num)]
        self.slr_interrupt_flag = [0 for _ in range(slr_num)]

        self.pvpu_sync_flag = [
            0 for _ in range(slr_num)
        ]  # 0-normal; 1-sync; 2-reprocess

        for slr_id in range(slr_num):
            self.slr.append(SLRModule(slr_id))
            self.slr[-1].hbm = self.hbm
            self.slr[-1].pvpu.communication_bus = self.PVPU_bus

        self.init_hbm_flag = False

    @property
    def is_slr_sync(self):
        return sum(self.slr_sync_flag) == self.slr_num

    @property
    def is_pvpu_sync(self):
        return sum(self.pvpu_sync_flag) == self.slr_num

    @property
    def is_slr_interrupt(self):
        return sum(self.slr_interrupt_flag) == self.slr_num

    def PVPU_sync(self):
        max_reg_list = [reg["max_register"] for reg in self.PVPU_bus]
        acc_reg_list = [reg["acc_register"] for reg in self.PVPU_bus]
        mm_dynamic_scale_list = [
            reg["_dynamic_scale_register"] for reg in self.PVPU_bus
        ]

        assert all(isinstance(reg, type(max_reg_list[0])) for reg in max_reg_list)
        assert all(isinstance(reg, type(acc_reg_list[0])) for reg in acc_reg_list)
        assert all(
            isinstance(reg, type(mm_dynamic_scale_list[0]))
            for reg in mm_dynamic_scale_list
        )

        data_key = acc_reg_list[0].keys() if acc_reg_list[0] is not None else []
        acc_reg_dict = dict()

        count_list = list()
        max_data = 0
        max_mm_dynamic_scale = None

        if len(data_key) == 0:
            assert np.min(mm_dynamic_scale_list) >= 0
            max_mm_dynamic_scale = np.max(mm_dynamic_scale_list).astype("float16")

        if "count" in data_key:
            count_list = list()
            for reg in acc_reg_list:
                count_list.append(reg["count"])
            count = np.sum(count_list).astype("float32")

            acc_reg_dict["count"] = count

        if "square_sum" in data_key:
            square_sum_list = list()
            for reg in acc_reg_list:
                square_sum_list.append(reg["square_sum"])
            # square_sum = np.average(square_sum_list, weights=count_list).astype("float32")
            square_sum = np.sum(square_sum_list)
            acc_reg_dict["square_sum"] = square_sum

        if "sum" in data_key:
            sum_list = list()
            for reg in acc_reg_list:
                sum_list.append(reg["sum"])
            acc_reg_dict["sum"] = np.sum(sum_list)

        if "exp_mean" in data_key:
            max_data = np.max(max_reg_list).astype("float32")
            max_diff = max_data - np.array(max_reg_list).astype("float32")

            exp_mean_list = list()
            for reg in acc_reg_list:
                exp_mean_list.append(reg["exp_mean"])

            exp_mean_list = exp_mean_list * (-max_diff)
            exp_mean = np.sum(exp_mean_list).astype("float32")

            acc_reg_dict["exp_mean"] = exp_mean

        for slr in self.slr:
            slr.pvpu.sync_flag = True
            if len(data_key) == 0:
                assert max_mm_dynamic_scale is not None
                slr.pvpu.sync_mm_dynamic_scale(max_mm_dynamic_scale)
            elif "exp_mean" in data_key:
                slr.pvpu.sync_acc_reg(acc_reg_dict, max_data)
            else:
                slr.pvpu.sync_acc_reg(acc_reg_dict)

    def SYS_exec(self, inst, slr_id=0):
        interrupt_flag = inst.op
        if interrupt_flag == SysOp.sync:
            self.slr_sync_flag[slr_id] = 1
        else:
            self.slr_interrupt_flag[slr_id] = 1

    def step_exec_inst(self, inst_list):
        assert len(inst_list) == self.slr_num

        for i, slr in enumerate(self.slr):
            for inst in inst_list[i]:
                if inst == []:  # 空字典说明slr被挂起了（等待同步/中断）
                    continue

                if isinstance(inst, SYSCInst):
                    self.SYS_exec(inst, slr_id=i)
                    continue

                if not self.slr_sync_flag[i] and not self.pvpu_sync_flag[i]:
                    slr.exec_inst(inst)

                if isinstance(inst, MISCInst):
                    if inst.last_flag:
                        self.pvpu_sync_flag[i] = inst.co_misc_flag

        if (
            "_dynamic_scale_register"
            in self.PVPU_bus[PVPUCore.get_slr_location()[0]].keys()
        ):
            mm_dynamic_scale = self.PVPU_bus[PVPUCore.get_slr_location()[0]][
                "_dynamic_scale_register"
            ]
            if mm_dynamic_scale is not None:
                for pvpu_id in PVPUCore.get_slr_location():
                    mm_dynamic_scale = np.max(
                        mm_dynamic_scale,
                        self.PVPU_bus[pvpu_id]["_dynamic_scale_register"],
                    )
                max_mm_dynamic_scale = mm_dynamic_scale.astype("float16")
                for slr in self.slr:
                    slr.pvpu.sync_mm_dynamic_scale(max_mm_dynamic_scale)

        if self.is_slr_sync:
            self.slr_sync_flag = [0 for _ in range(self.slr_num)]
        if self.is_pvpu_sync:
            self.PVPU_sync()
            self.pvpu_sync_flag = [0 for _ in range(self.slr_num)]

        block_flag = [
            self.slr_sync_flag[i] + self.pvpu_sync_flag[i] for i in range(self.slr_num)
        ]

        return block_flag


class RTLModel:
    def __init__(self, config_file_path: str, debug: bool = True):
        self.debug = debug
        self.config_file = config_file_path
        # read inst from json
        inst_path = os.path.join(config_file_path, "inst.json")
        self.model_inst_list: List[List[Inst]] = list()  # for multi slr

        # parse inst from json
        self.model_inst_list = self.load_inst_from_json(inst_path)

        self.slr_num = 1 if debug else HW_Info.get_slr_count()  # debug for 1

        # read addr info from yaml
        self.addr_info: Dict[str, List[Dict]] = tools.load_yaml(
            os.path.join(self.config_file, "info.yaml")
        )

        # Scheduler Config
        self.scheduler = list()
        self.total_inst_num = 0
        for slr_id, slr_inst in enumerate(self.model_inst_list):
            self.scheduler.append(Scheduler(slr_id, slr_inst))
            self.total_inst_num += len(slr_inst)

        logger.info(f"Decode the inst fifo, total inst num = {self.total_inst_num}")
        self.slr_block_flag = [0 for _ in range(self.slr_num)]  # ?

        # Hardware Config
        self.hardware = HardwareModule(
            1 if debug else HW_Info.get_slr_count(), multi_process=False
        )

        # load param and input into hbm
        self.load_bin_file_to_hbm()

        logger.info("RTL model init done!!!")

    def load_inst_from_json(self, json_file_dir: str) -> List[List[Inst]]:
        json_inst = tools.load_json(json_file_dir)
        return tools.parse_json_to_inst([json_inst])

    def load_bin_file_to_hbm(self):
        self._load_data_to_hbm("inputs_info")
        self._load_data_to_hbm("weights_info")
        self.hardware.init_hbm_flag = True

    def _load_data_to_hbm(self, data_type: str):
        logger.info(f"Load {data_type} to rtl model hbm!")
        for info in self.addr_info[data_type]:
            data_file_path = os.path.join(
                self.config_file, info["file_path"].split("/")[-1]
            )
            assert os.path.exists(data_file_path), f"File {data_file_path} not exists"
            data = np.fromfile(data_file_path, dtype=np.int8)
            assert (
                data.size == info["size"]
            ), f"Data size not match, expect {info['size']} but got {data.size}"
            assert (
                info["start"] + info["size"] <= HW_Info.get_hbm_size()
            ), f"Data size overflow"
            self.hardware.hbm[info["start"] : info["start"] + info["size"]] = data

    def run(self):
        logger.info(f"=======Run rtl model=========")
        exec_inst_num: int = 0
        while exec_inst_num < self.total_inst_num:
            exec_slrs_inst: List[List[Inst]] = list()
            for slr_id in range(self.slr_num):
                if self.slr_block_flag[slr_id]:
                    exec_slrs_inst.append([])
                    continue

                inst_list: List[Inst] = list()

                module_inst: List[Inst] = list()
                for i in range(self.scheduler[slr_id].inst_type_num):
                    module_inst.append(
                        self.scheduler[slr_id].send_module_inst(Opcode[INST_TYPE[i]])
                    )

                execute_inst_flag: bool = False
                for module_id in range(self.scheduler[slr_id].inst_type_num):
                    this_inst = module_inst[module_id]
                    if this_inst is not None:
                        inst_type = Opcode[INST_TYPE[module_id]]
                        execute_inst_flag = True
                        self.scheduler[slr_id].inst_type_cnt[inst_type] += 1
                        exec_inst_num += 1
                        wait_list = this_inst.wait
                        release_list = this_inst.release
                        self.scheduler[slr_id].write_depend_reg(
                            module_id, wait_list, release_list
                        )
                        inst_list.append(this_inst)
                if not execute_inst_flag:
                    if self.hardware.slr_interrupt_flag[slr_id]:
                        # inst_list.append(dict())
                        exec_slrs_inst.append(inst_list)
                    self.scheduler[slr_id].depend_reg_warning()
                    logger.error("Dependency wrong!")
                    raise Exception
                exec_slrs_inst.append(inst_list)
            self.slr_block_flag = self.hardware.step_exec_inst(exec_slrs_inst)

        if self.debug:
            for info in self.addr_info["outputs_info"]:
                output = self.hardware.hbm[info["start"] : info["start"] + info["size"]]
                golden = np.fromfile(
                    os.path.join(self.config_file, info["file_path"].split("/")[-1]),
                    dtype=np.int8,
                )
                cmp_result = output == golden
                logger.info(f"Compare {info['file_path']} data: {cmp_result.all()}")

    def dump_hbm_output_data_and_check(self):
        logger.debug("Dump rtl model data to " + self.rtl_model_output_dir)
        total_cmp_result = True
        for channel_info in self.addr_info["output"]:
            physic_id = channel_info["channel_physic_id"]
            rtl_model_output_file_dir = os.path.join(
                self.rtl_model_output_dir, f"rtl_model_output.ch{physic_id:02d}.rtl.bin"
            )

            data = self.hardware.hbm[
                channel_info["absolute_addr"] : channel_info["absolute_addr"]
                + channel_info["bytes"]
            ]
            data.tofile(rtl_model_output_file_dir)

            ref_output_file_dir = os.path.join(
                self.ref_output_dir, f"output.ch{physic_id:02d}.rtl.bin"
            )
            assert os.path.isfile(ref_output_file_dir)

            cmp_result = filecmp.cmp(rtl_model_output_file_dir, ref_output_file_dir)
            logger.info(f"rtl model compare channel {physic_id:02d} data: {cmp_result}")
            total_cmp_result = cmp_result & total_cmp_result

        if total_cmp_result:
            logger.info("rtl model check PASS!")
        else:
            logger.error("rtl model check FAIL!")

        return total_cmp_result


# class RTLModel:
#     def __init__(
#         self,
#         model_inst_list: List[List[Inst]],
#         config_file: Optional[str] = None,
#         debug: bool = True,
#     ):
#         assert debug, "Temporary function, it will be removed in future version"
#         self.debug = debug
#         if not debug:
#             # TODO: 添加数据的Path
#             # model_inst_list to be deleted
#             if config_file is None:
#                 config_file = "./testbench/case_1/config.json"
#                 # config_file = os.path.join(os.path.dirname(__file__), "config.json")

#             self.config_file = config_file
#             self.config = tools.load_json(config_file)

#             self.inst_output_dir = os.path.join("./output", "inst_output")
#             self.addr_info: Dict[str, List[Dict]] = dict()  # tools.load_json(os.path.join(self.inst_output_dir, "info.json"))
#             self.rtl_model_output_dir = os.path.join(self.inst_output_dir, "rtl_model_output")
#             # if os.path.exists(self.rtl_model_output_dir):
#             #     shutil.rmtree(self.rtl_model_output_dir)
#             # os.mkdir(self.rtl_model_output_dir)

#             self.ref_output_dir = os.path.join(self.inst_output_dir, "output")

#         else:
#             logger.debug("Debug mode is on")

#         # Instruction List
#         self.model_inst_list = model_inst_list  # 存储每条指令的dict

#         # Hardware Config
#         self.slr_num = HW_Info.get_slr_count() if not debug else 1

#         # Scheduler Config
#         self.scheduler = list()
#         self.total_inst_num = 0
#         for slr_id, slr_inst in enumerate(self.model_inst_list):
#             self.scheduler.append(Scheduler(slr_id, slr_inst))
#             self.total_inst_num += len(slr_inst)

#         self.slr_block_flag = [0 for _ in range(self.slr_num)]

#         # Hardware Config
#         self.hardware = HardwareModule(HW_Info.get_slr_count() if not debug else 1, multi_process=False)
#         # self.load_bin_file_to_hbm()
#         logger.debug(f"Decode the inst fifo, total inst num = {self.total_inst_num}")

#     def load_inst_from_json(self, json_file_dir):
#         self.model_inst_list = tools.load_json(json_file_dir)
#         self.total_inst_num = 0

# def run(self, **kwargs):
#     """
#     Run the model
#     :param
#     kwargs:
#         input_data: np.ndarray, input data
#         input_addr: int, input data address
#         output_data: np.ndarray, output data
#         output_addr: int, output data address
#         weight_data: np.ndarray, weight data
#         weight_addr: int, weight data address
#     :return:
#     """
#     logger.info(f"Run rtl model")
#     if self.debug:
#         logger.warning("Temporary function, it will be removed in future version")
#         input_data = kwargs["input_data"].flatten()
#         input_addr = kwargs["input_addr"]

#         input_data.dtype = np.int8
#         input_len_bytes = input_data.size
#         self.hardware.hbm[input_addr: input_addr + input_len_bytes] = input_data

#         if "weight_data" in kwargs:
#             weight_data = kwargs["weight_data"].flatten()
#             weight_addr = kwargs["weight_addr"]

#             weight_data.dtype = np.int8
#             weight_len = weight_data.size
#             self.hardware.hbm[weight_addr: weight_addr + weight_len] = weight_data
#         self.hardware.init_hbm_flag = True  # hbm数据初始化

#     # start run
#     exec_inst_cnt = 0
#     # for this_inst in inst_dict_list:
#     #     self._exec_inst(this_inst["TYPE"], this_inst["PARAM"])
#     while exec_inst_cnt < self.total_inst_num:
#         slr_inst_list = list()
#         for slr_id in range(self.slr_num):
#             if self.slr_block_flag[slr_id]:
#                 slr_inst_list.append([{}])
#                 continue
#             inst_list = list()
#             module_inst = [self.scheduler[slr_id].send_module_inst(Opcode[INST_TYPE[i]]) for i in range(self.scheduler[slr_id].inst_type_num)]
#             execute_inst_flag = False  # 这轮有没有执行指令
#             for module_id in range(self.scheduler[slr_id].inst_type_num):
#                 this_inst = module_inst[module_id]
#                 if this_inst is not None:
#                     inst_type = Opcode[INST_TYPE[module_id]]
#                     execute_inst_flag = True
#                     self.scheduler[slr_id].inst_type_cnt[inst_type] += 1
#                     exec_inst_cnt += 1
#                     wait_list = this_inst.wait
#                     release_list = this_inst.release
#                     self.scheduler[slr_id].write_depend_reg(module_id, wait_list, release_list)
#                     """
#                         inst_list: length equal the number of slr [slr-0, slr-1, slr-2, ...]
#                             - Inst_0, Inst_1, Inst_2, ...
#                             ...
#                     """
#                     inst_list.append(this_inst)

#             if not execute_inst_flag:  # 该轮没有执行任何指令，说明依赖出现了bug
#                 if self.hardware.slr_interrupt_flag[slr_id]:
#                     inst_list.append(dict())
#                     slr_inst_list.append(inst_list)
#                     continue
#                 self.scheduler[slr_id].depend_reg_warning()
#                 logger.error("Dependency wrong!")
#                 raise Exception
#             slr_inst_list.append(inst_list)
#         self.slr_block_flag = self.hardware.step_exec_inst(slr_inst_list)

#     for sche in self.scheduler:
#         sche.print_depend_reg()
#     if not self.hardware.is_slr_interrupt:
#         logger.error("Do not have final interrupt!")
#         raise InterruptedError

#     if self.debug:
#         output_data = kwargs["output_data"].flatten()
#         output_addr = kwargs["output_addr"]

#         output_data.dtype = np.int8
#         output_len_bytes = output_data.size

#         res_data = self.hardware.hbm[output_addr: output_addr + output_len_bytes]
#         cmp_result = (res_data.data == output_data.data)
#     else:
#         cmp_result = self.dump_hbm_output_data_and_check()

#     return cmp_result

# def load_bin_file_to_hbm(self):
#     for name in ["param", "input"]:
#         logger.debug(f"Load {name} to rtl model hbm")
#         for channel_info in self.addr_info[name]:
#             physic_id = channel_info["channel_physic_id"]
#             data_file_dir = os.path.join(self.inst_output_dir, name, f"{name}.ch{physic_id:02d}.rtl.bin")
#             assert os.path.exists(data_file_dir), data_file_dir
#             data = np.fromfile(data_file_dir, dtype=np.int8)
#             assert data.size == channel_info["bytes"]
#             assert channel_info["absolute_addr"] + channel_info["bytes"] <= HW_Info.get_hbm_size()
#             self.hardware.hbm[channel_info["absolute_addr"]: channel_info["absolute_addr"] + channel_info["bytes"]] = data
#     self.hardware.init_hbm_flag = True  # hbm数据初始化

#     def data_compare(self, golden_data, data):
#         pass

#     def dump_hbm_output_data_and_check(self):
#         logger.debug("Dump rtl model data to " + self.rtl_model_output_dir)
#         total_cmp_result = True
#         for channel_info in self.addr_info["output"]:
#             physic_id = channel_info["channel_physic_id"]
#             rtl_model_output_file_dir = os.path.join(self.rtl_model_output_dir, f"rtl_model_output.ch{physic_id:02d}.rtl.bin")

#             data = self.hardware.hbm[channel_info["absolute_addr"]: channel_info["absolute_addr"] + channel_info["bytes"]]
#             data.tofile(rtl_model_output_file_dir)

#             ref_output_file_dir = os.path.join(self.ref_output_dir, f"output.ch{physic_id:02d}.rtl.bin")
#             assert os.path.isfile(ref_output_file_dir)

#             cmp_result = filecmp.cmp(rtl_model_output_file_dir, ref_output_file_dir)
#             logger.info(f"rtl model compare channel {physic_id:02d} data: {cmp_result}")
#             total_cmp_result = cmp_result & total_cmp_result

#         if total_cmp_result:
#             logger.info("rtl model check PASS!")
#         else:
#             logger.error("rtl model check FAIL!")

#         return total_cmp_result
