# -*-coding:utf-8-*-
import logging
import math
import os
from enum import IntEnum, unique
from typing import List, Optional

from inst import (
    Inst, LDInst, MISCInst, MMInst, RSInst, STInst, SYSCInst,
    LDMode, STMode, RSDataType,
    PUType, MiscOp, MiscMode, MMInputMode, MMOutputMode)
from utils import HW_Info, MMCore, PVPUCore, RSModule, RSDType, MM_ParallelType

import multiprocessing as mp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
PATH = os.path.dirname(os.path.abspath(__file__))

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

def putype_to_inst_type(pu_type: PUType):
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


class InstProfiler:
    def __init__(
            self,
            model_inst_list: list,
            task_name: str = "default",
        ):
        self.MISC_overlap_time = 0.0

        self.task_name = task_name
        self.inst_performance_fig_dir = os.path.join(PATH, "time_performance.png")
        self.inst_performance_txt_dir = os.path.join(PATH, "time_performance.txt")
        self.TIME_THRESHOLD = 1e-7

        self.module_num = len(INST_TYPE)
        self.slr_module_inst_cnt = [dict() for _ in range(HW_Info.get_slr_count())]
        self.slr_module_part_time = [dict() for _ in range(HW_Info.get_slr_count())]  # 计算每个module的时间
        self.slr_module_cycle_cnt = [dict() for _ in range(HW_Info.get_slr_count())]
        self.slr_inst_total_time = [0 for _ in range(HW_Info.get_slr_count())]
        
        self.inst_total_time = 0
        
        self.total_LD_size_B = 0
        self.total_ST_size_B = 0

        self.total_LD_weight_size_B = 0
        self.max_module_part_time = 0
        self.pre_processed_time = 0
        self.post_processed_time = 0
        self.mac_num = 0

        for slr_idx in range(HW_Info.get_slr_count()):
            for module_name in INST_TYPE:
                self.slr_module_inst_cnt[slr_idx][Opcode[module_name]] = 0
                self.slr_module_part_time[slr_idx][Opcode[module_name]] = 0.
                self.slr_module_cycle_cnt[slr_idx][Opcode[module_name]] = 0
    
        self.total_inst_cnt = [0 for _ in range(HW_Info.get_slr_count())]
        self.model_inst_list = model_inst_list  # 存储每条指令的class

        # 自动check
        self.pub_array: list[dict[Opcode, dict[Opcode, int]]] = [{
            Opcode[model_name]: {Opcode[model_name]: 0 for model_name in INST_TYPE} for model_name in INST_TYPE
        } for _ in range(HW_Info.get_slr_count())]
        self.rev_array: list[dict[Opcode, dict[Opcode, int]]] = [{
            Opcode[model_name]: {Opcode[model_name]: 0 for model_name in INST_TYPE} for model_name in INST_TYPE
        } for _ in range(HW_Info.get_slr_count())]

        # 用于各单元利用情况画图的信息
        self.inst_plot_left: list[dict[Opcode, List[float]]] = [{Opcode[module_name]: [] for module_name in INST_TYPE} for _ in range(HW_Info.get_slr_count())]
        self.inst_plot_width: list[dict[Opcode, List[float]]] = [{Opcode[module_name]: [] for module_name in INST_TYPE} for _ in range(HW_Info.get_slr_count())]
        self.inst_wr_time: list[dict[Opcode, List[float]]] = [{Opcode[module_name]: [] for module_name in INST_TYPE} for _ in range(HW_Info.get_slr_count())]

        # 同步参数
        self.sync_plot_left = [[] for _ in range(HW_Info.get_slr_count())]
        self.sync_plot_width = [[] for _ in range(HW_Info.get_slr_count())]

        self.slr_pseudo_sync_num = [0 for _ in range(HW_Info.get_slr_count())]

        # 临时变量
        self.misc_init_flag = True
        self.now_misc_op: Optional[MiscOp] = None
        self.misc_op_cnt: dict[MiscOp, int] = {}
        self.misc_op_cnt_pre: dict[MiscOp, int] = {}

    @staticmethod
    def ld_start_latency(data_size_B: int):
        return math.ceil(0.000485 * data_size_B) + 8 # math.ceil(0.000485 * data_size_B) + 157
    
    @staticmethod
    def st_start_latency(data_size_B: int):
        return math.ceil(0.000485 * data_size_B) + 64

    def time_LD(self, inst_param: LDInst, slr_idx: int):
        if inst_param.mode == LDMode.HBM2Bbuffer:
            bits = HW_Info.get_hbm_axi_bits()
            factor = HW_Info.get_hbm_utilization()
            bandwidth = HW_Info.get_hbm_bandwidth() / 8
            channel_num = 1
        else:
            bits = HW_Info.get_ddr_axi_bits()
            factor = HW_Info.get_ddr_utilization()
            bandwidth = HW_Info.get_ddr_bandwidth() / 8
            channel_num = 1

        bandwidth_B_s = bandwidth * channel_num * factor
        freq = HW_Info.get_hbm_freq() * 1000 * 1000  # Hz

        total_data_B = inst_param.length_1d * inst_param.loop_1d
        self.total_LD_size_B += total_data_B
        if inst_param.mode in (LDMode.HBM2Bbuffer, LDMode.DDR2Meta):
            self.total_LD_weight_size_B += total_data_B
        
        total_cycle = math.ceil(total_data_B / (bits / 8 * factor))
        self.slr_module_cycle_cnt[slr_idx][Opcode.LD] += (total_cycle + self.ld_start_latency(total_data_B))

        total_time_ms = (total_data_B / bandwidth_B_s + self.ld_start_latency(total_data_B) / freq) * 1e3  # ms
        return total_time_ms

    def time_ST(self, inst_param, slr_idx: int):
        factor = HW_Info.get_ddr_utilization()
        bandwidth = HW_Info.get_ddr_bandwidth() / 8
        channel_num = 1

        bandwidth_B_s = bandwidth * channel_num * factor
        freq = HW_Info.get_hbm_freq() * 1000 * 1000  # Hz

        total_data_B = inst_param.length_1d * inst_param.loop_1d
        self.total_ST_size_B += total_data_B

        total_cycle = math.ceil(total_data_B / (HW_Info.get_ddr_axi_bits() / 8 * factor))
        self.slr_module_cycle_cnt[slr_idx][Opcode.ST] += (total_cycle + self.st_start_latency(total_data_B))

        total_time_ms = (total_data_B / bandwidth_B_s + self.st_start_latency(total_data_B) / freq) * 1e3  # ms
        return total_time_ms
    
    def time_MM(self, inst_param, slr_idx: int):
        M = inst_param.M + 1
        N = inst_param.N + 1
        K = inst_param.K + 1 if not inst_param.K <= 127 else 128

        self.mac_num += M * N * K
        parallelism = MMCore.get_parallel_dim(MM_ParallelType.int) \
            if inst_param.input_mode in (MMInputMode.w4a8, MMInputMode.w8a8, MMInputMode.w8a16) else MMCore.get_parallel_dim(MM_ParallelType.float)

        if inst_param.input_mode in (MMInputMode.w4a8, MMInputMode.w8a8):  # uint4/int8 * int8
            total_grp = (math.ceil(M / parallelism.M) *
                         math.ceil(K / parallelism.K) *
                         math.ceil(N / parallelism.N))
        elif inst_param.input_mode == MMInputMode.w8a16:  # uint4/int8 * int16
            total_grp = (math.ceil(M / parallelism.M) *
                         math.ceil(K / (parallelism.K // 2)) *
                         math.ceil(N / parallelism.N))
        elif inst_param.input_mode == MMInputMode.fp16:  # FP16/BF16
            total_grp = (math.ceil(M / parallelism.M) *
                         math.ceil(K / parallelism.K) *
                         math.ceil(N / parallelism.N))
        else:
            raise ValueError

        # Latency
        dsp_clk_time_ms = 1 / MMCore.get_dsp_freq() * 1e-3  # ms
        total_cycle = MMCore.get_cycle_data()["INIT_LATENCY"]

        total_cycle += total_grp

        self.slr_module_cycle_cnt[slr_idx][Opcode.MM] += total_cycle
        total_time_ms = total_cycle * dsp_clk_time_ms
        return total_time_ms

    def time_MISC(self, inst_param, slr_idx: int, mode="new", merge_flag=True):
        assert mode in ("new", "old")

        misc_op_type = inst_param.op
        parallelism = PVPUCore.get_parallel_dim() // 2 if inst_param.input_a_mode in (MiscMode.fp32, MiscMode.int32, MiscMode.tf32) else PVPUCore.get_parallel_dim()
        K = math.ceil(inst_param.K / parallelism)
        B = inst_param.batch

        # Init flag
        if not misc_op_type in (MiscOp.set_k, MiscOp.load_ins, MiscOp.elt_add, MiscOp.elt_mul,):
            if not self.now_misc_op == misc_op_type:
                self.misc_init_flag = True
                self.now_misc_op = misc_op_type

        if not misc_op_type in self.misc_op_cnt.keys():
            self.misc_op_cnt[misc_op_type] = 0
            self.misc_op_cnt_pre[misc_op_type] = 0
        self.misc_op_cnt[misc_op_type] += 1 * int(self.misc_init_flag)

        # latency
        pvpu_clk_time_ms = 1 / PVPUCore.get_pvpu_freq() * 1e-3  # ms
        total_cycle = 0
        
        # PreProcess
        if misc_op_type == MiscOp.gelu1:
            total_cycle += B * K + PVPUCore.get_cycle_data()[MiscOp.gelu0] * (1 if mode == "new" else B) * int(self.misc_init_flag)
            self.misc_op_cnt_pre[misc_op_type] += PVPUCore.get_cycle_data()[MiscOp.gelu0] * (1 if mode == "new" else B) * int(self.misc_init_flag)
        elif misc_op_type == MiscOp.layernorm:
            total_cycle += B * K + PVPUCore.get_cycle_data()[MiscOp.layernorm_pp] * (1 if mode == "new" else B) * int(self.misc_init_flag)
            self.misc_op_cnt_pre[misc_op_type] += PVPUCore.get_cycle_data()[MiscOp.layernorm_pp] * (1 if mode == "new" else B) * int(self.misc_init_flag)
        elif misc_op_type == MiscOp.rmsnorm:
            total_cycle += B * K + PVPUCore.get_cycle_data()[MiscOp.rmsnorm_pp] * (1 if mode == "new" else B) * int(self.misc_init_flag)
            self.misc_op_cnt_pre[misc_op_type] += PVPUCore.get_cycle_data()[MiscOp.rmsnorm_pp] * (1 if mode == "new" else B) * int(self.misc_init_flag)
        elif misc_op_type == MiscOp.softmax:
            total_cycle += B * K + PVPUCore.get_cycle_data()[MiscOp.softmax_pp] * (1 if mode == "new" else B) * int(self.misc_init_flag)
            total_cycle += B * K + PVPUCore.get_cycle_data()[MiscOp.softmax] * (1 if mode == "new" else B) * int(self.misc_init_flag)  # Piepline
            self.misc_op_cnt_pre[misc_op_type] += PVPUCore.get_cycle_data()[MiscOp.softmax_pp] * (1 if mode == "new" else B) * int(self.misc_init_flag)
        elif misc_op_type == MiscOp.abs_max:
            total_cycle += B * K + PVPUCore.get_cycle_data()[MiscOp.dynamic_int] * (1 if mode == "new" else B) * int(self.misc_init_flag)
            self.misc_op_cnt_pre[misc_op_type] += PVPUCore.get_cycle_data()[MiscOp.dynamic_int] * (1 if mode == "new" else B) * int(self.misc_init_flag)
    
        # Basic Latency
        total_cycle += B * K + PVPUCore.get_cycle_data()[misc_op_type] * (1 if mode == "new" else B) * int(self.misc_init_flag)
        self.misc_op_cnt_pre[misc_op_type] += PVPUCore.get_cycle_data()[misc_op_type] * (1 if mode == "new" else B) * int(self.misc_init_flag)
    
        # Dynamic Scale
        if inst_param.dynamic_scale and not misc_op_type in (MiscOp.dynamic_int, MiscOp.data_convert, MiscOp.abs_max):
            total_cycle += B * K + PVPUCore.get_cycle_data()[MiscOp.dynamic_int] * (1 if mode == "new" else B) * int(self.misc_init_flag)
            self.misc_op_cnt_pre[misc_op_type] += PVPUCore.get_cycle_data()[MiscOp.dynamic_int] * (1 if mode == "new" else B) * int(self.misc_init_flag)
        
        self.slr_module_cycle_cnt[slr_idx][Opcode.MISC] += total_cycle
        
        # Init flag
        if not misc_op_type in (MiscOp.set_k, MiscOp.load_ins, MiscOp.elt_add, MiscOp.elt_mul,):
            if self.now_misc_op == misc_op_type:
                self.misc_init_flag = False if merge_flag else True
        
        total_time_ms = total_cycle * pvpu_clk_time_ms
        return total_time_ms

    def time_RS(self, inst_param, slr_idx: int):
        total_data_B = inst_param.M * inst_param.K
        rs_type = inst_param.data_type

        if rs_type == RSDataType.int8:
            # self.rs_data_int8_amount += total_data_B
            parralelism = RSModule.get_parallel_dim(RSDType.int8)
        elif rs_type == RSDataType.int16:
            # self.rs_data_int16_amount += total_data_B
            parralelism = RSModule.get_parallel_dim(RSDType.int16)
        else:
            raise ValueError

        total_cycle = math.ceil(total_data_B / parralelism) + RSModule.get_start_latency()
        self.slr_module_cycle_cnt[slr_idx][Opcode.RS] += total_cycle

        total_time_ms = 2 * total_cycle / HW_Info.get_main_freq() * 1e-3  # ms
        return total_time_ms

    def time_SYS(self, inst_param, slr_idx: int):
        clk_time_ms = 1 / HW_Info.get_main_freq() * 1e-3  # ms
        total_cycle = 1
        self.slr_module_cycle_cnt[slr_idx][Opcode.SYS] += total_cycle
        total_time_ms = total_cycle * clk_time_ms
        return total_time_ms

    def get_inst_time_ms(self, inst_param, slr_idx: int):
        if isinstance(inst_param, LDInst):
            time = self.time_LD(inst_param, slr_idx)
        elif isinstance(inst_param, STInst):
            time = self.time_ST(inst_param, slr_idx)
        elif isinstance(inst_param, MMInst):
            time = self.time_MM(inst_param, slr_idx)
        elif isinstance(inst_param, MISCInst):
            time = self.time_MISC(inst_param, slr_idx)
        elif isinstance(inst_param, RSInst):
            time = self.time_RS(inst_param, slr_idx)
        else:
            assert isinstance(inst_param, SYSCInst)
            time = self.time_SYS(inst_param, slr_idx)
        return time
    
    def align_time(self):
        """
        方案：
        1.检索sys_inst指令的起始时间，然后切片，然后更新切片内的起始时间
        2.添加同步/中断时间间隔符
        """
        # Align sys sync time
        for i in range(len(self.inst_plot_left[0][Opcode.SYS])):
            real_sync_time = max([slr_plot_left[Opcode.SYS][i] for slr_plot_left in self.inst_plot_left])
            for slr_id in range(HW_Info.get_slr_count()):
                sync_diff_time = real_sync_time - self.inst_plot_left[slr_id][Opcode.SYS][i]
                for j_inst in INST_TYPE:
                    j = Opcode[j_inst]
                    if len(self.inst_plot_left[slr_id][j]) == 0:
                        continue
                    else:
                        for k in range(len(self.inst_plot_left[slr_id][j])):
                            if self.inst_plot_left[slr_id][j][k] > self.inst_plot_left[slr_id][Opcode.SYS][i]:
                                self.inst_plot_left[slr_id][j][k] += sync_diff_time
                self.sync_plot_left[slr_id].append(self.inst_plot_left[slr_id][Opcode.SYS][i]
                                                   + self.inst_plot_width[slr_id][Opcode.SYS][i])
                self.sync_plot_width[slr_id].append(sync_diff_time)



    def run(self, start_time = 0., dump_breakdown: bool = False, dump_txt: bool = False):
        """
        输入是每层的指令列表, 本层的起始时间, 本层初始的wait参数
        输出是本层每条指令的时间起点, 时长, 结束时间, wait release参数
        """
        # 遍历现有的inst_dict, 存储指令的依赖以及每条指令的执行时间
        logger.info("Run inst profiler")
        
        # 遍历所有的指令
        for slr_idx, layer_inst_list in enumerate(self.model_inst_list):
            for _, inst_dict in enumerate(layer_inst_list):
                inst_type = get_inst_type(inst_dict)
                self.slr_module_inst_cnt[slr_idx][inst_type] += 1
                self.total_inst_cnt[slr_idx] += 1

                # get wr and time
                wait_list = inst_dict.wait
                wait_id_list = list()
                for wait_name in wait_list:
                    wait_id_list.append(putype_to_inst_type(wait_name))

                release_list = inst_dict.release
                release_id_list = list()
                for release_name in release_list:
                    release_id_list.append(putype_to_inst_type(release_name))

                time = self.get_inst_time_ms(inst_dict, slr_idx)
                self.inst_wr_time[slr_idx][inst_type].append((wait_id_list, release_id_list, time))

                # record for wait release
                for wait_id in wait_id_list:
                    self.rev_array[slr_idx][wait_id][inst_type] += 1
                for release_id in release_id_list:
                    self.pub_array[slr_idx][inst_type][release_id] += 1

        # check for wait release number
        for slr_idx in range(HW_Info.get_slr_count()):
            for i_inst in INST_TYPE:
                for j_inst in INST_TYPE:
                    i = Opcode[i_inst]
                    j = Opcode[j_inst]
                    if self.pub_array[slr_idx][i][j] != self.rev_array[slr_idx][i][j] and i != Opcode.LD and j != Opcode.LD and \
                        (not (i == Opcode.MM and j == Opcode.ST and self.pub_array[slr_idx][i][j] == self.rev_array[slr_idx][i][j] + 1)) and \
                            (not (i == Opcode.MM and j == Opcode.ST and self.pub_array[slr_idx][i][j] == self.rev_array[slr_idx][i][j] - 1)):
                        raise Exception(f"{INST_TYPE[i]} -> {INST_TYPE[j]}, pub {self.pub_array[slr_idx][i][j]}, rev {self.rev_array[slr_idx][i][j]}")
        
        # 根据一个原则来执行如何通过单线程模拟不同模块之间的时间和依赖
        # 每一时刻选择起始时间最小的能执行的来执行
        wr_array = [{
            Opcode[module_name]: {
                Opcode[module_name]: [] for module_name in INST_TYPE
            } for module_name in INST_TYPE
        } for _ in range(HW_Info.get_slr_count())]
        wr_index = [{
            Opcode[module_name]: {
                Opcode[module_name]: 0 for module_name in INST_TYPE
            } for module_name in INST_TYPE
        } for _ in range(HW_Info.get_slr_count())]

        # if start_wait != None:
        #     wr_array[start_wait[0]][start_wait[1]].append(start_time)

        # 代表着当前每条指令的执行编号
        inst_index = [{Opcode[module_name]: 0 for module_name in INST_TYPE} for _ in range(HW_Info.get_slr_count())]
        inst_hardware_time = [{Opcode[module_name]: start_time for module_name in INST_TYPE} for _ in range(HW_Info.get_slr_count())]

        # 此原则是，找到五类指令中满足依赖关系的，开始时间最小的作为本次指令的执行时间开始
        # 并对wait和release列表进行更新
        for slr_idx in range(HW_Info.get_slr_count()):
            while True:
                # 判断是否有满足要求的指令可以执行
                inst_start_time = []
                for i_inst in INST_TYPE:
                    i = Opcode[i_inst]
                    index = inst_index[slr_idx][i]
                    # 是否已经是完成的指令序列
                    if index > (len(self.inst_wr_time[slr_idx][i]) - 1):
                        continue
                    # 判断依赖关系是否满足
                    if len(self.inst_wr_time[slr_idx][i][index][0]) == 0:
                        # 如果没有任何依赖，那么这个开始时间由硬件依赖决定，即上条指令的结束时间
                        inst_start_time.append((i, inst_hardware_time[slr_idx][i]))
                    else:
                        # 如果有依赖，那么需要查询这个依赖在当前状态下是否被满足
                        for w in self.inst_wr_time[slr_idx][i][index][0]:
                            # 依赖被满足需要所有的wr_index小于完整的长度
                            if wr_index[slr_idx][w][i] >= len(wr_array[slr_idx][w][i]):
                                break
                        else:
                            start_time = max([wr_array[slr_idx][w][i][wr_index[slr_idx][w][i]] for w in self.inst_wr_time[slr_idx][i][index][0]])
                            start_time = max(inst_hardware_time[slr_idx][i], start_time)
                            inst_start_time.append((i, start_time))
                # 对当前的状态进行一些判断
                if len(inst_start_time) == 0:
                    # 说明所有的指令都无法执行，那么必然要求两种条件
                    # 一是所有的指令完全执行
                    for i_inst in INST_TYPE:
                        i = Opcode[i_inst]
                        assert inst_index[slr_idx][i] == len(self.inst_wr_time[slr_idx][i])
                    # 二是所有的依赖都已经清空
                    wait_release_count = 0
                    for i_inst in INST_TYPE:
                        for j_inst in INST_TYPE:
                            i = Opcode[i_inst]
                            j = Opcode[j_inst]
                            if wr_index[slr_idx][i][j] < len(wr_array[slr_idx][i][j]):
                                wait_release_count += (len(wr_array[slr_idx][i][j]) - wr_index[slr_idx][i][j])
                    # assert wait_release_count == 0
                    break
                # 从所有开始时间中选择最小的一条指令
                inst_start_time = sorted(inst_start_time, key=lambda x: x[1])
                chosen_id = inst_start_time[0][0]
                start_time = inst_start_time[0][1]
                wait_id_list, release_id_list, time = self.inst_wr_time[slr_idx][chosen_id][inst_index[slr_idx][chosen_id]]
                end_time = start_time + time

                # 更新中间结果的值
                # wait
                for wait_id in wait_id_list:
                    wr_index[slr_idx][wait_id][chosen_id] += 1
                # release
                for release_id in release_id_list:
                    wr_array[slr_idx][chosen_id][release_id].append(end_time)
                # inst_index
                inst_index[slr_idx][chosen_id] += 1
                # hardware_time
                inst_hardware_time[slr_idx][chosen_id] = end_time

                # record time
                # check overlap
                # if len(self.inst_plot_left[chosen_id]) > 0:
                #     assert self.inst_plot_left[chosen_id][-1] + self.inst_plot_width[chosen_id][-1] - start_time < self.TIME_THRESHOLD
                if (len(self.inst_plot_left[slr_idx][chosen_id]) > 0
                        and abs(self.inst_plot_left[slr_idx][chosen_id][-1] + self.inst_plot_width[slr_idx][chosen_id][-1] - start_time) < self.TIME_THRESHOLD
                        and not chosen_id == Opcode.SYS):
                    # 如果与上条指令是相连的，合并时间，加速画图
                    self.inst_plot_width[slr_idx][chosen_id][-1] = self.inst_plot_width[slr_idx][chosen_id][-1] + time
                else:
                    self.inst_plot_left[slr_idx][chosen_id].append(start_time)
                    self.inst_plot_width[slr_idx][chosen_id].append(time)

                # SLR total time
                self.slr_inst_total_time[slr_idx] = max(inst_hardware_time[slr_idx].values())
        
        self.align_time()  # Align sync/interrupt time

        # total time
        self.inst_total_time = max(self.slr_inst_total_time)

        # module part-time
        for slr_idx in range(HW_Info.get_slr_count()):
            self.slr_inst_total_time[slr_idx] += sum(self.sync_plot_width[slr_idx])

            for i_inst in INST_TYPE:
                i = Opcode[i_inst]
                part_time = sum(self.inst_plot_width[slr_idx][i]) if len(self.inst_plot_width[slr_idx][i]) > 0 else 0
                self.slr_module_part_time[slr_idx][i] += part_time

        self.max_module_part_time = max(self.slr_module_part_time[slr_idx].values())

        for slr_idx in range(HW_Info.get_slr_count()):
            logger.info("┌" + 19 * "─" + "FPGA-SLR{:d}".format(slr_idx) + 19 * "─" + "┐")
            for i_inst in INST_TYPE:
                i = Opcode[i_inst]
                logger.info("│ %6s │  %8d  │  %.2e ms  │ %5.1f%%  │" % (
                    i_inst,
                    self.slr_module_inst_cnt[slr_idx][i],
                    self.slr_module_part_time[slr_idx][i],
                    (self.slr_module_part_time[slr_idx][i]) / self.slr_inst_total_time[slr_idx] * 100))
            logger.info("│" + 47 * "─" + "│")
            logger.info("│  %5s │  %8s  │  %.2e ms  │ %5.1f%%  │" % (
                "Sync.",
                "/",
                sum(self.sync_plot_width[slr_idx]),
                sum(self.sync_plot_width[slr_idx]) / self.slr_inst_total_time[slr_idx] * 100))
            # logger.info("│  PRE-  │  "+8*"-"+"  │  %.2e ms  │ %5.1f%%  │" % (self.pre_processed_time, self.pre_processed_time / self.inst_total_time * 100))
            # logger.info("│  POST- │  "+8*"-"+"  │  %.2e ms  │ %5.1f%%  │" % (self.post_processed_time, self.post_processed_time / self.inst_total_time * 100))
            logger.info("│  TOTAL │  %8d  │  %.2e ms  │   100%%  │" % (
                self.total_inst_cnt[slr_idx],
                self.slr_inst_total_time[slr_idx]))
            logger.info("└" + 47 * "─" + "┘")

        logger.info("LD Total  Amount:\t%6.2f MB" % (self.total_LD_size_B/1024/1024))
        logger.info("LD Weight Amount:\t%6.2f MB" % (self.total_LD_weight_size_B/1024/1024))
        logger.info("ST Total  Amount:\t%6.2f MB" % (self.total_ST_size_B/1024/1024))
        logger.info("MAC Total Amount:\t%6.2f M-MACs" % (self.mac_num/1e6))
        
        # logger.info("RS INT8 Amount:\t%6d KB" % (self.rs_data_int8_amount/1024))
        # print("MISC OP COUNT:", self.misc_op_cnt)
        # print("MISC OP PRE-Process COUNT:", self.misc_op_cnt_pre)
        
        if dump_txt:
            self.dump_inst_performance_txt()

        if dump_breakdown:
            self.dump_inst_performance_fig()

    def dump_inst_performance_fig(self):
        import matplotlib
        import matplotlib.pyplot as plt

        # Force matplotlib to not use any Xwindows backend.
        matplotlib.use('Agg')

        logger.info("Start plotting time sequence")

        plt.set_loglevel("notset")

        # figure
        plt.figure(figsize = (10, 10), dpi=300)
        plt.axis('off')
        plt.title(f"Total time = {max(self.slr_inst_total_time):.2f} ms")
        x_axi = (0, max(self.slr_inst_total_time))

        # plot
        for slr_idx in range(HW_Info.get_slr_count()):
            index = int("".join([str(HW_Info.get_slr_count()), "1", str(slr_idx + 1)]))
            plt.subplot(index)

            for i_inst in INST_TYPE:
                i = Opcode[i_inst]
                if len(self.inst_plot_width[slr_idx][i]) > 0:
                    plt.barh(
                        y = INST_TYPE.index(i_inst),
                        width = self.inst_plot_width[slr_idx][i],
                        height = 1,
                        left = self.inst_plot_left[slr_idx][i]
                    )
            if len(self.sync_plot_width[slr_idx]) > 0:
                for i in range(len(self.sync_plot_width[slr_idx])):
                    plt.barh(
                        y = 4,
                        width = self.sync_plot_width[slr_idx][i],
                        height = 1,
                        left = self.sync_plot_left[slr_idx][i],
                        color = "gray",
                        hatch = "////"
                    )
            plt.ylim((-0.5, self.module_num - 0.5))
            plt.yticks([i for i in range(self.module_num)], INST_TYPE)
            plt.ylabel("FPGA-SLR%d Performance" % slr_idx)
            plt.xlim(x_axi)
            plt.xlabel("Time (ms)")

            plt_legend = []
            for i_inst in INST_TYPE:
                i = Opcode[i_inst]
                if len(self.inst_plot_width[slr_idx][i]) > 0:
                    plt_str = f"{i_inst}: {self.slr_module_part_time[slr_idx][i]:.2e} ms; {self.slr_module_part_time[slr_idx][i] / self.slr_inst_total_time[slr_idx] * 100:.1f}%"
                    plt_legend.append(plt_str)
            plt.legend(plt_legend, loc="best", fontsize="x-small", ncol=3)

        plt.savefig(self.inst_performance_fig_dir)
        logger.info(f"Time sequence figure is saved in {self.inst_performance_fig_dir}")
        plt.close("all")
    
    def dump_inst_performance_txt(self):
        path = "./inst_cycle.txt"
        total_cycle = sum([sum(i.values()) for i in self.slr_module_cycle_cnt])
        with open(self.inst_performance_txt_dir, "a+") as f:
            f.write(f"Task: {self.task_name}\n")
            f.write(f"Total cycle: {total_cycle}\n")
            for slr_idx in range(HW_Info.get_slr_count()):
                slr_total_cycle = sum(self.slr_module_cycle_cnt[slr_idx].values())
                f.write(f"SLR{slr_idx}:\n")
                f.write(f"\tSLR Total cycle: {slr_total_cycle}\n")
                for i_inst in INST_TYPE:
                    i = Opcode[i_inst]
                    f.write(f"\t{i_inst}: {self.slr_module_cycle_cnt[slr_idx][i]}\n")
            f.write(f"\n")
