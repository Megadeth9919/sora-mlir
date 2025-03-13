from .graph_ir import *
from inst import *
from .mm_lower_hbm import lower_matmul, lower_linear
from .misc_lower import *
from .fuse_op_lower import *
from .pass_core_slice import core_slice_tensor
from .pass_mem_allocate import graph_mem_allocate ,ddr_address_alloc,CCTarget,recoverTensorAddr,getAddrTable
from .pass_fuse_cvt_linearw8 import fuse_cvt_linearw8
from .pass_fuse_div_cvt_matmul import fuse_div_cvt_matmul
from .pass_fuse_linearw8_act import fuse_linearw8_act
from .pass_fuse_linearw8_transpose import fuse_linearw8_transpose
from .pass_fuse_transpose23_cvt import fuse_transpose23_cvt
from .pass_fuse_misc_cvt import fuse_misc_cvt
from .pass_fuse_softmax_cvt_matmul import fuse_softmax_cvt_matmul
from .pass_cut_gelu_dscale import cut_gelu_dscale
from .pass_insert_view_before_linear import insert_view_before_linear
from .pass_insert_view_before_linear_act import insert_view_before_linear_act
from .pass_insert_view_before_cvt_linear import insert_view_before_cvt_linear
from .pass_mem_reside import pass_mem_reside
from .StatisticPass import Statistic_Pass
from Common import HookInfo, InfoCollector
from dataclasses import dataclass
import json
import yaml
from utils import hw_info
import time
import tqdm

import numpy as np
from collections import namedtuple
@dataclass
class CompileFlags:
    target: CCTarget = CCTarget.verify
    enable_slr_slice: bool = False
    enable_gen_golden: bool = False
    optimization:int = 0
    compile_only:bool = False

def graph_compile(g: StaticGraph, flags: CompileFlags, sparsification:bool = False,debug:bool = False):
    if(flags.optimization >0):
        cut_gelu_dscale(g)
        fuse_transpose23_cvt(g)
        fuse_cvt_linearw8(g)
        fuse_softmax_cvt_matmul(g)
        fuse_div_cvt_matmul(g)
        fuse_linearw8_act(g)
        fuse_misc_cvt(g)
        insert_view_before_linear_act(g)
        insert_view_before_cvt_linear(g)
    insert_view_before_linear(g)
    g.complete()
    pass_mem_reside(g)
    print("intermediate tensor",len(g.get_intermediate()))
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
    if sparsification:
        rt_info = recoverTensorAddr(g,getAddrTable())
    else :
        if flags.target == CCTarget.verify:
            rt_info = ddr_address_alloc(g, flags.target)
        else:
            rt_info = graph_mem_allocate(g, flags.target,static_allocate= debug)
    
    load_weights(g)
    p_model_forward(g)
    Statistic_Pass(g)
    
    # inst_collect = InstCollector()
    # ops = list(g.get_ops())
    # lower(ops, inst_collect)
    if flags.enable_slr_slice:
        op_lists = core_slice_tensor(g, hw_info.HW_Info.get_slr_count())
        hook_ret = [InfoCollector() for i in range(hw_info.HW_Info.get_slr_count())]
        ret = []
        for idx,ops in enumerate(op_lists):
            inst_collect = InstCollector()
            lower(ops, inst_collect,hook_ret[idx],debug)
            ret.append(inst_collect)
            # for op in ops:
            #     print(idx,"{")
            #     op.show()
            #     print("}")
        return rt_info, ret, hook_ret
    else:
        ret = [InstCollector() for i in range(hw_info.HW_Info.get_slr_count())]
        hook_ret = [InfoCollector() for i in range(hw_info.HW_Info.get_slr_count())]
        ops = list(g.get_ops())
        lower(ops, ret[0],hook_ret[0],debug)
        # gen_sync_inst(g, ret[1])
        # gen_sync_inst(g, ret[2])
        return rt_info, ret, hook_ret
    # return rt_info, inst_collect

def gen_sync_inst(g: StaticGraph, inst_collect: InstCollector):
    for op in g.get_ops():
        sync = SYSCInst()
        sync.op = SysOp.interrupt
        inst_collect.add(sync)
    return inst_collect

def tensor_size(t: Tensor):
    return t.size * get_size_by_type(t.data_type)

def pack_numpy_array(arr_list: list[np.ndarray]):
    b = bytes()
    for a in arr_list:
        b += a.tobytes()

    ret = np.frombuffer(b, dtype=np.int8)
    return ret



def load_weights(g: StaticGraph, weight_dict: Optional[dict] = None):
    if weight_dict is not None:
        for op in g.get_ops():
            op.load_weight_from_dict(weight_dict)
    return None


def p_model_forward(g: StaticGraph):
    return None

# 直接覆盖掉View op的输出tensor
def remove_view(g: StaticGraph):
    for op in g.get_ops():
        if isinstance(op, View):
            tensor_in = op.get_input
            assert tensor_in.addr
            tensor_out = op.get_output
            assert tensor_out.addr
            tensor_out.addr = tensor_in.addr
    return 


def lower(ops: list[Op], inst_collect: InstCollector, hook_collect: InfoCollector,dump:bool = False):
    if not dump:
        hook_info = HookInfo()
        hook_info.op_name = "graph"
        hook_info.input[hook_info.op_name] = []
        hook_collect.add(hook_info)
    with tqdm.tqdm(total=len(ops), desc='Lowering') as pbar:
        for op in ops:
            if isinstance(op, LinearW8Act):
                lower_Linearw8_act(op, inst_collect, op.act_scale_flag)
            elif isinstance(op, LinearW8Transpose):
                lower_linearw8_transpose(op, inst_collect)
            elif isinstance(op, DivCvtMatmul):
                lower_div_cvt_matmul(op, inst_collect)
            elif isinstance(op, SoftmaxCvtMatmul):
                lower_unary_cvt_matmul(op, inst_collect)
            elif isinstance(op, CvtLinearW8):
                lower_cvt_linearw8(op, inst_collect)
            elif isinstance(op, LinearW8):
                lower_linear(op, inst_collect)
            elif isinstance(op, Matmul):
                lower_matmul(op, inst_collect)
            elif isinstance(op, TransposeCvt):
                lower_transpose_cvt(op, inst_collect)
            elif (
                isinstance(op, Layernorm)
                or isinstance(op, RMSnorm)
            ):
                lower_tow_stage_for_parallel(op, inst_collect, op.act_scale_flag)
                # lower_tow_stage(op, inst_collect, op.act_scale_flag)
            elif isinstance(op, Eltwise) or isinstance(op, Div):
                lower_binary_for_parallel(op, inst_collect, op.act_scale_flag)
                # lower_binary(op, inst_collect, op.act_scale_flag)
            elif isinstance(op, Silu) or \
                isinstance(op, Gelu) or \
                isinstance(op, Softmax) or \
                isinstance(op, Convert):
                lower_unary_for_parallel(op, inst_collect, op.act_scale_flag)
                # lower_unary(op, inst_collect, op.act_scale_flag)
            elif isinstance(op, Transpose):
                trans_dims = op.get_trans_dims
                if (trans_dims[0] == 0 and trans_dims[1] == 1) or (
                    trans_dims[0] == 1 and trans_dims[1] == 0
                ):
                    # lower_transpose01(op, inst_collect)
                    lower_transpose01_for_parallel(op, inst_collect)
                elif (trans_dims[0] == 2 and trans_dims[1] == 3) or (
                    trans_dims[0] == 3 and trans_dims[1] == 2
                ):
                    lower_transpose23(op, inst_collect)
                elif (trans_dims[0] == 1 and trans_dims[1] == 2) or (
                        trans_dims[0] == 2 and trans_dims[1] == 1
                    ):
                    lower_transpose12(op, inst_collect)
                else:
                    raise NotImplementedError(f"not implemented transpose: {op}")
            elif isinstance(op, View):
                pass
            elif isinstance(op, Split):
                # lower_split(op, inst_collect)
                lower_split_for_parallel(op, inst_collect)
            elif isinstance(op, RoPE):
                lower_rope(op, inst_collect, op.act_scale_flag)
            elif isinstance(op,LoadInst):
                lower_load_inst(op, inst_collect)
            elif isinstance(op, Copy):
                # raise NotImplementedError(f"not implemented op: {op}")
                lower_copy(op, inst_collect)
            elif isinstance(op, FakeLoad):
                lower_fakeload(op, inst_collect)
            elif isinstance(op, Sync):
                lower_sync(op, inst_collect)
            elif isinstance(op, Nop):
                    pass
            else:
                raise NotImplementedError(f"not implemented op: {op}")
            if dump :
                if not isinstance(op, View):
                    hook_info = HookInfo()
                    hook_info.op_name = op.name
                    if len(op.get_inputs()) > 0:
                        hook_info.input[op.get_inputs()[0].name] = list(op.get_inputs()[0].shape)
                    hook_collect.add(hook_info)
                    sync = SYSCInst()
                    sync.op = SysOp.interrupt
                    if len(inst_collect) > 0:
                        last_ins = inst_collect[-1]
                        if not isinstance(last_ins, SYSCInst):
                            last_ins.release.append(PUType.SYS)
                            sync.wait.append(get_inst_type(last_ins))
                    inst_collect.add(sync)
            pbar.update(1)
    if not dump :
        sync = SYSCInst()
        sync.op = SysOp.interrupt
        if len(inst_collect) > 0:
            last_ins = inst_collect[-1]
            if not isinstance(last_ins, SYSCInst):
                last_ins.release.append(PUType.SYS)
                sync.wait.append(get_inst_type(last_ins))
        inst_collect.add(sync)