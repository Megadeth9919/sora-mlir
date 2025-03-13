from .graph_ir import *
from inst import *
from typing import Optional, Union, Any
import math
from utils import hw_info
from .misc_lower import get_inst_type
import numpy as np
import json

def hbm_layout_convert(t: Tensor, buffer) -> TensorView:
    # HBM的布局转换
    # 输入shape为(B, T, N, K)，输出shape为(B, T, N^, channels, line, K)
    assert t.shape.dim >= 2
    # expand_dim = shape.dim + 2
    expand_shape = Shape(*t.shape[0:-2], math.ceil(t.shape[-2] / (hw_info.HW_Info.get_hbm_channels() * buffer.get_connect_bank_num())), hw_info.HW_Info.get_hbm_channels(), buffer.get_connect_bank_num(), *t.shape[-1:])
    stride = [1 for _ in range(expand_shape.dim)]
    stride[-2] = stride[-1] * expand_shape[-1]
    stride[-3] = hw_info.HW_Info.get_channel_stride(get_size_by_type(t.data_type))
    stride[-4] = stride[-2] * expand_shape[-2]
    for i in range(expand_shape.dim-5, -1, -1):
        stride[i] = stride[i+1] * expand_shape[i+1]
    tensor_view = TensorView(t)
    tensor_view.shape = expand_shape
    tensor_view.reside_ddr = t.reside_ddr
    tensor_view.data_type = t.data_type
    tensor_view.addr = t.addr
    tensor_view.stride = Shape(*stride)
    return tensor_view

def get_hw_type(data_type: DataType):
    if data_type in (DataType.float16, ):
        return hw_info.MM_ParallelType.float
    if data_type in (DataType.int8, DataType.int16):
        return hw_info.MM_ParallelType.int

def get_feature_chunk_num_in_bank(feature: Tensor):
    M, K = feature.shape[-2:]
    m = hw_info.GlobalBuffer.get_chunk_num_in_bank(K * get_size_by_type(feature.data_type))
    # if K <= 32:
    #     m =  min(m, 1)
    hw_mm_type = get_hw_type(DataType.int8)
    mm_m = hw_info.MMCore.get_parallel_dim(hw_mm_type).M
    # 按并行度和bank容量计算feature能放多少m维度的数据
    return m, mm_m

def get_output_chunk_num_in_bank(output: Tensor, tile_N: Optional[int] = None):
    if tile_N is None:
        M, N = output.shape[-2:]
    else:
        N = min(tile_N * hw_info.WeightBuffer.get_bank_num(), output.shape[-1])
    m = hw_info.GlobalBuffer.get_chunk_num_in_bank(hw_info.high_align(N, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(output.data_type))
    # hw_mm_type = get_hw_type(output.data_type)
    # mm_m = hw_info.MMCore.get_parallel_dim(hw_mm_type).M
    # 按并行度和bank容量计算feature能放多少m维度的数据
    return m

def get_weight_chunk_num_in_bank(weight: Weight):
    N, K = weight.shape[-2:]
    n = hw_info.WeightBuffer.get_chunk_num_in_bank(K * get_size_by_type(weight.data_type))
    # if K <= 32:
    #     n =  min(n, 1)
    hw_mm_type = get_hw_type(weight.data_type)
    mm_n = hw_info.MMCore.get_parallel_dim(hw_mm_type).N
    # 按并行度和bank容量计算weight能放多少n维度的数据
    return n, mm_n


def tensor_slice_cube_hbm(t: Union[Tensor, TensorView], tile_size: int = 1, keep_dim: bool = False, slice_dim: int = 0) -> list[TensorView]:
    N = t.shape[slice_dim]
    ret = []
    for i in range(0, N, tile_size):
        if tile_size == 1 and not keep_dim:
            new_shape = Shape(*t.shape[slice_dim + 1:])
        else:
            dim = min(tile_size, N-i)
            if slice_dim > 0:
                new_shape = Shape(*t.shape[:slice_dim], dim, *t.shape[slice_dim + 1:])
            else:
                new_shape = Shape(dim, *t.shape[slice_dim + 1:])
        tensor_view = TensorView(t)
        tensor_view.shape = new_shape
        tensor_view.data_type = t.data_type
        tensor_view.reside_ddr = t.reside_ddr
        tensor_view.addr = t.addr + i * t.stride[0] * get_size_by_type(t.data_type)
        tensor_view.stride = t.stride if keep_dim or tile_size >1 else t.stride[1:]
        ret.append(tensor_view)
    return ret

def gen_2d_expand_tensor_view(t: Tensor):
    if len(t.shape) == 5:
        return tensor_slice_cube_hbm(t)
    elif len(t.shape) == 6:
        ret = []
        for e in tensor_slice_cube_hbm(t):
            ret += tensor_slice_cube_hbm(e)
        return ret
    elif len(t.shape) == 4:
        return [t]
    else:
        raise NotImplementedError("unsupported tensor shape")

def tensor_slice_first_dim(t: Union[Tensor, TensorView], tile_size: int = 1, keep_dim = False, repeat_num = 1) -> list[TensorView]:
    N = t.shape[0]
    ret = []
    for i in range(0, N, tile_size):
        tensor_view = TensorView(t)
        if tile_size == 1 and not keep_dim:
            new_shape = Shape(*t.shape[1:])
        else:
            dim = min(tile_size, N-i)
            new_shape = Shape(dim, *t.shape[1:])
        tensor_view.shape = new_shape
        tensor_view.data_type = t.data_type
        tensor_view.reside_ddr = t.reside_ddr
        tensor_view.addr = t.addr + i * int(np.prod(t.shape[1:])) * get_size_by_type(t.data_type)
        tensor_view.stride = Shape([0 for i in range(len(t.shape))])
        tensor_view.reside_ddr = t.reside_ddr
        for _ in range(repeat_num):
            ret.append(tensor_view)
    return ret

def gen_2dtensor_view(t: Tensor):
    if len(t.shape) == 3:
        return tensor_slice_first_dim(t)
    elif len(t.shape) == 4:
        ret = []
        for e in tensor_slice_first_dim(t):
            ret += tensor_slice_first_dim(e)
        return ret
    elif len(t.shape) == 2:
        return [t]

def gen_3dtensor_view_same(t: Tensor):
    if len(t.shape) == 4:
        return tensor_slice_first_dim(t)
    elif len(t.shape) == 5:
        ret = []
        for e in tensor_slice_first_dim(t):
            ret += tensor_slice_first_dim(e)
        return ret
    elif len(t.shape) == 3:
        return [t]

# 从m_start开始, 加载最多m_dim行数据
def gen_ld_feature(m_start: int, m_dim: int, feature: Union[Tensor, TensorView], inst_collect: InstCollector, wait_store_flag: bool = False):
    M, K = feature.shape
    bank_num = hw_info.MMCore.get_act_bank_num(get_hw_type(feature.data_type))
    K_line_in_bytes = K * get_size_by_type(feature.data_type)
    m_end = min(m_start + m_dim, M)
    last_inst = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else None
    
    # for i in range(m_start, m_end, bank_num):
    ld_feature = LDInst()
    if wait_store_flag and last_inst is not None:
        last_inst.release += [PUType.LD]
        ld_feature.wait += [get_inst_type(last_inst)]
        wait_store_flag = False
    ld_feature.src_addr = feature.addr + m_start * K_line_in_bytes #+ i * K_line_in_bytes
    ld_feature.length_1d = K_line_in_bytes
    ld_feature.loop_1d = m_end - m_start #min(bank_num, m_end - i)
    ld_feature.src_1d_stride = K_line_in_bytes
    ld_feature.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(K_line_in_bytes)
    ld_feature.mode = LDMode.DDR2global if feature.reside_ddr else LDMode.HBM2Global
    ld_feature.dst_group_id = 0
    ld_feature.dst_bank_id = 0
    ld_feature.dst_addr = 0 #+ math.floor((i - m_start) / bank_num) * hw_info.GlobalBuffer.get_addr_len_in_bank(K_line_in_bytes)
    inst_collect.add(ld_feature)

def gen_ld_act_scale(m_start: int, m_dim: int, scale: Union[Tensor, TensorView], inst_collect: InstCollector):
    assert scale.shape[-1] == 1
    M  = scale.shape[-2]
    m_end = min(m_start + m_dim, M)
    ld_scale = LDInst()
    elt_size = get_size_by_type(scale.data_type)
    elt_num = m_end - m_start
    ld_scale.src_addr = scale.addr + m_start * elt_size
    ld_scale.length_1d = elt_num * elt_size
    ld_scale.loop_1d = 1
    ld_scale.src_1d_stride = 0
    ld_scale.bank_addr_stride = hw_info.MetaBuffer.get_addr_len_in_bank(elt_num * elt_size)
    ld_scale.mode = LDMode.DDR2Meta
    ld_scale.dst_group_id = 0
    ld_scale.dst_bank_id = hw_info.MetaBuffer.get_bank_id(hw_info.MetaBankType.ActScale)
    ld_scale.dst_addr = 0
    inst_collect.add(ld_scale)
    return

def gen_ld_weight(n_start: int, n_dim: int, weight: Weight, inst_collect: InstCollector, wait_store_flag: bool = False, bank_start_addr: int = 0):
    # 当前硬件设计需要将weight转换成这个形状
    N, C, L, K = weight.shape
    bank_num = hw_info.MMCore.get_weight_bank_num(get_hw_type(weight.data_type))
    K_line_in_bytes = K * get_size_by_type(weight.data_type)
    n_end = min(n_start + n_dim, N * C * L)
    
    last_inst = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else None
    
    # for i in range(n_start, n_end, bank_num):
    ld_weight = LDInst()
    if wait_store_flag and last_inst is not None:
        last_inst.release += [PUType.LD]
        ld_weight.wait += [get_inst_type(last_inst)]
        wait_store_flag = False
    if n_start % (C*L) == 0:
        src_offeset = n_start // (C * L) * weight.stride[0]
    else:
        off_line = n_start % (C*L)
        src_offeset = n_start // (C * L) * weight.stride[0] + off_line // L * weight.stride[1] + (off_line % L) * weight.stride[2]
    ld_weight.src_addr = weight.addr + src_offeset * get_size_by_type(weight.data_type) 
    ld_weight.length_1d = K_line_in_bytes
    ld_weight.loop_1d = n_end - n_start #min(bank_num, n_end - i)
    ld_weight.ld_weight_step = L == 2
    ld_weight.src_1d_stride = K_line_in_bytes
    ld_weight.bank_addr_stride = hw_info.WeightBuffer.get_addr_len_in_bank(K_line_in_bytes)
    ld_weight.mode = LDMode.HBM2Bbuffer
    ld_weight.dst_group_id = 0
    ld_weight.dst_bank_id = 0
    ld_weight.dst_addr = bank_start_addr #+ math.floor((i - n_start) / bank_num) * hw_info.WeightBuffer.get_addr_len_in_bank(K_line_in_bytes)
    inst_collect.add(ld_weight)

def gen_ld_act_to_weight(n_start: int, n_dim: int, weight: Weight, inst_collect: InstCollector, wait_store_flag: bool = False, weight_bank_addr: int = 0):
    # 当前硬件设计需要将weight转换成这个形状
    N, K = weight.shape
    bank_num = hw_info.MMCore.get_weight_bank_num(get_hw_type(weight.data_type))
    K_line_in_bytes = K * get_size_by_type(weight.data_type)
    n_end = min(n_start + n_dim, N)
    
    last_inst = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else None
    
    # for i in range(n_start, n_end, bank_num):
    ld_weight = LDInst()
    if wait_store_flag and last_inst is not None:
        last_inst.release += [PUType.LD]
        ld_weight.wait += [get_inst_type(last_inst)]
        wait_store_flag = False
    ld_weight.src_addr = weight.addr + n_start * K_line_in_bytes #+ i * K_line_in_bytes
    ld_weight.length_1d = K_line_in_bytes
    ld_weight.loop_1d = n_end - n_start #min(bank_num, n_end - i)
    ld_weight.src_1d_stride = K_line_in_bytes
    ld_weight.bank_addr_stride = hw_info.WeightBuffer.get_addr_len_in_bank(K_line_in_bytes)
    ld_weight.mode = LDMode.DDR2Bbuffer if weight.reside_ddr else LDMode.HBM2BbufferNStride
    ld_weight.dst_group_id = 0
    ld_weight.dst_bank_id = 0
    ld_weight.dst_addr = weight_bank_addr #+ math.floor((i - n_start) / bank_num) * hw_info.WeightBuffer.get_addr_len_in_bank(K_line_in_bytes)
    inst_collect.add(ld_weight)
def gen_ld_weight_scale(n_start: int, n_dim: int, scale: Weight, inst_collect, meta_bank_addr: int = 0):
    if scale.shape.dim > 1:
        assert scale.shape[-1] == 1
        N = scale.shape[-2]
    else:
        N  = scale.shape[-1]
    n_end = min(n_start + n_dim, N)
    ld_scale = LDInst()
    elt_size = get_size_by_type(scale.data_type)
    elt_num = n_end - n_start
    ld_scale.src_addr = scale.addr + n_start * elt_size
    ld_scale.length_1d = elt_num * elt_size
    ld_scale.loop_1d = 1
    ld_scale.src_1d_stride = 0
    ld_scale.bank_addr_stride = hw_info.MetaBuffer.get_addr_len_in_bank(elt_num * elt_size)
    ld_scale.mode = LDMode.DDR2Meta
    ld_scale.dst_group_id = 0
    ld_scale.dst_bank_id = hw_info.MetaBuffer.get_bank_id(hw_info.MetaBankType.WeightScale)
    ld_scale.dst_addr = meta_bank_addr
    inst_collect.add(ld_scale)

def gen_ld_bias(n_start: int, n_dim: int, weight: Weight, inst_collect: InstCollector, bank_addr: int = 0):
    N  = weight.shape[-1]
    n_end = min(n_start + n_dim, N)
    ld_bias = LDInst()
    elt_size = get_size_by_type(weight.data_type)
    elt_num = n_end - n_start
    ld_bias.src_addr = weight.addr + n_start * elt_size
    ld_bias.length_1d = elt_num * elt_size
    ld_bias.loop_1d = 1
    ld_bias.src_1d_stride = 0
    ld_bias.bank_addr_stride = hw_info.MetaBuffer.get_addr_len_in_bank(elt_num * elt_size)
    ld_bias.mode = LDMode.DDR2Meta
    ld_bias.dst_group_id = 0
    ld_bias.dst_bank_id = hw_info.MetaBuffer.get_bank_id(hw_info.MetaBankType.Bias)
    ld_bias.dst_addr = bank_addr
    inst_collect.add(ld_bias)

def gen_st_output(m_info: tuple[int, int], n_info: tuple[int, int], output: Union[Tensor, TensorView], inst_collect: InstCollector, bank_start_addr: int = 0):
    # 将mm计算得到的数据写回ddr, 生成的数据有[m_dim x global_bank_num, n_dim x weight_bank_num]
    # 对应输出tensor的m_start, n_start的位置
    M, N = output.shape
    m_start, m_dim = m_info
    m_end = min(m_start + m_dim, M)
    n_start, n_dim = n_info
    n_end = min(n_start + n_dim, N)
    bank_num = hw_info.GlobalBuffer.get_bank_num_in_BGroup()
    inst_collect[-1].release += [PUType.ST]
    # for i in range(m_start, m_end, bank_num):
    st_res = STInst()
    st_res.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
    st_res.src_addr = bank_start_addr #math.floor((i - m_start) / bank_num) * hw_info.GlobalBuffer.get_addr_len_in_bank(n_dim * get_size_by_type(output.data_type))
    st_res.src_bank_id = 0
    st_res.src_group_id = 1
    st_res.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(n_dim, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(output.data_type))
    st_res.loop_1d = m_end - m_start #min(bank_num, m_end - i)
    st_res.length_1d = n_dim * get_size_by_type(output.data_type)
    st_res.dst_1d_stride = (N) * get_size_by_type(output.data_type)
    st_res.dst_addr = output.addr + (m_start * N + n_start) * get_size_by_type(output.data_type) #(i * N + n_start) * get_size_by_type(output.data_type)
    # if i == 0:
    st_res.wait += [PUType.MM]
    inst_collect.add(st_res)

def lower_matmul(op: Matmul, inst_collect: InstCollector):
    # if hasattr(op, "opt_func") and op.opt_func == "parallel_n":
    lower_matmul_parallel_N(op, inst_collect)
    return
    feature = op.get_matrix_A
    weight = op.get_matrix_B
    output = op.get_output
    # expand_feature = hbm_layout_convert(feature, hw_info.GlobalBuffer)
    # expand_weight = hbm_layout_convert(weight, hw_info.GlobalBuffer) # matmul weight 是另一个op的输出 
    # expand_output = hbm_layout_convert(output, hw_info.GlobalBuffer)
    
    iterator: Any = None
    if feature.data_type == weight.data_type == DataType.int8:
        feature_act_scale = feature.get_act_scale
        weight_act_scale = weight.get_act_scale
        input_mode = MMInputMode.w8a8
        iterator = zip(gen_2dtensor_view(feature), gen_2dtensor_view(feature_act_scale), gen_2dtensor_view(weight), gen_2dtensor_view(weight_act_scale), gen_2dtensor_view(output))
    else:
        raise NotImplementedError("unsupported data type")

    # [M, N] -> m_count x n_count x [m, n]
    M, K = feature.shape[-2:]
    # feature应该适配
    m_feature, m_parallel = get_feature_chunk_num_in_bank(feature)
    #  dim=K
    N, _ = weight.shape[-2:]
    n_weight, n_parallel = get_weight_chunk_num_in_bank(weight)
    
    m_output = get_output_chunk_num_in_bank(output, n_weight)

    m_feature = min(m_feature, m_output)

    # 每次在m方向取m_tiling_size，即(m_tiling_size, K)的一片数据到GlobalBuffer
    m_tiling_size = m_feature * hw_info.MMCore.get_act_bank_num(get_hw_type(feature.data_type))
    # 每次在n方向取n_tiling_size，即(n_tiling_size, K)的一片数据到WeightBuffer
    n_tiling_size = n_weight * hw_info.MMCore.get_weight_bank_num(get_hw_type(weight.data_type))

    m_tiling_size = min(m_tiling_size, 2**12)
    n_tiling_size = min(n_tiling_size, 2**12)

    wait_store_flag = True

    for eltment_2ds in iterator:
        if input_mode == MMInputMode.w8a8:
            feature2d, feature_act_scale2d, weight2d, weight_act_scale2d, out2d = eltment_2ds
        else:
            raise NotImplementedError("unsupported input mode")
        assert feature2d.addr is not None
        assert out2d.addr is not None
        for i in range(0, M, m_tiling_size):
            m_start = i
            m_end: int = min(m_start + m_tiling_size, M)
            gen_ld_feature(m_start, m_end - m_start, feature2d, inst_collect, wait_store_flag)
            if wait_store_flag:
                wait_store_flag = False
            if input_mode == MMInputMode.w8a8:
                gen_ld_act_scale(m_start, m_end - m_start, feature_act_scale2d, inst_collect)
            for j in range(0, N, n_tiling_size):
                n_start = j
                n_end: int = min(n_start + n_tiling_size, N)
                gen_ld_act_to_weight(n_start, n_end - n_start, weight2d, inst_collect, wait_store_flag)
                if wait_store_flag:
                    wait_store_flag = False
                if input_mode == MMInputMode.w8a8:
                    gen_ld_weight_scale(n_start, n_end - n_start, weight_act_scale2d, inst_collect)
                # gen_ld_out_scale
                # gen_ld_bias
                inst_collect[-1].release += [PUType.MM]
                mm = MMInst()
                mm.wait += [PUType.LD]
                mm.input_mode = input_mode
                if out2d.data_type == DataType.int8:
                    mm.output_mode = MMOutputMode.int8
                elif out2d.data_type == DataType.float16:
                    mm.output_mode = MMOutputMode.fp16
                mm.act_start_addr = 0
                mm.act_bank_group_id = 0
                mm.out_bank_group_id = 1
                mm.bias_flag = False
                mm.output_flag = True
                mm.weights_start_addr = 0
                mm.bias_start_addr = 0
                mm.out_start_addr = 0
                mm.M = (m_end - m_start) -1
                mm.N = (n_end - n_start) -1
                mm.K = K -1
                mm.act_scale_start_addr = 0
                mm.out_scale_start_addr = 0
                mm.weights_scale_start_addr = 0
                inst_collect.add(mm)
                gen_st_output((m_start, (m_end - m_start)), (n_start, (n_end - n_start)), out2d, inst_collect)
                wait_store_flag = True

def lower_matmul_parallel_N(op: Matmul, inst_collect: InstCollector):
    # print("Matmul lowering func: ", op.opt_func)
    feature = op.get_matrix_A
    weight = op.get_matrix_B
    output = op.get_output
    # expand_feature = hbm_layout_convert(feature, hw_info.GlobalBuffer)
    # expand_weight = hbm_layout_convert(weight, hw_info.GlobalBuffer) # matmul weight 是另一个op的输出 
    # expand_output = hbm_layout_convert(output, hw_info.GlobalBuffer)
    
    iterator: Any = None
    if feature.data_type == weight.data_type == DataType.int8:
        feature_act_scale = feature.get_act_scale
        weight_act_scale = weight.get_act_scale
        input_mode = MMInputMode.w8a8
        iterator = zip(gen_2dtensor_view(feature), gen_2dtensor_view(feature_act_scale), gen_2dtensor_view(weight), gen_2dtensor_view(weight_act_scale), gen_2dtensor_view(output))
    else:
        raise NotImplementedError("unsupported data type")

    # [M, N] -> m_count x n_count x [m, n]
    M, K = feature.shape[-2:]
    # feature应该适配
    m_feature, m_parallel = get_feature_chunk_num_in_bank(feature)
    #  dim=K
    N, _ = weight.shape[-2:]
    n_weight, n_parallel = get_weight_chunk_num_in_bank(weight)
    
    m_output = get_output_chunk_num_in_bank(output, n_weight)

    m_feature = min(m_feature, m_output)

    # 每次在m方向取m_tiling_size，即(m_tiling_size, K)的一片数据到GlobalBuffer
    m_tiling_size = m_feature * hw_info.MMCore.get_act_bank_num(get_hw_type(feature.data_type))
    # 每次在n方向取n_tiling_size，即(n_tiling_size, K)的一片数据到WeightBuffer
    n_tiling_size = n_weight * hw_info.MMCore.get_weight_bank_num(get_hw_type(weight.data_type))

    m_tiling_size = min(m_tiling_size, 2**12)
    n_tiling_size = min(n_tiling_size, 2**12)

    wait_store_flag = True

    for eltment_2ds in iterator:
        if input_mode == MMInputMode.w8a8:
            feature2d, feature_act_scale2d, weight2d, weight_act_scale2d, out2d = eltment_2ds
        else:
            raise NotImplementedError("unsupported input mode")
        assert feature2d.addr is not None
        assert out2d.addr is not None
        for i in range(0, M, m_tiling_size):
            m_start = i
            m_end: int = min(m_start + m_tiling_size, M)
            m_dim = m_end - m_start
            gen_ld_feature(m_start, m_end - m_start, feature2d, inst_collect, wait_store_flag)
            if wait_store_flag:
                wait_store_flag = False
            if input_mode == MMInputMode.w8a8:
                gen_ld_act_scale(m_start, m_end - m_start, feature_act_scale2d, inst_collect)
            for j in range(0, N, n_tiling_size):
                n_start = j
                n_end: int = min(n_start + n_tiling_size, N)
                n_dim = n_end - n_start
                
                real_n = min(n_weight * n_parallel, N)
                real_n = hw_info.high_align(real_n // 4, n_parallel)
                dependency_num = math.ceil(n_tiling_size / real_n)
                real_n_tiling_size = math.ceil(n_tiling_size / dependency_num)
                real_n_tiling_size = hw_info.high_align(real_n_tiling_size, real_n)
                real_dependency_num = math.ceil(n_dim / real_n_tiling_size)
                
                # real_weight_buffer_depth = (real_n_tiling_size // n_parallel) * hw_info.WeightBuffer.get_addr_len_in_bank(K * get_size_by_type(weight.data_type)) * dependency_num
                # real_w_scale_meta_buffer_depth = hw_info.WeightBuffer.get_addr_len_in_bank(real_n_tiling_size * get_size_by_type(weight_act_scale.data_type)) * dependency_num
                weight_bank_start_addr = 0
                w_scale_bank_addr = 0
                st_bank_addr = 0
                for idx, k in enumerate(range(n_start, n_end, real_n_tiling_size)):
                    real_n_start = k
                    real_n_end = min(real_n_start + real_n_tiling_size, n_end)
                    real_n_dim = real_n_end - real_n_start
                    weight_bank_start_addr = int(real_n_tiling_size / n_parallel) * idx * hw_info.WeightBuffer.get_addr_len_in_bank(K * get_size_by_type(weight.data_type))
                    gen_ld_act_to_weight(real_n_start, real_n_dim, weight2d, inst_collect, wait_store_flag, weight_bank_start_addr)
                    if input_mode == MMInputMode.w8a8:
                        w_scale_bank_addr = idx * hw_info.MetaBuffer.get_addr_len_in_bank(real_n_tiling_size * get_size_by_type(weight_act_scale.data_type))
                        gen_ld_weight_scale(real_n_start, real_n_dim, weight_act_scale2d, inst_collect, w_scale_bank_addr)
                    
                    st_bank_addr = idx * math.ceil(m_dim / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_tiling_size, n_parallel) * get_size_by_type(output.data_type))
                    inst_collect[-1].release += [PUType.MM]
                    mm = MMInst()
                    mm.wait += [PUType.LD]
                    mm.input_mode = input_mode
                    if out2d.data_type == DataType.int8:
                        mm.output_mode = MMOutputMode.int8
                    elif out2d.data_type == DataType.float16:
                        mm.output_mode = MMOutputMode.fp16
                    mm.act_start_addr = 0
                    mm.act_bank_group_id = 0
                    mm.out_bank_group_id = 1
                    mm.bias_flag = False
                    mm.output_flag = True
                    mm.weights_start_addr = weight_bank_start_addr
                    mm.bias_start_addr = 0
                    mm.out_start_addr = st_bank_addr
                    mm.M = (m_end - m_start) -1
                    mm.N = (real_n_end - real_n_start) -1
                    mm.K = K -1
                    mm.act_scale_start_addr = 0
                    mm.out_scale_start_addr = 0
                    mm.weights_scale_start_addr = w_scale_bank_addr
                    inst_collect.add(mm)
                    gen_st_output((m_start, (m_end - m_start)), (real_n_start, (real_n_end - real_n_start)), out2d, inst_collect, st_bank_addr)
                    if real_dependency_num == idx + 1:
                        wait_store_flag = True

def lower_linear(op: LinearW8, inst_collect: InstCollector):
    # if hasattr(op, "opt_func") and op.opt_func == "parallel_n":
    lower_linear_parallel_N(op, inst_collect)
    return
    
    feature = op.get_feature
    act_scale = op.get_act_scale
    weight, scale = op.get_weight
    bias = op.get_bias
    output = op.get_output

    # expand_feature = hbm_layout_convert(feature, hw_info.GlobalBuffer)
    expand_weight = hbm_layout_convert(weight, hw_info.WeightBuffer)
    # expand_output = hbm_layout_convert(output, hw_info.GlobalBuffer)

    # [M, N] -> m_count x n_count x [m, n]
    M, K = feature.shape[-2:]
    # feature应该适配
    m_feature, m_parallel = get_feature_chunk_num_in_bank(feature)
    #  dim=K
    N, _ = weight.shape
    n_weight, n_parallel = get_weight_chunk_num_in_bank(weight)
    
    m_output = get_output_chunk_num_in_bank(output, n_weight)

    m_feature = min(m_feature, m_output)

    # 每次在m方向取m_tiling_size，即(m_tiling_size, K)的一片数据到GlobalBuffer
    m_tiling_size = m_feature * hw_info.GlobalBuffer.get_bank_num_in_BGroup()
    # 每次在n方向取n_tiling_size，即(n_tiling_size, K)的一片数据到WeightBuffer
    n_tiling_size = n_weight * hw_info.WeightBuffer.get_bank_num()
    
    m_tiling_size = min(m_tiling_size, 2**12)
    n_tiling_size = min(n_tiling_size, 2**12)

    wait_store_flag = True

    for feature2d, act_scale2d, out2d in zip(gen_2dtensor_view(feature), gen_2dtensor_view(act_scale), gen_2dtensor_view(output)):
        assert feature2d.addr is not None
        assert act_scale2d.addr is not None
        assert out2d.addr is not None
        for i in range(0, M, m_tiling_size):
            m_start = i
            m_end: int = min(m_start + m_tiling_size, M)
            gen_ld_feature(m_start, m_end - m_start, feature2d, inst_collect, wait_store_flag)
            if wait_store_flag:
                wait_store_flag = False
            gen_ld_act_scale(m_start, m_end - m_start, act_scale2d, inst_collect)
            for j in range(0, N, n_tiling_size):
                n_start = j
                n_end: int = min(n_start + n_tiling_size, N)
                gen_ld_weight(n_start, n_end - n_start, expand_weight, inst_collect, wait_store_flag)
                if wait_store_flag:
                    wait_store_flag = False
                gen_ld_weight_scale(n_start, n_end - n_start, scale, inst_collect)
                # gen_ld_out_scale
                if op.bias_flag:
                    assert bias is not None
                    gen_ld_bias(n_start, n_end - n_start, bias, inst_collect)
                    pass
                inst_collect[-1].release += [PUType.MM]
                mm = MMInst()
                mm.wait += [PUType.LD]
                mm.input_mode = MMInputMode.w8a8
                if out2d.data_type == DataType.int8:
                    raise NotImplementedError("int8 output is not supported yet")
                elif out2d.data_type == DataType.float16:
                    mm.output_mode = MMOutputMode.fp16
                mm.act_start_addr = 0
                mm.act_bank_group_id = 0
                mm.out_bank_group_id = 1
                mm.bias_flag = op.bias_flag
                mm.output_flag = True
                mm.weights_start_addr = 0
                mm.bias_start_addr = 0
                mm.out_start_addr = 0
                mm.M = (m_end - m_start) -1
                mm.N = (n_end - n_start) -1 
                mm.K = K -1
                mm.act_scale_start_addr = 0
                mm.out_scale_start_addr = 0
                mm.weights_scale_start_addr = 0
                inst_collect.add(mm)
                gen_st_output((m_start, (m_end - m_start)), (n_start, (n_end - n_start)), out2d, inst_collect)
                wait_store_flag = True

def lower_linear_parallel_N(op: LinearW8, inst_collect: InstCollector):
    # print("Linear lowering func: ", op.opt_func)
    
    feature = op.get_feature
    act_scale = op.get_act_scale
    weight, scale = op.get_weight
    bias = op.get_bias
    output = op.get_output

    # expand_feature = hbm_layout_convert(feature, hw_info.GlobalBuffer)
    expand_weight = hbm_layout_convert(weight, hw_info.WeightBuffer)
    # expand_output = hbm_layout_convert(output, hw_info.GlobalBuffer)

    # [M, N] -> m_count x n_count x [m, n]
    M, K = feature.shape[-2:]
    # feature应该适配
    m_feature, m_parallel = get_feature_chunk_num_in_bank(feature)
    #  dim=K
    N, _ = weight.shape
    n_weight, n_parallel = get_weight_chunk_num_in_bank(weight)
    
    m_output = get_output_chunk_num_in_bank(output, n_weight)

    m_feature = min(m_feature, m_output)

    # 每次在m方向取m_tiling_size，即(m_tiling_size, K)的一片数据到GlobalBuffer
    m_tiling_size = m_feature * hw_info.GlobalBuffer.get_bank_num_in_BGroup()
    # 每次在n方向取n_tiling_size，即(n_tiling_size, K)的一片数据到WeightBuffer
    n_tiling_size = n_weight * hw_info.WeightBuffer.get_bank_num()
    
    m_tiling_size = min(m_tiling_size, 2**12)
    n_tiling_size = min(n_tiling_size, 2**12)

    wait_store_flag = True

    for feature2d, act_scale2d, out2d in zip(gen_2dtensor_view(feature), gen_2dtensor_view(act_scale), gen_2dtensor_view(output)):
        assert feature2d.addr is not None
        assert act_scale2d.addr is not None
        assert out2d.addr is not None
        for i in range(0, M, m_tiling_size):
            m_start = i
            m_end: int = min(m_start + m_tiling_size, M)
            m_dim = m_end - m_start
            gen_ld_feature(m_start, m_end - m_start, feature2d, inst_collect, wait_store_flag)
            if wait_store_flag:
                wait_store_flag = False
            gen_ld_act_scale(m_start, m_end - m_start, act_scale2d, inst_collect)
            for j in range(0, N, n_tiling_size):
                n_start = j
                n_end: int = min(n_start + n_tiling_size, N)
                n_dim = n_end - n_start
                
                real_n = min(n_weight * n_parallel, N)
                real_n = hw_info.high_align(real_n // 4, n_parallel)
                dependency_num = math.ceil(n_tiling_size / real_n)
                real_n_tiling_size = math.ceil(n_tiling_size / dependency_num)
                real_n_tiling_size = hw_info.high_align(real_n_tiling_size, real_n)
                real_dependency_num = math.ceil(n_dim / real_n_tiling_size)
                
                weight_bank_addr = 0
                w_scale_bank_addr = 0
                st_bank_addr = 0
                bias_bank_addr = 0
                for idx, k in enumerate(range(n_start, n_end, real_n_tiling_size)):
                    real_n_start = k
                    real_n_end = min(real_n_start + real_n_tiling_size, n_end)
                    real_n_dim = real_n_end - real_n_start
                    weight_bank_addr = int(real_n_tiling_size / n_parallel) * idx * hw_info.WeightBuffer.get_addr_len_in_bank(K * get_size_by_type(weight.data_type))
                    st_bank_addr = idx * math.ceil(m_dim / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_tiling_size, n_parallel) * get_size_by_type(output.data_type))
                    w_scale_bank_addr = idx * hw_info.MetaBuffer.get_addr_len_in_bank(real_n_tiling_size * get_size_by_type(scale.data_type))
                
                    gen_ld_weight(real_n_start, real_n_end - real_n_start, expand_weight, inst_collect, wait_store_flag, weight_bank_addr)
                    if wait_store_flag:
                        wait_store_flag = False
                    gen_ld_weight_scale(real_n_start, real_n_end - real_n_start, scale, inst_collect, w_scale_bank_addr)
                    # gen_ld_out_scale
                    if op.bias_flag:
                        assert bias is not None
                        bias_bank_addr = idx * hw_info.MetaBuffer.get_addr_len_in_bank(real_n_tiling_size * get_size_by_type(bias.data_type))
                        gen_ld_bias(real_n_start, real_n_end - real_n_start, bias, inst_collect, bias_bank_addr)
                        pass
                    inst_collect[-1].release += [PUType.MM]
                    mm = MMInst()
                    mm.wait += [PUType.LD]
                    mm.input_mode = MMInputMode.w8a8
                    if out2d.data_type == DataType.int8:
                        raise NotImplementedError("int8 output is not supported yet")
                    elif out2d.data_type == DataType.float16:
                        mm.output_mode = MMOutputMode.fp16
                    mm.act_start_addr = 0
                    mm.act_bank_group_id = 0
                    mm.out_bank_group_id = 1
                    mm.bias_flag = op.bias_flag
                    mm.output_flag = True
                    mm.weights_start_addr = weight_bank_addr
                    mm.bias_start_addr = bias_bank_addr
                    mm.out_start_addr = st_bank_addr
                    mm.M = (m_end - m_start) -1
                    mm.N = (real_n_end - real_n_start) -1 
                    mm.K = K -1
                    mm.act_scale_start_addr = 0
                    mm.out_scale_start_addr = 0
                    mm.weights_scale_start_addr = w_scale_bank_addr
                    inst_collect.add(mm)
                    gen_st_output((m_start, (m_end - m_start)), (real_n_start, (real_n_end - real_n_start)), out2d, inst_collect, st_bank_addr)
                    if real_dependency_num == idx + 1:
                        wait_store_flag = True
                        
BANK_SIZE = hw_info.GlobalBuffer.get_bank_bytes_num()
def get_tile_dim(dim: int, tile_size: int, index: int, max_index: int) -> int:
    return (
        tile_size
        if (index != max_index - 1)
        else dim % tile_size if dim % tile_size != 0 else tile_size
    )
def tensor_slice_cube_hbm_layout(t: Union[Tensor, TensorView], max_bank=1) -> list[TensorView]:
    assert t.shape.dim >= 5
    ret = []
    dim0 = t.shape[-1]
    dim3 = t.shape[-4]
    # assert(max_bank %2 ==0)
    assert BANK_SIZE >= dim3 * dim0 * get_size_by_type(t.data_type)
    length = len(t.shape)
    N = math.prod(t.shape[: length - 5])
    # dim0 = t.shape[-1]
    # dim1 = max_bank
    dim1 = 1
    dim1_tile_num = math.ceil(t.shape[-5] / dim1)
    addr = t.addr
    for i in range(N):
        for j in range(dim1_tile_num):
            tensor_view = TensorView(t)
            tensor_view.shape = Shape(
                get_tile_dim(t.shape[-2], dim1, j, dim1_tile_num), t.shape[-4], t.shape[-3], t.shape[-2], t.shape[-1]
            )
            tensor_view.data_type = t.data_type
            tensor_view.addr = addr
            tensor_view.stride = t.stride
            tensor_view.reside_ddr = t.reside_ddr
            addr += (j + 1) * t.stride[-5] * get_size_by_type(t.data_type)
            ret.append(tensor_view)
    return ret

def lower_copy_hbm(op: Copy, inst_collect: InstCollector):
    input = op.get_input
    output = op.get_output
    assert input.shape == output.shape
    assert input.shape.dim == output.shape.dim >= 2 
    
    # tensor 在hbm的layout与origin shape不同
    expand_input = hbm_layout_convert(input, hw_info.GlobalBuffer)
    expand_output = hbm_layout_convert(output, hw_info.GlobalBuffer)
    
    for input3d, output3d in zip(tensor_slice_cube_hbm_layout(expand_input), tensor_slice_cube_hbm_layout(expand_output)):
        assert input3d.shape == output3d.shape
        assert input3d.addr is not None
        assert output3d.addr is not None
        ld = LDInst()
        last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
        if last_ins is not None:
            last_ins.release.append(PUType.LD)
            ld.wait.append(get_inst_type(last_ins))
        ld.release.append(PUType.ST)
        ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
        ld.length_1d = input3d.shape[-1] * get_size_by_type(input3d.data_type)
        ld.loop_1d = math.prod(input3d.shape[-4:-1])
        ld.src_1d_stride = input3d.shape[-1] * get_size_by_type(input3d.data_type)
        ld.src_addr = input3d.addr
        ld.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(input3d.shape[-1] * get_size_by_type(input3d.data_type))
        inst_collect.add(ld)
        st = STInst()
        st.wait.append(PUType.LD)
        st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
        st.length_1d = output3d.shape[-1] * get_size_by_type(output3d.data_type)
        st.loop_1d = math.prod(output3d.shape[-4:-1])
        st.dst_1d_stride = output.shape[-1] * get_size_by_type(output.data_type)
        st.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(output3d.shape[-1] * get_size_by_type(input.data_type))
        st.dst_addr = output3d.addr
        inst_collect.add(st)