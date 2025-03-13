from .graph_ir import *
from inst import *
from typing import Optional, Union, Any
import math
from utils import hw_info
from .misc_lower import get_inst_type

def tensor_slice_first_dim(t: Union[Tensor, TensorView]) -> list[TensorView]:
    N = t.shape[0]
    ret = []
    new_shape = Shape(*t.shape[1:])
    for i in range(N):
        tensor_view = TensorView(t)
        tensor_view.reside_ddr = t.reside_ddr
        tensor_view.shape = new_shape
        tensor_view.data_type = t.data_type
        tensor_view.addr = t.addr + i * new_shape.prod() * get_size_by_type(t.data_type)
        tensor_view.stride = Shape([0 for i in range(len(t.shape))])
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

def low_align(n: int, m: int) -> int:
    return math.floor(n / m) * m

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
    hw_mm_type = get_hw_type(feature.data_type)
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

def gen_ld_out_scale(n_start: int, n_dim: int, scale: Weight, inst_collect: InstCollector):
    N  = scale.shape[-1]
    n_end = min(n_start + n_dim, N)
    ld_scale = LDInst()
    elt_size = get_size_by_type(scale.data_type)
    elt_num = n_end - n_start
    ld_scale.src_addr = scale.addr + n_start * elt_size
    ld_scale.length_1d = (n_end - n_start) * elt_size
    ld_scale.loop_1d = 1
    ld_scale.src_1d_stride = 0
    ld_scale.bank_addr_stride = 0
    ld_scale.mode = LDMode.DDR2Meta
    ld_scale.dst_group_id = 0
    ld_scale.dst_bank_id = hw_info.MetaBuffer.get_bank_id(hw_info.MetaBankType.OutScale)
    ld_scale.dst_addr = 0
    inst_collect.add(ld_scale)
    return

def gen_ld_weight(n_start: int, n_dim: int, weight: Weight, inst_collect: InstCollector, wait_store_flag: bool = False):
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
    ld_weight.mode = LDMode.HBM2Bbuffer
    ld_weight.dst_group_id = 0
    ld_weight.dst_bank_id = 0
    ld_weight.dst_addr = 0 #+ math.floor((i - n_start) / bank_num) * hw_info.WeightBuffer.get_addr_len_in_bank(K_line_in_bytes)
    inst_collect.add(ld_weight)


def gen_ld_weight_scale(n_start: int, n_dim: int, scale: Weight, inst_collect):
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
    ld_scale.dst_addr = 0
    inst_collect.add(ld_scale)


def gen_ld_bias(n_start: int, n_dim: int, weight: Weight, inst_collect: InstCollector):
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
    ld_bias.dst_addr = 0
    inst_collect.add(ld_bias)

def gen_st_output(m_info: tuple[int, int], n_info: tuple[int, int], output: Union[Tensor, TensorView], inst_collect: InstCollector):
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
    st_res.src_addr =  0 #math.floor((i - m_start) / bank_num) * hw_info.GlobalBuffer.get_addr_len_in_bank(n_dim * get_size_by_type(output.data_type))
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

def lower_linear(op: LinearW8, inst_collect: InstCollector):
    feature = op.get_feature
    act_scale = op.get_act_scale
    weight, scale = op.get_weight
    bias = op.get_bias
    output = op.get_output

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
                gen_ld_weight(n_start, n_end - n_start, weight, inst_collect, wait_store_flag)
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

def lower_matmul(op: Matmul, inst_collect: InstCollector):
    feature = op.get_matrix_A
    weight = op.get_matrix_B
    output = op.get_output
    iterator: Any = None
    if feature.data_type == weight.data_type == DataType.int8:
        feature_act_scale = feature.get_act_scale
        weight_act_scale = weight.get_act_scale
        input_mode = MMInputMode.w8a8
        iterator = zip(gen_2dtensor_view(feature), gen_2dtensor_view(feature_act_scale), gen_2dtensor_view(weight), gen_2dtensor_view(weight_act_scale), gen_2dtensor_view(output))
    elif feature.data_type == weight.data_type == DataType.float16:
        input_mode = MMInputMode.fp16
        iterator = zip(gen_2dtensor_view(feature), gen_2dtensor_view(weight), gen_2dtensor_view(output)) # type: ignore
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
        elif input_mode == MMInputMode.fp16:
            assert len(eltment_2ds) == 3
            feature2d, weight2d, out2d = eltment_2ds
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
                gen_ld_weight(n_start, n_end - n_start, weight2d, inst_collect, wait_store_flag)
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
