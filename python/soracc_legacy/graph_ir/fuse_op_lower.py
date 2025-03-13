from .mm_lower_hbm import *
from .misc_lower import get_max_banks_half
import numpy as np
def hbm_layout_convert_merg_high_dim(t: Tensor, buffer) -> TensorView:
    # HBM的布局转换
    assert t.shape.dim >= 2
    
    if buffer is hw_info.MetaBuffer or buffer is hw_info.GlobalBuffer:
        if t.shape.dim == 2:
            expand_shape = Shape(1, *t.shape[-2:])
        else:
            expand_shape = Shape(prod(t.shape[:-2]), *t.shape[-2:])
        stride = [1 for _ in range(expand_shape.dim)]
        for i in range(expand_shape.dim-2, -1, -1):
            stride[i] = stride[i+1] * expand_shape[i+1]
        tensor_view = TensorView(t)
        tensor_view.shape = expand_shape
        tensor_view.data_type = t.data_type
        tensor_view.reside_ddr = t.reside_ddr
        tensor_view.addr = t.addr
        tensor_view.stride = Shape(*stride)
        return tensor_view
    # 输入shape为(B, T, N, K)，输出shape为(B*T, N^, channels, line, K)
    # expand_dim = shape.dim + 2
    expand_shape = Shape(prod(t.shape[0:-2]), math.ceil(t.shape[-2] / (hw_info.HW_Info.get_hbm_channels() * buffer.get_connect_bank_num())), hw_info.HW_Info.get_hbm_channels(), buffer.get_connect_bank_num(), *t.shape[-1:])
    stride = [1 for _ in range(expand_shape.dim)]
    stride[-2] = stride[-1] * expand_shape[-1]
    stride[-3] = hw_info.HW_Info.get_channel_stride(get_size_by_type(t.data_type))
    stride[-4] = stride[-2] * expand_shape[-2]
    for i in range(expand_shape.dim-5, -1, -1):
        stride[i] = stride[i+1] * expand_shape[i+1]
    tensor_view = TensorView(t)
    tensor_view.shape = expand_shape
    tensor_view.data_type = t.data_type
    tensor_view.reside_ddr = t.reside_ddr
    tensor_view.addr = t.addr
    tensor_view.stride = Shape(*stride)
    return tensor_view
    
def gen_3d_expand_tensor_view(t: Tensor):
    if len(t.shape) == 6:
        return tensor_slice_cube_hbm(t)
    elif len(t.shape) == 5:
        return [t]
    else:
        raise NotImplementedError("unsupported tensor shape")

def gen_4d_merge_expand_tensor_view(t: Tensor, tile_size: int = 1):
    if len(t.shape) == 5:
        ret = []
        for e in tensor_slice_cube_hbm(t):
            ret += tensor_slice_cube_hbm(e, tile_size, keep_dim=True)
        return ret
    else:
        raise NotImplementedError("unsupported tensor shape")

def gen_5d_merge_expand_tensor_view(t: Tensor, tile_size: int = 1, slice_dim: int = 0):
    if len(t.shape) == 5:
        if slice_dim == 0:
            return tensor_slice_cube_hbm(t, tile_size, keep_dim=True)
        elif slice_dim == 1:
            ret = []
            for e in tensor_slice_cube_hbm(t, keep_dim=True):
                ret += tensor_slice_cube_hbm(e, tile_size, keep_dim=True, slice_dim=slice_dim)
            return ret
        else:
            raise NotImplementedError("unsupported slice dim")
    else:
        raise NotImplementedError("unsupported tensor shape")
    
def gen_3dtensor_view(t: Tensor, tile_size: int = 1, keep_dim = False, repeat_num = 1):
    if len(t.shape) == 3:
        return tensor_slice_first_dim(t,tile_size, keep_dim=keep_dim, repeat_num=repeat_num)
    else:
        raise NotImplementedError

def gen_3dtensor_view_slice_dim1(t: Tensor, tile_size: int = 1):
    if len(t.shape) == 3:
        res = []
        for e in tensor_slice_first_dim(t):
            res += tensor_slice_first_dim(e,tile_size)
        return res
    else:
        raise NotImplementedError

def prod(seq, start=0) -> int:
    ret = 1
    for x in seq[start:]:
        ret *= x
    return ret

def add_inst_dependency(inst, inst_collect: InstCollector):
    last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
    if last_ins is not None:
        last_ins.release.append(get_inst_type(inst))
        inst.wait.append(get_inst_type(last_ins))

def compute_max_tile_size_by_mm(feature, weight, output):
    m_feature, m_parallel = get_feature_chunk_num_in_bank(feature)
    n_weight, n_parallel = get_weight_chunk_num_in_bank(weight)
    # 约束输出tile
    #
    m_output = get_output_chunk_num_in_bank(output, n_weight)

    m_feature = min(m_feature, m_output)
    
    m_tiling_size = m_feature * hw_info.GlobalBuffer.get_bank_num_in_BGroup()
    n_tiling_size = n_weight * hw_info.WeightBuffer.get_bank_num()
    
    m_tiling_size = min(m_tiling_size, 2**12)
    n_tiling_size = min(n_tiling_size, 2**12)
    return m_tiling_size, n_tiling_size, m_parallel, n_parallel

def compute_max_tile_size_by_rs(feature):
    M, K = feature.shape[-2:]
    # 列完整优先
    m_tiling_size = min(M, 2**14 - 1)
    k = hw_info.GlobalBuffer.get_chunk_num_in_bank(m_tiling_size * get_size_by_type(feature.data_type))
    k_tiling_size = min(k, 2**14 - 1)
    return m_tiling_size, k_tiling_size

def lower_linearw8_transpose(op: LinearW8Transpose, inst_collect: InstCollector):
    feature = op.get_feature
    act_scale = op.get_act_scale
    weight, scale = op.get_weight
    bias = op.get_bias
    output = op.get_output
    head_num = op.head_num
    dim = op.dim

    # expand_feature = hbm_layout_convert(feature, hw_info.GlobalBuffer)
    expand_weight = hbm_layout_convert(weight, hw_info.WeightBuffer)
    # expand_output = hbm_layout_convert(output, hw_info.GlobalBuffer)

    # [M, N] -> m_count x n_count x [m, n]
    M, K = feature.shape[-2:]
    # feature应该适配
    m_feature, m_parallel = get_feature_chunk_num_in_bank(feature)
    #  dim=K
    N, _ = weight.shape
    n_weight, _ = get_weight_chunk_num_in_bank(weight)
    n_parallel = hw_info.MMCore.get_parallel_dim(get_hw_type(weight.data_type)).N
    # 约束输出tile
    #
    m_output = get_output_chunk_num_in_bank(output, n_weight)

    m_feature = min(m_feature, m_output)

    # 每次在m方向取m_tiling_size，即(m_tiling_size, K)的一片数据到GlobalBuffer
    m_tiling_size = m_feature * hw_info.GlobalBuffer.get_bank_num_in_BGroup()
    # 每次在n方向取n_tiling_size，即(n_tiling_size, K)的一片数据到WeightBuffer
    n_tiling_size = n_weight * hw_info.WeightBuffer.get_bank_num()
    
    m_tiling_size = min(m_tiling_size, 2**12)
    n_tiling_size = min(n_tiling_size, 2**12)

    wait_store_flag = True

    for feature2d, act_scale2d, out3d in zip(gen_2dtensor_view(feature), gen_2dtensor_view(act_scale), gen_3dtensor_view_same(output)):
        assert feature2d.addr is not None
        assert act_scale2d.addr is not None
        assert out3d.addr is not None
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
                
                # dependency_num = math.ceil(n_tiling_size / n_parallel)
                # real_n_tiling_size = math.ceil(n_tiling_size / dependency_num)
                # real_n_tiling_size = hw_info.low_align(real_n_tiling_size, n_parallel)
                real_n_tiling_size = dim
                real_dependency_num = math.ceil(n_dim / real_n_tiling_size)
                
                weight_bank_addr = 0
                w_scale_bank_addr = 0
                st_bank_addr = 0
                bias_bank_addr = 0
                for idx, k in enumerate(range(n_start, n_end, real_n_tiling_size)):
                    real_n_start = k
                    real_n_end = min(real_n_start + real_n_tiling_size, n_end)
                    real_n_dim = real_n_end - real_n_start
                    weight_bank_addr = math.ceil(real_n_tiling_size / n_parallel) * idx * hw_info.WeightBuffer.get_addr_len_in_bank(K * get_size_by_type(weight.data_type))
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
                        gen_ld_bias(n_start, n_end - n_start, bias, inst_collect, bias_bank_addr)
                        pass
                    inst_collect[-1].release += [PUType.MM]
                    mm = MMInst()
                    mm.wait += [PUType.LD]
                    mm.input_mode = MMInputMode.w8a8
                    if out3d.data_type == DataType.int8:
                        raise NotImplementedError("int8 output is not supported yet")
                    elif out3d.data_type == DataType.float16:
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
                    
                    H, M, N = out3d.shape
                    head_id = k // real_n_tiling_size
                    assert head_id < H
                    st_hbm_addr = out3d.addr + (head_id * H * N + m_start * N) * get_size_by_type(out3d.data_type)
                    inst_collect[-1].release += [PUType.ST]
                    st_res = STInst()
                    st_res.src_addr = st_bank_addr 
                    st_res.src_bank_id = 0
                    st_res.src_group_id = 1
                    st_res.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                    st_res.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_dim, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(out3d.data_type))
                    st_res.loop_1d = m_end - m_start 
                    st_res.length_1d = real_n_dim * get_size_by_type(out3d.data_type)
                    st_res.dst_1d_stride = (N) * get_size_by_type(out3d.data_type)
                    st_res.hbm_stride = 1
                    st_res.dst_addr = st_hbm_addr
                    st_res.wait += [PUType.MM]
                    inst_collect.add(st_res)
                    if real_dependency_num == idx + 1:
                        wait_store_flag = True

def ld_data(data: Union[Tensor, TensorView], inst_collect: InstCollector, gb = 0, bank_addr = 0, auto_dependency = True):
    ld0 = LDInst()
    if auto_dependency:
        add_inst_dependency(ld0, inst_collect)
    ld0.length_1d = data.shape[-1] * get_size_by_type(data.data_type)
    ld0.loop_1d = prod(data.shape[:-1])
    ld0.src_1d_stride = data.shape[-1] * get_size_by_type(data.data_type)
    assert data.addr is not None
    ld0.src_addr = data.addr
    ld0.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(ld0.length_1d)
    ld0.mode =  LDMode.DDR2global if data.reside_ddr else LDMode.HBM2Global
    ld0.dst_group_id = gb
    ld0.dst_bank_id = 0
    ld0.dst_addr = bank_addr
    inst_collect.add(ld0)

def misc_config(inst_collect: InstCollector):
    return
    misc0 = MISCInst()
    add_inst_dependency(misc0, inst_collect)
    misc0.op = MiscOp.set_k
    misc0.K = 0x2008 #?
    misc0.reg_index = 107 # ?
    inst_collect.add(misc0)

def misc_op(op: MiscOp, inst_collect: InstCollector, batch: int, k: int, in_a_gb: int = 1, in_b_gb:int = 2, out_gb:int=0, in_a_addr = 0, in_b_addr = 0, out_addr = 0, meta_addr=0):
    misc = MISCInst()
    add_inst_dependency(misc, inst_collect)
    misc.op = op
    misc.release.append(PUType.MM)
    misc.input_a_mode = MiscMode.fp16 
    misc.input_b_mode = MiscMode.fp16 
    misc.output_mode = MiscMode.int8 
    misc.in_a_start_addr = in_a_addr
    misc.in_b_start_addr = in_b_addr
    misc.in_a_bank_id = 0
    misc.in_a_bank_group = in_a_gb
    misc.in_b_bank_id = 0
    misc.in_b_bank_group = in_b_gb
    misc.out_bank_id = 0
    misc.out_bank_group = out_gb
    misc.out_start_addr = out_addr
    misc.K = k
    misc.fK = int(np.float32(k).view(np.uint32))
    misc.batch = batch
    misc.meta_addr = meta_addr
    misc.dynamic_scale = True
    inst_collect.add(misc)
    assert misc.batch <= 1024 # for new pvpu.

def lower_div_cvt_matmul(op: DivCvtMatmul, inst_collect: InstCollector):
    feature = op.get_matrix_A
    weight = op.get_matrix_B
    divisor = op.get_divisor
    mm_input_type = op.get_mm_type
    weight_scale = op.get_act_scale_B
    output = op.get_output
    
    assert feature.data_type == DataType.float16
    assert divisor.data_type == DataType.float16
    assert mm_input_type == DataType.int8
    
    m_tiling_size, n_tiling_size, m_parallel, n_parallel = compute_max_tile_size_by_mm(feature, weight, output)
    
    # 3d shape
    expand_weight = hbm_layout_convert_merg_high_dim(weight, hw_info.GlobalBuffer)
    expand_feature = hbm_layout_convert_merg_high_dim(feature, hw_info.GlobalBuffer) 
    expand_divisor = hbm_layout_convert_merg_high_dim(divisor, hw_info.GlobalBuffer) 
    expand_output = hbm_layout_convert_merg_high_dim(output, hw_info.GlobalBuffer)
    expand_weight_scale = hbm_layout_convert_merg_high_dim(weight_scale, hw_info.MetaBuffer)
    
    gb_bank_num = hw_info.GlobalBuffer.get_bank_num_in_BGroup()
    max_merge_high_dim = m_tiling_size // hw_info.high_align(expand_feature.shape[1], m_parallel)#prod(expand_feature.shape[-4:-1])
    max_merge_high_dim = min(1024 // hw_info.high_align(expand_feature.shape[1], m_parallel), max_merge_high_dim)
    if max_merge_high_dim == 0:
        max_merge_high_dim = min(1024, m_tiling_size)
        repeat_num = math.ceil(expand_feature.shape[-2] / max_merge_high_dim)
        iterator = zip(gen_3dtensor_view_slice_dim1(expand_feature, max_merge_high_dim), 
                       gen_3dtensor_view_slice_dim1(expand_divisor, max_merge_high_dim), 
                       gen_3dtensor_view(expand_weight, 1, keep_dim=True, repeat_num=repeat_num), 
                       gen_3dtensor_view(expand_weight_scale, 1, keep_dim=True, repeat_num=repeat_num), 
                       gen_3dtensor_view_slice_dim1(expand_output, max_merge_high_dim))
        # iterator = zip(gen_2d_merge_expand_tensor_view(expand_feature, m_tiling_size), 
        #                gen_2d_merge_expand_tensor_view(expand_divisor, m_tiling_size), 
        #                gen_2d_expand_tensor_view(expand_weight), 
        #                gen_2dtensor_view(weight_act_scale), 
        #                gen_2d_expand_tensor_view(expand_output))
        pass
    else:
        iterator = zip(gen_3dtensor_view(expand_feature, max_merge_high_dim), 
                       gen_3dtensor_view(expand_divisor, max_merge_high_dim), 
                       gen_3dtensor_view(expand_weight, max_merge_high_dim), 
                       gen_3dtensor_view(expand_weight_scale, max_merge_high_dim), 
                       gen_3dtensor_view(expand_output, max_merge_high_dim))
        
    for bidx, (feature3d, divisor3d, weight3d, weight_act_scale3d, out3d) in enumerate(iterator):
        align_m_dim =  hw_info.high_align(feature3d.shape[-2], m_parallel)
        merge_batch_dim = prod(feature3d.shape[:-2]) * align_m_dim
        K = feature3d.shape[-1]
        if feature3d.shape.dim == 3 and feature3d.shape[-2] % m_parallel != 0:
            act_bank_addr = 0
            for idx, (feature2d, divisor2d) in enumerate(zip(gen_2dtensor_view(feature3d), gen_2dtensor_view(divisor3d))):
                ld_data(feature2d, inst_collect, gb=1, bank_addr=act_bank_addr, auto_dependency=idx==0)
                ld_data(divisor2d, inst_collect, gb=2, bank_addr=act_bank_addr, auto_dependency=False)
                act_bank_addr += math.ceil(feature2d.shape[-2] / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(K*get_size_by_type(feature2d.data_type))
        else:
            ld_data(feature3d, inst_collect, gb=1)
            ld_data(divisor3d, inst_collect, gb=2)
        
        misc_config(inst_collect)
        
        misc_op(
            MiscOp.elt_div,
            inst_collect,
            merge_batch_dim,
            K,
        )
        wait_misc_flag = True
        
        m_dim = feature3d.shape[-2]
        wait_store_flag = False
        act_bank_addr = 0
        act_scale_bank_addr = 0
        for idx, (weight2d, weight_act_scale2d, out2d) in enumerate(zip(
            gen_2dtensor_view(weight3d),
            gen_2dtensor_view(weight_act_scale3d), 
            gen_2dtensor_view(out3d)
            )):
            N, _ = weight2d.shape
            for j in range(0, N, n_tiling_size):
                n_start = j
                n_end: int = min(n_start + n_tiling_size, N)
                n_dim = n_end - n_start
                
                real_n = min(n_dim, N)
                real_n = hw_info.high_align(real_n // 4, n_parallel)
                dependency_num = math.ceil(n_tiling_size / real_n)
                real_n_tiling_size = math.ceil(n_tiling_size / dependency_num)
                real_n_tiling_size = hw_info.high_align(real_n_tiling_size, real_n)
                real_dependency_num = math.ceil(n_dim / real_n_tiling_size)
                
                weight_bank_start_addr = 0
                w_scale_bank_addr = 0
                st_bank_addr = 0
                for idx, k in enumerate(range(n_start, n_end, real_n_tiling_size)):
                    real_n_start = k
                    real_n_end = min(real_n_start + real_n_tiling_size, n_end)
                    real_n_dim = real_n_end - real_n_start
                    weight_bank_start_addr = int(real_n_tiling_size / n_parallel) * idx * hw_info.WeightBuffer.get_addr_len_in_bank(K * get_size_by_type(weight.data_type))
                    
                    gen_ld_act_to_weight(real_n_start, real_n_dim, weight2d, inst_collect, wait_store_flag, weight_bank_start_addr)
                    if wait_store_flag:
                        wait_store_flag = False
                    w_scale_bank_addr = idx * hw_info.MetaBuffer.get_addr_len_in_bank(real_n_tiling_size * get_size_by_type(weight_act_scale2d.data_type))
                    gen_ld_weight_scale(real_n_start, real_n_end - real_n_start, weight_act_scale2d, inst_collect, w_scale_bank_addr)
                    
                    st_bank_addr = idx * math.ceil(m_dim / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_tiling_size, n_parallel) * get_size_by_type(output.data_type))
                    mm = MMInst()
                    add_inst_dependency(mm, inst_collect)
                    if wait_misc_flag:
                        mm.wait += [PUType.MISC]
                        wait_misc_flag = False
                    mm.input_mode = MMInputMode.w8a8
                    mm.output_mode = MMOutputMode.fp16
                    mm.act_start_addr = act_bank_addr
                    mm.act_bank_group_id = 0
                    mm.out_bank_group_id = 1
                    mm.bias_flag = False
                    mm.output_flag = True
                    mm.weights_start_addr = weight_bank_start_addr
                    mm.bias_start_addr = 0
                    mm.out_start_addr = st_bank_addr
                    mm.M = m_dim -1
                    mm.N = real_n_dim -1
                    mm.K = K -1
                    mm.act_scale_start_addr = act_scale_bank_addr
                    mm.out_scale_start_addr = 0
                    mm.weights_scale_start_addr = w_scale_bank_addr
                    inst_collect.add(mm)
                    
                    st_res = STInst()
                    add_inst_dependency(st_res, inst_collect)
                    st_res.src_addr =  st_bank_addr
                    st_res.src_bank_id = 0
                    st_res.src_group_id = 1
                    st_res.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                    st_res.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_dim, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(out2d.data_type))
                    st_res.loop_1d = m_dim 
                    st_res.length_1d = real_n_dim * get_size_by_type(out2d.data_type)
                    st_res.dst_1d_stride = (N) * get_size_by_type(out2d.data_type)
                    st_res.dst_addr = out2d.addr + real_n_start * get_size_by_type(out2d.data_type)
                    inst_collect.add(st_res)
                    
                    if real_dependency_num == idx + 1:
                        wait_store_flag = True
                
            act_bank_addr += math.ceil(feature3d.shape[-2] / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(K * get_size_by_type(mm_input_type))
            act_scale_bank_addr += hw_info.MetaBuffer.get_addr_len_in_bank(align_m_dim * get_size_by_type(weight_act_scale2d.data_type))

def lower_unary_cvt_matmul(op: DivCvtMatmul, inst_collect: InstCollector):
    feature = op.get_matrix_A
    weight = op.get_matrix_B
    mm_input_type = op.get_mm_type
    weight_scale = op.get_act_scale_B
    output = op.get_output
    
    assert feature.data_type == DataType.float16
    assert mm_input_type == DataType.int8
    
    m_tiling_size, n_tiling_size, m_parallel, n_parallel = compute_max_tile_size_by_mm(feature, weight, output)
    
    
    # 3d shape
    expand_weight = hbm_layout_convert_merg_high_dim(weight, hw_info.GlobalBuffer)
    expand_feature = hbm_layout_convert_merg_high_dim(feature, hw_info.GlobalBuffer) 
    expand_output = hbm_layout_convert_merg_high_dim(output, hw_info.GlobalBuffer)
    expand_weight_scale = hbm_layout_convert_merg_high_dim(weight_scale, hw_info.MetaBuffer)
    
    gb_bank_num = hw_info.GlobalBuffer.get_bank_num_in_BGroup()
    max_merge_high_dim = m_tiling_size // hw_info.high_align(expand_feature.shape[1], m_parallel)#prod(expand_feature.shape[-4:-1])
    max_merge_high_dim = min(1024 // hw_info.high_align(expand_feature.shape[1], m_parallel), max_merge_high_dim)
    if max_merge_high_dim == 0:
        max_merge_high_dim = min(1024, m_tiling_size)
        repeat_num = math.ceil(expand_feature.shape[-2] / max_merge_high_dim)
        iterator = zip(gen_3dtensor_view_slice_dim1(expand_feature, max_merge_high_dim), 
                       gen_3dtensor_view(expand_weight, 1, keep_dim=True, repeat_num=repeat_num), 
                       gen_3dtensor_view(expand_weight_scale, 1, keep_dim=True, repeat_num=repeat_num), 
                       gen_3dtensor_view_slice_dim1(expand_output, max_merge_high_dim))
    else:
        iterator = zip(gen_3dtensor_view(expand_feature, max_merge_high_dim), 
                       gen_3dtensor_view(expand_weight, max_merge_high_dim), 
                       gen_3dtensor_view(expand_weight_scale, max_merge_high_dim), 
                       gen_3dtensor_view(expand_output, max_merge_high_dim))
    for feature3d, weight3d, weight_act_scale3d, out3d in iterator:
        align_m_dim =  hw_info.high_align(feature3d.shape[-2], m_parallel)
        merge_batch_dim = prod(feature3d.shape[:-2]) * align_m_dim
        K = feature3d.shape[-1]
        
        if feature3d.shape.dim == 3 and feature3d.shape[-2] % m_parallel != 0:
            act_bank_addr = 0
            for bidx, feature2d in enumerate(gen_2dtensor_view(feature3d)):
                ld_data(feature2d, inst_collect, gb=1, bank_addr=act_bank_addr, auto_dependency=bidx==0)
                act_bank_addr += math.ceil(feature2d.shape[-2] / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(K*get_size_by_type(feature2d.data_type))
        else:
            ld_data(feature3d, inst_collect, gb=1)
        
        misc_config(inst_collect)
        
        misc_op(MiscOp.softmax, inst_collect, merge_batch_dim, K)
        wait_misc_flag = True
        wait_store_flag = False
        
        m_dim = feature3d.shape[-2]
        act_bank_addr = 0
        act_scale_bank_addr = 0
        for idx, (weight2d, weight_act_scale2d, out2d) in enumerate(zip(
            gen_2dtensor_view(weight3d),
            gen_2dtensor_view(weight_act_scale3d), 
            gen_2dtensor_view(out3d)
            )):
            N, _ = weight2d.shape
            for j in range(0, N, n_tiling_size):
                n_start = j
                n_end: int = min(n_start + n_tiling_size, N)
                n_dim = n_end - n_start
                
                real_n = min(n_dim, N)
                real_n = hw_info.high_align(real_n // 4, n_parallel)
                dependency_num = math.ceil(n_tiling_size / real_n)
                real_n_tiling_size = math.ceil(n_tiling_size / dependency_num)
                real_n_tiling_size = hw_info.high_align(real_n_tiling_size, real_n)
                real_dependency_num = math.ceil(n_dim / real_n_tiling_size)
                
                weight_bank_start_addr = 0
                w_scale_bank_addr = 0
                st_bank_addr = 0
                for idx, k in enumerate(range(n_start, n_end, real_n_tiling_size)):
                    real_n_start = k
                    real_n_end = min(real_n_start + real_n_tiling_size, n_end)
                    real_n_dim = real_n_end - real_n_start
                    weight_bank_start_addr = int(real_n_tiling_size / n_parallel) * idx * hw_info.WeightBuffer.get_addr_len_in_bank(K * get_size_by_type(weight.data_type))
                    
                    gen_ld_act_to_weight(real_n_start, real_n_end - real_n_start, weight2d, inst_collect, wait_store_flag, weight_bank_start_addr)
                    if wait_store_flag:
                        wait_store_flag = False
                    w_scale_bank_addr = idx * hw_info.MetaBuffer.get_addr_len_in_bank(real_n_tiling_size * get_size_by_type(weight_act_scale2d.data_type))
                    gen_ld_weight_scale(real_n_start, real_n_end - real_n_start, weight_act_scale2d, inst_collect, w_scale_bank_addr)
                    
                    st_bank_addr = idx * math.ceil(m_dim / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_tiling_size, n_parallel) * get_size_by_type(output.data_type))
                    mm = MMInst()
                    if wait_misc_flag:
                        mm.wait.append(PUType.MISC)
                        wait_misc_flag = False
                    add_inst_dependency(mm, inst_collect)
                    mm.input_mode = MMInputMode.w8a8
                    mm.output_mode = MMOutputMode.fp16
                    mm.act_start_addr = act_bank_addr
                    mm.act_bank_group_id = 0
                    mm.out_bank_group_id = 1
                    mm.bias_flag = False
                    mm.output_flag = True
                    mm.weights_start_addr = weight_bank_start_addr
                    mm.bias_start_addr = 0
                    mm.out_start_addr = st_bank_addr
                    mm.M = m_dim -1
                    mm.N = real_n_dim -1
                    mm.K = K -1
                    mm.act_scale_start_addr = act_scale_bank_addr
                    mm.out_scale_start_addr = 0
                    mm.weights_scale_start_addr = w_scale_bank_addr
                    inst_collect.add(mm)
                    
                    st_res = STInst()
                    add_inst_dependency(st_res, inst_collect)
                    st_res.src_addr =  st_bank_addr
                    st_res.src_bank_id = 0
                    st_res.src_group_id = 1
                    st_res.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                    st_res.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_dim, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(out2d.data_type))
                    st_res.loop_1d = m_dim 
                    st_res.length_1d = real_n_dim * get_size_by_type(out2d.data_type)
                    st_res.dst_1d_stride = (N) * get_size_by_type(out2d.data_type)
                    st_res.hbm_stride = 1
                    st_res.dst_addr = out2d.addr + real_n_start * get_size_by_type(out2d.data_type)
                    inst_collect.add(st_res)
                    
                    if real_dependency_num == idx + 1:
                        wait_store_flag = True
            act_bank_addr += math.ceil(feature3d.shape[-2] / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(K * get_size_by_type(mm_input_type))
            act_scale_bank_addr += hw_info.MetaBuffer.get_addr_len_in_bank(align_m_dim * get_size_by_type(weight_act_scale2d.data_type))

def lower_cvt_linearw8(op: CvtLinearW8, inst_collect: InstCollector):
    feature = op.get_feature
    weight, scale = op.get_weight
    bias = op.get_bias
    output = op.get_output
    assert feature.data_type == DataType.float16
    mm_in_type = DataType.int8

    m_tiling_size, n_tiling_size, m_parallel, n_parallel = compute_max_tile_size_by_mm(feature, weight, output)
    
    # 3d shape
    expand_feature = hbm_layout_convert_merg_high_dim(feature, hw_info.GlobalBuffer) 
    expand_output = hbm_layout_convert_merg_high_dim(output, hw_info.GlobalBuffer)
    
    # 4d
    expand_weight = hbm_layout_convert(weight, hw_info.WeightBuffer)
    
    

    max_merge_high_dim = m_tiling_size // hw_info.high_align(expand_feature.shape[-2], m_parallel)#prod(expand_feature.shape[-4:-1])
    max_merge_high_dim = min(1024 // hw_info.high_align(expand_feature.shape[1], m_parallel), max_merge_high_dim)
    if max_merge_high_dim == 0:
        max_merge_high_dim = min(1024, m_tiling_size)
        # 注意，实际返回是2d tensor，当作3d tensor使用
        iterator = zip(gen_3dtensor_view_slice_dim1(expand_feature, max_merge_high_dim), 
                       gen_3dtensor_view_slice_dim1(expand_output, max_merge_high_dim))
    else:
        iterator = zip(gen_3dtensor_view(expand_feature, max_merge_high_dim), 
                       gen_3dtensor_view(expand_output, max_merge_high_dim))
    
    wait_mm_flag = False    
    for feature3d, out3d in iterator:
        assert feature3d.addr is not None
        assert out3d.addr is not None
        
        # merge_batch_dim = prod(feature3d.shape[:-2]) * hw_info.high_align(feature3d.shape[-2], m_parallel)
        merge_batch_dim = prod(feature3d.shape[:-1])
        K = feature3d.shape[-1]
        
        # if feature3d.shape.dim == 3 and feature3d.shape[-2] % m_parallel != 0:
        #     act_bank_addr = 0
        #     for bidx, feature2d in enumerate(gen_2dtensor_view(feature3d)):
        #         ld_data(feature2d, inst_collect, gb=1, bank_addr=act_bank_addr, auto_dependency=bidx==0)
        #         act_bank_addr += math.ceil(feature2d.shape[-2] / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(K*get_size_by_type(feature2d.data_type))
        # else:
        ld_data(feature3d, inst_collect, gb=1)
        
        misc_config(inst_collect)
        
        misc_op(MiscOp.abs_max, inst_collect, merge_batch_dim, K)
        
        m_dim = merge_batch_dim#prod(feature3d.shape[:-1])
        # for i in range(0, M, m_tiling_size):
        #     m_start = i
        #     m_end: int = min(m_start + m_tiling_size, M)
        #     gen_ld_feature(m_start, m_end - m_start, feature3d, inst_collect, wait_store_flag)
        #     if wait_store_flag:
        #         wait_store_flag = False
        #     gen_ld_act_scale(m_start, m_end - m_start, act_scale3d, inst_collect)
        N = weight.shape[-2]
        wait_store_flag = False
        wait_misc_flag = True
        for j in range(0, N, n_tiling_size):
            n_start = j
            n_end: int = min(n_start + n_tiling_size, N)
            n_dim = n_end - n_start
            
            real_n = min(n_dim, N)
            real_n = hw_info.high_align(real_n // 4, n_parallel)
            dependency_num = math.ceil(n_tiling_size / real_n)
            real_n_tiling_size = math.ceil(n_tiling_size / dependency_num)
            real_n_tiling_size = hw_info.high_align(real_n_tiling_size, real_n)
            real_dependency_num = math.ceil(n_dim / real_n_tiling_size)
            
            weight_bank_start_addr = 0
            w_scale_bank_addr = 0
            st_bank_addr = 0
            for idx, k in enumerate(range(n_start, n_end, real_n_tiling_size)):
                real_n_start = k
                real_n_end = min(real_n_start + real_n_tiling_size, n_end)
                real_n_dim = real_n_end - real_n_start
                weight_bank_start_addr = int(real_n_tiling_size / n_parallel) * idx * hw_info.WeightBuffer.get_addr_len_in_bank(K * get_size_by_type(weight.data_type))
                    
                gen_ld_weight(real_n_start, real_n_dim, expand_weight, inst_collect, wait_store_flag, weight_bank_start_addr)
                if wait_store_flag:
                    wait_store_flag = False
                w_scale_bank_addr = idx * hw_info.MetaBuffer.get_addr_len_in_bank(real_n_tiling_size * get_size_by_type(scale.data_type))
                gen_ld_weight_scale(real_n_start, real_n_dim, scale, inst_collect, w_scale_bank_addr)
                # gen_ld_out_scale
                if op.bias_flag:
                    assert bias is not None
                    bias_bank_addr = idx * hw_info.MetaBuffer.get_addr_len_in_bank(real_n_tiling_size * get_size_by_type(bias.data_type))
                    gen_ld_bias(real_n_start, real_n_dim, bias, inst_collect, bias_bank_addr)
                
                st_bank_addr = idx * math.ceil(m_dim / m_parallel) * hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_tiling_size, n_parallel) * get_size_by_type(output.data_type))
                mm = MMInst()
                if wait_misc_flag:
                    mm.wait.append(PUType.MISC)
                    wait_misc_flag = False
                add_inst_dependency(mm, inst_collect)
                mm.input_mode = MMInputMode.w8a8
                if out3d.data_type == DataType.int8:
                    raise NotImplementedError("int8 output is not supported yet")
                elif out3d.data_type == DataType.float16:
                    mm.output_mode = MMOutputMode.fp16
                mm.act_start_addr = 0
                mm.act_bank_group_id = 0
                mm.out_bank_group_id = 1
                mm.bias_flag = op.bias_flag
                mm.output_flag = True
                mm.weights_start_addr = weight_bank_start_addr
                mm.bias_start_addr = bias_bank_addr
                mm.out_start_addr = st_bank_addr
                mm.M = m_dim -1
                mm.N = real_n_dim -1 
                mm.K = K -1
                mm.act_scale_start_addr = 0
                mm.out_scale_start_addr = 0
                mm.weights_scale_start_addr = w_scale_bank_addr
                inst_collect.add(mm)
                
                st_res = STInst()
                add_inst_dependency(st_res, inst_collect)
                st_res.src_addr =  st_bank_addr
                st_res.src_bank_id = 0
                st_res.src_group_id = 1
                st_res.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                st_res.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_dim, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(out3d.data_type))
                st_res.loop_1d = m_dim 
                st_res.length_1d = real_n_dim * get_size_by_type(out3d.data_type)
                st_res.dst_1d_stride = (N) * get_size_by_type(out3d.data_type)
                st_res.dst_addr = out3d.addr + real_n_start * get_size_by_type(out3d.data_type)
                inst_collect.add(st_res)
                
                if real_dependency_num == idx + 1:
                    wait_store_flag = True
            
def lower_transpose_cvt(op: TransposeCvt, inst_collect: InstCollector):
    assert len(op.get_inputs()) == 1
    input = op.get_inputs()[0]
    output = op.get_outputs()[0]
    assert input.shape[-1] == output.shape[-2]
    assert input.shape[-2] == output.shape[-1]

    in_stride2 = math.prod(input.shape[2:])
    in_stride3 = input.shape[-1]
    
    output_stride2 = math.prod(output.shape[2:])
    output_stride3 = output.shape[-1]
    scale_addrs = [0, 256]
    addrs= [0, 2048]
    half_size = (hw_info.GlobalBuffer.get_bank_bytes_num() //2) // get_size_by_type(input.data_type)
    line_size = half_size // input.shape[-2]
    if line_size > 1024 :
        line_size = 1024
    ktiles = math.ceil(input.shape[-1] / line_size)
    N = math.prod(input.shape[:-2])
    max_index = N * ktiles
    index = 0
    for i in range(N):
        for k in range(ktiles):
            ks = get_tile_dim(input.shape[-1], line_size, k, ktiles)
            ld = LDInst()
            if(index == 0):
                last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
                if last_ins is not None:
                    last_ins.release.append(PUType.LD)
                    ld.wait.append(get_inst_type(last_ins))
            elif index >= 2:
                ld.wait.append(PUType.ST)
            ld.release.append(PUType.RS)
            ld.length_1d = ks * get_size_by_type(input.data_type)
            ld.loop_1d = input.shape[-2]
            ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
            ld.src_addr =input.addr + ( i*in_stride2  + k * line_size )* get_size_by_type(input.data_type)
            ld.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(ks, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(input.data_type))
            ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
            ld.dst_addr = addrs[index % 2]
            ld.dst_group_id = 0
            inst_collect.add(ld)
            assert input.data_type == DataType.float16
            rs = RSInst()
            rs.wait.append(PUType.LD)
            rs.release.append(PUType.MISC)
            rs.src_addr = addrs[index % 2]
            rs.dst_addr = addrs[index % 2]
            rs.src_bank_group_id = 0
            rs.dst_bank_group_id = 1
            rs.M = input.shape[-2]
            rs.K = ks
            rs.data_type = RSDataType.int16
            inst_collect.add(rs)
            misc = MISCInst()
            misc.wait.append(PUType.RS)
            misc.release.append(PUType.ST)
            misc.op = MiscOp.abs_max
            misc.dynamic_scale = True
            misc.input_a_mode = MiscMode.fp16  
            misc.output_mode = MiscMode.int8 
            misc.in_a_bank_group = 1
            misc.in_a_start_addr = addrs[index %2]
            misc.out_bank_group = 2
            misc.out_start_addr = addrs[index %2]
            misc.meta_addr = scale_addrs[index %2]
            misc.K = input.shape[-2]
            misc.fK = int(np.float32(input.shape[-2]).view(np.uint32))
            misc.batch = ks
            inst_collect.add(misc)
            st = STInst()
            st.wait.append(PUType.MISC)
            st.length_1d = input.shape[-2] * get_size_by_type(output.data_type)
            st.loop_1d = ks
            st.dst_1d_stride = input.shape[-2] * get_size_by_type(output.data_type)
            st.src_group_id = 2
            st.src_addr = addrs[index %2]
            st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
            st.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(input.shape[-2], hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(output.data_type))
            st.dst_addr = output.addr + (i * output_stride2 + k *line_size*output_stride3) * get_size_by_type(output.data_type)
            inst_collect.add(st)
            out_dynamic_scale = op.get_outputs()[1]
            st_dynamic_scale = STInst()
            if index < max_index-2:
                st_dynamic_scale.release.append(PUType.LD)
            st_dynamic_scale.mode = STMode.Meta2DDR
            st_dynamic_scale.src_addr = scale_addrs[index %2]
            st_dynamic_scale.length_1d = ks * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.loop_1d = 1
            st_dynamic_scale.dst_1d_stride = out_dynamic_scale.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(ks, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(out_dynamic_scale.data_type))
            st_dynamic_scale.dst_addr = out_dynamic_scale.addr + (i * out_dynamic_scale.shape[-2] + k*line_size)* get_size_by_type(out_dynamic_scale.data_type)
            inst_collect.add(st_dynamic_scale)
            index += 1
    return
    
    
        
          
def lower_Linearw8_act(op: LinearW8Act, inst_collect: InstCollector, dynamic_scale:bool = False):
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
    
    m_tiling_size = min(m_tiling_size, 2**10)
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
                
                # dependency_num = math.ceil(n_tiling_size / n_parallel)
                # real_n_tiling_size = math.ceil(n_tiling_size / dependency_num)
                # real_n_tiling_size = hw_info.low_align(real_n_tiling_size, n_parallel)
                # real_dependency_num = math.ceil(n_dim / real_n_tiling_size)
                
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
                        # gen_ld_bias(n_start, n_end - n_start, bias, inst_collect, bias_bank_addr)
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
                    mm.release.append(PUType.MISC)
                    inst_collect.add(mm)
                    
                    # #  activation
                    # if dynamic_scale:
                    #     misc0 = MISCInst()
                    #     misc0.op = MiscOp.set_k
                    #     misc0.wait.append(PUType.MM)
                    #     misc0.K = 0x2000
                    #     misc0.reg_index = 107
                    #     inst_collect.add(misc0)
                    misc = MISCInst()
                    last_ins = inst_collect.get_insts()[-1]
                    if not isinstance(last_ins,MISCInst):
                        misc.wait.append(PUType.MM)
                    misc.dynamic_scale = dynamic_scale
                    if op.act_type == "silu":
                        misc.op = MiscOp.silu
                    elif op.act_type == "gelu":
                        misc.op = MiscOp.gelu1
                    misc.input_a_mode = MiscMode.fp16
                    misc.output_mode = MiscMode.fp16
                    misc.in_a_bank_group = 1
                    misc.in_a_bank_id = 0
                    misc.in_a_start_addr = st_bank_addr
                    misc.out_bank_group = 2
                    misc.out_bank_id = 0
                    misc.out_start_addr = st_bank_addr
                    misc.meta_addr = 0
                    misc.K = real_n_end - real_n_start
                    misc.fK = int(np.float32(misc.K).view(np.uint32))
                    misc.batch = m_end - m_start
                    misc.release.append(PUType.ST)
                    inst_collect.add(misc)
                    assert misc.batch <= 1024 # for new pvpu.
                    
                    # import pdb
                    # pdb.set_trace()
                    st_res = STInst()
                    st_res.src_addr =  st_bank_addr
                    st_res.src_bank_id = 0
                    st_res.src_group_id = 2
                    st_res.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                    st_res.bank_addr_stride = hw_info.GlobalBuffer.get_addr_len_in_bank(hw_info.high_align(real_n_dim, hw_info.WeightBuffer.get_bank_num()) * get_size_by_type(output.data_type))
                    st_res.loop_1d = m_end - m_start #min(bank_num, m_end - i)
                    st_res.length_1d = real_n_dim * get_size_by_type(output.data_type)
                    st_res.dst_1d_stride = out2d.shape[-1] * get_size_by_type(output.data_type)
                    st_res.dst_addr = out2d.addr + (m_start * out2d.shape[-1] + real_n_start) * get_size_by_type(output.data_type) #(m_start * N + n_start) * get_size_by_type(output.data_type) #(i * N + n_start) * get_size_by_type(output.data_type)
                    st_res.wait += [PUType.MISC]
                    inst_collect.add(st_res)
                    
                    # if dynamic_scale:
                    #     out_dynamic_scale = op.get_outputs()[1]
                    #     st_dynamic_scale = STInst()
                    #     st_dynamic_scale.mode = STMode.Meta2HBM
                    #     st_dynamic_scale.length_1d = (m_end - m_start) * get_size_by_type(out_dynamic_scale.data_type)
                    #     st_dynamic_scale.loop_1d = 1
                    #     st_dynamic_scale.src_addr = addrs[index %3]
                    #     st_dynamic_scale.dst_1d_stride = out_dynamic_scale.shape[-1] * get_size_by_type(out_dynamic_scale.data_type)
                    #     st_dynamic_scale.bank_addr_stride = get_bank_stride(st_dynamic_scale.length_1d)
                    #     st_dynamic_scale.dst_addr = out_dynamic_scale.addr + index * st_dynamic_scale.length_1d
                    #     inst_collect.add(st_dynamic_scale)
                    if real_dependency_num == idx + 1:
                        wait_store_flag = True
