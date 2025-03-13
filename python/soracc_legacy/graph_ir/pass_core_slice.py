from .graph_ir import *
from typing import Optional, Sequence
import math
import copy

# 在这种条件下，该类op无法slice
class NoSliceOp(Exception):
    pass

def get_op_config(op: Op):
    config = copy.copy(op.__dict__)
    config.pop('_prev')
    config.pop('_next')
    config.pop('_inputs')
    config.pop('_outputs')
    config.pop('_weights')
    config.pop('act_scale_flag')
    config.pop('opt_func')
    return config

def op_clone(op):
    clone = type(op)(**get_op_config(op))
    in_clone = []
    for in_tensor in op.get_inputs():
        in_clone.append(copy.copy(in_tensor))
    clone.set_inputs(in_clone)
    out_clone = []
    for out_tensor in op.get_outputs():
        out_clone.append(copy.copy(out_tensor))
    clone.set_outputs(out_clone)
    # update dynamic scale 
    if isinstance(op, (Gelu, Silu, RMSnorm, Layernorm, Convert, Softmax, RoPE, Div, Eltwise, LinearW8Act)):
        if op.act_scale_flag:
            clone.act_scale_flag = True
            clone.get_output.set_act_scale(clone.get_act_scale)
    if isinstance(op, (DivCvtMatmul, SoftmaxCvtMatmul)):
        # clone.get_matrix_A.set_act_scale(clone.get_inputs()[-2])
        clone.get_matrix_B.set_act_scale(clone.get_inputs()[-1])
    elif isinstance(op, Matmul) and op.get_matrix_A.data_type == op.get_matrix_B.data_type == DataType.int8:
        clone.get_matrix_A.set_act_scale(clone.get_inputs()[-2])
        clone.get_matrix_B.set_act_scale(clone.get_inputs()[-1])
            
    raw_weight = {}
    cloned_weights = op.get_raw_weights()
    for k, v in cloned_weights.items():
        raw_weight[k] = copy.copy(v)
    clone.set_raw_weights(raw_weight)
    return clone 

def core_slice_tensor(g: StaticGraph, total_num: int) -> list[list[Op]]:
    ret = []
    for i in range(total_num):
        cur_slice = []
        for op in g.get_ops():
            cur_slice.append(op_clone(op))
        ret.append(cur_slice)

    for i in range(total_num):
        core_id = i
        for j, op in enumerate(ret[i]):
            if isinstance(op, LinearW8):
                _op = core_slice_linear(op, core_id, total_num)
            elif isinstance(op, DivCvtMatmul):
                _op = core_slice_div_cvt_matmul(op, core_id, total_num)
            elif isinstance(op, Matmul):
                _op = core_slice_matmul(op, core_id, total_num)
            elif isinstance(op, Softmax):
                _op = core_slice_softmax(op, core_id, total_num)
            elif isinstance(op, (Eltwise, Div)):
                _op = core_slice_eltwise(op, core_id, total_num)
            elif isinstance(op, (Silu, Gelu,Copy)):
                _op = core_slice_activation(op, core_id, total_num)
            elif isinstance(op, (Layernorm, RMSnorm)):
                _op = core_slice_normalization(op, core_id, total_num)
            elif isinstance(op, RoPE):
                _op = core_slice_rope(op, core_id, total_num)
            elif isinstance(op, Convert):
                _op = core_slice_convert(op, core_id, total_num)
            elif isinstance(op, Transpose):
                if can_slice_by_high_dim(op):
                    _op = core_slice_transpose(op, core_id, total_num)
                else:
                    _op = Nop(op.name) if core_id != 0 else op
            elif isinstance(op, (View, Split)):
                _op = Nop(op.name) if core_id != 0 else op
            elif isinstance(op, LoadInst):
                _op = op
            else:
                raise NotImplementedError(f"core slice not implemented for {type(op)}")
            ret[i][j] = _op
    for i in range(total_num):
        graph_len = len(ret[i])
        idx = 0
        visted_op_num = 0
        while visted_op_num < graph_len:
            op = ret[i][idx]
            ret[i].insert(idx+1, Sync(f"sync_{op.name}"))
            idx += 2
            visted_op_num += 1
    return ret

def n_part_division(org: int, n: int):
    part = math.ceil(org / n)
    ret = []
    for i in range(n):
        ret.append(min(org, part))
        org -= min(org, part)
    return list(reversed(ret))

def n_part_division_matmul(org: int, n: int):
    if n == 0:
        return []
    
    # 计算保证所有元素非零的最小需求
    min_non_zero = n * (n + 1) // 2  # 首项1公差1的等差数列和
    
    if org >= min_non_zero:
        # 基础等差数列生成
        a = (org - min_non_zero) // n
        remainder = (org - min_non_zero) % n
        sequence = [1 + a + i for i in range(n)]
        # 余数均匀分配到后部元素
        for i in range(n-remainder, n):
            sequence[i] += 1
    else:
        # 当资源不足时，生成最小递增序列
        sequence = []
        avg = org // n
        remainder = org % n
        
        # 前n-remainder项为avg，后remainder项为avg+1
        sequence = [avg]*(n-remainder) + [avg+1]*remainder
        
        # 保证严格递增（当可能时）
        if avg == 0 and remainder > 0:
            # 处理含0的特殊情况
            sequence = [0]*(n-remainder-1) + [1]*remainder + [org - (remainder + n-remainder-1)]
            sequence = sorted(sequence)
    
    return sequence

def n_part_division_linear(org: int, n: int):
    if n == 0:
        return []
    
    min_non_zero = n * (n + 1) // 2
    
    if org >= min_non_zero:
        a = (org - min_non_zero) // n
        remainder = (org - min_non_zero) % n
        # 生成逆序等差数列（从大到小）
        sequence = [1 + a + (n-1 - i) for i in range(n)]
        # 余数分配到前部元素
        for i in range(remainder):
            sequence[i] += 1
    else:
        base = org // n
        remainder = org % n
        # 生成逆序序列并优先填充后方元素
        sequence = [base+1]*remainder + [base]*(n-remainder)
        # 降序排列并优化尾部零
        sequence.sort(reverse=True)
        
        # 优化尾部零：将尾部零转换为最小非零值
        zero_count = sequence.count(0)
        if zero_count > 0:
            non_zero_sum = org - zero_count
            # 重新分配非零部分
            new_base = non_zero_sum // (n - zero_count)
            new_remainder = non_zero_sum % (n - zero_count)
            sequence = [new_base+1]*new_remainder + [new_base]*(n-zero_count-new_remainder) + [0]*zero_count
            sequence.sort(reverse=True)
    
    return sequence

def prod(it: Sequence[int]):
    ret = 1
    for n in it:
        ret *= n
    return ret

def first_none_trivial_dim(s: Shape):
    for i, x in enumerate(s):
        if x > 1:
            return i, x
    assert False, 'not a valid shape'

def first_none_trivial_dim_not_last(s: Shape, last_limit: int = 1):
    last_limit = min(s.dim - 1, last_limit)
    dim_idx = s.dim - 1
    for i, x in enumerate(s):
        if x > 1:
            dim_idx = i
            break
    if dim_idx >= s.dim - last_limit:
        dim_idx = s.dim - last_limit - 1
    return dim_idx, s[dim_idx]

def refine_tesnor(x: Tensor, parition: list[int], cur_id: int, refine_dim: int = 0):
    offset = sum(parition[0:cur_id])
    x.addr = x.addr + offset * prod(x.shape[refine_dim + 1:]) * get_size_by_type(x.data_type)
    x.shape = Shape(*x.shape[:refine_dim], parition[cur_id], *x.shape[refine_dim + 1:])

def can_slice_by_high_dim(op: Op) -> bool:
    if isinstance(op, Transpose):
        dimidx, dim = first_none_trivial_dim_not_last(op.get_input.shape)
        dima, dimb = op.get_trans_dims
        dima = dima if dima >= 0 else dima + op.get_input.shape.dim
        dimb = dimb if dimb >= 0 else dimb + op.get_input.shape.dim
        return True if dima > dimidx and dimb > dimidx else False
    else:
        raise NotImplementedError(f"can_slice_by_high_dim not implemented for {type(op)}")
            
# linear只对weight做切分
# cur_id: 0, 1, 2
def discrate_weights(x: Tensor, cur_id:int):
    x.addr += 8*cur_id*1073741824
def core_slice_linear(op: LinearW8, cur_id: int, total_num: int) -> Op:
    feature = op.get_feature
    weight, scale = op.get_weight
    output = op.get_output

    # divide by n
    # parts = n_part_division(weight.shape[0], total_num)
    
    # weight.addr = weight.addr + offset * weight.shape[-1] * get_size_by_type(weight.data_type)
    # weight.shape = Shape(parts[cur_id], weight.shape[-1])
    # scale.addr = scale.addr + offset * get_size_by_type(scale.data_type)
    # scale.shape = Shape(parts[cur_id])
    
    # divide by batch
    idx, dim = first_none_trivial_dim_not_last(feature.shape)
    parts = n_part_division(dim, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    
    refine_tesnor(feature, parts, cur_id, idx)
    if feature.data_type == DataType.int8:
        act_scale = op.get_act_scale
        refine_tesnor(act_scale, parts, cur_id, idx)

    refine_tesnor(output, parts, cur_id, idx)
    discrate_weights(weight,cur_id)
    
    return op

def core_slice_matmul(op: Matmul, cur_id: int, total_num: int):
    matrix_a = op.get_matrix_A
    matrix_b = op.get_matrix_B
    res = op.get_output
    assert matrix_a.shape.dim == matrix_b.shape.dim
    idx, dim = first_none_trivial_dim_not_last(matrix_a.shape)

    parts = n_part_division(dim, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    refine_tesnor(matrix_a, parts, cur_id, idx)
    if matrix_a.data_type == DataType.int8:
        refine_tesnor(matrix_a.get_act_scale, parts, cur_id, idx)
    if idx < matrix_b.shape.dim - 2:
        refine_tesnor(matrix_b, parts, cur_id, idx)
        if matrix_b.data_type == DataType.int8:
            refine_tesnor(matrix_b.get_act_scale, parts, cur_id, idx)
    refine_tesnor(res, parts, cur_id, idx)
    return op

def core_slice_div_cvt_matmul(op: DivCvtMatmul, cur_id: int, total_num: int):
    matrix_a = op.get_matrix_A
    matrix_b = op.get_matrix_B
    divisor = op.get_divisor
    res = op.get_output
    assert matrix_a.shape.dim == matrix_b.shape.dim
    assert matrix_a.data_type == DataType.float16
    assert divisor.data_type == DataType.float16
    idx, dim = first_none_trivial_dim_not_last(matrix_a.shape)

    parts = n_part_division(dim, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    refine_tesnor(matrix_a, parts, cur_id, idx)
    refine_tesnor(divisor, parts, cur_id, idx)
    if idx < matrix_b.shape.dim - 2:
        refine_tesnor(matrix_b, parts, cur_id, idx)
        if matrix_b.data_type == DataType.int8:
            refine_tesnor(matrix_b.get_act_scale, parts, cur_id, idx)
    refine_tesnor(res, parts, cur_id, idx)
    return op

# cur_id: 0, 1, 2
def core_slice_softmax(op: Softmax, cur_id: int, total_num: int):
    feature_in = op.get_input
    feature_out = op.get_output

    idx, batch = first_none_trivial_dim_not_last(feature_in.shape)

    parts = n_part_division(batch, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    
    refine_tesnor(feature_in, parts, cur_id, idx)
    refine_tesnor(feature_out, parts, cur_id, idx)
    if op.act_scale_flag:
        refine_tesnor(feature_out.get_act_scale, parts, cur_id, idx)
    return op

def core_slice_transpose(op: Transpose, cur_id: int, total_num: int):
    feature_in = op.get_input
    feature_out = op.get_output

    idx, batch = first_none_trivial_dim_not_last(feature_in.shape, last_limit=2)
    parts = n_part_division(batch, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)

    refine_tesnor(feature_in, parts, cur_id, idx)
    refine_tesnor(feature_out, parts, cur_id, idx)
    if feature_in.data_type == DataType.float16 and feature_out.data_type == DataType.int8:
        out_scale = op.get_outputs()[1]
        refine_tesnor(out_scale, parts, cur_id, idx)
    return op

# 检查是否存在src到target的boardcast
def check_board_cast(target: Shape, src: Shape) -> Optional[int]:
    assert target.dim == src.dim

    for i, n in enumerate(src):
        if n == 1 and target[i] != 1:
            return i
    return None

# cur_id: 0, 1, 2
# eltwise需要考虑boardcast
def core_slice_eltwise(op: Union[Eltwise, Div], cur_id: int, total_num: int):
    left = op.get_input_A
    right = op.get_input_B
    out = op.get_output

    # keep right tensor be board cast
    board_cast_idx = check_board_cast(left.shape, right.shape)
    if board_cast_idx is None and check_board_cast(right.shape, left.shape):
        left, right = right, left
        board_cast_idx = check_board_cast(left.shape, right.shape)

    ref_shape = left.shape
    idx, dim = first_none_trivial_dim_not_last(ref_shape)
    parts = n_part_division(dim, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    if board_cast_idx is None or board_cast_idx > idx:
        refine_tesnor(left, parts, cur_id, idx)
        refine_tesnor(right, parts, cur_id, idx)
        refine_tesnor(out, parts, cur_id, idx)
    elif board_cast_idx == idx:
        # [3, 6, 1152] x [1, 6, 1152] => [3, 6, 1152]
        refine_tesnor(left, parts, cur_id, idx)
        refine_tesnor(out, parts, cur_id, idx)
    else:
        raise NotImplementedError
    if op.act_scale_flag:
        refine_tesnor(out.get_act_scale, parts, cur_id, idx)
    return op

def core_slice_activation(op: Union[Silu, Gelu], cur_id: int, total_num: int):
    feature = op.get_input
    output = op.get_output
    
    dim_idx, dim = first_none_trivial_dim_not_last(feature.shape) 
    parts = n_part_division(dim, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    
    refine_tesnor(feature, parts, cur_id, dim_idx)
    refine_tesnor(output, parts, cur_id, dim_idx)
    if op.act_scale_flag:
        refine_tesnor(output.get_act_scale, parts, cur_id, dim_idx)
    return op

def core_slice_normalization(op: Union[RMSnorm, Layernorm], cur_id: int, total_num: int):
    feature = op.get_input
    output = op.get_output
    
    dim_idx, dim = first_none_trivial_dim_not_last(feature.shape) 
    parts = n_part_division(dim, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    
    refine_tesnor(feature, parts, cur_id, dim_idx)
    refine_tesnor(output, parts, cur_id, dim_idx)
    if op.act_scale_flag:
        refine_tesnor(output.get_act_scale, parts, cur_id, dim_idx)
    return op

def core_slice_rope(op: RoPE, cur_id: int, total_num: int):
    feature = op.get_input
    output = op.get_output
    
    dim_idx, dim = first_none_trivial_dim_not_last(feature.shape, last_limit=2) 
    parts = n_part_division(dim, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    
    refine_tesnor(feature, parts, cur_id, dim_idx)
    refine_tesnor(output, parts, cur_id, dim_idx)
    if op.act_scale_flag:
        refine_tesnor(output.get_act_scale, parts, cur_id, dim_idx)
    return op

def core_slice_convert(op: Convert, cur_id: int, total_num: int):
    feature = op.get_input
    output = op.get_output
    
    dim_idx, dim = first_none_trivial_dim_not_last(feature.shape) 
    parts = n_part_division(dim, total_num)
    if parts[cur_id] == 0:
        return Nop(op.name)
    
    refine_tesnor(feature, parts, cur_id, dim_idx)
    refine_tesnor(output, parts, cur_id, dim_idx)
    if op.act_scale_flag:
        refine_tesnor(output.get_act_scale, parts, cur_id, dim_idx)
    return op
