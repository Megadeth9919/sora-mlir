from .graph_ir import *
from inst import *
from utils import hw_info, GlobalBuffer
import math
import numpy as np
GROUP_SIZE = hw_info.GlobalBuffer.get_bank_num_in_BGroup()
BANK_SIZE = hw_info.GlobalBuffer.get_bank_bytes_num()
GROUP_NUM = hw_info.GlobalBuffer.get_Group_num()


def get_tile_dim(dim: int, tile_size: int, index: int, max_index: int) -> int:
    return (
        tile_size
        if (index != max_index - 1)
        else dim % tile_size if dim % tile_size != 0 else tile_size
    )

def get_tile_dim0_num(
    residual: bool, tile_dim1_size: int, max_dim1_size: int, index: int, max_index: int
) -> int:
    if index != max_index - 1:
        tile_num = math.ceil(tile_dim1_size / max_dim1_size)
        return get_tile_dim(tile_dim1_size, max_dim1_size, index, tile_num)
    else:
        if residual:
            return 1
        else:
            return get_tile_dim(tile_dim1_size, max_dim1_size, index, max_index)
def get_bank_stride(K_in_byte:int ) -> int:
    stride = math.ceil(K_in_byte/ hw_info.WeightBuffer.get_bytewidth())
    # if stride %  2 ==1 :
    #     stride += 1
    return stride
def tensor_slice_row(t: Union[Tensor, TensorView]) -> list[TensorView]:
    N = math.prod(t.shape[: len(t.shape) - 1])
    ret = []
    new_shape = Shape(t.shape[-1])
    assert BANK_SIZE >= t.shape[-1] * get_size_by_type(t.data_type)
    for i in range(N):
        tensor_view = TensorView(t)
        tensor_view.shape = new_shape
        tensor_view.data_type = t.data_type
        tensor_view.addr = t.addr + i * new_shape.prod() * get_size_by_type(t.data_type)
        tensor_view.stride = Shape([0 for i in range(len(t.shape))])
        ret.append(tensor_view)
    return ret


def get_dim0_bank_factor(dim0: int, dim1: int, type: DataType) -> float:
    factor = 1.0
    while dim0 * get_size_by_type(type) < BANK_SIZE and dim1 > 2 and dim1 % 2 == 0:
        dim0 *= 2
        dim1 //= 2
        factor /= 2

    while dim0 * get_size_by_type(type) > BANK_SIZE and dim0 % 2 == 0:
        dim0 //= 2
        factor *= 2
    return factor


def get_max_banks_org(K :int, type:DataType, dynamic_scale: bool) -> int:
    size = K * get_size_by_type(type)
    # align to 32
    size = math.ceil(size / hw_info.GlobalBuffer.get_bytewidth()) * hw_info.GlobalBuffer.get_bytewidth()
    maxsize = math.floor(BANK_SIZE/size)
    max_bank_size = maxsize * GROUP_SIZE
    if dynamic_scale:
        max_bank_size = min(max_bank_size, hw_info.MetaBuffer.get_bank_bytes_num() / 4)
    return int(max_bank_size)
def get_max_banks(K :int, type:DataType, dynamic_scale: bool) -> int:
    max_bank_size = get_max_banks_org(K,type,dynamic_scale)
    if max_bank_size > 1024:
        max_bank_size = 1024
    return max_bank_size

def get_max_banks_half(K:int, type:DataType,dynamic_scale: bool) -> int:
    maxbanks = get_max_banks_org(K,type,dynamic_scale) // 2
    if maxbanks > 1024:
        maxbanks = 1024
    return maxbanks
    
def tensor_slice_plane_merge(t: Union[Tensor, TensorView], max_bank) -> list[TensorView]:
    ret = []
    dim0 = t.shape[-1]
    dim1 = t.shape[-2]
    assert BANK_SIZE >= dim0 * get_size_by_type(t.data_type)

    max_dim1_size = math.floor(
        BANK_SIZE / (dim0 * get_size_by_type(t.data_type))
    )  # 每个bank最多能存的行数

    length = len(t.shape)
    N = math.prod(t.shape[: length - 2])

    tile_dim1 = max_bank
    tile_dim1_size = math.floor(t.shape[-2] / tile_dim1)  # 31
    residual = True if t.shape[-2] % tile_dim1 else False

    tile_num = math.ceil(tile_dim1_size / max_dim1_size)  # 16
    if residual:
        tile_num = tile_num + 1

    addr = t.addr
    for i in range(N):
        for j in range(tile_num):
            tensor_view = TensorView(t)
            tensor_view.shape = Shape(
                1,
                get_tile_dim(t.shape[-2], tile_dim1, j, tile_num),
                t.shape[-1]
                * get_tile_dim0_num(
                    residual, tile_dim1_size, max_dim1_size, j, tile_num
                ),
            )
            tensor_view.data_type = t.data_type
            tensor_view.addr = addr
            tensor_view.stride = Shape([0 for i in range(len(t.shape))])
            addr += math.prod(tensor_view.shape[-2:]) * get_size_by_type(t.data_type)
            ret.append(tensor_view)
    return ret

def tensor_slice_plane(t: Union[Tensor, TensorView], max_bank) -> list[TensorView]:
    ret = []
    dim0 = t.shape[-1]
    dim1 = t.shape[-2]
    assert(max_bank %2 ==0)
    assert BANK_SIZE >= dim0 * get_size_by_type(t.data_type)
    length = len(t.shape)
    N = math.prod(t.shape[: length - 2])
    dim0 = t.shape[-1]
    dim1 = max_bank
    dim1_tile_num = math.ceil(t.shape[-2] / dim1)
    addr = t.addr
    for i in range(N):
        for j in range(dim1_tile_num):
            tensor_view = TensorView(t)
            tensor_view.shape = Shape(
                i, j, get_tile_dim(t.shape[-2], dim1, j, dim1_tile_num), t.shape[-1]
            )
            tensor_view.data_type = t.data_type
            tensor_view.addr = addr
            tensor_view.stride = Shape([0 for i in range(len(t.shape))])
            addr += math.prod(tensor_view.shape[-2:]) * get_size_by_type(t.data_type)
            ret.append(tensor_view)
    return ret

def tensor_slice_plane_dim_merge(t: Union[Tensor, TensorView], max_bank) -> list[TensorView]:
    assert BANK_SIZE >= t.shape[-1] * get_size_by_type(t.data_type)
    
    ret = []
    N = math.prod(t.shape[: -1])
    dim0 = t.shape[-1]
    # real rows per wave
    dim1 = max_bank
    addr = t.addr
    remain_num = N
    
    tile_num = math.ceil(N / dim1)
    for i in range(tile_num):
        tensor_view = TensorView(t)
        tensor_view.shape = Shape(min(remain_num, dim1), dim0)
        tensor_view.data_type = t.data_type
        tensor_view.addr = addr
        tensor_view.stride = Shape(dim0, 1)
        addr += math.prod(tensor_view.shape[-2:]) * get_size_by_type(t.data_type)
        remain_num -= dim1
        ret.append(tensor_view)
    assert remain_num <= 0
    return ret

def get_inst_type(inst: Inst) -> PUType:
    if isinstance(inst,LDInst) :
        return PUType.LD
    elif isinstance(inst,STInst) :
        return PUType.ST
    elif isinstance(inst,MMInst) :
        return PUType.MM
    elif isinstance(inst,MISCInst) :
        return PUType.MISC
    elif isinstance(inst,SYSCInst) :
        return PUType.SYS
    elif isinstance(inst,RSInst) :
        return PUType.RS
    else:
        raise ValueError("Inst type error")

def lower_binary_for_parallel(op: Op, inst_collect: InstCollector,dynamic_scale:bool = False):
    assert len(op.get_inputs()) == 2
    input0 = op.get_inputs()[0]
    input1 = op.get_inputs()[1]
    output = op.get_outputs()[0]
    broadcast = False
    broadcast_dim = 0
    if input0.shape != input1.shape:
        broadcast = True
        for i in range(len(input0.shape)):
            if input0.shape[i] != input1.shape[i] and input1.shape[i] == 1:
                broadcast_dim = i
    if broadcast:
        assert input0.shape[broadcast_dim] != input1.shape[broadcast_dim]
    # assume broadcast on operand1
    assert input0.shape == output.shape
    ref = input0
    if (
        get_size_by_type(input0.data_type) < get_size_by_type(input1.data_type)
        and not broadcast
    ):
        ref = input1
        if get_size_by_type(input1.data_type) < get_size_by_type(output.data_type):
            ref = output
    else:
        if get_size_by_type(input0.data_type) < get_size_by_type(output.data_type):
            ref = output
    max_size = get_max_banks_half(ref.shape[-1],ref.data_type,dynamic_scale)
    tiles = tensor_slice_plane(ref, max_size)
    plane_offset:int = 0
    addrs = [0, 2048]
    meta_addrs = [0, 256]
    dscale_addr = 0
    for index,tile in enumerate(tiles):
        assert tile.addr is not None
        assert ref.addr is not None
        tile_offset = (
            lambda t: (tile.addr - ref.addr)
            * get_size_by_type(t)
            / get_size_by_type(ref.data_type)
        )

        input1_size = input1.shape.prod() * get_size_by_type(input1.data_type)
        ld0 = LDInst()
        if(index == 0):
            last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
            if last_ins is not None:
                last_ins.release.append(PUType.LD)
                ld0.wait.append(get_inst_type(last_ins))
        elif index >= 2:
            ld0.wait.append(PUType.ST)
        ld0.length_1d = tile.shape[-1] * get_size_by_type(input0.data_type)
        ld0.loop_1d = tile.shape[-2]
        ld0.src_1d_stride = tile.shape[-1] * get_size_by_type(input0.data_type)
        assert input0.addr is not None
        ld0.src_addr = input0.addr + int(tile_offset(input0.data_type))
        ld0.bank_addr_stride = get_bank_stride(ld0.length_1d)
        ld0.mode = LDMode.DDR2global if input0.reside_ddr else LDMode.HBM2Global
        ld0.dst_group_id = 0
        ld0.dst_bank_id = 0
        ld0.dst_addr = addrs[index % 2]
        inst_collect.add(ld0)
        ld1 = LDInst()
        ld1.release.append(PUType.MISC)
        ld1.length_1d = tile.shape[-1] * get_size_by_type(input1.data_type)
        if broadcast:
            if len(input0.shape) == 3:
                # assert tile.shape[-2] > input1.shape[broadcast_dim]
                if broadcast_dim == 0:
                    ld1.loop_1d = tile.shape[-2]
                    ld1.src_1d_stride = input1.shape[-1] * get_size_by_type(
                        input1.data_type
                    )
                    if tile.shape[1] == 0:
                        plane_offset = 0                     
                    ld1.src_addr = int(input1.addr + ( plane_offset) * get_size_by_type(input1.data_type))
                    plane_offset += tile.shape[-2] * tile.shape[-1] 
                elif broadcast_dim == 1:
                    ld1.loop_1d = tile.shape[-2]
                    ld1.src_1d_stride = 0
                    ld1.src_addr = int(input1.addr + tile.shape[0] * tile.shape[-1] * get_size_by_type(input1.data_type))
            elif len(input0.shape) == 4:
                assert len(input1.shape) == 4
                if tile.shape[1] == 0:
                    plane_offset = 0                  
                if broadcast_dim == 0:
                    ld1.loop_1d = tile.shape[-2]
                    ld1.src_1d_stride = tile.shape[-1] * get_size_by_type(
                        input1.data_type
                    )
                    ld1.src_addr = int(input1.addr + 
                                    ((tile.shape[0] % input0.shape[1]) * math.prod(input1.shape[-2:])  + plane_offset) * get_size_by_type(input1.data_type))
                elif broadcast_dim == 1:
                    ld1.loop_1d = tile.shape[-2]
                    ld1.src_1d_stride = tile.shape[-1] * get_size_by_type(input1.data_type)
                    ld1.src_addr = int(input1.addr + (math.floor(tile.shape[0] / input0.shape[1]) * math.prod(
                        input1.shape[-2:]
                    ) + plane_offset) * get_size_by_type(input1.data_type))
                plane_offset += tile.shape[-2] * tile.shape[-1] 
        else:
            ld1.loop_1d = tile.shape[-2]
            ld1.src_1d_stride = input1.shape[-1] * get_size_by_type(input1.data_type)
            assert input1.addr is not None
            ld1.src_addr = input1.addr + int(tile_offset(input1.data_type))
        ld1.dst_group_id = 1
        ld1.dst_bank_id = 0
        ld1.dst_addr = addrs[index % 2]
        ld1.bank_addr_stride = get_bank_stride(ld1.length_1d)
        ld1.mode = LDMode.DDR2global if input1.reside_ddr else LDMode.HBM2Global
        inst_collect.add(ld1)
        assert not( (output.data_type != DataType.float16) ^ dynamic_scale )
        
        misc = MISCInst()
        misc.wait.append(PUType.LD)
        if isinstance(op, Eltwise) and op.type == "add":
            misc.op = MiscOp.elt_add
        elif isinstance(op, Eltwise) and op.type == "mul":
            misc.op = MiscOp.elt_mul
        elif isinstance(op, Eltwise) and op.type == "sub":
            misc.op = MiscOp.elt_sub
        elif isinstance(op, Div):
            misc.op = MiscOp.elt_div
        misc.release.append(PUType.ST)
        misc.input_a_mode = MiscMode.fp16 if input0.data_type == DataType.float16 else MiscMode.int8 
        misc.input_b_mode = MiscMode.fp16 if input1.data_type == DataType.float16 else MiscMode.int8 
        misc.output_mode = MiscMode.fp16 if output.data_type == DataType.float16 else MiscMode.int8 
        misc.in_a_start_addr = addrs[index % 2]
        misc.in_b_start_addr = addrs[index % 2]
        misc.in_b_bank_id = 0
        misc.in_b_bank_group = 1
        misc.out_bank_id = 0
        misc.out_bank_group = 2
        misc.out_start_addr = addrs[index % 2]
        misc.K = tile.shape[-1]
        misc.fK = int(np.float32(tile.shape[-1]).view(np.uint32))
        misc.batch = tile.shape[-2]
        misc.meta_addr = meta_addrs[index % 2]
        misc.dynamic_scale = dynamic_scale
        inst_collect.add(misc)
        st = STInst()
        if not dynamic_scale:
            if index < len(tiles)-2:
                st.release.append(PUType.LD)
        st.wait.append(PUType.MISC)
        st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
        st.length_1d = tile.shape[-1] * get_size_by_type(output.data_type)
        st.loop_1d = tile.shape[-2]
        st.dst_1d_stride = tile.shape[-1] * get_size_by_type(output.data_type)
        st.src_group_id = 2
        st.src_addr =  addrs[index % 2]
        st.bank_addr_stride = get_bank_stride(st.length_1d)
        st.dst_addr = output.addr + int(tile_offset(output.data_type))
        inst_collect.add(st)

        if dynamic_scale :
            out_dynamic_scale = op.get_outputs()[1]
            st1 = STInst()
            if index < len(tiles)-2:
                st1.release.append(PUType.LD)
            st1.mode = STMode.Meta2DDR
            st1.length_1d = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st1.loop_1d = 1
            st1.src_addr =  meta_addrs[index % 2]
            st1.dst_1d_stride = out_dynamic_scale.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st1.bank_addr_stride = get_bank_stride(st1.length_1d)
            st1.dst_addr = out_dynamic_scale.addr + dscale_addr 
            dscale_addr += tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            inst_collect.add(st1)
    return

def lower_load_inst(op: Op ,inst_collect: InstCollector):
    input = op.get_inputs()[0]
    size = input.shape[0]
    ld0 = LDInst()
    ld0.release.append(PUType.MISC)
    ld0.length_1d = size 
    ld0.loop_1d = 1
    ld0.src_1d_stride = size
    assert input.addr is not None
    ld0.src_addr = input.addr
    ld0.bank_addr_stride =  get_bank_stride(size)
    ld0.mode = LDMode.DDR2global
    ld0.dst_group_id = 0
    ld0.dst_bank_id = 0
    inst_collect.add(ld0)
    misc = MISCInst()
    misc.op = MiscOp.load_ins
    misc.wait.append(PUType.LD)
    misc.input_a_mode =  MiscMode.int8 
    misc.in_b_bank_id = 0
    misc.out_bank_id = 0
    misc.K = size // hw_info.WeightBuffer.get_bytewidth()
    misc.batch = 1
    inst_collect.add(misc)
    return 
   
# row continous
def lower_binary(op: Op, inst_collect: InstCollector, dynamic_scale:bool = False):
    assert len(op.get_inputs()) == 2
    input0 = op.get_inputs()[0]
    input1 = op.get_inputs()[1]
    output = op.get_outputs()[0]
    broadcast = False
    broadcast_dim = 0
    if input0.shape != input1.shape:
        broadcast = True
        for i in range(len(input0.shape)):
            if input0.shape[i] != input1.shape[i] and input1.shape[i] == 1:
                broadcast_dim = i
    if broadcast:
        assert input0.shape[broadcast_dim] != input1.shape[broadcast_dim]
    # assume broadcast on operand1
    assert input0.shape == output.shape
    ref = input0
    if (
        get_size_by_type(input0.data_type) < get_size_by_type(input1.data_type)
        and not broadcast
    ):
        ref = input1
        if get_size_by_type(input1.data_type) < get_size_by_type(output.data_type):
            ref = output
    else:
        if get_size_by_type(input0.data_type) < get_size_by_type(output.data_type):
            ref = output
    max_size = get_max_banks(ref.shape[-1],ref.data_type,dynamic_scale)
    tiles = tensor_slice_plane(ref, max_size)
    plane_offset:int = 0
    dscale_addr = 0
    for index ,tile in enumerate(tiles):
        assert tile.addr is not None
        assert ref.addr is not None
        tile_offset = (
            lambda t: (tile.addr - ref.addr)
            * get_size_by_type(t)
            / get_size_by_type(ref.data_type)
        )

        input1_size = input1.shape.prod() * get_size_by_type(input1.data_type)
        ld0 = LDInst()
        last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
        if last_ins is not None:
            last_ins.release.append(PUType.LD)
            ld0.wait.append(get_inst_type(last_ins))
        ld0.length_1d = tile.shape[-1] * get_size_by_type(input0.data_type)
        ld0.loop_1d = tile.shape[-2]
        ld0.src_1d_stride = tile.shape[-1] * get_size_by_type(input0.data_type)
        assert input0.addr is not None
        ld0.src_addr = input0.addr + int(tile_offset(input0.data_type))
        ld0.bank_addr_stride = get_bank_stride(ld0.length_1d)
        ld0.mode = LDMode.DDR2global if input0.reside_ddr else LDMode.HBM2Global
        ld0.dst_group_id = 0
        ld0.dst_bank_id = 0
        inst_collect.add(ld0)
        ld1 = LDInst()
        ld1.release.append(PUType.MISC)
        ld1.length_1d = tile.shape[-1] * get_size_by_type(input1.data_type)
        if broadcast:
            assert input1.const
            if len(input0.shape) == 3:
                # assert tile.shape[-2] > input1.shape[broadcast_dim]
                if broadcast_dim == 0:
                    ld1.loop_1d = tile.shape[-2]
                    ld1.src_1d_stride = input1.shape[-1] * get_size_by_type(
                        input1.data_type
                    )
                    if tile.shape[1] == 0:
                        plane_offset = 0                     
                    ld1.src_addr = int(input1.addr + ( plane_offset) * get_size_by_type(input1.data_type))
                    plane_offset += tile.shape[-2] * tile.shape[-1] 
                elif broadcast_dim == 1:
                    ld1.loop_1d = tile.shape[-2]
                    ld1.src_1d_stride = 0
                    ld1.src_addr = int(input1.addr + tile.shape[0] * tile.shape[-1] * get_size_by_type(input1.data_type))
            elif len(input0.shape) == 4:
                assert len(input1.shape) == 4
                if tile.shape[1] == 0:
                    plane_offset = 0
                if broadcast_dim == 0:
                    ld1.loop_1d = tile.shape[-2]
                    ld1.src_1d_stride = tile.shape[-1] * get_size_by_type(
                        input1.data_type
                    )
                    ld1.src_addr = int(input1.addr + 
                                       ((tile.shape[0] % input0.shape[1]) * math.prod(input1.shape[-2:])  + plane_offset) * get_size_by_type(input1.data_type))
                elif broadcast_dim == 1:
                    ld1.loop_1d = tile.shape[-2]
                    ld1.src_1d_stride = tile.shape[-1] * get_size_by_type(input1.data_type)
                    ld1.src_addr = int(input1.addr + (math.floor(tile.shape[0] / input0.shape[1]) * math.prod(
                        input1.shape[-2:]
                    ) + plane_offset) * get_size_by_type(input1.data_type))
                plane_offset += tile.shape[-2] * tile.shape[-1] 
        else:
            ld1.loop_1d = tile.shape[-2]
            ld1.src_1d_stride = input1.shape[-1] * get_size_by_type(input1.data_type)
            assert input1.addr is not None
            ld1.src_addr = input1.addr + int(tile_offset(input1.data_type))
        ld1.dst_group_id = 1
        ld1.dst_bank_id = 0
        ld1.bank_addr_stride = get_bank_stride(ld1.length_1d)
        ld1.mode = LDMode.DDR2global if input1.reside_ddr else LDMode.HBM2Global
        inst_collect.add(ld1)
        assert not( (output.data_type != DataType.float16) ^ dynamic_scale )
        
        misc = MISCInst()
        misc.wait.append(PUType.LD)
        if isinstance(op, Eltwise) and op.type == "add":
            misc.op = MiscOp.elt_add
        elif isinstance(op, Eltwise) and op.type == "mul":
            misc.op = MiscOp.elt_mul
        elif isinstance(op, Eltwise) and op.type == "sub":
            misc.op = MiscOp.elt_sub
        elif isinstance(op, Div):
            misc.op = MiscOp.elt_div
        misc.release.append(PUType.ST)
        misc.input_a_mode = MiscMode.fp16 if input0.data_type == DataType.float16 else MiscMode.int8 
        misc.input_b_mode = MiscMode.fp16 if input1.data_type == DataType.float16 else MiscMode.int8 
        misc.output_mode = MiscMode.fp16 if output.data_type == DataType.float16 else MiscMode.int8 
        misc.in_b_bank_id = 0
        misc.in_b_bank_group = 1
        misc.out_bank_id = 0
        misc.out_bank_group = 2
        misc.K = tile.shape[-1]
        misc.fK = int(np.float32(tile.shape[-1]).view(np.uint32))
        misc.batch = tile.shape[-2]
        misc.dynamic_scale = dynamic_scale
        inst_collect.add(misc)
        st = STInst()
        st.wait.append(PUType.MISC)
        st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
        st.length_1d = tile.shape[-1] * get_size_by_type(output.data_type)
        st.loop_1d = tile.shape[-2]
        st.dst_1d_stride = tile.shape[-1] * get_size_by_type(output.data_type)
        st.src_group_id = 2
        st.bank_addr_stride = get_bank_stride(st.length_1d)
        st.dst_addr = output.addr + int(tile_offset(output.data_type))
        inst_collect.add(st)

        if dynamic_scale :
            out_dynamic_scale = op.get_outputs()[1]
            st1 = STInst()
            st1.mode = STMode.Meta2DDR
            st1.length_1d = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st1.loop_1d = 1
            st1.dst_1d_stride = out_dynamic_scale.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st1.bank_addr_stride = get_bank_stride(st1.length_1d)
            st1.dst_addr = out_dynamic_scale.addr + dscale_addr 
            dscale_addr += tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            inst_collect.add(st1)
    return

def lower_rope(op: Op, inst_collect: InstCollector, dynamic_scale:bool = False):
    # assert len(op.get_inputs()) == 2
    input = op.get_inputs()[0]
    sin_cos_table = op.get_weights()[0]
    output = op.get_outputs()[0]
    ref = input
    if get_size_by_type(input.data_type) < get_size_by_type(output.data_type):
        ref = output
    max_size = get_max_banks_half(ref.shape[-1],ref.data_type,dynamic_scale)
    if max_size > 1024 :
        max_size = 1024
    tiles = tensor_slice_plane(ref, max_size)
    assert input.shape == output.shape
    assert input.data_type == DataType.float16
    assert sin_cos_table.data_type == DataType.float16
    assert input.shape[-1] % 2 == 0
    addrs = [0, 256]
    dscale_addr = 0
    for index,tile in enumerate(tiles) :
        assert tile.addr is not None
        assert ref.addr is not None
        assert input.addr is not None
        tile_offset = (
            lambda t: (tile.addr - ref.addr)
            * get_size_by_type(t)
            / get_size_by_type(ref.data_type)
        )
        #loadx
        ld0 = LDInst()
        last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
        if last_ins is not None and not isinstance(last_ins,LDInst):
            last_ins.release.append(PUType.LD)
            ld0.wait.append(get_inst_type(last_ins))
        ld0.length_1d = tile.shape[-1] * get_size_by_type(input.data_type)
        ld0.loop_1d = tile.shape[-2]
        ld0.src_1d_stride = tile.shape[-1] * get_size_by_type(input.data_type)
        ld0.src_addr = input.addr + int(tile_offset(input.data_type))
        ld0.bank_addr_stride = get_bank_stride(ld0.length_1d)
        ld0.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
        ld0.dst_group_id = 0
        ld0.dst_bank_id = 0
        inst_collect.add(ld0)
        #load cos
        ld1 = LDInst()
        ld1.release.append(PUType.MISC)
        ld1.length_1d = tile.shape[-1] * get_size_by_type(sin_cos_table.data_type)
        ld1.loop_1d = tile.shape[-2]
        ld1.src_1d_stride = sin_cos_table.shape[-1] * get_size_by_type(
                    sin_cos_table.data_type)
        ld1.src_addr = sin_cos_table.addr + (tile.shape[1]*tile.shape[-2]*sin_cos_table.shape[-1]) * get_size_by_type(sin_cos_table.data_type) 
        ld1.dst_group_id = 1
        ld1.dst_bank_id = 0
        ld1.bank_addr_stride = get_bank_stride(ld1.length_1d)
        ld1.mode = LDMode.DDR2global
        inst_collect.add(ld1)
        #x*cos
        misc = MISCInst()
        misc.op = MiscOp.elt_mul
        misc.wait.append(PUType.LD)
        misc.input_a_mode = MiscMode.fp16
        misc.input_b_mode = MiscMode.fp16 
        misc.output_mode = MiscMode.fp16 
        misc.in_b_bank_id = 0
        misc.in_b_bank_group = 1
        misc.out_bank_id = 0
        misc.out_bank_group = 2
        misc.K = tile.shape[-1]
        misc.batch = tile.shape[-2]
        inst_collect.add(misc)
        #load sin
        ld2 = LDInst()
        ld2.release.append(PUType.MISC)
        ld2.length_1d = tile.shape[-1] * get_size_by_type(sin_cos_table.data_type)
        ld2.loop_1d = tile.shape[-2]
        ld2.src_1d_stride = sin_cos_table.shape[-1] * get_size_by_type(
                        sin_cos_table.data_type)
        ld2.src_addr = sin_cos_table.addr + ((tile.shape[1]*tile.shape[-2]*sin_cos_table.shape[-1]) + sin_cos_table.shape[-1]//2) *  get_size_by_type(sin_cos_table.data_type) 
        ld2.dst_group_id = 2
        ld2.dst_bank_id = 0
        ld2.dst_addr = 2048
        ld2.bank_addr_stride = get_bank_stride(ld2.length_1d)
        ld2.mode = LDMode.DDR2global
        inst_collect.add(ld2)
        #x*sin
        misc3 = MISCInst()
        misc3.wait.append(PUType.LD)
        misc3.op = MiscOp.elt_mul
        misc3.input_a_mode = MiscMode.fp16
        misc3.input_b_mode = MiscMode.fp16 
        misc3.output_mode = MiscMode.fp16
        misc3.in_b_bank_id = 0
        misc3.in_b_bank_group = 2
        misc3.in_b_start_addr=2048
        misc3.out_bank_group = 1
        misc3.out_start_addr=2048
        misc3.out_swap = 1
        misc3.K = tile.shape[-1]
        misc3.batch = tile.shape[-2]
        inst_collect.add(misc3)
        assert not( (output.data_type != DataType.float16) ^ dynamic_scale )
        #xsin+xcos
        misc4 = MISCInst()
        misc4.op = MiscOp.elt_add
        if not dynamic_scale :
            misc4.release.append(PUType.ST)
        misc4.input_a_mode = MiscMode.fp16 
        misc4.input_b_mode = MiscMode.fp16 
        misc4.output_mode = MiscMode.fp16
        misc4.in_a_bank_id = 0
        misc4.in_a_bank_group = 2
        misc4.in_b_start_addr= 2048
        misc4.in_b_bank_group = 1
        misc4.out_bank_group = 0
        misc4.out_start_addr=2048
        misc4.K = tile.shape[-1]
        misc4.batch = tile.shape[-2]
        inst_collect.add(misc4)
        if dynamic_scale :
            misc5 = MISCInst()
            misc5.op = MiscOp.abs_max
            misc5.release.append(PUType.ST)
            misc5.input_a_mode = MiscMode.fp16 
            misc5.input_b_mode = MiscMode.int8
            misc5.output_mode = MiscMode.int8
            misc5.in_a_start_addr = 2048
            misc5.in_a_bank_group = 0
            misc5.out_bank_group = 1
            misc5.out_start_addr=0
            misc5.meta_addr = addrs[index % 2]
            misc5.K = tile.shape[-1]
            misc5.batch = tile.shape[-2]
            misc5.dynamic_scale = dynamic_scale
            inst_collect.add(misc5)
            #stror result 
            st2 = STInst()
            st2.wait.append(PUType.MISC)
            st2.length_1d = tile.shape[-1] * get_size_by_type(output.data_type)
            st2.loop_1d = tile.shape[-2]
            st2.dst_1d_stride = input.shape[-1] * get_size_by_type(output.data_type)
            st2.src_group_id = 1
            st2.src_addr = 0
            st2.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
            st2.bank_addr_stride = get_bank_stride(st2.length_1d)
            st2.dst_addr = output.addr + int(tile_offset(output.data_type))
            inst_collect.add(st2)
            #store scale
            out_dynamic_scale = op.get_outputs()[1]
            st3 = STInst()
            st3.mode = STMode.Meta2DDR
            st3.length_1d = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st3.loop_1d = 1
            st3.src_addr = addrs[index % 2]
            st3.dst_1d_stride = out_dynamic_scale.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st3.bank_addr_stride = get_bank_stride(st3.length_1d)
            st3.dst_addr = out_dynamic_scale.addr + dscale_addr
            dscale_addr += st3.length_1d 
            inst_collect.add(st3)
        else :
            st2 = STInst()
            st2.wait.append(PUType.MISC)
            st2.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
            st2.length_1d = tile.shape[-1] * get_size_by_type(output.data_type)
            st2.loop_1d = tile.shape[-2]
            st2.dst_1d_stride = input.shape[-1] * get_size_by_type(output.data_type)
            st2.src_group_id = 0
            st2.src_addr = 2048
            st2.bank_addr_stride = get_bank_stride(st2.length_1d)
            st2.dst_addr = output.addr + int(tile_offset(output.data_type))
            inst_collect.add(st2)
    return



def lower_unary_for_parallel(op: Op, inst_collect: InstCollector, dynamic_scale:bool = False):
    assert len(op.get_inputs()) == 1
    input = op.get_inputs()[0]
    output = op.get_outputs()[0]
    assert input.shape == output.shape
    ref = input
    if get_size_by_type(input.data_type) < get_size_by_type(output.data_type):
        ref = output

    max_size = get_max_banks(ref.shape[-1],ref.data_type,dynamic_scale)
    if  max_size > 1024:
        max_size = 1024
    tiles = tensor_slice_plane_dim_merge(ref, max_size)
    addrs = [0, 160, 320]
    group_id = [0, 2, 1]
    out_group_id = [1, 0, 2]
    dscale_addr = 0
    for index, tile in enumerate(tiles):
        assert tile.addr is not None
        assert ref.addr is not None
        tile_offset = (
            lambda t: (tile.addr - ref.addr)
            * get_size_by_type(t)
            / get_size_by_type(ref.data_type)
        )

        ld = LDInst()
        if(index == 0):
            last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
            if last_ins is not None:
                last_ins.release.append(PUType.LD)
                ld.wait.append(get_inst_type(last_ins))
        elif index >= 2:
            ld.wait.append(PUType.ST)
        ld.release.append(PUType.MISC)
        ld.length_1d = tile.shape[-1] * get_size_by_type(input.data_type)
        ld.loop_1d = tile.shape[-2]
        ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
        assert input.addr is not None
        ld.src_addr = input.addr + int(tile_offset(input.data_type))
        ld.bank_addr_stride = get_bank_stride(ld.length_1d)
        ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
        ld.dst_group_id = group_id[index % 3]
        inst_collect.add(ld)
        assert not( (output.data_type != DataType.float16) ^ dynamic_scale )
        misc = MISCInst()
        misc.wait.append(PUType.LD)
        misc.dynamic_scale = dynamic_scale
        if isinstance(op, Silu):
            misc.op = MiscOp.silu
        elif isinstance(op, Gelu):
            misc.op = MiscOp.gelu1
        elif isinstance(op, Softmax):
            misc.op = MiscOp.softmax
        elif isinstance(op, Convert):
            misc.op = MiscOp.abs_max
            misc.dynamic_scale = True

        misc.release.append(PUType.ST)
        misc.input_a_mode = MiscMode.fp16 if input.data_type == DataType.float16 else MiscMode.int8 
        misc.output_mode = MiscMode.fp16 if output.data_type == DataType.float16 else MiscMode.int8 
        misc.in_a_bank_group = group_id[index % 3]
        misc.in_a_bank_id = 0
        misc.in_a_start_addr = 0
        misc.out_bank_group = out_group_id[index % 3]
        misc.out_bank_id = 0
        misc.out_start_addr = 0
        misc.meta_addr = addrs[index %3]
        misc.K = tile.shape[-1]
        misc.fK = int(np.float32(tile.shape[-1]).view(np.uint32))
        misc.batch = tile.shape[-2]
        inst_collect.add(misc)

        st = STInst()
        st.wait.append(PUType.MISC)
        if not dynamic_scale:
            if index < len(tiles)-2:
                st.release.append(PUType.LD)
        st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
        st.length_1d = tile.shape[-1] * get_size_by_type(output.data_type)
        st.loop_1d = tile.shape[-2]
        st.dst_1d_stride = tile.shape[-1] * get_size_by_type(output.data_type)
        st.src_group_id = out_group_id[index % 3]
        st.bank_addr_stride = get_bank_stride(st.length_1d)
        st.dst_addr = output.addr + int(tile_offset(output.data_type))
        inst_collect.add(st)

        if dynamic_scale:
            out_dynamic_scale = op.get_outputs()[1]
            st_dynamic_scale = STInst()
            if index < len(tiles)-2:
                st_dynamic_scale.release.append(PUType.LD)
            st_dynamic_scale.mode = STMode.Meta2DDR
            st_dynamic_scale.length_1d = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.loop_1d = 1
            st_dynamic_scale.src_addr = addrs[index %3]
            st_dynamic_scale.dst_1d_stride = out_dynamic_scale.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.bank_addr_stride = get_bank_stride(st_dynamic_scale.length_1d)
            st_dynamic_scale.dst_addr = out_dynamic_scale.addr + dscale_addr
            dscale_addr += tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            inst_collect.add(st_dynamic_scale)
    return


def lower_unary(op: Op, inst_collect: InstCollector, dynamic_scale:bool = False):
    assert len(op.get_inputs()) == 1
    input = op.get_inputs()[0]
    output = op.get_outputs()[0]
    assert input.shape == output.shape
    ref = input
    if get_size_by_type(input.data_type) < get_size_by_type(output.data_type):
        ref = output

    max_size = get_max_banks(ref.shape[-1],ref.data_type,dynamic_scale)
    tiles = tensor_slice_plane_dim_merge(ref, max_size)
    dscale_addr = 0
    for index,tile in enumerate(tiles):
        assert tile.addr is not None
        assert ref.addr is not None
        tile_offset = (
            lambda t: (tile.addr - ref.addr)
            * get_size_by_type(t)
            / get_size_by_type(ref.data_type)
        )

        ld = LDInst()
        last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
        if last_ins is not None:
            last_ins.release.append(PUType.LD)
            ld.wait.append(get_inst_type(last_ins))
        ld.release.append(PUType.MISC)
        ld.length_1d = tile.shape[-1] * get_size_by_type(input.data_type)
        ld.loop_1d = tile.shape[-2]
        ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
        assert input.addr is not None
        ld.src_addr = input.addr + int(tile_offset(input.data_type))
        ld.bank_addr_stride = get_bank_stride(ld.length_1d)
        ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
        ld.dst_group_id = 0
        inst_collect.add(ld)
        assert not( (output.data_type != DataType.float16) ^ dynamic_scale )
        misc = MISCInst()
        misc.wait.append(PUType.LD)
        misc.dynamic_scale = dynamic_scale
        if isinstance(op, Silu):
            misc.op = MiscOp.silu
        elif isinstance(op, Gelu):
            misc.op = MiscOp.gelu1
        elif isinstance(op, Softmax):
            misc.op = MiscOp.softmax
        elif isinstance(op, Convert):
            misc.op = MiscOp.abs_max
            misc.dynamic_scale = True
        misc.release.append(PUType.ST)
        misc.input_a_mode = MiscMode.fp16 if input.data_type == DataType.float16 else MiscMode.int8 
        misc.output_mode = MiscMode.fp16 if output.data_type == DataType.float16 else MiscMode.int8 
        misc.in_a_bank_group = 0
        misc.in_a_bank_id = 0
        misc.in_a_start_addr = 0
        misc.out_bank_group = 1
        misc.out_bank_id = 0
        misc.out_start_addr = 0
        misc.K = tile.shape[-1]
        misc.fK = int(np.float32(tile.shape[-1]).view(np.uint32))
        misc.batch = tile.shape[-2]
        inst_collect.add(misc)

        st = STInst()
        st.wait.append(PUType.MISC)
        st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
        st.length_1d = tile.shape[-1] * get_size_by_type(output.data_type)
        st.loop_1d = tile.shape[-2]
        st.dst_1d_stride = tile.shape[-1] * get_size_by_type(output.data_type)
        st.src_group_id = 1
        st.bank_addr_stride = get_bank_stride(st.length_1d)
        st.dst_addr = output.addr + int(tile_offset(output.data_type))
        inst_collect.add(st)

        if dynamic_scale:
            out_dynamic_scale = op.get_outputs()[1]
            st_dynamic_scale = STInst()
            st_dynamic_scale.mode = STMode.Meta2DDR
            st_dynamic_scale.length_1d = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.loop_1d = 1
            st_dynamic_scale.dst_1d_stride = out_dynamic_scale.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.bank_addr_stride = get_bank_stride(st_dynamic_scale.length_1d)
            st_dynamic_scale.dst_addr = out_dynamic_scale.addr + dscale_addr
            dscale_addr += tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            inst_collect.add(st_dynamic_scale)
    return


def get_optional_addr(value):
    return value if value is not None else 0

# hotspot
def lower_tow_stage(op: Op, inst_collect: InstCollector, dynamic_scale: bool = False):
    input = op.get_inputs()[0]
    output = op.get_outputs()[0]
    ref = input
    if get_size_by_type(input.data_type) < get_size_by_type(output.data_type):
        ref = output
        
    max_size = get_max_banks(ref.shape[-1],ref.data_type,dynamic_scale)
    tiles = tensor_slice_plane_dim_merge(ref, max_size)
    weight = op.get_weights()[0]

    assert weight.shape[-1] == input.shape[-1]
    # only load a times for weight
    ld_weight = LDInst()
    last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
    if not isinstance(last_ins,LDInst) and last_ins is not None:
        last_ins.release.append(PUType.LD)
        ld_weight.wait.append(get_inst_type(last_ins))
    ld_weight.length_1d = input.shape[-1] * get_size_by_type(weight.data_type)
    ld_weight.loop_1d = min(max_size,tiles[0].shape[-2])
    ld_weight.src_1d_stride = 0
    ld_weight.src_addr = weight.addr
    ld_weight.dst_bank_id = 0
    ld_weight.dst_group_id = 2
    ld_weight.bank_addr_stride = get_bank_stride(ld_weight.length_1d)
    ld_weight.mode = LDMode.DDR2global
    inst_collect.add(ld_weight)
    dscale_addr = 0
    for index,tile in enumerate(tiles):
        assert tile.addr is not None
        assert ref.addr is not None
        assert input.addr is not None
        tile_offset = (
            lambda t: (tile.addr - ref.addr)
            * get_size_by_type(t)
            / get_size_by_type(ref.data_type)
        )

        ld = LDInst()
        last_ins = inst_collect.get_insts()[-1]
        if not isinstance(last_ins,LDInst):
            last_ins.release.append(PUType.LD)
            ld.wait.append(get_inst_type(last_ins))
        ld.release.append(PUType.MISC)
        ld.length_1d = tile.shape[-1] * get_size_by_type(input.data_type)
        ld.loop_1d = tile.shape[-2]
        ld.src_1d_stride = tile.shape[-1] * get_size_by_type(input.data_type)
        ld.src_addr = input.addr + int(tile_offset(input.data_type))
        ld.bank_addr_stride = get_bank_stride(ld.length_1d)
        ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
        inst_collect.add(ld)

        assert not( (output.data_type != DataType.float16) ^ dynamic_scale )
        misc = MISCInst()
        misc.wait.append(PUType.LD)
        if isinstance(op, Layernorm):
            misc.op = MiscOp.layernorm
        elif isinstance(op, RMSnorm):
            misc.op = MiscOp.rmsnorm
        misc.release.append(PUType.ST)
        misc.input_a_mode = MiscMode.fp16 if input.data_type == DataType.float16 else MiscMode.int8 
        misc.input_b_mode = MiscMode.fp16 if weight.data_type == DataType.float16 else MiscMode.int8 
        misc.output_mode = MiscMode.fp16 if output.data_type == DataType.float16 else MiscMode.int8 
        misc.in_a_bank_group = 0
        misc.in_a_bank_id = 0
        misc.in_a_start_addr = 0
        misc.in_b_bank_group = 2
        misc.in_b_bank_id = 0
        misc.in_b_start_addr = 0
        misc.out_bank_group = 1
        misc.out_bank_id = 0
        misc.out_start_addr = 0
        misc.dynamic_scale = dynamic_scale
        misc.K = tile.shape[-1]
        misc.fK = int(np.float32(tile.shape[-1]).view(np.uint32))
        misc.batch = tile.shape[-2]
        inst_collect.add(misc)

        st = STInst()
        st.wait.append(PUType.MISC)
        st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
        st.length_1d = output.shape[-1] * get_size_by_type(output.data_type)
        st.loop_1d = tile.shape[-2]
        st.dst_1d_stride = output.shape[-1] * get_size_by_type(output.data_type)
        st.src_group_id = 1
        st.bank_addr_stride = get_bank_stride(st.length_1d)
        st.dst_addr = output.addr + int(tile_offset(output.data_type))
        inst_collect.add(st)

        if dynamic_scale:
            out_dynamic_scale = op.get_outputs()[1]
            st_dynamic_scale = STInst()
            st_dynamic_scale.mode = STMode.Meta2DDR
            st_dynamic_scale.length_1d = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.loop_1d = 1
            st_dynamic_scale.dst_1d_stride = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.bank_addr_stride = get_bank_stride(st_dynamic_scale.length_1d)
            st_dynamic_scale.dst_addr = out_dynamic_scale.addr + dscale_addr
            dscale_addr += tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            inst_collect.add(st_dynamic_scale)
        

    return
def lower_tow_stage_for_parallel(op: Op, inst_collect: InstCollector, dynamic_scale: bool = False):
    input = op.get_inputs()[0]
    output = op.get_outputs()[0]
    ref = input
    if get_size_by_type(input.data_type) < get_size_by_type(output.data_type):
        ref = output
        
    max_size = get_max_banks_half(ref.shape[-1],ref.data_type,dynamic_scale)
    tiles = tensor_slice_plane_dim_merge(ref, max_size)
    weight = op.get_weights()[0]

    assert weight.shape[-1] == input.shape[-1]
    # only load a times for weight
    ld_weight = LDInst()
    last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
    if last_ins is not None and not isinstance(last_ins,LDInst):
        last_ins.release.append(PUType.LD)
        ld_weight.wait.append(get_inst_type(last_ins))
    ld_weight.length_1d = input.shape[-1] * get_size_by_type(weight.data_type)
    ld_weight.loop_1d = min(max_size,tiles[0].shape[-2]) # FIXME: is min(max_size,tiles[0].shape[-2]) right?
    ld_weight.src_1d_stride = 0
    ld_weight.src_addr = weight.addr
    ld_weight.dst_bank_id = 0
    ld_weight.dst_group_id = 2
    ld_weight.bank_addr_stride = get_bank_stride(ld_weight.length_1d)
    ld_weight.mode = LDMode.DDR2global
    inst_collect.add(ld_weight)
    addrs = [0, 2048]
    meta_addrs = [0, 256]
    dscale_addr = 0
    for index,tile in enumerate(tiles) :
        assert tile.addr is not None
        assert ref.addr is not None
        assert input.addr is not None
        tile_offset = (
            lambda t: (tile.addr - ref.addr)
            * get_size_by_type(t)
            / get_size_by_type(ref.data_type)
        )

        ld = LDInst()
        if(index == 0):
            last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
            if last_ins is not None and not isinstance(last_ins,LDInst):
                last_ins.release.append(PUType.LD)
                ld.wait.append(get_inst_type(last_ins))
        elif index >= 2:
            ld.wait.append(PUType.ST)
        ld.release.append(PUType.MISC)
        ld.length_1d = tile.shape[-1] * get_size_by_type(input.data_type)
        ld.loop_1d = tile.shape[-2]
        ld.dst_addr = addrs[index %2]
        ld.src_1d_stride = tile.shape[-1] * get_size_by_type(input.data_type)
        ld.src_addr = input.addr + int(tile_offset(input.data_type))
        ld.bank_addr_stride = get_bank_stride(ld.length_1d)
        ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
        inst_collect.add(ld)

        assert not( (output.data_type != DataType.float16) ^ dynamic_scale )
        
        misc = MISCInst()
        misc.wait.append(PUType.LD)
        if isinstance(op, Layernorm):
            misc.op = MiscOp.layernorm
        elif isinstance(op, RMSnorm):
            misc.op = MiscOp.rmsnorm
        misc.release.append(PUType.ST)
        misc.input_a_mode = MiscMode.fp16 if input.data_type == DataType.float16 else MiscMode.int8 
        misc.input_b_mode = MiscMode.fp16 if weight.data_type == DataType.float16 else MiscMode.int8 
        misc.output_mode = MiscMode.fp16 if output.data_type == DataType.float16 else MiscMode.int8 
        misc.in_a_bank_group = 0
        misc.in_a_bank_id = 0
        misc.in_a_start_addr = addrs[index %2]
        
        misc.in_b_bank_group = 2
        misc.in_b_bank_id = 0
        misc.in_b_start_addr = 0
        
        misc.out_bank_group = 1
        misc.out_bank_id = 0
        misc.out_start_addr = addrs[index %2]
        
        misc.meta_addr = meta_addrs[index %2]
        misc.dynamic_scale = dynamic_scale
        misc.K = tile.shape[-1]
        misc.batch = tile.shape[-2]
        misc.fK = int(np.float32(tile.shape[-1]).view(np.uint32))
        inst_collect.add(misc)

        st = STInst()
        st.wait.append(PUType.MISC)
        if not dynamic_scale:
            if index < len(tiles)-2:
                st.release.append(PUType.LD)
        st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
        st.length_1d = output.shape[-1] * get_size_by_type(output.data_type)
        st.loop_1d = tile.shape[-2]
        st.dst_1d_stride = output.shape[-1] * get_size_by_type(output.data_type)
        st.src_group_id = 1
        st.src_addr = addrs[index %2]
        st.bank_addr_stride = get_bank_stride(st.length_1d)
        st.dst_addr = output.addr + int(tile_offset(output.data_type))
        inst_collect.add(st)

        if dynamic_scale:
            out_dynamic_scale = op.get_outputs()[1]
            st_dynamic_scale = STInst()
            if index < len(tiles)-2:
                st_dynamic_scale.release.append(PUType.LD)
            st_dynamic_scale.mode = STMode.Meta2DDR
            st_dynamic_scale.length_1d = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.loop_1d = 1
            st_dynamic_scale.src_addr = meta_addrs[index %2]
            st_dynamic_scale.dst_1d_stride = tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            st_dynamic_scale.bank_addr_stride = get_bank_stride(st_dynamic_scale.length_1d)
            st_dynamic_scale.dst_addr = out_dynamic_scale.addr + dscale_addr
            dscale_addr += tile.shape[-2] * get_size_by_type(out_dynamic_scale.data_type)
            inst_collect.add(st_dynamic_scale)
    return


def get_align_size(size: int, align: int):
    return (size + align - 1) // align * align
def lower_transpose01(op: Transpose, inst_collect: InstCollector):
    input = op.get_inputs()[0]
    assert len(input.shape) == 4
    output = op.get_outputs()[0]
    assert input.shape[0] == output.shape[1]
    assert input.shape[1] == output.shape[0]
    max_size = get_max_banks(input.shape[-1],input.data_type,False)
    stride0 = math.prod(input.shape[1:])
    stride1 = math.prod(input.shape[2:])
    stride2 = input.shape[-1]
    output_stride0 = math.prod(output.shape[1:])
    output_stride1 = math.prod(output.shape[2:])
    output_stride2 = output.shape[-1]
    tile_num = math.ceil(input.shape[2] / max_size)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            kidx:int = 0
            for k in range(tile_num):
                loop_1d = get_tile_dim(input.shape[2], max_size, k, tile_num)
                ld = LDInst()
                last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
                if last_ins is not None:
                    last_ins.release.append(PUType.LD)
                    ld.wait.append(get_inst_type(last_ins))
                ld.release.append(PUType.ST)
                ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
                ld.length_1d = input.shape[-1] * get_size_by_type(input.data_type)
                ld.loop_1d = loop_1d
                ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
                ld.src_addr = input.addr + ((
                    i * stride0 + j * stride1 + kidx * stride2
                ) * get_size_by_type(input.data_type))
                ld.bank_addr_stride = get_bank_stride(ld.length_1d)
                inst_collect.add(ld)
                st = STInst()
                st.wait.append(PUType.LD)
                st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                st.length_1d = output.shape[-1] * get_size_by_type(output.data_type)
                st.loop_1d = loop_1d
                st.dst_1d_stride = output.shape[-1] * get_size_by_type(output.data_type)
                st.bank_addr_stride = get_bank_stride(st.length_1d)
                # st.dst_addr=output.addr + (i*stride1 + j*stride0 +k*stride2) *get_size_by_type(input.data_type)
                st.dst_addr = output.addr + ((
                    j * output_stride0 + i * output_stride1 + kidx * output_stride2
                ) * get_size_by_type(input.data_type))
                inst_collect.add(st)
                kidx = kidx + loop_1d
    return

def lower_transpose01_for_parallel(op: Transpose, inst_collect: InstCollector):
    input = op.get_inputs()[0]
    assert len(input.shape) == 4
    output = op.get_outputs()[0]
    assert input.shape[0] == output.shape[1]
    assert input.shape[1] == output.shape[0]
    max_size = get_max_banks(input.shape[-1],input.data_type,False)
    stride0 = math.prod(input.shape[1:])
    stride1 = math.prod(input.shape[2:])
    stride2 = input.shape[-1]
    output_stride0 = math.prod(output.shape[1:])
    output_stride1 = math.prod(output.shape[2:])
    output_stride2 = output.shape[-1]
    tile_num = math.ceil(input.shape[2] / max_size)
    group_id = [id for id in range(GROUP_NUM)]
    index : int = 0
    max_index = input.shape[0] * input.shape[1] * tile_num
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            kidx:int = 0
            for k in range(tile_num):
                loop_1d = get_tile_dim(input.shape[2], max_size, k, tile_num)
                ld = LDInst()
                if(index == 0):
                    last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
                    if (last_ins is not None) and (not isinstance(last_ins, LDInst)):
                        last_ins.release.append(PUType.LD)
                        ld.wait.append(get_inst_type(last_ins))
                elif index >=2:
                    ld.wait.append(PUType.ST)
                ld.release.append(PUType.ST)
                ld.length_1d = input.shape[-1] * get_size_by_type(input.data_type)
                ld.loop_1d = loop_1d
                ld.dst_group_id = group_id[index % GROUP_NUM]
                ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
                ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
                ld.src_addr = input.addr + ((
                    i * stride0 + j * stride1 + kidx * stride2
                ) * get_size_by_type(input.data_type))
                ld.bank_addr_stride = get_bank_stride(ld.length_1d)
                inst_collect.add(ld)
                st = STInst()
                st.wait.append(PUType.LD)
                if index < max_index - 2:
                    st.release.append(PUType.LD)
                st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                st.length_1d = output.shape[-1] * get_size_by_type(output.data_type)
                st.loop_1d = loop_1d
                st.dst_1d_stride = output.shape[-1] * get_size_by_type(output.data_type)
                st.src_group_id = group_id[index % GROUP_NUM]
                st.bank_addr_stride = get_bank_stride(st.length_1d)
                st.dst_addr = output.addr + ((
                    j * output_stride0 + i * output_stride1 + kidx * output_stride2
                ) * get_size_by_type(input.data_type))
                inst_collect.add(st)
                index += 1
                kidx = kidx + loop_1d
    return

def lower_transpose23_old(op: Transpose, inst_collect: InstCollector):
    input = op.get_input
    assert len(input.shape) == 4
    output = op.get_output
    msize = input.shape[2]
    ksize = input.shape[3]

    in_stride2 = math.prod(input.shape[2:])
    in_stride3 = input.shape[-1]
    output_stride2 = math.prod(output.shape[2:])
    output_stride3 = output.shape[-1]
    # rs only use one bank
    while (msize * ksize * get_size_by_type(input.data_type)) > BANK_SIZE:
        msize //= 2
        ksize //= 2
    mtnum = math.ceil(input.shape[2] / msize)
    ktnum = math.ceil(input.shape[3] / ksize)
    bank_id = [id for id in range(GROUP_SIZE)]
    group_id = [i for i in range(GROUP_NUM)]
    index : int = 0
    N = input.shape[0] * input.shape[1]
    max_index = N * mtnum * ktnum
    for i in range(N):
        for m in range(mtnum):
            ms = get_tile_dim(input.shape[2], msize, m, mtnum)
            for k in range(ktnum):
                ks = get_tile_dim(input.shape[3], ksize, k, ktnum)
                ld = LDInst()
                if(index == 0):
                    last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
                    if last_ins is not None:
                        last_ins.release.append(PUType.LD)
                        ld.wait.append(get_inst_type(last_ins))
                elif index >= 2:
                    ld.wait.append(PUType.ST)
                ld.release.append(PUType.RS)
                ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
                ld.length_1d = ks * get_size_by_type(input.data_type)
                ld.loop_1d = ms
                ld.loop_direction = LoopDir.in_bank
                ld.dst_bank_id = bank_id[index % GROUP_SIZE]
                ld.dst_group_id = group_id[index % GROUP_NUM]
                ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
                ld.src_addr = input.addr + (
                    i * in_stride2 + m * msize * in_stride3 + k * ksize
                ) * get_size_by_type(input.data_type) 
                ld.bank_addr_stride = get_bank_stride(ld.length_1d)
                inst_collect.add(ld)
                rs = RSInst()
                rs.wait.append(PUType.LD)

                rs.release.append(PUType.ST)
                rs.src_bank_id = bank_id[index % GROUP_SIZE]
                rs.dst_bank_id = bank_id[index % GROUP_SIZE]
                rs.src_bank_group_id = group_id[index % GROUP_NUM]
                rs.dst_bank_group_id = group_id[(index+1) % GROUP_NUM]
                rs.M = ms
                rs.K = ks
                rs.data_type = RSDataType.int8 if get_size_by_type(input.data_type) == 1 else RSDataType.int16
                inst_collect.add(rs)
                st = STInst()
                if index < max_index-2:
                    st.release.append(PUType.LD)
                st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                st.wait.append(PUType.RS)
                st.loop_direction = LoopDir.in_bank
                st.src_bank_id = bank_id[index % GROUP_SIZE]
                st.src_group_id = group_id[(index+1) % GROUP_NUM]
                st.length_1d = ms * get_size_by_type(output.data_type)
                st.loop_1d = ks
                st.dst_1d_stride = output.shape[-1] * get_size_by_type(output.data_type)
                st.bank_addr_stride = get_bank_stride(st.length_1d)
                st.dst_addr = output.addr + (i * output_stride2 + k * ksize * output_stride3  + m * msize ) * get_size_by_type(output.data_type) 
                inst_collect.add(st)
                index += 1
    return
def lower_transpose23(op: Op, inst_collect: InstCollector):
    assert len(op.get_inputs()) == 1
    input = op.get_inputs()[0]
    output = op.get_outputs()[0]
    assert input.shape[-1] == output.shape[-2]
    assert input.shape[-2] == output.shape[-1]

    in_stride2 = math.prod(input.shape[2:])
    in_stride3 = input.shape[-1]
    
    output_stride2 = math.prod(output.shape[2:])
    output_stride3 = output.shape[-1]
    addrs = [0, 2048]
    max_size0 = get_max_banks_half(input.shape[-1],input.data_type,False)
    max_size1 = get_max_banks_half(input.shape[-2],input.data_type,False)
    max_size = min(max_size0, max_size1)
    mtiles = math.ceil(input.shape[-2] / max_size)
    N = math.prod(input.shape[:-2])
    max_index = N * mtiles
    index = 0
    for i in range(N):
        for m in range(mtiles):
            ms = get_tile_dim(input.shape[-2], max_size, m, mtiles)
            ld = LDInst()
            if(index == 0):
                last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
                if last_ins is not None:
                    last_ins.release.append(PUType.LD)
                    ld.wait.append(get_inst_type(last_ins))
            elif index >= 2:
                ld.wait.append(PUType.ST)
            ld.release.append(PUType.RS)
            ld.length_1d = input.shape[-1] * get_size_by_type(input.data_type)
            ld.loop_1d = ms
            ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
            ld.src_addr =input.addr + (i*in_stride2 + m*max_size*in_stride3) * get_size_by_type(input.data_type)
            ld.bank_addr_stride = get_bank_stride(ld.length_1d)
            ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
            ld.dst_addr = addrs[index %2]
            ld.dst_group_id = 0
            inst_collect.add(ld)
            assert output.data_type == DataType.float16
            rs = RSInst()
            rs.wait.append(PUType.LD)
            rs.release.append(PUType.ST)
            rs.src_addr = addrs[index %2]
            rs.src_bank_group_id = 0
            rs.dst_addr = addrs[index %2]
            rs.dst_bank_group_id = 1
            rs.M = ms
            rs.K = input.shape[-1]
            rs.data_type = RSDataType.int16
            inst_collect.add(rs)

            st = STInst()
            st.wait.append(PUType.RS)
            if index < max_index-2:
                st.release.append(PUType.LD)
            st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
            st.length_1d = ms * get_size_by_type(output.data_type)
            st.loop_1d = input.shape[-1]
            st.dst_1d_stride = ms * get_size_by_type(output.data_type)
            st.src_group_id = 1
            st.src_addr = addrs[index %2]
            st.bank_addr_stride = get_bank_stride(st.length_1d)
            st.dst_addr = output.addr + (i * output_stride2 + m * max_size) * get_size_by_type(input.data_type)
            inst_collect.add(st)
            index += 1
    return
def lower_transpose23_serial(op: Op, inst_collect: InstCollector):
    assert len(op.get_inputs()) == 1
    input = op.get_inputs()[0]
    output = op.get_outputs()[0]
    assert input.shape[-1] == output.shape[-2]
    assert input.shape[-2] == output.shape[-1]

    in_stride2 = math.prod(input.shape[2:])
    in_stride3 = input.shape[-1]
    
    output_stride2 = math.prod(output.shape[2:])
    output_stride3 = output.shape[-1]
    max_size0 = get_max_banks(input.shape[-1],input.data_type,False)
    max_size1 = get_max_banks(input.shape[-2],input.data_type,False)
    max_size = min(max_size0, max_size1)
    mtiles = math.ceil(input.shape[-2] / max_size)
    N = math.prod(input.shape[:-2])
    max_index = N * mtiles
    index = 0
    for i in range(N):
        for m in range(mtiles):
            ms = get_tile_dim(input.shape[-2], max_size, m, mtiles)
            ld = LDInst()
            last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
            if last_ins is not None:
                last_ins.release.append(PUType.LD)
                ld.wait.append(get_inst_type(last_ins))
            ld.release.append(PUType.RS)
            ld.length_1d = input.shape[-1] * get_size_by_type(input.data_type)
            ld.loop_1d = ms
            ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
            ld.src_addr =input.addr + (i*in_stride2 + m*max_size*in_stride3) * get_size_by_type(input.data_type)
            ld.bank_addr_stride = get_bank_stride(ld.length_1d)
            ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
            ld.dst_group_id = 0
            inst_collect.add(ld)
            assert output.data_type == DataType.float16
            rs = RSInst()
            rs.wait.append(PUType.LD)
            rs.release.append(PUType.ST)
            rs.src_bank_group_id = 0
            rs.dst_bank_group_id = 1
            rs.M = ms
            rs.K = input.shape[-1]
            rs.data_type = RSDataType.int16
            inst_collect.add(rs)

            st = STInst()
            st.wait.append(PUType.RS)
            st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
            st.length_1d = ms * get_size_by_type(output.data_type)
            st.loop_1d = input.shape[-1]
            st.dst_1d_stride = ms * get_size_by_type(output.data_type)
            st.src_group_id = 1
            st.bank_addr_stride = get_bank_stride(st.length_1d)
            st.dst_addr = output.addr + (i * output_stride2 + m * max_size )* get_size_by_type(input.data_type)
            inst_collect.add(st)
            index += 1
    return
def lower_split(op: Split, inst_collect: InstCollector):
    split_num, split_dim = op.get_split_size
    input = op.get_input
    outputs = op.get_output
    assert split_dim == len(input.shape) - 2
    # shape to 3dim
    # before dim
    shape = [1 if split_dim == 0 else math.prod(input.align_shape[:split_dim])]
    # dim
    shape.append(input.shape[split_dim])
    # after dim
    shape.append(
        1
        if split_dim >= (len(input.align_shape) - 1)
        else math.prod(input.align_shape[split_dim + 1 :])
    )
    split_size = math.ceil(input.shape[split_dim] / split_num)  # ?
    assert shape[-1] <= BANK_SIZE
    assert split_num == len(op.get_output)
    assert split_size == 1
    # assert input.addr >= 24<<30
    
    for j in range(split_num):
        for i in range(shape[0]):
            real_splite_size = get_tile_dim(shape[-2], split_size, j, split_num)
            for k in range(real_splite_size):
                loop_1d = 1
                ld = LDInst()
                last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
                if last_ins is not None:
                    last_ins.release.append(PUType.LD)
                    ld.wait.append(get_inst_type(last_ins))
                ld.release.append(PUType.ST)
                ld.mode = LDMode.DDR2global
                ld.length_1d = shape[2] * get_size_by_type(input.data_type)
                ld.loop_1d = loop_1d
                ld.src_1d_stride = shape[2] * get_size_by_type(input.data_type)
                ld.src_addr = input.addr + (
                    i * math.prod(shape[-2:])
                    + j * split_size * shape[2]
                    + k * shape[2]
                ) * get_size_by_type(input.data_type)
                ld.bank_addr_stride = get_bank_stride(ld.length_1d)
                inst_collect.add(ld)
                st = STInst()
                st.mode = STMode.Global2DDR if outputs[j].reside_ddr else STMode.Global2HBM
                st.wait.append(PUType.LD)
                st.length_1d = shape[2] * get_size_by_type(outputs[j].data_type)
                st.loop_1d = loop_1d
                st.dst_1d_stride = shape[2] * get_size_by_type(outputs[j].data_type)
                st.bank_addr_stride = get_bank_stride(st.length_1d)
                st.dst_addr = outputs[j].addr + (i* split_size * shape[2]) * get_size_by_type(outputs[j].data_type)
                inst_collect.add(st)

    return
G=1073741824
def lower_split_for_parallel(op: Split, inst_collect: InstCollector):
    split_num, split_dim = op.get_split_size
    input = op.get_input
    outputs = op.get_output
    # shape to 3dim
    # before dim
    shape: list[int] = []
    shape.append(1 if split_dim == 0 else math.prod(list(input.shape[:split_dim])))
    # dim
    shape.append(input.shape[split_dim])
    # after dim
    shape.append(
        1
        if split_dim >= (len(input.shape) - 1)
        else math.prod(input.shape[split_dim + 1 :])
    )
    split_size = math.ceil(input.shape[split_dim] / split_num)  # ?
    assert shape[-1] <= BANK_SIZE
    assert split_num == len(op.get_output)
    group_id = [0, 1, 2]
    index : int = 0
    max_index = shape[0] * split_num * split_size
    for i in range(shape[0]):
        for j in range(split_num):
            real_splite_size = get_tile_dim(shape[-2], split_size, j, split_num)
            for k in range(real_splite_size):
                loop_1d = 1
                ld = LDInst()
                if(index == 0):
                    last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
                    if (last_ins is not None) and (not isinstance(last_ins, LDInst)):
                        last_ins.release.append(PUType.LD)
                        ld.wait.append(get_inst_type(last_ins))
                elif index >=2:
                    ld.wait.append(PUType.ST)
                ld.release.append(PUType.ST)
                ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
                ld.dst_group_id = group_id[index % 3]
                ld.length_1d = shape[2] * get_size_by_type(input.data_type)
                ld.loop_1d = loop_1d
                ld.src_1d_stride = shape[2] * get_size_by_type(input.data_type)
                ld.src_addr = input.addr + (
                    i * math.prod(shape[-2:])
                    + j * split_size * shape[2]
                    + k * shape[2]
                ) * get_size_by_type(input.data_type)
                ld.bank_addr_stride = get_bank_stride(ld.length_1d)
                inst_collect.add(ld)
                st = STInst()
                st.mode = STMode.Global2DDR if outputs[j].reside_ddr else STMode.Global2HBM
                st.wait.append(PUType.LD)
                if index < max_index-2:
                    st.release.append(PUType.LD)
                st.length_1d = shape[2] * get_size_by_type(outputs[j].data_type)
                st.loop_1d = loop_1d
                st.src_group_id = group_id[index % 3]
                st.dst_1d_stride = shape[2] * get_size_by_type(outputs[j].data_type)
                st.bank_addr_stride = get_bank_stride(st.length_1d)
                st.dst_addr = outputs[j].addr + ((i * split_size * shape[2] )) * get_size_by_type(outputs[j].data_type)
                inst_collect.add(st)
                index += 1
    return

    
def lower_fakeload(op, inst_collect):
    input = op.get_input
    stride = op.get_stride
    ld = LDInst()
    ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
    ld.length_1d = input.shape[-1]
    ld.loop_1d = math.prod(input.shape[-4:-1])
    ld.src_1d_stride = stride
    ld.src_addr = input.addr
    ld.bank_addr_stride = 256
    inst_collect.add(ld)
def lower_sync(op: Sync, inst_collect: InstCollector):
    sync = SYSCInst()
    sync.op = SysOp.sync
    last_inst = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
    if last_inst is not None:
        last_inst.release.append(PUType.SYS)
        sync.wait.append(get_inst_type(last_inst))
    inst_collect.add(sync)

def lower_transpose12(op: Transpose, inst_collect: InstCollector):
    assert len(op.get_input.shape) == 4
    input = op.get_inputs()[0]
    assert len(input.shape) == 4
    output = op.get_outputs()[0]
    assert input.shape[1] == output.shape[2]
    assert input.shape[2] == output.shape[1]
    max_size = get_max_banks(input.shape[-1],input.data_type,False)
    stride0 = math.prod( input.shape[1:])
    stride1 = math.prod( input.shape[2:])
    stride2 =  input.shape[-1]
    output_stride0 = math.prod(output.shape[1:])
    output_stride1 = math.prod(output.shape[2:])
    output_stride2 = output.shape[-1]
    tile_num = math.ceil(input.shape[2] / max_size)
    group_id = [id for id in range(GROUP_NUM)]
    index : int = 0
    max_index = input.shape[0] * input.shape[1] * tile_num
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            kidx:int = 0
            for k in range(tile_num):
                loop_1d = get_tile_dim(input.shape[2], max_size, k, tile_num)
                ld = LDInst()
                if(index == 0):
                    last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
                    if (last_ins is not None) and (not isinstance(last_ins, LDInst)):
                        last_ins.release.append(PUType.LD)
                        ld.wait.append(get_inst_type(last_ins))
                elif index >=2:
                    ld.wait.append(PUType.ST)
                ld.release.append(PUType.ST)
                ld.length_1d = input.shape[-1] * get_size_by_type(input.data_type)
                ld.loop_1d = loop_1d
                ld.dst_group_id = group_id[index % GROUP_NUM]
                ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
                ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
                ld.src_addr = input.addr + ((
                    i * stride0 + j * stride1 + kidx * stride2
                ) * get_size_by_type(input.data_type))
                ld.bank_addr_stride = get_bank_stride(ld.length_1d)
                inst_collect.add(ld)
                st = STInst()
                st.wait.append(PUType.LD)
                if index < max_index-2:
                    st.release.append(PUType.LD)
                st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
                st.length_1d = output.shape[-1] * get_size_by_type(output.data_type)
                st.loop_1d = loop_1d
                st.dst_1d_stride = output_stride1 * get_size_by_type(output.data_type)
                st.src_group_id = group_id[index % GROUP_NUM]
                st.bank_addr_stride = get_bank_stride(st.length_1d)
                st.dst_addr = output.addr + ((
                    i * output_stride0 + kidx * output_stride1 + j*output_stride2 
                ) * get_size_by_type(input.data_type)) 
                inst_collect.add(st)
                index += 1
                kidx = kidx + loop_1d
def lower_copy(op: Copy, inst_collect: InstCollector):
    assert len(op.get_inputs()) == 1
    input = op.get_inputs()[0]
    output = op.get_outputs()[0]
    assert input.shape == output.shape
    ref = input
    if get_size_by_type(input.data_type) < get_size_by_type(output.data_type):
        ref = output

    max_size = get_max_banks(ref.shape[-1],ref.data_type,False)
    tiles = tensor_slice_plane_dim_merge(ref, max_size)
    for index,tile in enumerate(tiles):
        assert tile.addr is not None
        assert ref.addr is not None
        tile_offset = (
            lambda t: (tile.addr - ref.addr)
            * get_size_by_type(t)
            / get_size_by_type(ref.data_type)
        )

        ld = LDInst()
        last_ins = inst_collect.get_insts()[-1] if len(inst_collect.get_insts()) > 0 else  None
        if last_ins is not None:
            last_ins.release.append(PUType.LD)
            ld.wait.append(get_inst_type(last_ins))
        ld.release.append(PUType.ST)
        ld.length_1d = tile.shape[-1] * get_size_by_type(input.data_type)
        ld.loop_1d = tile.shape[-2]
        ld.src_1d_stride = input.shape[-1] * get_size_by_type(input.data_type)
        assert input.addr is not None
        ld.src_addr = input.addr + int(tile_offset(input.data_type))
        ld.bank_addr_stride = get_bank_stride(ld.length_1d)
        ld.mode = LDMode.DDR2global if input.reside_ddr else LDMode.HBM2Global
        ld.dst_group_id = 0
        inst_collect.add(ld)
        st = STInst()
        st.wait.append(PUType.LD)
        st.mode = STMode.Global2DDR if output.reside_ddr else STMode.Global2HBM
        st.length_1d = tile.shape[-1] * get_size_by_type(output.data_type)
        st.loop_1d = tile.shape[-2]
        st.dst_1d_stride = tile.shape[-1] * get_size_by_type(output.data_type)
        st.src_group_id = 0
        st.bank_addr_stride = get_bank_stride(st.length_1d)
        st.dst_addr = output.addr + int(tile_offset(output.data_type))
        inst_collect.add(st)

    return