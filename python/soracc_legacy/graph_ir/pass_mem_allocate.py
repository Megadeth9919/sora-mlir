from .graph_ir import *
from inst import *
from collections import namedtuple
from typing import NamedTuple
import math
import json


class InforSection(NamedTuple):
    name: str = " "
    start: int = 0
    size: int = 0
    type: str = " "
    file_path: str = " "


CCTarget = Enum("CCTarget", ("runtime", "verify"))


def getStrtype(data_type: DataType) -> str:
    match data_type:
        case DataType.float16:
            return "fp16"
        case DataType.int8:
            return "int8"
        case DataType.int16:
            return "int16"
        case DataType.float32:
            return "float32"
        case _:
            assert False


# @dataclass
class RtInfo:
    def __init__(self) -> None:
        self.inputs: list[InforSection] = []
        self.outputs: list[InforSection] = []
        # self.output_dscales: list[InforSection] = []
        self.weights: list[InforSection] = []
        self.consts: list[InforSection] = []
        self.intermediate: list[InforSection] = []
        self.cached: list[InforSection] = []
        self.dscales: list[InforSection] = []
    def clear(self):
        self.inputs.clear()
        self.outputs.clear()
        self.weights.clear()
        self.consts.clear()
        self.intermediate.clear()
        self.cached.clear()
        self.dscales.clear()
        
    def create_section_info(self, sections: list[InforSection]) -> list:
        """Convert sections to info dictionaries."""
        info_list = []
        for section in sections:
            if section.size == 0:
                continue
            info = {
                "name": section.name,
                "start": section.start,
                "size": section.size,
                "type": section.type,
            }
            if section.file_path:
                info["file_path"] = self.legalize_file_path(section.file_path)
            info_list.append(info)
        return info_list

    def legalize_file_path(self, file_path: str) -> str:
        if file_path:
            return "./" + file_path + ".bin"
        return file_path

    def misc_serilize(self) -> dict:
        ret = {
            "inputs_info": self.create_section_info(self.inputs),
            "outputs_info": self.create_section_info(self.outputs),
            # "output_dscales_info": self.create_section_info(self.output_dscales),
            "weights_info": self.create_section_info(self.weights),
            "consts_info": self.create_section_info(self.consts),
            "cached_info": self.create_section_info(self.cached),
            "dscales_info": self.create_section_info(self.cached),
            "intermediate_info": self.create_section_info(self.intermediate),
        }
        return ret


def align_64(addr: int) -> int:
    if addr % 64 != 0:
        return (addr + 63) // 64 * 64
    return addr


def get_aligned_size(shape: Shape, size: int) -> int:
    if len(shape) == 1:
        return math.prod(shape)
    else:
        align_size = 8 * size
        if shape[-2] % align_size == 0:
            return math.prod(shape[:-2]) * shape[-2] // 8 * shape[-1]
        else:
            return (
                math.prod(shape[:-2])
                * ((shape[-2] + align_size - shape[-2] % align_size) // 8)
                * shape[-1]
            )
rt_info = RtInfo()
def ddr_address_alloc(g: StaticGraph, mode: CCTarget = CCTarget.verify) -> RtInfo:
    global rt_info
    addr = 0
    # allocate DDR memory for inputs
    addr = align_64(addr)
    prev = addr
    for t in sorted(g.get_inputs(), key=lambda t: t.get_id()):
        # assert t.addr is None
        t.addr = addr
        addr += t.bytes_size()
        assert addr < (0x1 << 35)  # TODO?
        inputs_info = InforSection(
            name=t.name,
            start=prev,
            size=addr - prev,
            type=getStrtype(t.data_type),
            file_path=t.name,
        )
        addr = align_64(addr)
        prev = addr
        rt_info.inputs.append(inputs_info)

    # allocate DDR memory for outputs
    addr = align_64(addr)
    prev = addr
    for t in sorted(g.get_outputs(), key=lambda t: t.get_id()):
        # assert t.addr is None
        t.addr = addr
        addr += t.bytes_size()
        assert addr < (0x1 << 35)
        outputs_info = InforSection(
            name=t.name,
            start=prev,
            size=addr - prev,
            type=getStrtype(t.data_type),
            file_path=t.name,
        )
        addr = align_64(addr)
        prev = addr
        rt_info.outputs.append(outputs_info)

    # allocate DDR memory for const tensor
    addr = align_64(addr)
    prev = addr
    for t in g.get_const():
        # assert t.addr is None
        t.addr = addr
        addr += t.bytes_size()
        assert addr < (0x1 << 35)
    consts_info = InforSection(
        name="consts", start=prev, size=addr - prev, file_path="consts"
    )
    rt_info.consts.append(consts_info)
    
    addr = align_64(addr)
    prev = addr
    for t in g.get_cached():
        assert t.addr is None
        t.addr = addr
        addr += t.bytes_size()
        assert addr < (0x1 << 35)
    cached_info = InforSection(
        name="cached", start=prev, size=addr - prev, file_path="cached"
    )
    rt_info.cached.append(cached_info)

    # allocate DDR memory for cached Intermediate tensor
    addr = align_64(addr)
    prev = addr
    act_hbm_addr = 24<<30
    act_hbm_prev = act_hbm_addr
    intermediates = g.get_intermediate()
    for op in g.get_ops():
        if isinstance(op, View):
            assert op.get_inputs()[0].addr is not None
            # assert op.get_outputs()[0].addr is None
            op.get_outputs()[0].addr = op.get_inputs()[0].addr
            # intermediate_info = InforSection(
            #             name=op.get_outputs()[0].name,
            #             start=op.get_outputs()[0].addr,
            #             size=op.get_outputs()[0].bytes_size(),
            #             type=getStrtype(op.get_outputs()[0].data_type),
            #             file_path=op.get_outputs()[0].name,
            #         )
            # rt_info.intermediate.append(intermediate_info)
        else :
            for output in op.get_outputs():
                if output in intermediates:
                    if not output.reside_ddr:
                        output.addr = act_hbm_addr
                        act_hbm_addr += output.bytes_size()
                        intermediate_info = InforSection(
                            name=output.name,
                            start=act_hbm_prev,
                            size=act_hbm_addr - act_hbm_prev,
                            type=getStrtype(output.data_type),
                            file_path=output.name,
                        )
                        rt_info.intermediate.append(intermediate_info)
                        act_hbm_prev = act_hbm_addr
                    else :
                        assert output.addr is None
                        output.addr = addr
                        addr += output.bytes_size()
                        intermediate_info = InforSection(
                            name=output.name,
                            start=prev,
                            size=addr - prev,
                            type=getStrtype(output.data_type),
                            file_path=output.name,
                        )
                        rt_info.intermediate.append(intermediate_info)
                        prev = addr

    # allocate HBM memory for weights
    hbm_addr = 256<<20
    hbm_addr = align_64(hbm_addr)
    prev = hbm_addr
    for w in sorted(g.get_weights(), key=lambda t: t.get_id()):
        if w in g.get_const():
            continue
        # assert w.addr is None
        w.addr = hbm_addr
        hbm_addr += get_aligned_size(w.shape, 4) * get_size_by_type(w.data_type)
        assert hbm_addr < (0x1 << 35)
    weights_info = InforSection(
        name="weights",
        start=prev,
        size=hbm_addr - prev,
        file_path="weights",
    )
    rt_info.weights.append(weights_info)

    return rt_info

class Node:
    def __init__(self, addr: int, size: int) -> None:
        self.addr = addr
        self.size = size


class DynamicMemoryPool:
    def __init__(self, start: int, size: int, reisde_ddr:bool=True) -> None:
        self.start = start
        self.size = size
        self.Occupy: list[Node] = []
        self.Free: list[Node] = [Node(start, size)]
        self.max: int = 0
        self.DDR :bool = reisde_ddr

    def allocate(self, size: int) -> Optional[int]:
        for node in self.Free:
            if node.size >= size:
                node.size -= size
                oc = Node(node.addr, size)
                print("alocate:", "DDR" if self.DDR else "HBM",node.addr, size)
                self.Occupy.append(oc)
                node.addr += size
                self.max = max(node.addr, self.max)
                return oc.addr
        raise ValueError(f"Memory not enough : {self.max/(1024*1024) }")
        return None

    def free(self, addr: int) -> None:
        for node in self.Occupy:
            if node.addr == addr:
                print("free:","DDR" if self.DDR else "HBM", node.addr, node.size)
                self.Free.append(node)
                self.Occupy.remove(node)
                self.merge()
                break
        else:
            raise ValueError(f"free {addr} not found")

    def merge(self):
        self.Free.sort(key=lambda x: x.addr)
        if len(self.Free) < 2:
            return
        for i in reversed(range(len(self.Free) - 1)):
            if self.Free[i].addr + self.Free[i].size == self.Free[i + 1].addr:
                self.Free[i].size += self.Free[i + 1].size
                # print("merge:", self.Free[i].addr, self.Free[i + 1].addr)
                self.Free.remove(self.Free[i + 1])

    def getMaxSize(self) -> int:
        return self.max - self.start


class StaticMemoryPool:
    def __init__(self, start: int, size: int) -> None:
        self.start = start
        self.offset = 0
        self.size = size

    def allocate(self, size: int) -> Optional[int]:
        ret = self.offset
        self.offset += size
        assert self.offset <= self.size
        return self.start + ret

    def get_size(self) -> int:
        return self.offset


class TLife:
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

tensor_map=dict()
def get_org_tensor(t: Tensor | TensorView) -> Tensor:
    ret =t
    while(ret in tensor_map):
        ret = tensor_map[ret]
    return ret

def get_tensor_life(g: StaticGraph) -> dict[Tensor | TensorView, TLife]:
    ret = dict()
    global tensor_map
    for idx, op in enumerate(g.get_ops()):
        outputs = op.get_outputs()
        if isinstance(op,View):
            tensor_map[op.get_outputs()[0]] = op.get_inputs()[0]
            continue
        for out in outputs:
            if out not in ret:
                ret[out] = TLife(idx, 0)
        inputs = op.get_inputs()
        for input in inputs:
            org = get_org_tensor(input)
            if org in ret:
                ret[org].end = idx
                print(org.get_id(), ret[org].start, ret[org].end ,org.name, org.name)

    return ret


class AddrTable:
    def __init__(self) -> None:
        self.table: dict[str, int] = dict()

    def add(self, name: str, addr: int):
        if name in self.table:
            raise ValueError("tensor already exists", name)
        else:
            self.table[name] = addr
    def has(self, name: str) -> bool:
        return name in self.table
    def get(self, name: str) -> int:
        return self.table[name]

    def clear(self):
        self.table.clear()


addr_table = AddrTable()


def getAddrTable() -> AddrTable:
    return addr_table

def graph_mem_allocate(g: StaticGraph, mode: CCTarget = CCTarget.verify, static_allocate: bool = True) -> RtInfo:
    global rt_info
    rt_info.clear()
    if not static_allocate:
        addr_table.clear()
        # WeightPool = StaticMemoryPool(0, 192 << 20)  # TODO
        # InputPool = StaticMemoryPool(192 << 20, 64 << 20)
        # OutPutPool = StaticMemoryPool(256 << 20, 64 << 20)
        # CachePool = StaticMemoryPool(320 << 20, 256 << 20)
        # IntermediatePool = DynamicMemoryPool(570 << 20, 448 << 20)
        # ConstPool = StaticMemoryPool(24 << 30, 512 << 20)
        # DscalePool = StaticMemoryPool((24 << 30) + (512 << 20), 512 << 20)
        # OutDscalePool = StaticMemoryPool((24 << 30) + (1024 << 20), 512 << 20)

        WeightPool = StaticMemoryPool(0, 192 << 20)  # TODO
        InputPool = StaticMemoryPool(0, 64 << 20)
        OutPutPool = StaticMemoryPool(64 << 20, 64 << 20)
        ConstPool = StaticMemoryPool(128 << 20, 256 << 20)
        # DscalePool = StaticMemoryPool(640 << 20, 128 << 20)
        # OutDscalePool = StaticMemoryPool(768 << 20, 128 << 20)
        
        CachePool = StaticMemoryPool(384 << 20, 6000 << 20)
        IntermediatePool = DynamicMemoryPool(6384 << 20, 8000 << 20)
        IntermediatePoolHBM = DynamicMemoryPool(24 << 30, 8000 << 20, reisde_ddr=False)
        lifes = get_tensor_life(g)

        g_inputs = sorted(g.get_inputs(), key=lambda t: t.get_id())
        for input in g_inputs:
            assert input.addr is None
            addr = InputPool.allocate(input.bytes_size())
            assert addr is not None
            addr_table.add(input.name, addr)
            input.addr = addr
            if mode == CCTarget.runtime:
                rt_info.inputs.append(
                    InforSection(
                        name=input.name,
                        start=addr,
                        size=input.bytes_size(),
                        type=getStrtype(input.data_type),
                        file_path=input.name,
                    )
                )

        g_outputs = sorted(g.get_outputs(), key=lambda t: t.get_id())
        for output in g_outputs:
            assert output.addr is None
            addr = OutPutPool.allocate(output.bytes_size())
            assert addr is not None
            addr_table.add(output.name, addr)
            output.addr = addr
            rt_info.outputs.append(
                InforSection(
                    name=output.name,
                    start=addr,
                    size=output.bytes_size(),
                    file_path=output.name,
                    type=getStrtype(output.data_type),
                )
            )

        g_weights = sorted(g.get_weights(), key=lambda t: t.get_id())
        for weight in g_weights:
            assert weight.addr is None
            if weight in g.get_const():
                continue
            addr = WeightPool.allocate(
                get_aligned_size(weight.shape, 4) * get_size_by_type(weight.data_type)
            )
            assert addr is not None
            weight.addr = addr
            addr_table.add(weight.name, addr)

        g_const = g.get_const()
        for cst in g_const:
            assert cst.addr is None
            addr = ConstPool.allocate(cst.bytes_size())
            assert addr is not None
            addr_table.add(cst.name, addr)
            cst.addr = addr

        # g_dscale = g.get_dscale()
        # for dscale in g_dscale:
        #     assert dscale.addr is None
        #     addr = DscalePool.allocate(cst.bytes_size())
        #     assert addr is not None
        #     addr_table.add(dscale.name, addr)
        #     dscale.addr = addr

        # g_output_dscale = g.get_dscale_outputs()
        # for output_dscale in g_output_dscale:
        #     assert output_dscale.addr is None
        #     addr = OutDscalePool.allocate(cst.bytes_size())
        #     assert addr is not None
        #     addr_table.add(output_dscale.name, addr)
        #     output_dscale.addr = addr

        for cached in sorted(g.get_cached(), key=lambda t: t.get_id()):
            assert cached.addr is None
            addr = CachePool.allocate(
                cached.bytes_size()
            )
            assert addr is not None
            addr_table.add(cached.name, addr)
            cached.addr = addr

        for idx, op in enumerate(g.get_ops()):
            outputs = op.get_outputs()
            for out in outputs:
                if out in g_inputs:
                    continue
                elif out in g_outputs:
                    continue
                elif out in g_weights:
                    continue
                elif out in g_const:
                    continue
                # elif out in g_dscale:
                #     continue
                elif out in g.get_cached():
                    continue
                else:
                    if isinstance(out.get_def(), View):
                        assert out.addr is None
                        assert len(out.get_def().get_inputs()) == 1
                        assert out.get_def().get_inputs()[0].addr is not None
                        out.addr = out.get_def().get_inputs()[0].addr
                        assert out.reside_ddr == out.get_def().get_inputs()[0].reside_ddr
                        out.reside_ddr = out.get_def().get_inputs()[0].reside_ddr
                    else:
                        assert out.addr is None
                        if out.reside_ddr:
                            addr = IntermediatePool.allocate(out.bytes_size())
                        else:
                            addr = IntermediatePoolHBM.allocate(out.bytes_size())
                        print(f"allocate {out.get_id()} at {idx} {out.name} {out.bytes_size()}")
                        assert addr is not None
                        addr_table.add(out.name, addr)
                        out.addr = addr
                        intermediate_info = InforSection(
                            name=out.name,
                            start=out.addr,
                            size=out.bytes_size(),
                            type=getStrtype(out.data_type),
                            file_path=out.name,
                        )
                        rt_info.intermediate.append(intermediate_info)
            for input in op.get_inputs():
                org = get_org_tensor(input)
                if input in g_const:
                    continue
                elif input in g_weights:
                    continue
                elif input in g_inputs:
                    continue
                elif input in g_outputs:
                    continue
                elif input in g.get_cached():
                    continue
                elif isinstance(op, View):
                    continue
                elif org in lifes:
                    if idx == lifes[org].end :
                        assert input.addr is not None
                        if org.reside_ddr :
                            assert (org.addr >= IntermediatePool.start and org.addr < IntermediatePool.start + IntermediatePool.size)
                            IntermediatePool.free(org.addr)
                        else:
                            assert (org.addr >= IntermediatePoolHBM.start and org.addr < IntermediatePoolHBM.start + IntermediatePoolHBM.size)
                            IntermediatePoolHBM.free(org.addr)
                        print(f"free {org.get_id()} at {idx} {org.name} { org.reside_ddr }")
                else:
                    print("skip:",input.name)
        # with open("/home/fpga5/sora-case/addr_table.txt", "w") as f:
        #     for key, value in addr_table.table.items():
        #         f.write(f"{key}: {value}\n")

        weights_info = InforSection(
            name="weights",
            start=WeightPool.start,
            size=WeightPool.offset,
            file_path="weights",
        )
        rt_info.weights.append(weights_info)

        # if mode == CCTarget.verify:
        #     rt_info.inputs.append(
        #         IORtSection("inputs", InputPool.start, InputPool.offset, "inputs")
        #     )
        # outputs_info = InforSection(
        #     "outputs", OutPutPool.start, OutPutPool.offset, "outputs"
        # )
        # rt_info.outputs.append(outputs_info)

        cached_info = InforSection(
            name="cached", start=CachePool.start, size=CachePool.offset, file_path="cached"
        )
        rt_info.cached.append(cached_info)

        consts_info = InforSection(
            name="consts", start=ConstPool.start, size=ConstPool.offset, file_path="consts"
        )
        rt_info.consts.append(consts_info)

        # dscales_info = InforSection(
        #     name="dscale",
        #     start=DscalePool.start,
        #     size=DscalePool.offset,
        #     file_path="dscale",
        # )
        # rt_info.dscales.append(dscales_info)

        # out_dscales_info = InforSection(
        #     name="outdscale",
        #     start=OutDscalePool.start,
        #     size=OutDscalePool.offset,
        #     file_path="outdscale",
        # )
        # rt_info.output_dscales.append(out_dscales_info)

        print("weights size:", WeightPool.offset)
        print("inputs size:", InputPool.offset)
        print("outputs size:", OutPutPool.offset)
        print("consts size:", ConstPool.offset)
        print("cached size:", CachePool.offset)
        # print("dscales size:", DscalePool.offset)
        # print("outdscale size:", OutDscalePool.offset)
        print("Intermediate size:", IntermediatePool.getMaxSize())
        print("IntermediateHBM size:", IntermediatePoolHBM.getMaxSize())
        return rt_info
    else:
        rt_info =ddr_address_alloc(g,mode)

        return rt_info


def recoverTensorAddr(g: StaticGraph, addr_dict: dict):
    ret = RtInfo()
    IntermediatePool = DynamicMemoryPool(6384 << 20, 8000 << 20)
    IntermediatePoolHBM = DynamicMemoryPool(24 << 30, 8000 << 20,reisde_ddr=False)
    lifes = get_tensor_life(g)
    for input in g.get_inputs():
        if not addr_dict.has(input.name):
            raise ValueError("input name not in addr_dict")
        else:
            input.addr = addr_dict.get(input.name)
        ret.inputs.append(
                    InforSection(
                        name=input.name,
                        start=input.addr,
                        size=input.bytes_size(),
                        type=getStrtype(input.data_type),
                        file_path=input.name,
                    )
                )
    for output in g.get_outputs():
        if not addr_dict.has(output.name):
            raise ValueError("output name not in addr_dict")
        else:
            output.addr = addr_dict.get(output.name)
        ret.outputs.append(
                InforSection(
                    name=output.name,
                    start= output.addr,
                    size=output.bytes_size(),
                    file_path=output.name,
                )
            )
    for weight in g.get_weights():
        if not addr_dict.has(weight.name):
            raise ValueError("weight name not in addr_dict")
        else:
            weight.addr = addr_dict.get(weight.name)
    weights_info = InforSection(
            name="weights",
            start=rt_info.weights[0].start,
            size=rt_info.weights[0].size,
            file_path="weights",
    )
    ret.weights.append(weights_info)       
    for cst in g.get_const():
        if not addr_dict.has(cst.name):
            raise ValueError("cst name not in addr_dict")
        else:
            cst.addr = addr_dict.get(cst.name)
    consts_info = InforSection(
        name="consts", start=rt_info.consts[0].start , size=rt_info.consts[0].size, file_path="consts"
    )
    ret.consts.append(consts_info)
    # for dscale in g.get_dscale():
    #     if dscale.name not in addr_dict:
    #         raise ValueError("dscale name not in addr_dict")
    #     else:
    #         dscale.addr = addr_dict[dscale.name]
    # for output_dscale in g.get_dscale_outputs():
    #     if output_dscale.name not in addr_dict:
    #         raise ValueError("output_dscale name not in addr_dict")
    #     else:
    #         output_dscale.addr = addr_dict[output_dscale.name]
    for cached in g.get_cached():
        if not addr_dict.has(cached.name):
            raise ValueError("cached name not in addr_dict")
        else:
            cached.addr = addr_dict.get(cached.name)
    cached_info = InforSection(
            name="cached", start=rt_info.cached[0].start , size=rt_info.cached[0].size, file_path="cached"
        )
    ret.cached.append(cached_info)
    for idx, op in enumerate(g.get_ops()):
            outputs = op.get_outputs()
            for out in outputs:
                if out in g.get_inputs():
                    continue
                elif out in g.get_outputs():
                    continue
                elif out in g.get_weights():
                    continue
                elif out in g.get_const():
                    continue
                # elif out in g_dscale:
                #     continue
                elif out in g.get_cached():
                    continue
                else:
                    if isinstance(out.get_def(), View):
                        assert out.addr is None
                        assert len(out.get_def().get_inputs()) == 1
                        assert out.get_def().get_inputs()[0].addr is not None
                        out.addr = out.get_def().get_inputs()[0].addr
                        assert out.reside_ddr == out.get_def().get_inputs()[0].reside_ddr
                        out.reside_ddr = out.get_def().get_inputs()[0].reside_ddr
                    else:
                        assert out.addr is None
                        if out.reside_ddr:
                            addr = IntermediatePool.allocate(out.bytes_size())
                        else:
                            addr = IntermediatePoolHBM.allocate(out.bytes_size())
                        print(f"allocate {out.get_id()} at {idx} {out.name} {out.bytes_size()}")
                        assert addr is not None
                        out.addr = addr
                        intermediate_info = InforSection(
                            name=out.name,
                            start=out.addr,
                            size=out.bytes_size(),
                            type=getStrtype(out.data_type),
                            file_path=out.name,
                        )
                        rt_info.intermediate.append(intermediate_info)
            for input in op.get_inputs():
                org = get_org_tensor(input)
                if input in g.get_inputs():
                    continue
                elif input in g.get_outputs():
                    continue
                elif input in g.get_weights():
                    continue
                elif input in g.get_const():
                    continue
                elif input in g.get_cached():
                    continue
                elif isinstance(op, View):
                    continue
                elif org in lifes:
                    if idx == lifes[org].end :
                        assert input.addr is not None
                        if org.reside_ddr :
                            assert (org.addr >= IntermediatePool.start and org.addr < IntermediatePool.start + IntermediatePool.size)
                            IntermediatePool.free(org.addr)
                        else:
                            assert (org.addr >= IntermediatePoolHBM.start and org.addr < IntermediatePoolHBM.start + IntermediatePoolHBM.size)
                            IntermediatePoolHBM.free(org.addr)
                        print(f"free {org.get_id()} at {idx} {org.name} { org.reside_ddr }")
                else:
                    print("skip:",input.name)
    return ret
