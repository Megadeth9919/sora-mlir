import graph_ir.pass_impl
import graph_ir.graph_ir as ir
from graph_ir.pass_mem_allocate import RtInfo, InforSection
from inst import *
import numpy as np
import yaml
import json
import os
from typing import Union, List, Sequence
from safetensors.numpy import load_file
from p_model import *
import math
from graph_ir.pass_mem_allocate import CCTarget
from datetime import datetime
from utils.inst2onnx import inst2onnx
import copy


DDR_OFFSET = 0
def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        os.system("rm -rf " + path)
        os.makedirs(path)


def call_pmodel(g: ir.StaticGraph, mode: CCTarget = CCTarget.verify, sparsification:bool = False,fake_pmodel:bool = False):
    if mode == CCTarget.verify:
        p = PModel(graph=g)
        output_list = p.run(fake_pmodel=fake_pmodel)
    else:
        p = PModel(graph=g, param_path="/home/fpga5/OpenSora/model_quant_new.safetensors")
        # p = PModel(graph=g, param_path="/data1/shared/OpenSora/model_quant_new.safetensors")
        output_list = p.run(fake_pmodel=fake_pmodel)


def dump_information(
    g: ir.StaticGraph, root_path: str, op_name: str, mode, call_pmode: int = 1,  
    sparsification:bool = False, 
    dump:bool = True, 
    dump_json:bool = True,
    dump_onnx:bool = True,
    debug:bool = False,
    complete:bool = True,fake_pmodel:bool = False
):
    op_name = op_name + "_" + datetime.now().strftime("%m-%d")
    mkdir(root_path + "/" + op_name)
    print("============Compile Start============")
    if complete:
        g.complete()
    if call_pmode:
        call_pmodel(g, mode.target,fake_pmodel=fake_pmodel)

    print("======Lower Op======")
    rtl_info, inst, hook_info = graph_ir.pass_impl.graph_compile(g, flags =  mode, sparsification=sparsification,debug=debug)
    
    print("======Dump inst Dependency======")
    dump_inst_dependency(inst, op_name, root_path)

    print("======Dump Inst to json======")
    dump_inst_to_json(inst, op_name, root_path, mode)
    
    print("======Dump Graph======")
    dump_graph_to_json(g, op_name, root_path)

    print("======Dump Inst to bin======")
    dump_inst_to_bin(inst, op_name, hook_info, root_path, mode,debug)
    
    print("======Dump Info to yaml======")
    # dump_info_to_multichannel_yaml(rtl_info, op_name + "/info.yaml", root_path)

    dump_info_to_yaml(rtl_info, op_name + "/info.yaml", root_path,sparsification=sparsification)
    if dump_json:
        print("======Dump graph to json======")
        dump_graph_to_json(g, op_name, root_path)
    print("======Dump Data======")
    dump_data(g, op_name, root_path, mode.target, mode.enable_slr_slice,debug=debug,dump = dump)

    print("======Dump Inst to onnx======")
    if dump_onnx:
        if mode.enable_slr_slice:
            for i in range(3):
                inst2onnx(f'{root_path}/{op_name}/inst{i}.json',i)
        else:
            inst2onnx(f'{root_path}/{op_name}/inst.json')
    print("============Compile End============")

def dump_graph_to_json(g: ir.StaticGraph, case_name: str, root_path):
    with open(root_path + "/" + case_name + f"/graph.json", "w") as f:
        json.dump(g.to_json(), f, indent=4)
def dump_inst_to_json(insts: List[InstCollector], case_name: str, root_path, mode):
    if mode.enable_slr_slice:
        for i, inst in enumerate(insts):
            with open(root_path + "/"+ case_name + f"/inst{i}.json", "w") as f:
                json.dump(inst.to_json(), f, indent=4)
    else:
        with open(root_path + "/" + case_name + f"/inst.json", "w") as f:
            json.dump(insts[0].to_json(), f, indent=4)


def dump_data(
    g: ir.StaticGraph, case_name: str, root_path: str, mode: CCTarget = CCTarget.verify , slr_en :bool = False, debug:bool = False, dump:bool = True
):
    inputs_data = sorted(g.get_inputs(), key=lambda t: t.addr)
    outpus_data = sorted(g.get_outputs(), key=lambda t: t.addr)
    all_weights_data = sorted(g.get_weights(), key=lambda x: x.addr)
    weights_data = [w for w in all_weights_data if w not in g.get_const()]
    const_data = g.get_const()
    const_data = sorted(const_data, key=lambda x: x.addr)
    outdscale_data = g.get_dscale_outputs()
    intermediate_data = sorted(g.get_intermediate(), key = lambda t: t.get_id())
    if mode == CCTarget.verify:
        print("Dump Consts")
        dump_data_to_bin(const_data, case_name, root_path, "consts")
        print("Dump Weights")
        dump_data_to_channel_bin(weights_data, case_name, root_path, "weights", mode, slr_en)
        if not dump:
            return 
        print("Dump Inputs")
        dump_data_to_bin(inputs_data, case_name, root_path, "inputs")
        print("Dump Outputs")
        dump_data_to_bin(outpus_data, case_name, root_path, "outputs")
        # dump_data_to_channel_bin(const_data, case_name, root_path, "const",mode)
        # dump_data_to_channel_bin(outdscale_data, case_name, root_path, "outdscale",mode)
    elif mode == CCTarget.runtime:


        print("Dump Weights")
        dump_data_to_channel_bin(weights_data, case_name, root_path, "weights", mode, slr_en)
        print("Dump Consts")
        dump_data_to_bin(const_data, case_name, root_path, "consts")
        if not dump:
            return 
        print("Dump Inputs")
        dump_data_to_bin(inputs_data, case_name, root_path, "inputs")

        print("Dump Outputs")
        dump_data_to_bin(outpus_data, case_name, root_path, "outputs")
        if debug :
            print("Dump Consts")
            dump_data_to_bin(const_data, case_name, root_path, "inputs")
            print("Dump Weights seperate")
            dump_data_to_bin(weights_data, case_name, root_path, "inputs")
            for inter in intermediate_data:
                if inter is not None and not inter.get_def().isView():
                    dump_seperate_data_to_bin(inter, case_name, root_path)


def dump_data_to_bin(
    tensors: Union[List[ir.Tensor], List[ir.Weight]],
    file_path: str,
    root_path: str,
    type: str,
):
    global DDR_OFFSET
    # 64对齐
    DDR_OFFSET = 16*G
    if type == "inputs" or type == "outputs":
        for t in tensors:
            if t.get_data() is not None:
                with open(
                    root_path + "/" + file_path + "/" + "config.txt", "a"
                ) as file:
                    if type == "inputs":
                        file.write(
                            f"inputs:{t.addr}:{t.size * ir.get_size_by_type(t.data_type)}:./{t.name}.bin\n"
                        )
                    else:
                        file.write(
                            f"outputs:{t.addr}:{t.size * ir.get_size_by_type(t.data_type)}:{getStrtype(t.data_type)}:./{t.name}.bin\n"
                        )
                file.close()
                with open(
                    root_path + "/" + file_path + "/" + t.name + ".bin", "wb"
                ) as f:
                    f.write(graph_ir.pack_numpy_array([t.get_data()]))
            else:
                raise ValueError(f"The tensor {t.name} has no data to dump!")
    elif type == "consts":
        if len(tensors) == 0:
            return
        tensors_data = list()
        addr = 0
        size = 0
        for idx, t in enumerate(tensors):
            if t.get_data() is not None:
                tensors_data.append(t.get_data())
                addr = t.addr if idx == 0 else addr
                size += t.size * ir.get_size_by_type(t.data_type)
        with open(root_path + "/" + file_path + "/" + "config.txt", "a") as file:
            file.write(
                f"inputs:{addr}:{size}:./consts.bin\n"
            )
        file.close()
        with open(root_path + "/" + file_path + "/" + type + ".bin", "wb") as f:
            f.write(graph_ir.pack_numpy_array(tensors_data))
    else:
        raise ValueError(f"The tensor {t.name} has no data to dump!")
    
    if type == "outputs":
        last_output = tensors[-1]


        with open(root_path + "/" + file_path + "/" + "config.txt", "a") as file:
            file.write(f"ddr:{DDR_OFFSET}\n")
        file.close()

def align_data(tensor: Union[ir.Tensor, ir.Weight], size: int) -> List[np.ndarray]:
    if len(tensor.shape) == 1:
        return [tensor.get_data()]
    else:
        ret = list()
        align_size = size * 8
        data = tensor.get_data().copy()
        shape = data.shape
        if shape[-2] % align_size != 0:
            pad_width = (
                [(0, 0)] * (len(shape) - 2)
                + [(0, align_size - shape[-2] % align_size)]
                + [(0, 0)]
            )
            data = np.pad(data, pad_width, mode="constant")
        # print("pad:",data)
        new_shape = shape[:-2] + (
            math.ceil(shape[-2] / (align_size)),
            8,
            size,
            shape[-1],
        )
        data = data.reshape(new_shape)
        permute_order = (
            (len(data.shape) - 3,)
            + tuple(range(len(data.shape) - 3))
            + tuple(range(len(data.shape) - 2, len(data.shape), 1))
        )
        data = np.transpose(data, permute_order)
        for i in range(data.shape[0]):
            ret.append(data[i])
        return ret


G = 1073741824


def dump_data_to_channel_bin(
    tensor: Union[List[ir.Tensor], List[ir.Weight]],
    file_path: str,
    root_path: str,
    type: str,
    mode: CCTarget,
    slr_en:bool = False
):
    if type == "outputs":
        tensor_data = list(list())
        with open(root_path + "/" + file_path + "/" + "config.txt", "a") as file:
            for t in tensor:
                data = t.get_data()
                if data is not None:
                    aligned_data = align_data(t, 2)
                    tensor_data.append(aligned_data)
                    for idx, it in enumerate(aligned_data):
                        assert t.addr is not None
                        file.write(
                            f"""outputs:{t.addr+idx*G}:{it.size*ir.get_size_by_type(t.data_type)}:{"fp16" if t.data_type ==  ir.DataType.float16 else "int8"}:./{t.name}_{idx}.bin\n"""
                        )
                        with open(
                            root_path + "/" + file_path + "/" + f"{t.name}_{idx}.bin",
                            "wb",
                        ) as f:
                            f.write(graph_ir.pack_numpy_array([it]))
                else:
                    raise ValueError(f"The tensor {t.name} has no data to dump!")
            file.close()
        # elif mode == CCTarget.verify:
        #     for t in tensor:
        #         data = t.get_data()
        #         if data is not None:
        #             aligned_data = align_data(t,2)
        #             tensor_data.append(aligned_data)
        #         else:
        #             raise ValueError(f"The tensor {t.name} has no data to dump!")

        #     outdata = list()
        #     for i in range(8):
        #         size =0
        #         for idx,out in enumerate(tensor_data):
        #             outdata.append(out[i])
        #             size += math.prod(out[i].shape[0:]) * ir.get_size_by_type(tensor[idx].data_type)
        #         if len(outdata)!= 0:
        #             with open(root_path + "/" + file_path + "/" +"config.txt", 'a') as file:
        #                 assert tensor[0].addr is not None
        #                 file.write(f"""outputs:{tensor[0].addr+i*G}:{size}:{"fp16" if t.data_type ==  ir.DataType.float16 else "int8"}:./outputs_{i}.bin\n""")
        #             with open(root_path + "/" + file_path + "/"+ f"outputs_{i}.bin", 'wb') as f:
        #                 f.write(graph_ir.pack_numpy_array(outdata))
        #         outdata.clear()
    elif type == "inputs":
        tensor_data = list(list())
        if mode == CCTarget.runtime:
            with open(root_path + "/" + file_path + "/" + "config.txt", "w") as file:
                for t in tensor:
                    data = t.get_data()
                    if data is not None:
                        aligned_data = align_data(t, 2)
                        tensor_data.append(aligned_data)
                        for idx, it in enumerate(aligned_data):
                            assert t.addr is not None
                            file.write(
                                f"inputs:{t.addr+idx*G}:{it.size*ir.get_size_by_type(t.data_type)}:./{t.name}_{idx}.bin\n"
                            )
                            with open(
                                root_path
                                + "/"
                                + file_path
                                + "/"
                                + f"{t.name}_{idx}.bin",
                                "wb",
                            ) as f:
                                f.write(graph_ir.pack_numpy_array([it]))
                    else:
                        raise ValueError(f"The tensor {t.name} has no data to dump!")
                file.close()
        elif mode == CCTarget.verify:
            for t in tensor:
                data = t.get_data()
                if data is not None:
                    aligned_data = align_data(t, 2)
                    tensor_data.append(aligned_data)
                else:
                    raise ValueError(f"The tensor {t.name} has no data to dump!")

            indata = list()
            for i in range(8):
                size = 0
                for idx, ind in enumerate(tensor_data):
                    indata.append(ind[i])
                    size += math.prod(ind[i].shape[0:]) * ir.get_size_by_type(
                        tensor[idx].data_type
                    )
                if len(indata) != 0:
                    with open(
                        root_path + "/" + file_path + "/" + "config.txt", "a"
                    ) as file:
                        assert tensor[0].addr is not None
                        file.write(
                            f"inputs:{tensor[0].addr+i*G}:{size}:./inputs_{i}.bin\n"
                        )
                    with open(
                        root_path + "/" + file_path + "/" + f"inputs_{i}.bin", "wb"
                    ) as f:
                        f.write(graph_ir.pack_numpy_array(indata))
                indata.clear()
    elif type == "weights":
        tensor_wdata = list(list())
        for t in tensor:
            data = t.get_data()
            if data is not None:
                aligned_data = align_data(t, 2)
                tensor_wdata.append(aligned_data)
            else:
                raise ValueError(f"The tensor {t.name} has no data to dump!")

        outdata = list()
        for i in range(8):
            size = 0
            for idx, w in enumerate(tensor_wdata):
                outdata.append(w[i])
                size += math.prod(w[i].shape[0:]) * ir.get_size_by_type(
                    tensor[idx].data_type
                )
            if len(outdata) != 0:
                num = 3 if slr_en else 1
                for slr in range(num):
                    with open(
                        root_path + "/" + file_path + "/" + "config.txt", "a"
                    ) as file:
                        assert tensor[0].addr is not None
                        global DDR_OFFSET
                        file.write(
                            f"inputs:{tensor[0].addr+slr*8*G+ i*G+DDR_OFFSET}:{size}:./weights_{i}.bin\n"
                        )
                with open(
                    root_path + "/" + file_path + "/" + f"weights_{i}.bin", "wb"
                ) as f:
                    f.write(graph_ir.pack_numpy_array(outdata))
            outdata.clear()
    elif type == "const":
        tensor_cdata = list()
        size = 0
        for t in tensor:
            data = t.get_data()
            size += math.prod(t.shape[:]) * ir.get_size_by_type(t.data_type)
            if data is not None:
                tensor_cdata.append(data)
            else:
                raise ValueError(f"The tensor {t.name} has no data to dump!")
        if len(tensor_cdata) != 0:
            with open(root_path + "/" + file_path + "/" + "config.txt", "a") as file:
                assert tensor[0].addr is not None
                file.write(f"inputs:{tensor[0].addr}:{size}:./consts.bin\n")
            with open(root_path + "/" + file_path + "/" + f"consts.bin", "wb") as f:
                f.write(graph_ir.pack_numpy_array(tensor_cdata))
    elif type == "outdscale":
        tensor_cdata = list()
        size = 0
        for t in tensor:
            data = t.get_data()
            size += math.prod(t.shape[:]) * ir.get_size_by_type(t.data_type)
            if data is not None:
                tensor_cdata.append(data)
            else:
                raise ValueError(f"The tensor {t.name} has no data to dump!")
        if len(tensor_cdata) != 0:
            with open(root_path + "/" + file_path + "/" + "config.txt", "a") as file:
                assert tensor[0].addr is not None
                file.write(f"outputs:{tensor[0].addr}:{size}:fp16:./outdscale.bin\n")
            with open(root_path + "/" + file_path + "/" + f"outdscale.bin", "wb") as f:
                f.write(graph_ir.pack_numpy_array(tensor_cdata))
    else:
        raise ValueError(f"The tensor has no data to dump!")


def dump_seperate_data_to_bin(
    tensor: Union[ir.Tensor, ir.Weight], file_path: str, root_path
):
    # Check if file_path exists before dumping
    with open(root_path + "/" + file_path + "/" + "config.txt", "a") as file:
        if tensor.get_data() is not None:
            bin_data = graph_ir.pass_impl.pack_numpy_array([tensor.get_data()])
            file.write(
                f"inputs:{tensor.addr}:{tensor.size*ir.get_size_by_type(tensor.data_type)}:./{tensor.name}.bin\n"
            )
        else:
            raise ValueError(
                f"file_path cannot be None for tensor with data: {file_path}"
            )
    file.close()

    with open(root_path + "/" + file_path + "/" + tensor.name + ".bin", "wb") as f:
        f.write(bin_data)


def dump_inst_to_bin(insts: List[InstCollector], case_name: str,hook_info, root_path: str,mode, debug:bool = False):
    if mode.enable_slr_slice:
        for i, inst in enumerate(insts):
            inst_bits,info = insts_to_bin(inst.get_insts(),hook_info[i])
            with open(root_path + "/" + case_name + f"/inst{i}.bin", "ab") as f:
                f.write(inst_bits)
            with open(root_path + "/" + case_name + f"/inst{i}.txt", "w") as f:
                f.write("".join(["%02X " % b for b in inst_bits]))
            if debug :
                with open(root_path + "/" + case_name + f"/hook_info{i}.yaml", "w") as f:
                    yaml.dump(info.to_dict(), f, default_flow_style=None, sort_keys=False)
    else:
        inst_bits,info = insts_to_bin(insts[0].get_insts(),hook_info[0])
        with open(root_path + "/" + case_name + f"/inst.bin", "wb") as f:
            f.write(inst_bits)
        with open(root_path + "/" + case_name + f"/inst.txt", "w") as f:
            f.write("".join(["%02X " % b for b in inst_bits]))
        if debug :
            with open(root_path + "/" + case_name + f"/hook_info.yaml", "w") as f:
                yaml.dump(info.to_dict(), f, default_flow_style=None, sort_keys=False)

def dump_info_to_yaml(rtl_info, file_path: str, root_path: str, sparsification:bool = False):
    
    inst_info = []
    inst_info.append({"file_path": "./inst0.bin"})
    inst_info.append({"file_path": "./inst1.bin"})
    inst_info.append({"file_path": "./inst2.bin"})

    # if not sparsification:
    size = sum([weight.size for weight in rtl_info.weights])
    if size > 0:
        weights = copy.deepcopy(rtl_info.weights)
        rtl_info.weights.clear()
        for it in weights:
            for i in range(8):
                new_it = InforSection(
                    name="weights",
                    start=it.start + i * G,
                    size=max(size, 1),
                    file_path=it.file_path + f"_{i}",
                )
                rtl_info.weights.append(new_it)
    ret = rtl_info.misc_serilize()
    ret["inst_info"] = inst_info
    with open(root_path + "/" + file_path, "w") as f:
        yaml.dump(ret, f)


def dump_info_to_multichannel_yaml(rtl_info, file_path, root_path):
    muli_rtinfo = RtInfo()
    size = sum([input.size for input in rtl_info.inputs])
    if size > 0:
        for it in rtl_info.inputs:
            for i in range(8):
                new_it = InforSection(
                    it.name + f"_{i}", it.start + i * G, it.size, it.file_path + f"_{i}"
                )
                muli_rtinfo.inputs.append(new_it)
    else:
        for it in rtl_info.inputs:
            muli_rtinfo.inputs.append(
                InforSection(it.name, it.start, it.size, it.file_path)
            )
    size = sum([output.size for output in rtl_info.outputs])
    if size > 0:
        for it in rtl_info.outputs:
            for i in range(8):
                new_it = InforSection(
                    it.name + f"_{i}", it.start + i * G, it.size, it.file_path + f"_{i}"
                )
                muli_rtinfo.outputs.append(new_it)
    else:
        for it in rtl_info.outputs:
            muli_rtinfo.outputs.append(
                InforSection(it.name, it.start, it.size, it.file_path)
            )
    size = sum([weight.size for weight in rtl_info.weights])
    if size > 0:
        for it in rtl_info.weights:
            for i in range(8):
                new_it = InforSection(
                    name="weights",
                    start=it.start + i * G,
                    size=max(size, 1),
                    file_path=it.file_path + f"_{i}",
                )
                muli_rtinfo.weights.append(new_it)
    else:
        for it in rtl_info.weights:
            muli_rtinfo.weights.append(
                InforSection(it.start, it.size, it.type, it.file_path)
            )
    for it in rtl_info.consts:
        muli_rtinfo.consts.append(
            InforSection(it.start, it.size, it.type, it.file_path)
        )
    for it in rtl_info.dscales:
        muli_rtinfo.dscales.append(
            InforSection(it.start, it.size, it.type, it.file_path)
        )
    for it in rtl_info.output_dscales:
        muli_rtinfo.output_dscales.append(
            InforSection(it.name, it.start, it.size, it.file_path)
        )
    for it in rtl_info.cached:
        muli_rtinfo.cached.append(
            InforSection(it.start, it.size, it.type, it.file_path)
        )
    for it in rtl_info.intermediate:
        muli_rtinfo.intermediate.append(
            InforSection(it.start, it.size, it.type, it.file_path)
        )
    out = muli_rtinfo.misc_serilize()
    inst_info = []
    inst_info.append({"file_path": "./inst.bin"})
    out["inst_info"] = inst_info
    with open(root_path + "/" + file_path, "w") as f:
        yaml.dump(out, f)


def gen_random_data(tensors: Union[list[ir.Tensor], set[ir.Weight]]):
    np.random.seed(0)
    for t in tensors:
        if t.data is not None:
            continue
        fakedata = np.random.randint(low=-2, high=2, size=list(t.shape), dtype=np.int8)
        if t.data_type == ir.DataType.int16:
            fakedata = fakedata.astype(np.int16)
        if t.data_type == ir.DataType.float16:
            fakedata = fakedata.astype(np.float16)
        if t.data_type == ir.DataType.float32:
            fakedata = fakedata.astype(np.float32)
        if t.data_type == ir.DataType.int8:
            fakedata = fakedata.astype(np.int8)
        t.set_data(fakedata)
    return


def load_inputs_data(tensors: list[ir.Tensor], file_path: str):
    golden_inputs = load_file(file_path)
    for i, t in enumerate(tensors):
        if t.data is not None:
            raise ValueError(f"Tensor {t.name} has data already!")
        name = t.name
        input = golden_inputs[name]
        t.set_data(input)


def getStrtype(data_type: ir.DataType) -> str:
    match data_type:
        case ir.DataType.float16:
            return "fp16"
        case ir.DataType.int8:
            return "int8"
        case ir.DataType.int16:
            return "int16"
        case ir.DataType.float32:
            return "float32"
        case _:
            assert False


def get_inst_type(inst: Union[Inst, PUType]) -> str:
    if isinstance(inst, LDInst) or inst == PUType.LD:
        return "LD"
    elif isinstance(inst, STInst) or inst == PUType.ST:
        return "ST"
    elif isinstance(inst, MMInst) or inst == PUType.MM:
        return "MM"
    elif isinstance(inst, MISCInst) or inst == PUType.MISC:
        return "MISC"
    elif isinstance(inst, RSInst) or inst == PUType.RS:
        return "ReShape"
    elif isinstance(inst, SYSCInst) or inst == PUType.SYS:
        return "SYS"
    else:
        raise ValueError("Undefined instruction type!")


def relay_pass(insts: List[Inst], inst_type):
    count = len(inst_type)
    wait_release_counts: dict[str, dict[str, dict[str, int]]] = {}
    for i in range(count):
        wait_release_counts[inst_type[i]] = {}
        wait_release_counts[inst_type[i]] = {}
        # release_counts[inst_type[i]] = []
    for item in wait_release_counts:
        wait_release_counts[item]["wait"] = {}
        wait_release_counts[item]["release"] = {}
    for inst in insts:
        for wait in inst.wait:
            if (
                get_inst_type(wait)
                in wait_release_counts[get_inst_type(inst)]["wait"]
            ):
                wait_release_counts[get_inst_type(inst)]["wait"][
                    get_inst_type(wait)
                ] += 1
            else:
                wait_release_counts[get_inst_type(inst)]["wait"][
                    get_inst_type(wait)
                ] = 1
        for release in inst.release:
            if (
                get_inst_type(release)
                in wait_release_counts[get_inst_type(inst)]["release"]
            ):
                wait_release_counts[get_inst_type(inst)]["release"][
                    get_inst_type(release)
                ] += 1
            else:
                wait_release_counts[get_inst_type(inst)]["release"][
                    get_inst_type(release)
                ] = 1
    return wait_release_counts



def parse_inst(insts: list[InstCollector]):
    inst_type = set()
    ret = []
    for inst in insts:
        for item in inst.get_insts():
            inst_type.add(get_inst_type(item))
        dependency_info = relay_pass(inst.get_insts(), list(inst_type))
        ret.append(dependency_info)
    return ret


def dump_inst_dependency(insts: list[InstCollector], op_name: str, root_path: str):
    dependency_info = parse_inst(insts)
    with open(root_path + "/" + op_name + "/dependency.yaml", "w") as f:
        yaml.dump(dependency_info, f, default_flow_style=None)


def generate_cross_attn_mask(x_B, x_N, cond_size):
    # x_B * x_N = B * T * S
    # Initialize mask_data as a float16 numpy array
    mask_data = np.empty((x_B * x_N * cond_size), dtype=np.float16)

    # Generate the mask
    for i in range(x_B * x_N):
        for j in range(cond_size):
            if (i < (x_B * x_N) // 2 and j < cond_size // 2) or (
                i >= (x_B * x_N) // 2 and j >= cond_size // 2
            ):
                mask_data[i * cond_size + j] = 0  # equivalent to 0
            else:
                mask_data[i * cond_size + j] =  -np.inf  # equivalent to -inf
    mask_data = mask_data.reshape((1, 1, x_B * x_N, cond_size))
    return mask_data
