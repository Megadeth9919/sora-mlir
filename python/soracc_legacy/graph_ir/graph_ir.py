from collections.abc import Iterable
from enum import Enum
from typing import Optional, Union

from numpy import ndarray

DataType = Enum("DataType", ("float16", "float32", "unit4", "int8", "int16"))


def get_size_by_type(x: DataType):
    if x == DataType.float16:
        return 2
    elif x == DataType.float32:
        return 4
    elif x == DataType.unit4:
        return 1
    elif x == DataType.int8:
        return 1
    elif x == DataType.int16:
        return 2


class Shape(tuple):
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return super().__new__(cls, args[0])
        else:
            return super().__new__(cls, args)
    def __init__(self, *args):
        self.dim = len(self)

    def prod(self, start=0) -> int:
        ret = 1
        for x in self[start:]:
            ret *= x
        return ret

    def to_json(self):
        return self


class Count:
    def __init__(self):
        self.n = 0

    def get(self):
        self.n += 1
        return self.n


class TensorView:
    def __init__(self, t: Union["Tensor", "TensorView"]):
        self.origin = t
        self.shape: Shape
        self.data_type: DataType
        self.addr: Optional[int] = None
        # 对应的shape的数据取完以后取下一组数据需要跨越的距离
        # stride和shape的维度需要是相同的
        self.stride: Shape
        self.reside_ddr :bool = True

    def __str__(self):
        ret = (f"""origin: {self.origin}\n""" +
               f"""shape: {self.shape}\n""" +
               f"""data_type: {self.data_type}\n""" +
               f"""addr: {self.addr}\n""" +
               f"""stride: {self.stride}""")
        return ret


class Tensor:
    id = Count()

    def __init__(self, shape: Shape, op_def=None, dtype: Optional[DataType] = None, name: str = '', const: bool = False, cached: bool = False, dscale:bool = False):
        """
        Args:
            shape: Shape of tensor
            op_def: The op that define this tensor
            dtype: Data type of tensor
            name: Name of tensor 
                ({layername}.{op_name}.{output/input}.{act_scale(if exists)} exemple: "atten.0.linear.output")
            const: Whether this tensor is a constant
        """
        self._id = Tensor.id.get()
        self.shape = shape
        self.name = name
        self.data_type: DataType = DataType.float16 if dtype is None else dtype
        self._def: 'Op' = op_def
        self._users: set['Op'] = set()
        self.addr: Optional[int] = None
        self.data: Optional[ndarray] = None
        self.act_scale: Optional[Tensor] = None
        self.stride:Shape
        self.const = const
        self.cached = cached
        self.dynamicscale = dscale
        self.reside_ddr :bool = True
        self.force_output : bool = False
    def set_reside_HBM(self):
        self.reside_ddr = False
    def get_reside_DDR(self):
        return self.reside_ddr
    def set_reside_DDR(self):
        self.reside_ddr = True
    def get_def(self) -> Optional['Op']:
        return self._def

    def set_def(self, op_def: 'Op'):
        self._def = op_def

    def add_user(self, user: 'Op') -> None:
        self._users.add(user)

    def get_users(self) -> set['Op']:
        return self._users
    
    def clear_users(self):
        self._users.clear()
    def delete_user(self,user):
        self._users.remove(user)

    def get_id(self) -> int:
        return self._id

    def set_data(self, data: ndarray):
        assert (
            self.shape == data.shape
        ), f"shape of data {data.shape} not match tensor shape {self.shape}"
        # assert data.dtype == 'float16', "data type of tensor {self.data_type} not match data type {DataType.float16}"
        self.data = data

    def get_data(self):
        return self.data
    
    def set_act_scale(self, act_scale: 'Tensor'):
        self.act_scale = act_scale
    
    @property
    def get_act_scale(self) -> Optional['Tensor']:
        return self.act_scale

    @property
    def size(self) -> int:
        return self.shape.prod()

    def bytes_size(self) -> int:
        return self.shape.prod() * get_size_by_type(self.data_type)

    def __str__(self):
        ret = (f"""tensor: {self.name}\n""" +
               f"""id: {self._id}\n""" +
               f"""shape: {self.shape}\n""" +
               f"""data_type: {self.data_type}\n""" +
               f"""def: {self._def is None}\n""" +
               f"""users: {len(self._users)}\n""" +
               f"""addr: {self.addr}""")
        return ret

    # self._op不做处理，这里只有创建计算图的时候用到
    def to_json(self):
        ret = {"id": self._id,
               "shape": self.shape.to_json(),
               "name": self.name,
               "data_type": self.data_type.name,
               "addr": self.addr,
               "dscale": self.dynamicscale,
               "const": self.const,
               "cached" : self.cached,
               "reside_ddr": self.reside_ddr,}
        return ret

    @staticmethod
    def from_json(js: dict):
        ret = Tensor(Shape(js["shape"]))
        ret._id = js["id"]
        ret.name = js["name"]
        ret.data_type = DataType[js["data_type"]]
        ret.addr = js["addr"]
        return ret


class StaticGraph:
    registered_op: dict[str, type] = dict()

    def __init__(self) -> None:
        self._ops: list[Op] = []
        self._tensors: set[Tensor] = set()
        self._inputs: list[Tensor] = []
        self._outputs: list[Tensor] = []
        self._intermediate: list[Tensor] = []
        self._cached: list[Tensor] = []
        self._const: set[Tensor] = set()
        self._dscale: list[Tensor] = []
        self._dscale_outputs: list[Tensor] = []

    def get_outputs(self) -> list[Tensor]:
        return self._outputs
    def add_output(self, tensors):
        for t in tensors:
            self._outputs.append(t)
        
    def get_dscale_outputs(self) -> list[Tensor]:
        return self._dscale_outputs
    def get_inputs(self) -> list[Tensor]:
        return self._inputs

    def get_intermediate(self) -> list[Tensor]:
        return self._intermediate
    
    def get_cached(self) -> list[Tensor]:
        return self._cached
    
    def set_inputs(self, tensors):
        if isinstance(tensors, list):
            self._inputs += tensors
            for t in tensors:
                t.add_user(self)
                def_op = t.get_def()
                if def_op:
                    self._prev.append(def_op)
                    def_op.add_next(self)
        else:
            tensors.add_user(self)
            def_op = tensors.get_def()
            if def_op:
                self._prev.append(def_op)
                def_op.add_next(self)
            self._inputs.append(tensors)

    def get_tensors(self) -> list[Tensor]:
        sorted_tensors = list(self._tensors)
        sorted_tensors.sort(key=lambda t: t.get_id())
        return sorted_tensors
    def del_tensor(self, tensor:Tensor):
        if tensor in self._tensors:
            self._tensors.remove(tensor)
    def del_op(self, op: 'Op'):
        if op in self._ops:
            self._ops.remove(op)

    def set_input_data(self, data_list: list[ndarray]):
        for i, data in enumerate(data_list):
            self._inputs[i].set_data(data)

    def get_weights(self) -> set['Weight']:
        ret: set[Weight] = set()
        for op in self._ops:
            weights = op.get_weights()
            for weight in weights:
                ret.add(weight)
        return ret
    
    def get_const(self) -> list[Tensor]:
        sorted_const = list(self._const)
        sorted_const.sort(key=lambda t: t.get_id())
        # sorted_const.sort(key=lambda t: t.get_id())
        return sorted_const
    def get_dscale(self) -> list[Tensor]:
        sorted_dscale = list(self._dscale)
        sorted_dscale.sort(key=lambda t: t.get_id())
        # sorted_const.sort(key=lambda t: t.get_id())
        return sorted_dscale

    def complete(self):
        self._intermediate.clear()
        self._inputs.clear()
        self._outputs.clear()

        sorted_tensors = list(self._tensors)
        sorted_tensors.sort(key=lambda t: t.get_id())
        for t in self._tensors:
            if t.get_def() is not None and len(t.get_users()):
                if t.cached :
                    if t not in self._cached:
                        self._cached.append(t)
                # elif t.dynamicscale:
                #     if t not in self._dscale:
                #         self._dscale.append(t)
                elif t.force_output:
                    if t not in self._outputs:
                        self._outputs.append(t)
                elif t not in self._intermediate:
                    self._intermediate.append(t)
            elif len(t.get_users()):
                if t not in self._inputs and not t.const and not t.cached:
                    self._inputs.append(t)
            else:
                # if t.dynamicscale :
                #     if t not in self._dscale_outputs:
                #         self._dscale_outputs.append(t)
                if t.cached :
                    if t not in self._cached:
                        self._cached.append(t)
                elif t not in self._outputs or t.force_output:
                    self._outputs.append(t)
    def to_json(self):
        ret = {}

        ops = []
        for op in self._ops:
            op_js = op.to_json()
            op_js["op_type"] = type(op).op_type
            inputs_id = [t.get_id() for t in op.get_inputs()]
            op_js["inputs"] = inputs_id
            outputs_id = [t.get_id() for t in op.get_outputs()]
            op_js["outputs"] = outputs_id
            ops.append(op_js)
        ret["ops"] = ops

        tensors = []
        for t in self._tensors:
            tensors.append(t.to_json())
        ret["tensors"] = tensors
        return ret

    def from_json(self, js: dict):
        js_tensors = js["tensors"]
        for t in js_tensors:
            tensor = Tensor.from_json(t)
            self._tensors.add(tensor)

        tensor_map = {}
        for t in self._tensors:
            tensor_map[t.get_id()] = t

        ops = js["ops"]
        for op_js in ops:
            op_type = StaticGraph.registered_op[op_js["op_type"]]
            op_ins = op_type(op_js["name"])
            op_info = {k: v for k, v in op_js.items() if k != "op_type"}
            op_ins.from_json(op_info)
            self._ops.append(op_ins)

        for i, op in enumerate(self._ops):
            op_inputs = []
            for k in ops[i]['inputs']:
                t_input = tensor_map[k]
                t_input.add_user(op)
                op_inputs.append(t_input)
            op.set_inputs(op_inputs)

            op_outputs = []
            for k in ops[i]['outputs']:
                t_output = tensor_map[k]
                t_output.set_def(op)
                op_outputs.append(t_output)
            op.set_outputs(op_outputs)

        for i in range(len(self._ops)):
            for j in range(i + 1, len(self._ops)):
                if set.intersection(set(ops[i]["inputs"]), set(ops[j]["outputs"])):
                    self._ops[i].add_prev(self._ops[j])
                    self._ops[j].add_next(self._ops[i])

                if set.intersection(set(ops[i]["outputs"]), set(ops[j]["inputs"])):
                    self._ops[i].add_next(self._ops[j])
                    self._ops[j].add_prev(self._ops[i])

    # add op之前需要将op相关的输入输出tensor设置好
    def add(self, op):
        self._ops.append(op)
        for t in op.get_inputs():
            if t not in self._tensors and not t.const and not t.cached:
                self._inputs.append(t)
            if t.const:
                self._const.add(t)
            if t.cached:
                self._cached.append(t)
            self._tensors.add(t)
        raw_weights = op.get_raw_weights()
        for k in list(raw_weights):
            if raw_weights[k].const:
                self._const.add(raw_weights[k])
                # raw_weights.pop(k)
        for t in op.get_outputs():
            self._tensors.add(t)
    def insert_after(self, op , new_op):
        index = self._ops.index(op)
        self._ops.insert(index + 1, new_op)
        for t in new_op.get_inputs():
            if t not in self._tensors and not t.const and not t.cached:
                self._inputs.append(t)
            if t.const:
                self._const.add(t)
            self._tensors.add(t)
        raw_weights = new_op.get_raw_weights()
        for k in list(raw_weights):
            if raw_weights[k].const:
                self._const.add(raw_weights[k])
                # raw_weights.pop(k)
        for t in new_op.get_outputs():
            self._tensors.add(t)
    def insert_before(self, op , new_op):
        index = self._ops.index(op)
        self._ops.insert(index, new_op)
        for t in new_op.get_inputs():
            if t not in self._tensors and not t.const and not t.cached:
                self._inputs.append(t)
            if t.const:
                self._const.add(t)
            self._tensors.add(t)
        raw_weights = new_op.get_raw_weights()
        for k in list(raw_weights):
            if raw_weights[k].const:
                self._const.add(raw_weights[k])
                # raw_weights.pop(k)
        for t in new_op.get_outputs():
            self._tensors.add(t)
    def get_ops(self):
        return self._ops

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ops[key]
        else:
            raise TypeError("key must be int")

    def __iter__(self):
        self.__iter = -1
        return self

    def __next__(self):
        if self.__iter < len(self) - 1:
            self.__iter += 1
            ret = self._ops[self.__iter]
            return ret
        else:
            raise StopIteration

    def __len__(self):
        return len(self._ops)

    def __str__(self) -> str:
        ret = 'graph: \n'
        for op in self._ops:
            ret += str(op) + '\n'
        return ret


def register_op(name: str):
    def wrapper(cls: type[Op]):
        StaticGraph.registered_op[name] = cls
        cls.op_type = name
        return cls

    return wrapper


class Op:
    op_type: str = "op"

    def __init__(self, name: str):
        self.name = name
        self._prev: list[Op] = []
        self._next: list[Op] = []
        self._inputs: list[Tensor] = []
        self._outputs: list[Tensor] = []
        self._weights: dict[str, Weight] = {}
        self.opt_func:str = ""
        self.act_scale_flag = False

    def to_json(self):
        weight_dict = {}
        for k, v in self._weights.items():
            weight_dict[k] = v.to_json()
        ret = {"name": self.name, 
               "weights": weight_dict, 
               "dscale":self.act_scale_flag,
               "input":self._inputs[0].to_json(),
               "output":self._outputs[0].to_json(),
               "act_scale":{} if not self.act_scale_flag else self._outputs[-1].to_json()}
        return ret

    def from_json(self, js: dict):
        self.name = js["name"]
        weight_dict = js["weights"]
        for k, v in weight_dict.items():
            w = Weight.from_json(v)
            self._weights[k] = w

    def add_prev(self, op: "Op"):
        self._prev.append(op)

    def add_next(self, op: "Op"):
        self._next.append(op)

    def get_prev(self):
        return self._prev

    def get_next(self):
        return self._next

    def get_inputs(self) -> list[Tensor]:
        return self._inputs

    def set_inputs(self, tensors: list[Tensor]):
        self._inputs = tensors

    def get_outputs(self):
        return self._outputs

    def set_outputs(self, tensors: Iterable[Tensor]):
        for t in tensors:
            t.set_def(self)
            self._outputs.append(t)

    def add_outputs(self, tensors: Iterable[Tensor]):
        for t in tensors:
            t.set_def(self)
            self._outputs.append(t)
    def clear_outputs(self) :
        self._outputs.clear()

    def get_weights(self):
        return list(self._weights.values())
    def get_users(self):
        users = [] 
        for t in self.get_outputs() :
            for u in t.get_users() :
                if u not in users:
                    users.append(u)
        return users
    def get_defs(self):
        defs = []
        for t in self.get_inputs() :
            if t.get_def() is not None:
                defs.append(t.get_def())
        return defs
    @property
    def get_act_scale(self) -> Tensor:
        assert self.act_scale_flag, "act_scale is not set"
        return self._outputs[-1]

    def get_raw_weights(self):
        return self._weights

    def set_raw_weights(self, weights: dict[str, 'Weight']):
        self._weights = weights

    def add_input_tensors(self, tensors: Iterable[Tensor]):
        for t in tensors:
            self._inputs.append(t)
            t.add_user(self)
            def_op = t.get_def()
            if def_op:
                self._prev.append(def_op)
                def_op.add_next(self)
    def isView(self):
        return False

    def show(self):
        print(f"op_type: {type(self).op_type}, op_name: {self.name}")
        print("prev:")
        for e in self._prev:
            print(f"  {str(e)}")
        print("next:")
        for e in self._next:
            print(f"  {str(e)}")
        print("input: ")
        for e in self._inputs:
            print(f"  {str(e)}")
        print("out: ")
        for e in self._outputs:
            print(f"  {str(e)}")

    def show_input_shape(self, name: str):
        input_shape_info = list()
        weigh_shape_info = list()
        for idx,e in enumerate(self._inputs):
            # print(f"{name} input: {str(e.shape)}")
            input_shape_info.append(e.shape)
        if len(self.get_weights())>0:
            for w in self.get_weights():
                # print(f"{name} weight: {str(w.shape)}")
                weigh_shape_info.append(w.shape)
        return (input_shape_info, weigh_shape_info)

    def create_tensor(self, shape: Shape, dtype: Optional[DataType] = None, name: str = '' , dscale:bool = False) -> Tensor:
        ret = Tensor(shape, self, dtype, name,dscale=dscale)
        self._outputs.append(ret)
        return ret

    def load_weight_from_dict(self, weight_dict: dict):
        return None

    def __str__(self) -> str:
        ret = f"""name: {self.name}\n""" + f"""type: {type(self).op_type}\n"""
        ret += 'inputs:\n'
        for in_ in self._inputs:
            ret += str(in_) + '\n'
        ret += 'outputs:\n'
        for out_ in self._outputs:
            ret += str(out_) + '\n'
        ret += 'weights:\n'
        for k, w in self._weights.items():
            ret += f'\nkey: {k}\n{str(w)}'
        ret += '\n'
        return ret


class Weight:
    id = Count()
    def __init__(self, name: str, shape: Shape, data_type: DataType, const: bool = False):
        self.name = name
        self.shape = shape
        self.data_type = data_type
        self.addr: Optional[int] = None
        self.data: Optional[ndarray] = None
        self.const = const
        self._id = Weight.id.get()
        self.align_shape: Shape
        self.reside_ddr :bool = True

    def get_id(self) -> int:
        return self._id
    
    def set_data(self, data: ndarray):
        assert (
            self.shape == data.shape
        ), f"shape of data {data.shape} not match tensor shape {self.shape}"
        # assert data.dtype == 'float16', "data type of tensor {self.data_type} not match data type {DataType.float16}"
        self.data = data

    def get_data(self):
        return self.data

    def to_json(self):
        ret = {'name': getattr(self, 'name'),
               'shape': getattr(self, 'shape'),
               'data_type': getattr(self, 'data_type').name,
               'addr': getattr(self, 'addr'),
               'const': getattr(self, 'const'),
               'reside_ddr': getattr(self, 'reside_ddr'),}
        return ret

    @staticmethod
    def from_json(js: dict):
        ret = Weight(js["name"], Shape(js["shape"]), DataType(js["data_type"]))
        ret.addr = js["addr"]
        return ret

    @property
    def size(self):
        return self.shape.prod()

    def bytes_size(self) -> int:
        return self.shape.prod() * get_size_by_type(self.data_type)

    def __str__(self) -> str:
        ret = (f"""weight: {self.name}\n""" +
               f"""shape: {self.shape}\n""" +
               f"""data_type: {self.data_type}\n""" +
               f"""addr: {self.addr}""")
        return ret


@register_op("linear_w4")
class LinearW4(Op):
    def __init__(self, name: str):
        super().__init__(name)

    def set_weight_scale(self, *, weight: Weight, weight_scale: Weight, zero_point: Weight):
        assert weight.data_type == DataType.unit4, "weight data type must be unit4"
        assert weight_scale.data_type == DataType.float16, "scale data type must be int16"
        assert zero_point.data_type == DataType.int8, "zero_point data type must be int8"
        self._weights['weight'] = weight
        self._weights['weight_scale'] = weight_scale
        self._weights['zero_point'] = zero_point

    @property
    def get_feature(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]


@register_op("linear_w8")
class LinearW8(Op):
    def __init__(self, name: str, bias_flag: bool = False):
        super().__init__(name)
        self.bias_flag = bias_flag

    # weight shape is (N, K) or (out_feature, K)
    def set_weight_scale(self, *, weight: Weight, weight_scale: Weight, bias: Optional[Weight] = None):
        assert weight.data_type == DataType.int8, "weight data type must be int8"
        assert weight_scale.data_type == DataType.float32, "scale data type must be float32"
        assert self.bias_flag == (bias is not None), "bias must be set if bias is True"
        self._weights['weight'] = weight
        self._weights['weight_scale'] = weight_scale
        if bias is not None:
            assert bias.data_type == DataType.float16, "bias data type must be int32"
            self._weights['bias'] = bias

    def load_weight_from_dict(self, weight_dict: dict):
        for t in self._weights.values():
            if t.name in weight_dict.keys():
                t.set_data(weight_dict[t.name])
            else:
                raise ValueError(f"Warning: weight {t.name} not found in weight dict {weight_dict.keys}")
        return True

    @property
    def get_feature(self) -> Tensor:
        return self._inputs[0]
    
    @property
    def get_act_scale(self) -> Tensor:
        return self._inputs[1]

    @property
    def get_weight(self) -> tuple[Weight, Weight]:
        return self._weights['weight'], self._weights['weight_scale']

    @property
    def get_bias(self) -> Optional[Weight]:
        if self.bias_flag:
            return self._weights['bias']
        return None

    @property
    def get_output(self):
        return self._outputs[0]

@register_op("matmul")
class Matmul(Op):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def get_matrix_A(self):
        return self._inputs[0]

    @property
    def get_matrix_B(self):
        return self._inputs[1]

    @property
    def get_act_scale_A(self) -> Tensor:
        return self._inputs[2]
    
    @property
    def get_act_scale_B(self) -> Tensor:
        return self._inputs[3]
    
    @property
    def get_output(self):
        return self._outputs[0]


class Conv3D(Op):
    def __init__(self, name: str, kernel_size: int, stride: int, padding: int, dilation: int, groups: int, bias: bool):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def set_weight_scale(self, *, weight: Weight, weight_scale: Weight, bias: Optional[Weight] = None):
        assert weight.data_type == DataType.int8, "weight data type must be int8"
        assert weight_scale.data_type == DataType.float16, "scale data type must be int16"
        assert self.bias == (bias is not None), "bias must be set if bias is True"
        self._weights['weight'] = weight
        self._weights['weight_scale'] = weight_scale
        if bias is not None:
            assert bias.data_type == DataType.float16, "bias data type must be int32"
            self._weights['bias'] = bias

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_weight(self):
        return self._weights['weight'], self._weights['weight_scale']

    @property
    def get_bias(self):
        if self.bias:
            return self._weights['bias']
        return None

    @property
    def get_output(self):
        return self._outputs[0]


class Conv2D(Op):
    def __init__(self, name: str, kernel_size: int, stride: int, padding: int, dilation: int, groups: int, bias: bool):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def set_weight_scale(self, *, weight: Weight, weight_scale: Weight, bias: Optional[Weight] = None):
        assert weight.data_type == DataType.int8, "weight data type must be int8"
        assert weight_scale.data_type == DataType.float16, "scale data type must be int16"
        assert self.bias == (bias is not None), "bias must be set if bias is True"
        self._weights['weight'] = weight
        self._weights['weight_scale'] = weight_scale
        if bias is not None:
            assert bias.data_type == DataType.float16, "bias data type must be int32"
            self._weights['bias'] = bias

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_weight(self):
        return self._weights['weight'], self._weights['weight_scale']

    @property
    def get_bias(self):
        if self.bias:
            return self._weights['bias']
        return None

    @property
    def get_output(self):
        return self._outputs[0]

@register_op("LoadInst")
class LoadInst(Op):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def get_input(self):
        return self._inputs[0]


@register_op("eltwise")
class Eltwise(Op):
    def __init__(self, name: str, type: str):
        super().__init__(name)
        assert type in [
            "add",
            "mul",
        ], f"The type of eltwise must be add or mul, but got {type}"
        self.type = type

    @property
    def get_input_A(self):
        return self._inputs[0]

    @property
    def get_input_B(self):
        return self._inputs[1]

    @property
    def get_output(self):
        return self._outputs[0]

    def show_input_shape(self, name: str):
        shape_info = super().show_input_shape(name)
        return shape_info + (self.type,)

@register_op("rotary")
class RoPE(Op):
    def __init__(self, name: str, dim: int):
        super().__init__(name)
        self.dim = dim

    def set_cos_sin_table(self, cos_sin_table: Weight) -> None:
        assert cos_sin_table.data_type == DataType.float16, "cos_sin_table data type must be float16"
        self._weights['cos_sin_table'] = cos_sin_table

    def get_cos_sin_table(self) -> Weight:
        return self._weights['cos_sin_table']

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]


@register_op("layernorm")
class Layernorm(Op):
    def __init__(self, name: str, var_epsilon: float = 1e-5, affine:bool = False):
        super().__init__(name)
        self.var_epsilon = var_epsilon
        self.affine = affine

    def set_weight_bias(self, *, gamma: Weight, beta: Optional[Weight] = None):
        assert gamma.data_type == DataType.float16, "gamma data type must be float16"
        self._weights["gamma"] = gamma
        if beta is not None:
            if not self.affine:
                raise ValueError("layernorm without affine can't set beta")
            assert beta.data_type == DataType.float16, "beta data type must be float16"
            self._weights["beta"] = beta
    
    @property
    def get_gamma(self):
        return self._weights['gamma']
    
    @property
    def get_beta(self):
        return self._weights['beta']

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]


@register_op("rmsnorm")
class RMSnorm(Op):
    def __init__(self, name: str, var_epsilon: float = 1e-5, affine: bool = False):
        super().__init__(name)
        self.var_epsilon = var_epsilon
        self.affine = affine

    def set_weight(self, weight: Weight):
        assert weight.data_type == DataType.float16, "weight data type must be float16"
        self._weights['weight'] = weight
    
    @property
    def get_weight(self):
        return self._weights['weight']

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]


@register_op("softmax")
class Softmax(Op):
    def __init__(self, name: str, dim: int = -1):
        super().__init__(name)
        self.dim: int = dim

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]


@register_op("gelu")
class Gelu(Op):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]


@register_op("silu")
class Silu(Op):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]


@register_op("transpose")
class Transpose(Op):
    def __init__(self, name: str, dim_a: int, dim_b: int):
        super().__init__(name)
        self.dim_a: int = dim_a
        self.dim_b: int = dim_b

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]

    @property
    def get_trans_dims(self):
        return [self.dim_a, self.dim_b]

    def __str__(self) -> str:
        ret = f'''
name: {self.name}
type: {type(self).op_type}
dim_a: {self.dim_a}, dim_b: {self.dim_b}\n'''
        ret += 'inputs:\n'
        for in_ in self._inputs:
            ret += str(in_) + '\n'
        ret += 'outputs:\n'
        for out_ in self._outputs:
            ret += str(out_) + '\n'
        ret += 'weights:\n'
        for k, w in self._weights.items():
            ret += f'\nkey: {k}\n{str(w)}'
        ret += '\n'
        return ret

    def show_input_shape(self, name: str):
        Shape_info = super().show_input_shape(name)
        output_shape_info = list()
        for output in self._outputs:
            # print(f"{name} output: {str(output.shape)}")
            output_shape_info.append(output.shape)
        return Shape_info+ (output_shape_info, )

@register_op("split")
class Split(Op):
    def __init__(self, name: str, split_size: int, dim: int):
        super().__init__(name)
        self.split_size: int = split_size
        self.dim: int = dim

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs

    @property
    def get_split_size(self):
        return [self.split_size, self.dim]


@register_op("view")
class View(Op):
    def __init__(self, name: str, shape: Shape):
        super().__init__(name)
        self.shape = shape

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]
    def isView(self):
        return True

@register_op('convert')
class Convert(Op):
    def __init__(self, name: str, out_type: DataType):
        super().__init__(name)
        self.out_type = out_type

    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]

# TODO: to fuse
@register_op("div")
class Div(Op):
    def __init__(self, name: str, divisor: Tensor):
        super().__init__(name)
        self.divisor = divisor

    @property
    def get_input(self):
        return self._inputs[0]
    
    @property
    def get_input_A(self):
        return self._inputs[0]

    @property
    def get_input_B(self):
        return self._inputs[1]

    @property
    def get_divisor(self):
        return self.divisor

    @property
    def get_output(self):
        return self._outputs[0]

# for tensor slice pass, only sync instruction
@register_op('sync_op')
class Sync(Op):
    def __init__(self, name: str):
        super().__init__(name)
        
@register_op('cpoy')
class Copy(Op):
    def __init__(self, name: str):
        super().__init__(name)
    
    @property
    def get_input(self):
        return self._inputs[0]

    @property
    def get_output(self):
        return self._outputs[0]


@register_op('fakeload')
class FakeLoad(Op):
    def __init__(self, name: str, stride:int =0):
        super().__init__(name)
        self.stride : int = stride
    @property
    def get_input(self):
        return self._inputs[0]
    @property
    def get_stride(self):
        return self.stride
    @property
    def get_output(self):
        return self._outputs[0]
@register_op('nop_op')
class Nop(Op):
    def __init__(self, name: str):
        super().__init__(name)

# fuse op
@register_op("linearw8_transpose")
class LinearW8Transpose(LinearW8):
    # (B, S, D) -> (B, S, H, _D) -> (B, H, S, _D)
    def __init__(self, name: str, head_num: int, dim: int, bias_flag: bool = False):
        super().__init__(name, bias_flag)
        self.head_num = head_num
        self.dim = dim

@register_op("div_cvt_matmul")
class DivCvtMatmul(Matmul):
    def __init__(self, name: str, divisor: Tensor, out_type: DataType):
        super().__init__(name)
        self.divisor = divisor
        self.out_type = out_type # cvt 输出type
        
    @property
    def get_divisor(self):
        return self.divisor
    
    @property
    def get_mm_type(self):
        return self.out_type

@register_op("softmax_cvt_matmul")
class SoftmaxCvtMatmul(Matmul):
    def __init__(self, name: str, out_type: DataType, dim: int = -1):
        super().__init__(name)
        self.dim = dim
        self.out_type = out_type
    
    @property
    def get_mm_type(self):
        return self.out_type

@register_op("transpose_cvt")
class TransposeCvt(Transpose):
    def __init__(self, name: str, dim_a: int, dim_b: int, out_type: DataType):
        super().__init__(name, dim_a, dim_b)
        self.out_type = out_type
        if dim_a == 2 and dim_b == 3: #only support transpose23
            pass
        else:
            raise NotImplementedError
        
@register_op("linearw8_act")
class LinearW8Act(LinearW8):
    def __init__(self, name: str, bias_flag: bool, act_type: str):
        super().__init__(name, bias_flag)
        self.act_type = act_type

@register_op("cvt_linearw8")
class CvtLinearW8(LinearW8):
    def __init__(self, name: str, bias_flag: bool = False):
        super().__init__(name, bias_flag)
    