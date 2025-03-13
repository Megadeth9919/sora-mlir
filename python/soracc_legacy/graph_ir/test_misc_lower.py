from .graph_ir import *
from .misc_lower import *
from .pass_impl import *
import json

def test_copy_lower():
    in_shape = (1, 3, 224, 224)
    out_shape = (1, 3, 224, 224)
    x = Tensor(Shape(*in_shape), dtype=DataType.float16, name="x")
    op = Copy("copy")
    op.add_input_tensors((x,))
    y = op.create_tensor(Shape(*in_shape), dtype=DataType.float16, name=f"y")
    x.addr = 0
    y.addr = 1000000
    inst = InstCollector()
    lower_copy_hbm(op, inst)
    with open("test_copy_lower.json", "w") as f:
        json.dump(inst.to_json(), f, indent=4)