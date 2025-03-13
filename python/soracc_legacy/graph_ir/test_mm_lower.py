from .graph_ir import *
from .mm_lower import *
from .pass_impl import *

def test_mm_lower():
    return 
    in_feature = 4
    out_feature = 8

    weight = Weight('test_weight', Shape(in_feature, out_feature), DataType.int8)
    weight_scale = Weight('test_weight_scale', Shape(in_feature, out_feature), DataType.float16)

    g = StaticGraph()
    x = Tensor(Shape(2, 3, in_feature))
    op = LinearW8('test_linear_w8')
    op.add_input_tensors((x, ))
    op.set_weight_scale(weight=weight, weight_scale=weight_scale)
    g.add(op)
    new_shape = x.shape[0:-1] + (out_feature, )
    print(new_shape)
    ret = Tensor(Shape(*new_shape))
    op.set_outputs([ret])

    ddr_address_alloc(g)

    inst_collect = InstCollector()

    lower_linear(op, inst_collect)

    print(inst_collect.to_json())


