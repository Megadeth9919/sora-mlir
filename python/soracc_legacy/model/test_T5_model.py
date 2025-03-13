from .sora_model import *


def test_gate_mlp():
    g = StaticGraph()

    mlp_layer = MLP(graph=g, in_features=512, hidden_features=1024, out_features=512, gated_flag=True)
    in_tensor = Tensor(Shape(2, 4, 6, 512), dtype=DataType.int8)
    out_tensor = mlp_layer(in_tensor)
    assert out_tensor.shape == in_tensor.shape


def test_T5_block():
    g = StaticGraph()

    t5_block = T5Block(graph=g)
    in_tensor = Tensor(Shape(2, 54, 4096), dtype=DataType.float16)
    out_tensor = t5_block(in_tensor)
    assert out_tensor.shape == in_tensor.shape


def test_T5_xxl():
    pass
