from .sora_model import *
import pytest


def test_mlp():
    g = StaticGraph()

    mlp_layer = MLP(graph=g, in_features=512, hidden_features=1024, out_features=512)
    in_tensor = Tensor(Shape(2, 4, 6, 512), dtype=DataType.int8)
    out_tensor = mlp_layer(in_tensor)
    assert out_tensor.shape == in_tensor.shape


def test_attention():
    g = StaticGraph()

    attention = Attention(
        graph=g,
        dim=512,
    )
    in_tensor = Tensor(Shape(4, 6, 512), dtype=DataType.int8)
    out_tensor = attention(in_tensor)
    assert out_tensor.shape == in_tensor.shape

@pytest.mark.skip(reason="not implemented yet")
def test_cross_attention():
    g = StaticGraph()

    cross_attention = CrossAttention(graph=g, dim=512)
    in_tensor = Tensor(Shape(4, 6, 512), dtype=DataType.int8)
    cond_tensor = Tensor(Shape(4, 6, 512), dtype=DataType.int8)
    out_tensor = cross_attention(in_tensor, cond_tensor)
    assert out_tensor.shape == in_tensor.shape

@pytest.mark.skip(reason="not implemented yet")
def test_STDiT3Block():
    g = StaticGraph()

    B, T, S = 2, 4, 144
    hidden_size = 1152
    cond_size = 54
    block = STDiT3Block(graph=g, hidden_size=hidden_size, num_heads=16)
    x_tensor = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16)
    y_tensor = Tensor(Shape(B, cond_size, hidden_size), dtype=DataType.int8)
    t_tensor = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16)
    mask = Tensor(Shape(4, T * S, 1), dtype=DataType.float16)
    out_tensor = block(x_tensor, y_tensor, t_tensor, T=T, S=S)
    assert x_tensor.shape == out_tensor.shape

@pytest.mark.skip(reason="not implemented yet")
def test_STDiT3BlockOnly():
    g = StaticGraph()

    T, S = 4, 144
    B = 2
    cond_size = 52
    hidden_size = 1152
    block = STDiT3BlockOnly(graph=g, hidden_size=hidden_size, num_heads=16)
    x_tensor = Tensor(Shape(B, T * S, hidden_size), dtype=DataType.float16)
    y_tensor = Tensor(Shape(B, cond_size, hidden_size), dtype=DataType.float16)
    t_tensor = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16)
    t_mlp_tensor = Tensor(Shape(B, 6, hidden_size), dtype=DataType.float16)

    out_tensor = block(x_tensor, y_tensor, t_tensor, t_mlp_tensor, T=T, S=S)
    assert out_tensor.shape[:-1] == x_tensor.shape[:-1] and out_tensor.shape[-1] == hidden_size


@pytest.mark.skip(reason="not implemented yet")
def test_STDiT3():
    g = StaticGraph()

    B, C, T, H, W = 2, 4, 16, 12, 12
    cond_size = 52

    caption_size = 4096

    video = Tensor(Shape(B, C, T, H, W), dtype=DataType.float16)
    text = Tensor(Shape(B, cond_size, caption_size), dtype=DataType.int8)
    t = Tensor(Shape(B, T), dtype=DataType.int8)
    fps = Tensor(Shape(B, T), dtype=DataType.float16)

    model = STDiT3(graph=g)
    out_tensor = model(video, t, text, fps=fps, height=H, width=W)
    assert out_tensor.shape == (B, C, T, H, W)
