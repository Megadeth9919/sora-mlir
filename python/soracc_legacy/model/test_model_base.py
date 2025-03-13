from .model_base import *
import numpy as np


def test_Linear():
    g = StaticGraph()

    linear = Linear(g, 1024, 512, bias=False)
    in_tensor = Tensor(Shape(3, 1024), dtype=DataType.int8)
    out_tensor = linear(in_tensor)
    assert out_tensor.shape == (3, 512)


def test_transpose():
    g = StaticGraph()

    transpose = Transpose(g, 0, 2)
    in_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.int8)
    out_tensor = transpose(in_tensor)
    assert out_tensor.shape == (6, 4, 2, 8)


def test_view():
    g = StaticGraph()

    in_tensor = Tensor(Shape(2, 4, 48), dtype=DataType.int8)
    view = View(g, Shape(2, 4, 8, 6))
    out_tensor = view(in_tensor)
    assert out_tensor.shape == (2, 4, 8, 6)


def test_split():
    g = StaticGraph()

    split = Split(g, 2, 1)
    in_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.int8)
    out_tensors = split(in_tensor)
    assert len(out_tensors) == 2
    assert out_tensors[0].shape == (2, 2, 6, 8)
    assert out_tensors[1].shape == (2, 2, 6, 8)


def test_repeat():
    g = StaticGraph()

    repeat = Repeat(g, 2, 1)
    in_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.int8)
    out_tensor = repeat(in_tensor)
    assert out_tensor.shape == (2, 8, 6, 8)


def test_eltwise():
    g = StaticGraph()

    elt = Eltwise(g, 'add')
    a_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.float16)
    b_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.float16)
    out_tensor = elt(a_tensor, b_tensor)
    assert out_tensor.shape == (2, 4, 6, 8)


def test_matmul():
    g = StaticGraph()

    matmul = Matmul(g)
    a_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.float16)
    b_tensor = Tensor(Shape(2, 4, 10, 8), dtype=DataType.float16)
    out_tensor = matmul(a_tensor, b_tensor)
    assert out_tensor.shape == (2, 4, 6, 10)


def test_softmax():
    g = StaticGraph()

    softmax = Softmax(g, -1)
    in_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.float16)
    out_tensor = softmax(in_tensor)
    assert out_tensor.shape == in_tensor.shape


def test_normalize():
    g = StaticGraph()

    normalize = RMSNorm(g, Shape(1, 1, 8))
    in_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.float16)
    out_tensor = normalize(in_tensor)
    assert out_tensor.shape == in_tensor.shape


def test_activation():
    g = StaticGraph()

    silu = Silu(g)
    in_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.float16)
    out_tensor = silu(in_tensor)
    assert out_tensor.shape == (2, 4, 6, 8)


def test_div():
    g = StaticGraph()

    div = Div(g)
    a_tensor = Tensor(Shape(2, 4, 6, 8), dtype=DataType.float16)
    divisor = Tensor(Shape(1, 1, 1, 1), dtype=DataType.float16)
    divisor.set_data(np.array([[[[2.0]]]], dtype=np.float16))
    out_tensor = div(a_tensor, divisor)
    assert out_tensor.shape == (2, 4, 6, 8)
