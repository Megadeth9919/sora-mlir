from .graph_ir import StaticGraph, Tensor, LinearW8, Shape

def test_graph_serilize():
    g = StaticGraph()
    op = LinearW8('linear_w8')
    x = Tensor(Shape((2, 3, 4)))
    op.add_input_tensors((x, ))
    g.add(op)
    new_shape = Shape(3, 4, 5)
    y = Tensor(new_shape, op)
    ret = g.to_json()

    return 

def test_graph_serilize_unserilize():
    g = StaticGraph()
    op = LinearW8('linear_w8')
    x = Tensor(Shape((2, 3, 4)))
    op.add_input_tensors((x, ))
    g.add(op)
    new_shape = Shape(3, 4, 5)
    y = Tensor(new_shape, op)
    ret = g.to_json()

    ng = StaticGraph()
    ng.from_json(ret)
    return 

def test_print_graph():
    g = StaticGraph()
    op = LinearW8('linear_w8')
    x = Tensor(Shape(2, 3, 4), name='x')
    op.add_input_tensors((x, ))
    g.add(op)
    new_shape = Shape(3, 4, 5)
    y = op.create_tensor(new_shape, name='y')

    print(g)
    return
