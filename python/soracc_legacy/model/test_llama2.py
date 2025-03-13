from graph_ir import *
from .sora_model import MLP

def llama_linear():
    g = StaticGraph()
    N = 5
    hid = 4096
    q_proj_layer_0 = Linear(g, hid, hid, bias=False)

    hidden_states_in = Tensor(Shape(1, N, hid))
    hidden_states_in.name = "model.layers.0.self_attn.q_proj.act_scale"
    hidden_states_in.data_type = DataType.float16

    q_proj_out = q_proj_layer_0(hidden_states_in)
    q_proj_out.name = 'b'
    q_proj_out.data_type = DataType.float16

    # g.show()

    for op in g.get_ops():
        print(f'type: {op.get_type()}')

    print(f'len: {len(g.get_ops())}')
    print(StaticGraph.registered_op)


def llama_mm():
    g = StaticGraph()
    matmul = Matmul(g)
    sm = Softmax(g, dim=2)

    heads = 32
    head_dim = 128
    N = 16
    query = Tensor(Shape(heads, N, head_dim))
    key_t = Tensor(Shape(heads, head_dim , N))
    qkt_mm = matmul(query, key_t)
    qkt_softmax = sm(qkt_mm)

    g.show()


def test_sora_block():
    g = StaticGraph()

    mlp_layer = MLP(graph=g, in_features=512, hidden_features=1024, out_features=512)
    in_tensor = Tensor(Shape(2, 4, 6, 512), dtype=DataType.int8)
    out_tensor = mlp_layer(in_tensor)
    assert out_tensor.shape == in_tensor.shape

    print(f'len: {len(g.get_ops())}')
