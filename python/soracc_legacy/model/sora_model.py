from typing import Callable

import numpy as np
from numpy.random import randn
from utils.compile_utils import generate_cross_attn_mask
from .model_base import *

_scale_ones = Tensor(shape=Shape(2, 1, 1152), dtype=DataType.float16, const=True, name="scale_ones")
_crs_attn_scale = None
_tem_attn_scale = None
_spa_attn_scale = None
_qkt_mask = None

_scale_ones.set_data(np.ones((2, 1, 1152), dtype=np.float16))
def t2i_modulate(x: Tensor, shift: Tensor, scale: Tensor,
                 graph: StaticGraph, out_dtype: Optional[DataType] = None, name: str = "x") -> Tensor:
    new_dtype = out_dtype if out_dtype is not None else x.data_type
    scale = Eltwise(graph, t="add", name=name + "_scale_add_ones")(scale, _scale_ones, out_dtype=DataType.float16)
    # scale.const = True
    x = Eltwise(graph, t="mul", name=name + "_mul_scale")(x, scale)
    x = Eltwise(graph, t="add", name=name + "_add_shift")(x, shift, out_dtype=new_dtype)
    return x


class MLP:
    def __init__(
            self,
            graph: StaticGraph,
            in_features: int,
            hidden_features: int = 1152,
            out_features: int = 1152,
            act_layer: str = "gelu",
            norm_layer: str = "instance",
            linear_bias: bool = True,
            gated_flag: bool = False,
            **kwargs
    ):
        assert graph is not None, "graph must be provided"
        assert act_layer in ["relu", "gelu", "silu"], "unsupported activation layer: {}".format(act_layer)
        assert norm_layer in ["layer", "instance", "rms"], "unsupported normalization layer: {}".format(norm_layer)
        if kwargs:
            if "name" in kwargs:
                self.prefix = kwargs["name"] + "."
            else:
                assert "stage" in kwargs and "layer_idx" in kwargs, "kwargs must contain stage and layer_idx"
                self.prefix = f"{kwargs['stage']}_blocks.{kwargs['layer_idx']}.mlp."
        else:
            self.prefix = "mlp."

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = Linear(graph, name=self.prefix + "fc1", in_feature=in_features, out_feature=hidden_features,
                          bias=linear_bias)
        self.act = act_func(type=act_layer, graph=graph, name=self.prefix + act_layer)
        self.norm = norm_func(type=norm_layer, graph=graph, hidden_size=hidden_features, name=self.prefix + norm_layer)
        self.fc2 = Linear(graph, name=self.prefix + "fc2", in_feature=hidden_features, out_feature=out_features,
                          bias=linear_bias)

        self.gated_flag = gated_flag
        self.gated_linear = Linear(graph, name=self.prefix + "gate", in_feature=in_features, out_feature=hidden_features,
                                   bias=linear_bias) if gated_flag else None

        self._graph = graph

    def __call__(self, x: Tensor, out_dtype: Optional[DataType] = None):
        new_dtype = out_dtype if out_dtype is not None else x.data_type
        if x.data_type == DataType.float16:
            x = Convert(self._graph, DataType.int8, name=self.prefix + "convert_to_int8")(x)

        if self.gated_linear:
            gated_x = self.gated_linear(x, act_scale=x.get_act_scale, out_dtype=DataType.float16)

        x = self.fc1(x, act_scale=x.get_act_scale, out_dtype=DataType.float16)
        x = self.act(x, out_dtype=(DataType.float16 if (self.norm or self.gated_flag) else DataType.int8))

        if self.gated_flag:
            x = Eltwise(self._graph, t="mul", name=self.prefix + "gated_x")(x, gated_x, out_dtype=DataType.float16 if self.norm else DataType.int8)

        x = self.norm(x, out_dtype=DataType.int8) if self.norm else x
        x = self.fc2(x, act_scale=x.get_act_scale, out_dtype=new_dtype)
        return x


class Attention:    
    def __init__(
            self,
            graph: StaticGraph,
            dim: int,
            num_heads: int = 16,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            norm_layer: str = "rms",
            rope: bool = False,
            matmul_int: bool = False,
            **kwargs,
    ):
        """
            Attention as used in Vision Transformer and related networks
        """
        assert graph is not None, "graph must be provided"
        assert norm_layer in ["layer", "instance", "rms"], "unsupported normalization layer: {}".format(norm_layer)

        self._stage = kwargs.get("stage", "unknown")
        self._layer_idx = kwargs.get("layer_idx", 0)
        self._prefix = f"{self._stage}_blocks.{self._layer_idx}.attn."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.matmul_int = matmul_int
        
        self.linear_q = Linear(graph, in_feature=dim, out_feature=dim, bias=qkv_bias, name=self._prefix + "q")
        self.linear_k = Linear(graph, in_feature=dim, out_feature=dim, bias=qkv_bias, name=self._prefix + "k")
        self.linear_v = Linear(graph, in_feature=dim, out_feature=dim, bias=qkv_bias, name=self._prefix + "v")
        self.linear_out = Linear(graph, in_feature=dim, out_feature=dim, bias=True, name=self._prefix + "proj")

        self.mutual_qkt = Matmul(graph, name=self._prefix + "qkt_mutual")
        self.mutual_attn = Matmul(graph, name=self._prefix + "attn_matmul")

        self.softmax = Softmax(graph, dim=-1, name=self._prefix + "softmax")

        self.rope_flag = rope
        if rope:
            self.q_rope: Optional[RoPE] = RoPE(graph, dim=dim, name=self._prefix + "q_rope")
            self.k_rope: Optional[RoPE] = RoPE(graph, dim=dim, name=self._prefix + "k_rope")
        else:
            self.q_rope = None
            self.k_rope = None

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = norm_func(graph=graph, hidden_size=self.head_dim, type=norm_layer,
                                    name=self._prefix + "q_norm")
            self.k_norm = norm_func(graph=graph, hidden_size=self.head_dim, type=norm_layer,
                                    name=self._prefix + "k_norm")
        else:
            self.q_norm = None
            self.k_norm = None

        self._graph = graph

    def __call__(self, x: Tensor, out_dtype: Optional[DataType] = None):
        assert len(x.shape) == 3, "Input tensor must have 3 dimensions (B, N, C)"
        new_dtype = out_dtype if out_dtype is not None else x.data_type

        B, N, C = x.shape
        q = self.linear_q(x, act_scale=x.get_act_scale, out_dtype=DataType.float16)
        k = self.linear_k(x, act_scale=x.get_act_scale, out_dtype=DataType.float16)
        v = self.linear_v(x, act_scale=x.get_act_scale, out_dtype=DataType.float16)

        qkv_shape = Shape(B, N, self.num_heads, self.head_dim)
        q = View(self._graph, qkv_shape, name=self._prefix + "q_view")(q)
        k = View(self._graph, qkv_shape, name=self._prefix + "k_view")(k)
        v = View(self._graph, qkv_shape, name=self._prefix + "v_view")(v)
        q = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "q_head_permute")(q)
        k = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "k_head_permute")(k)
        v = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "v_head_permute")(v)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k, out_dtype=DataType.int8 if (self.matmul_int and self.k_rope is None) else DataType.float16)

        if self.q_rope is not None and self.k_rope is not None:
            q = self.q_rope(q)
            k = self.k_rope(k, DataType.float16)
            if self.matmul_int:
                k = Convert(self._graph, DataType.int8, name=self._prefix + "rope_k_convert_to_int8")(k)
        
        if self._stage == "temporal":
            global _tem_attn_scale
            if _tem_attn_scale is None:
                _tem_attn_scale = Tensor(shape=q.shape, dtype=DataType.float16, name="tem_attn.q_scale", const=True)
                scale_data = np.zeros(q.shape)
                scale_data.fill(self.scale)
                _tem_attn_scale.set_data(scale_data.astype(np.float16))
        elif self._stage == "spatial":
            global _spa_attn_scale
            if _spa_attn_scale is None:
                _spa_attn_scale = Tensor(shape=q.shape, dtype=DataType.float16, name="spa_attn.q_scale", const=True)
                scale_data = np.zeros(q.shape)
                scale_data.fill(self.scale)
                _spa_attn_scale.set_data(scale_data.astype(np.float16))
        
        if self._stage == "temporal":
            assert _tem_attn_scale is not None
            q = Div(self._graph, name=self._prefix + "q_div")(q, _tem_attn_scale, out_dtype=DataType.int8 if self.matmul_int else DataType.float16)
        elif self._stage == "spatial":
            assert _spa_attn_scale is not None
            q = Div(self._graph, name=self._prefix + "q_div")(q, _spa_attn_scale, out_dtype=DataType.int8 if self.matmul_int else DataType.float16)

        if self.matmul_int:
            pass
            # q = Convert(self._graph, DataType.int8, name=self._prefix + "q_convert_to_int8")(q)
            # k = Convert(self._graph, DataType.int8, name=self._prefix + "k_convert_to_int8")(k)

        qkt = self.mutual_qkt(q, k)

        atten = self.softmax(qkt, out_dtype=DataType.int8 if self.matmul_int else DataType.float16)

        v_trans = Transpose(self._graph, dim_a=3, dim_b=2, name=self._prefix + "v_trans")(v)
        
        if self.matmul_int:
            # atten = Convert(self._graph, DataType.int8, name=self._prefix + "atten_convert_to_int8")(atten)
            v_trans = Convert(self._graph, DataType.int8, name=self._prefix + "v_trans_convert_to_int8")(v_trans)

        x = self.mutual_attn(atten, v_trans)
        
        x = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "attn_o_permute")(x)
        x = View(self._graph, Shape(B, N, C), name=self._prefix + "output_view")(x)

        x = Convert(self._graph, DataType.int8, name=self._prefix + "convert_to_int8")(x)
        output = self.linear_out(x, act_scale=x.get_act_scale, out_dtype=new_dtype)

        return output


class CrossAttention:
    def __init__(
            self,
            graph: StaticGraph, 
            dim: int,
            #cross_attn_mask: Tensor,
            num_heads: int = 16,
            matmul_int: bool = False,
            **kwargs,
    ):
        """
            Cross-Attention as used in Vision Transformer and related networks
        """
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self._stage = kwargs.get("stage", "unknown")
        self._layer_idx = kwargs.get("layer_idx", 0)
        self._prefix = f"{self._stage}_blocks.{self._layer_idx}.cross_attn."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.matmul_int = matmul_int

        self.q_linear = Linear(graph, in_feature=dim, out_feature=dim, bias=True, name=self._prefix + "q_linear")
        self.k_linear = Linear(graph, in_feature=dim, out_feature=dim, bias=True, name=self._prefix + "k_linear")
        self.v_linear = Linear(graph, in_feature=dim, out_feature=dim, bias=True, name=self._prefix + "v_linear")
        self.output_proj = Linear(graph, in_feature=dim, out_feature=dim, bias=True, name=self._prefix + "proj")

        self.mutual_qkt = Matmul(graph, name=self._prefix + "qkt_mutual")
        self.mutual_attn = Matmul(graph, name=self._prefix + "attn_matmul")
        self.softmax = Softmax(graph, dim=-1, name=self._prefix + "softmax")

        self.scale = self.head_dim ** 0.5

        # self.cross_attn_mask = cross_attn_mask # cross attn mask for CFG cross attn

        self._graph = graph

    def __call__(self, x: Tensor, cond: Tensor, out_dtype: Optional[DataType] = None):
        new_dtype = out_dtype if out_dtype is not None else x.data_type
        x_B, x_N, x_C = x.shape
        cond_B, cond_N, cond_C = cond.shape

        q = self.q_linear(x, act_scale=x.get_act_scale, out_dtype=DataType.float16)
        k = self.k_linear(cond, act_scale=cond.get_act_scale, out_dtype=DataType.float16)
        v = self.v_linear(cond, act_scale=cond.get_act_scale, out_dtype=DataType.float16)

        q_shape = Shape(1, x_B * x_N, self.num_heads, self.head_dim)
        kv_shape = Shape(1, cond_B * cond_N, self.num_heads, self.head_dim)

        q = View(self._graph, q_shape, name=self._prefix + "q_view")(q)
        k = View(self._graph, kv_shape, name=self._prefix + "k_view")(k)
        v = View(self._graph, kv_shape, name=self._prefix + "v_view")(v)
        q = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "q_head_permute")(q)
        k = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "k_head_permute")(k)
        v = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "v_head_permute")(v)
        
        global _crs_attn_scale
        if _crs_attn_scale is None:
            _crs_attn_scale = Tensor(shape=q.shape, dtype=DataType.float16, name="cross_attn.q_scale", const=True)
            scale_data = np.zeros(q.shape)
            scale_data.fill(self.scale)
            _crs_attn_scale.set_data(scale_data.astype(np.float16))

        assert _crs_attn_scale is not None
        q = Div(self._graph, name=self._prefix + "q_div")(q, _crs_attn_scale, out_dtype=DataType.int8 if self.matmul_int else DataType.float16)

        if self.matmul_int:
            # q = Convert(self._graph, DataType.int8, name=self._prefix + "q_convert_to_int8")(q)
            k = Convert(self._graph, DataType.int8, name=self._prefix + "k_convert_to_int8")(k)

        qkt = self.mutual_qkt(q, k)
        

        global _qkt_mask
        if _qkt_mask is None:
            _qkt_mask = Tensor(shape=Shape(1, 1, x_B * x_N, cond_N), dtype=DataType.float16, name="cross_attn_mask", const=True)
            _cross_attn_mask = generate_cross_attn_mask(x_B, x_N, cond_N)
            _cross_attn_mask = _cross_attn_mask.astype(np.float16)
            _qkt_mask.set_data(_cross_attn_mask)

        qkt = Eltwise(self._graph, t="add", name=self._prefix + "cross_attn_mask_add")(qkt, _qkt_mask)

        atten = self.softmax(qkt, out_dtype=DataType.int8 if self.matmul_int else DataType.float16)

        v_trans = Transpose(self._graph, dim_a=3, dim_b=2, name=self._prefix + "v_trans")(v)
        
        if self.matmul_int:
            # atten = Convert(self._graph, DataType.int8, name=self._prefix + "atten_convert_to_int8")(atten)
            v_trans = Convert(self._graph, DataType.int8, name=self._prefix + "v_trans_convert_to_int8")(v_trans)

        x = self.mutual_attn(atten, v_trans)
        
        x = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "attn_o_permute")(x)
        x = View(self._graph, Shape(x_B, x_N, x_C), name=self._prefix + "output_view")(x)
        
        x = Convert(self._graph, DataType.int8, name=self._prefix + "convert_to_int8")(x)

        output = self.output_proj(x, act_scale=x.get_act_scale, out_dtype=new_dtype)
        return output


# Encoder-Only
class T5Block:
    def __init__(
            self,
            graph: StaticGraph,
            d_model: int = 4096,
            d_ff: int = 10240,
            ff_act: str = "gelu",
            num_heads: int = 64,
            encoder_flag: bool = True,
            layer_idx: int = 0,
    ):
        assert encoder_flag
        self._stage = "encoder" if encoder_flag else "decoder"
        self._layer_idx = layer_idx
        self._prefix = f"T5_{self._stage}_blocks.{self._layer_idx}."

        self._graph = graph

        self.atten_layer_norm = norm_func(graph, type="layer", hidden_size=d_model, affine=True,
                                          name=self._prefix + "atten_layer_norm")
        self.attn = Attention(
            graph,
            dim=d_model,
            num_heads=num_heads,
            qkv_bias=False,
            qk_norm=False,
            norm_layer="layer",
            stage=self._stage,
            layer_idx=self._layer_idx,
        )

        self.mlp_layer_norm = norm_func(graph, type="layer", hidden_size=d_model, affine=True,
                                        name=self._prefix + "mlp_layer_norm")
        self.T5_ff = MLP(
            graph,
            in_features=d_model,
            hidden_features=d_ff,
            out_features=d_model,
            act_layer=ff_act,
            gated_flag=True,
        )

    def __call__(self, x: Tensor, out_dtype: Optional[DataType] = None):
        new_dtype = out_dtype if out_dtype is not None else x.data_type

        att_in = self.atten_layer_norm(x, out_dtype=DataType.int8)
        att_out = self.attn(att_in, out_dtype=DataType.float16)
        att_out = Eltwise(self._graph, t="add", name=self._prefix + "att_out")(att_out, x)

        mlp_in = self.mlp_layer_norm(att_out, out_dtype=DataType.int8)
        mlp_out = self.T5_ff(mlp_in, out_dtype=DataType.float16)
        mlp_out = Eltwise(self._graph, t="add", name=self._prefix + "mlp_out")(mlp_out, att_out, out_dtype=new_dtype)

        res_out = Eltwise(self._graph, t="add", name=self._prefix + "res_out")(mlp_out, x)

        return res_out


class T5Stack:
    def __init__(
            self,
    ):
        pass


class STDiT3Block:
    def __init__(
            self,
            graph: StaticGraph,
            hidden_size: int,
            num_heads: int,
            # cross_attn_mask: Tensor,
            mlp_ratio: float = 4.0,
            rope: bool = False,
            temporal: bool = False,
            layer_idx: int = 0,
            matmul_int: bool = False,
            do_cache: bool = False,
            re_compute: bool = True,
    ):
        self._stage = "temporal" if temporal else "spatial"
        self._layer_idx = layer_idx
        self._prefix = f"{self._stage}_blocks.{self._layer_idx}."

        self._graph = graph
        self.matmul_int = matmul_int
        self.do_cache = do_cache
        self.re_compute = re_compute

        self.temporal = temporal
        self.hidden_size = hidden_size

        self.attn_cls = Attention
        self.mha_cls = CrossAttention

        self.norm1 = norm_func(type="layer", graph=graph, hidden_size=hidden_size, name=self._prefix + "norm1", affine=False)
        self.attn = self.attn_cls(
            graph=graph,
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer="rms",
            rope=rope,
            stage=self._stage,
            layer_idx=self._layer_idx,
            matmul_int=matmul_int,
        )

        # cross attention
        self.cross_attn = self.mha_cls(
            dim=hidden_size,
            num_heads=num_heads,
            graph=graph,
            stage=self._stage,
            layer_idx=self._layer_idx,
            matmul_int=matmul_int,
            # cross_attn_mask=cross_attn_mask,
        )
        self.norm2 = norm_func(type="layer", graph=graph, hidden_size=hidden_size, name=self._prefix + "norm2", affine=False)

        # mlp
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            graph=graph,
            stage=self._stage,
            layer_idx=self._layer_idx,
        )
        self.scale_shift_table = Tensor(Shape(1, 6, hidden_size), name=self._prefix + "scale_shift_table",
                                        dtype=DataType.float16, const=True)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        B, _, C = x.shape
        x = View(self._graph, Shape(B, T, S, C), name=self._prefix + "x_view")(x)
        masked_x = View(self._graph, Shape(B, T, S, C), name=self._prefix + "masked_x_view")(masked_x)
        x_mask = View(self._graph, Shape(B, T, 1, 1), name=self._prefix + "x_mask_view")(x_mask)

        res = Eltwise(self._graph, t="add", name=self._prefix + "mask_select_add_0")(x, masked_x)
        x = Eltwise(self._graph, t="mul", name=self._prefix + "mask_select_mul")(x_mask, res)
        x = Eltwise(self._graph, t="add", name=self._prefix + "mask_select_add_1")(x, masked_x)

        x = View(self._graph, Shape(B, T * S, C))(x)
        return x

    def __call__(
            self,
            x,
            y,
            t,
            mask=None,  # text mask
            x_mask=None,  # temporal mask
            t0=None,  # t with timestamp=0
            T=None,  # number of frames
            S=None,
            out_dtype: Optional[DataType] = None,
    ): 
        if not self.re_compute:  # Use cached tensors
            x_m_s = Tensor(shape=x.shape, dtype=DataType.float16, name=f"{self._prefix}x_m_s_mul.output", cached=True)
            cattn_mlp_res = Tensor(shape=x.shape, dtype=DataType.float16, name=f"{self._prefix}cattn_mlp_res.output", cached=True)
        
        # prepare modulate parameters
        if y.data_type == DataType.float16 and self.re_compute:
            y = Convert(self._graph, DataType.int8, name=self._prefix + "y_int8")(y)
        new_dtype = out_dtype if out_dtype is not None else x.data_type
        B, N, C = x.shape
        
        if self.re_compute:
            t = View(self._graph, Shape(B, 6, C), name=self._prefix + "t_view")(t)
            scale_shift_table = Eltwise(self._graph, t="add", name=self._prefix + "scale_shift_table_add")(t, self.scale_shift_table)
            split_name = [f"{self._prefix}{name}" for name in ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                Split(self._graph, 6, dim=1, name=self._prefix + "scale_shift_table_split")(scale_shift_table, split_name=split_name)
            # shift_msa.const = scale_msa.const = gate_msa.const = shift_mlp.const = scale_mlp.const = gate_mlp.const = True
            
            if x_mask is not None:
                t0 = View(self._graph, Shape(B, 6, C), name=self._prefix + "t0_view")(t0)
                scale_shift_table_zero = Eltwise(self._graph, t="add", name=self._prefix + "scale_shift_table_zero_add")(
                    self.scale_shift_table, t0)
                split_name = [f"{self._prefix}{name}" for name in ["shift_msa_zero", "scale_msa_zero", "gate_msa_zero", "shift_mlp_zero", "scale_mlp_zero", "gate_mlp_zero"]]
                shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = \
                    Split(self._graph, 6, dim=1, name=self._prefix + "scale_shift_table_zero_split")(scale_shift_table_zero, split_name=split_name)
                # shift_msa_zero.const = scale_msa_zero.const = gate_msa_zero.const = shift_mlp_zero.const = scale_mlp_zero.const = gate_mlp_zero.const = True

        # modulate (attention)
        if self.re_compute:
            x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa, self._graph, out_dtype=DataType.int8,
                            name=self._prefix + "x_m_modulate")
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero, self._graph,
                                        name=self._prefix + "x_m_zero_modulate")
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # attention
        if self.re_compute:
            if self.temporal:
                x_m = View(self._graph, Shape(B, T, S, C), name=self._prefix + "x_m_view_for_transpose")(x_m)
                x_m = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "x_m_transpose")(x_m)
                x_m = View(self._graph, Shape(B * S, T, C), name=self._prefix + "x_m_view")(x_m)
                x_m = self.attn(x_m, out_dtype=DataType.float16)
                x_m = View(self._graph, Shape(B, S, T, C), name=self._prefix + "x_m_view_back_for_transpose")(x_m)
                x_m = Transpose(self._graph, dim_a=1, dim_b=2, name=self._prefix + "x_m_back_transpose")(x_m)
                x_m = View(self._graph, Shape(B, T * S, C), name=self._prefix + "x_m_view_back")(x_m)
            else:
                x_m = View(self._graph, Shape(B * T, S, C), name=self._prefix + "x_m_view")(x_m)
                x_m = self.attn(x_m, out_dtype=DataType.float16)
                x_m = View(self._graph, Shape(B, T * S, C), name=self._prefix + "x_m_view_back")(x_m)

            # modulate (attention)
            x_m_s = Eltwise(self._graph, t="mul", name=self._prefix + "x_m_s_mul")(x_m, gate_msa)
            if self.do_cache:
                x_m_s.cached = True

            if x_mask is not None:
                x_m_s_zero = Eltwise(self._graph, t="mul", name=self._prefix + "x_m_s_zero_mul")(x_m, gate_msa_zero)
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            # residual
            x = Eltwise(self._graph, t="add", name=self._prefix + "x_residual")(x, x_m_s)

        else:
            # cache residual
            x = Eltwise(self._graph, t="add", name=self._prefix + "x_residual")(x, x_m_s)


        if self.re_compute:
            # cross attention
            x_int8 = Convert(self._graph, out_dtype=DataType.int8, name=self._prefix + "x_int8_for_mlp")(x)

            cattn_x = self.cross_attn(x_int8, y, out_dtype=DataType.float16)
            
            # residual
            x = Eltwise(self._graph, t="add", name=self._prefix + "x_cross_residual")(x, cattn_x)

            # modulate (MLP)
            x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp, self._graph, out_dtype=DataType.int8,
                            name=self._prefix + "x_m_modulate_mlp")
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero, self._graph,
                                        name=self._prefix + "x_m_zero_modulate_mlp")
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            # MLP
            x_m = self.mlp(x_m, out_dtype=DataType.float16)

            # modulate (MLP)
            x_m_s = Eltwise(self._graph, t="mul", name=self._prefix + "x_m_s_mlp_mul")(x_m, gate_mlp)
            if x_mask is not None:
                x_m_s_zero = Eltwise(self._graph, t="mul")(gate_mlp_zero, x_m)
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            # residual
            x = Eltwise(self._graph, t="add", name=self._prefix + "x_residual_mlp")(x, x_m_s, out_dtype=new_dtype)

            if self.do_cache:
                cattn_mlp_res = Eltwise(self._graph, t="add", name=self._prefix + "cattn_mlp_res")(cattn_x, x_m_s)
                cattn_mlp_res.cached = True
        else:
            # cache residual
            x = Eltwise(self._graph, t="add", name=self._prefix + "x_residual_mlp")(x, cattn_mlp_res, out_dtype=new_dtype)

        return x


class PositionEmbedding2D:  # FIXME: have not supported yet
    def __init__(self, graph: StaticGraph, dim: int, ) -> None:
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        self.half_dim = dim // 2
        # self.inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.emb_sin = Tensor(Shape(self.half_dim // 2), dtype=DataType.float16)
        self.emb_cos = Tensor(Shape(self.half_dim // 2), dtype=DataType.float16)

        self._graph = graph

    def _get_sin_cos_emb(self, t: Tensor):  # t: [h * w] inv_freq: [half_dim // 2]
        pass

    def _get_cached_emb(
            self,
            h: int,
            w: int,
            scale: float = 1.0,
            base_size: int = 1,
    ):
        pass

    def __call__(
            self,
            x: Tensor,
            h: int,
            w: int,
            scale: Optional[float] = 1.0,
            base_size: Optional[int] = None,
    ) -> Tensor:
        # return self._get_cached_emb(h, w, scale, base_size)
        return Tensor(Shape(1, h * w, self.dim), dtype=DataType.float16)


class PatchEmbed3D:
    """Video to Patch Embedding.

    Args:
        patch_size (Shape): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
            self,
            graph: StaticGraph,
            patch_size=Shape(2, 4, 4),
            in_chans=3,
            embed_dim=96,
            norm_layer=None,
            flatten=True,
    ):
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv(name="patch_embed.proj", graph=graph, in_channel=in_chans, out_channel=embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_func(graph, type="layer", hidden_size=embed_dim)
        else:
            self.norm = None

        self._graph = graph

    def __call__(self, x: Tensor):
        """Forward function."""
        # padding
        assert len(x.shape) == 5
        _, _, D, H, W = x.shape
        # if W % self.patch_size[2] != 0:
        #     x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        # if H % self.patch_size[1] != 0:
        #     x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # if D % self.patch_size[0] != 0:
        #     x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            B, C, D, Wh, Ww = x.shape
            x = View(self._graph, Shape(B, C, D * Wh * Ww))(x)
            x = Transpose(self._graph, dim_a=3, dim_b=2)(x)
            x = self.norm(x)
            x = Transpose(self._graph, dim_a=3, dim_b=2)(x)
            x = View(self._graph, Shape(B, C, D, Wh, Ww))(x)

        if self.flatten:
            # BCTHW -> BNC
            B, C, D, Wh, Ww = x.shape
            x = View(self._graph, Shape(B, C, D * Wh * Ww))(x)
            x = Transpose(self._graph, dim_a=3, dim_b=2)(x)
        return x


class TimestepEmbedder:  # FIXME: have not supported yet
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, graph: StaticGraph, hidden_size, frequency_embedding_size=256):
        self.fc1 = Linear(name="t_embedder.mlp.0", graph=graph, in_feature=frequency_embedding_size, out_feature=hidden_size, bias=True)
        self.act = Silu(graph=graph)
        self.fc2 = Linear(name="t_embedder.mlp.2", graph=graph, in_feature=hidden_size, out_feature=hidden_size, bias=True)

        self.frequency_embedding_size = frequency_embedding_size

        self._graph = graph

    @staticmethod
    def timestep_embedding(t: Tensor, dim, max_period=10000):
        pass

    def __call__(self, t: Tensor):
        # t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = Tensor(Shape(1, t.shape[0] * t.shape[-1], self.frequency_embedding_size), dtype=DataType.int8)
        t_emb.set_act_scale(Tensor(Shape(1, t.shape[0] * t.shape[-1], 1), dtype=DataType.float16))

        t_emb = self.fc1(t_emb, act_scale=t_emb.get_act_scale, out_dtype=DataType.float16)
        t_emb = self.act(t_emb, out_dtype=DataType.int8)
        t_emb = self.fc2(t_emb, act_scale=t_emb.get_act_scale, out_dtype=DataType.float16)

        return t_emb


class SizeEmbedder:  # FIXME: have not supported yet
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, graph: StaticGraph, hidden_size: int, frequency_embedding_size=256):

        self.fc1 = Linear(name="fps_embedder.mlp.0", graph=graph, in_feature=frequency_embedding_size, out_feature=hidden_size, bias=True)
        self.act = Silu(graph=graph)
        self.fc2 = Linear(name="fps_embedder.mlp.2", graph=graph, in_feature=hidden_size, out_feature=hidden_size, bias=True)

        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

        self._graph = graph

    @staticmethod
    def timestep_embedding(t: Tensor, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        # half = dim // 2
        # freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        # args = t[:, None].float() * freqs[None]
        # embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # if dim % 2:
        #     embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        embedding = Tensor(Shape(1, t.shape[0] * t.shape[-1], dim), dtype=DataType.int8)
        embedding.set_act_scale(Tensor(Shape(1, t.shape[0] * t.shape[-1], 1), dtype=DataType.float16))
        return embedding

    def __call__(self, s: Tensor, bs: int):
        if len(s.shape) == 1:
            s = View(self._graph, Shape(1, s.shape[0]))(s)
        assert len(s.shape) == 2
        if s.shape[0] != bs:
            s = Repeat(self._graph, bs // s.shape[0], 0)(s)
            assert s.shape[0] == bs
        b, dims = s.shape
        s = View(self._graph, Shape(1, b * dims))(s)
        s_emb = self.timestep_embedding(s, self.frequency_embedding_size)

        s_emb = self.fc1(s_emb, act_scale=s_emb.get_act_scale, out_dtype=DataType.float16)
        s_emb = self.act(s_emb, out_dtype=DataType.int8)
        s_emb = self.fc2(s_emb, act_scale=s_emb.get_act_scale, out_dtype=DataType.float16)

        # s_emb = View(self._graph, Shape(1, b, dims * self.outdim))(s_emb)
        # , "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb


class T2IFinalLayer:
    def __init__(
            self,
            graph: StaticGraph,
            hidden_size: int,
            num_patches: int,
            out_channels: int,
            d_t=None,
            d_s=None,
            **kwargs
    ):
        self._graph = graph

        self.norm_layer = norm_func(name="final_layer", type="layer", graph=graph, hidden_size=hidden_size, affine=False)
        self.linear = Linear(graph, name="final_layer.linear", in_feature=hidden_size, out_feature=num_patches * out_channels, bias=True)

        self.scale_shift_table = Tensor(Shape(2, 1, hidden_size), name="final_layer.scale_shift_table",
                                        dtype=DataType.float16, const=True)

        self._hidden_size = hidden_size
        self._num_patches = num_patches
        self._out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        B, _, C = x.shape
        x = View(self._graph, Shape(B, T, S, C))(x)
        masked_x = View(self._graph, Shape(B, T, S, C))(masked_x)
        x_mask = View(self._graph, Shape(B, T, 1, 1))(x_mask)  # TODO: Default 1-True; 0-False

        res = Eltwise(self._graph, t="add")(x, masked_x)
        x = Eltwise(self._graph, t="mul")(x_mask, res)
        x = Eltwise(self._graph, t="add")(x, masked_x)

        x = View(self._graph, Shape(B, T * S, C))(x)
        return x

    def __call__(self, x: Tensor, t: Tensor, x_mask=None, t0=None, T=None, S=None, out_dtype=DataType.float16):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s

        scale_shift_table = Eltwise(self._graph, t="add")(t, self.scale_shift_table)
        split_name = [f"T2Final.{name}" for name in ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]]
        _, _, _, shift, scale, _ = Split(self._graph, 6, dim=1)(scale_shift_table, split_name=split_name)
        x = t2i_modulate(self.norm_layer(x), shift, scale, self._graph, out_dtype=DataType.int8)
        if x_mask is not None:
            t0 = View(self._graph, Shape(t0.shape[0], t0.shape[1], 1))(t0)
            scale_shift_table_zero = Eltwise(self._graph, t="add")(t0, self.scale_shift_table)
            split_name = [f"T2Final.{name}" for name in ["shift_zero", "scale_zero"]]
            shift_zero, scale_zero = Split(self._graph, 2, dim=1)(scale_shift_table_zero, split_name=split_name)
            x_zero = t2i_modulate(self.norm_layer(x), shift_zero, scale_zero, self._graph)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x, act_scale=x.get_act_scale, out_dtype=out_dtype)
        return x


class STDiT3BlockOnly: # The model define we used
    def __init__(
            self,
            graph: StaticGraph,
            # cross_attn_mask: Tensor, # Here the cross attn mask should be generated inside each Attention block, here just read from file for fast implementation
            input_size=Shape(1, 32, 32),
            input_sq_size=512,
            in_channels=4,
            patch_size=Shape(1, 2, 2),
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            pred_sigma=True,
            caption_channels=4096,
            matmul_int=True,
            do_cache: bool = False,
            re_compute: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels

        # ir generator
        self._graph = graph
        self.do_cache = do_cache
        self.re_compute = re_compute

        # model size related
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # input size related
        self.patch_size = patch_size
        self.input_sq_size = input_sq_size

        # embedding
        
        # spatial blocks
        self.spatial_blocks = [
            STDiT3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                # spatial
                temporal=False,
                graph=graph,
                layer_idx=idx,
                matmul_int=matmul_int,
                #cross_attn_mask=cross_attn_mask,
                do_cache=do_cache,
                re_compute=re_compute,
            )
            for idx in range(depth)
        ]

        # temporal blocks
        self.temporal_blocks = [
            STDiT3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                rope=True,
                # temporal
                temporal=True,
                graph=graph,
                layer_idx=idx,
                matmul_int=matmul_int,
                #cross_attn_mask=cross_attn_mask,
                do_cache=do_cache,
                re_compute=re_compute,
            )
            for idx in range(depth)
        ]

        # final layer
        self.final_layer = T2IFinalLayer(
            graph=graph,
            hidden_size=hidden_size,
            num_patches=self.patch_size.prod(),
            out_channels=self.out_channels,
        )

    def unpatchify(self, x: Tensor, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]
            N_t, N_h, N_w:
            R_t, R_h, R_w:
        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        B, N, C = x.shape
        T_p, H_p, W_p = self.patch_size
        assert self.out_channels == C // (T_p * H_p * W_p)
        x = View(self._graph, Shape(B, self.out_channels, N_t * T_p, N_h * H_p, N_w * W_p))(x)

        # unpad
        # x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def __call__(self, x: Tensor, y: Tensor, t: Tensor, t_mlp: Tensor, T:int, S:int,
                 mask: Optional[Tensor] = None, x_mask: Optional[Tensor] = None,
                 ):
        B = x.shape[0]
        # === get pos embed ===

        # === get timestep embed ===
        
        # === get y embed ===
        assert y.data_type == DataType.float16, "Input y should be float16"
        if self.re_compute:
            y = Convert(graph=self._graph, out_dtype=DataType.int8, name="y_convert_int8")(y)

        # === get x embed ===

        # === blocks ===
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = spatial_block(x, y, t_mlp, T=T, S=S)
            x = temporal_block(x, y, t_mlp, T=T, S=S)

        # === final layer ===
        # x = self.final_layer(x, t, T=T, S=S)
        # x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        return x


class STDiT3:
    def __init__(
            self,
            graph: StaticGraph,
            input_size=Shape(1, 32, 32),
            input_sq_size=512,
            in_channels=4,
            patch_size=Shape(1, 2, 2),
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            pred_sigma=True,
            caption_channels=4096,
            matmul_int=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels

        # ir generator
        self._graph = graph

        # model size related
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # input size related
        self.patch_size = patch_size
        self.input_sq_size = input_sq_size
        self.pos_embed = PositionEmbedding2D(graph, hidden_size)

        # embedding
        self.skip_y_embedder = False
        self.x_embedder = PatchEmbed3D(graph, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(graph, hidden_size)
        self.fps_embedder = SizeEmbedder(graph, hidden_size)
        self.t_block: list[Callable] = [
            Silu(graph),
            Linear(name="t_block.1", graph=graph, in_feature=hidden_size, out_feature=6 * hidden_size, bias=True),
        ]

        self.y_embedder = MLP(
            in_features=caption_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            graph=graph,
            name="y_embedder.y_proj",
        )

        # spatial blocks
        self.spatial_blocks = [
            STDiT3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                # spatial
                temporal=False,
                graph=graph,
                layer_idx=idx,
                matmul_int=matmul_int,
            )
            for idx in range(depth)
        ]

        # temporal blocks
        self.temporal_blocks = [
            STDiT3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                rope=True,
                # temporal
                temporal=True,
                graph=graph,
                layer_idx=idx,
                matmul_int=matmul_int,
            )
            for idx in range(depth)
        ]

        # final layer
        self.final_layer = T2IFinalLayer(
            graph=graph,
            hidden_size=hidden_size,
            num_patches=self.patch_size.prod(),
            out_channels=self.out_channels,
        )

    def encode_text(self, y: Tensor, mask=None):
        y = Convert(graph=self._graph, out_dtype=DataType.int8, name="text_convert_int8")(y)
        y = self.y_embedder(y, out_dtype=DataType.int8)  # [B, 1, N_token, C]
        if mask is not None:
            raise NotImplementedError("mask is not supported yet")
        else:
            y_lens = Shape(*tuple([y.shape[2]] * y.shape[0]))
            y = View(self._graph, Shape(1, y.shape.prod() // self.hidden_size, self.hidden_size))(y)
        return y, y_lens

    def get_dynamic_size(self, x: Tensor):
        _, _, T, H, W = x.shape
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return T, H, W

    def unpatchify(self, x: Tensor, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]
            N_t, N_h, N_w:
            R_t, R_h, R_w:
        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        B, N, C = x.shape
        T_p, H_p, W_p = self.patch_size
        assert self.out_channels == C // (T_p * H_p * W_p)
        x = View(self._graph, Shape(B, self.out_channels, N_t * T_p, N_h * H_p, N_w * W_p))(x)

        # unpad
        # x = x[:, :, :R_t, :R_h, :R_w]  # TODO: Item
        return x

    def __call__(self, x: Tensor, t: Tensor, y: Tensor, fps: Tensor, height: int, width: int,
                 mask: Optional[Tensor] = None, x_mask: Optional[Tensor] = None,
                 ):
        B = x.shape[0]

        # === get pos embed ===
        assert len(x.shape) == 5
        _, _, Tx, Hx, Wx = x.shape
        T, H, W = self.get_dynamic_size(x)

        S = H * W
        base_size = round(S ** 0.5)
        resolution_sq = (height * width) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(t)  # [B, C]
        fps = self.fps_embedder(fps, B)
        t_mlp = t = Eltwise(self._graph, t="add")(t, fps)

        t_mlp = self.t_block[0](t_mlp, DataType.int8)
        t_mlp = self.t_block[1](t_mlp, act_scale=t_mlp.get_act_scale, out_dtype=DataType.float16)

        t0 = t0_mlp = None
        if x_mask is not None:
            raise NotImplementedError("x_mask is not supported yet")
            # t0_timestep = torch.zeros_like(t)
            # t0 = self.t_embedder(t0_timestep)
            # t0_mlp = t0 = t0 + fps
            # for fn in self.t_block:
            #     t0_mlp = fn(t0_mlp)

        # === get y embed ===
        if self.skip_y_embedder:
            raise NotImplementedError("skip_y_embedder is not supported yet")
            # y_lens = mask
            # if isinstance(y_lens, torch.Tensor):
            #     y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        _, _, C = x.shape
        x = View(self._graph, Shape(B, T, S, C))(x)

        x = Eltwise(self._graph, t="add")(x, pos_emb)

        x = View(self._graph, Shape(B, T * S, C))(x)

        # === blocks ===
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = spatial_block(x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            x = temporal_block(x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        return x


class PostCFG:
    def __init__(self, graph: StaticGraph, guidance_scale: float,):
        self._graph = graph
        self.guidance_scale = guidance_scale

    def __call__(self, x: Tensor):
        B = x.shape[0]
        assert B == 2
        guidance_scale = Tensor(shape=x.shape, dtype=DataType.float16)
        guidance_scale.set_data(np.full(shape=x.shape, fill_value=self.guidance_scale, dtype=np.float16))
        
        # v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        pred_cond, pred_uncond = Split(self._graph, 2, dim=0)(x)
        diff_cond = Eltwise(self._graph, t="add")(pred_cond, pred_uncond)  # FIXME: 这里是减才对
        scale_cond = Eltwise(self._graph, t="mul")(diff_cond, guidance_scale)
        v_pred = Eltwise(self._graph, t="add")(pred_uncond, scale_cond)

        return v_pred


def get_mlp(x_tensor: Tensor) -> StaticGraph:
    g = StaticGraph()

    mlp = MLP(in_features=1152, hidden_features=1152 * 4, graph=g, stage="spatial", layer_idx=0)
    out_tensor = mlp(x_tensor,out_dtype=DataType.float16)
    assert x_tensor.shape == out_tensor.shape
    return g


def get_single_STDiT3Block(x_tensor: Tensor, y_tensor: Tensor, t_tensor: Tensor, T: int, S: int, cross_attenion_mask: Tensor) -> StaticGraph:
    g = StaticGraph()

    block = STDiT3Block(graph=g, hidden_size=1152, num_heads=16, layer_idx=0, temporal=True)

    #g.set_inputs([x_tensor, y_tensor, t_tensor])
    out_tensor = block(x_tensor, y_tensor, t_tensor, T=T, S=S, cross_attn_mask=cross_attenion_mask)
    assert x_tensor.shape == out_tensor.shape
    return g


def get_STDiT3(video: Tensor, text:Tensor,t:Tensor,fps: Tensor,
               B:int, C:int, T:int, H:int, W:int,cond_size:int,
               feature_size:int, caption_size:int):
    g = StaticGraph()
    model = STDiT3(graph=g)
    out_tensor = model(video, t, text, fps=fps, height=H, width=W)
    return g


if __name__ == '__main__':
    pass
