from dataclasses import dataclass
from typing import Optional
from transform.MLIRAdaptor import *
import mlir.dialects.SoraOps as sora
from mlir.ir import *

mlir = MLIRAdaptor()

def add(prefix: str, lhs: Value, rhs: Value) -> Value:
  output_type = lhs
  new_op = sora.ElementwiseOp(output=output_type,
                              lhs=lhs,
                              rhs=rhs,
                              op_type=self.mlir.get_string_attr(op.type),
                              dynamic_scale=op.act_scale_flag,
                              loc=self.get_loc(op.name),
                              ip=self.mlir.insert_point)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

class B_Tensor:
  def __init__(self):
    pass

class RMSNorm:
  def __init__(self, name: str, dim: int, eps: float = 1e-6):
    self.name = name
    self.eps = eps
    
  def __call__(self, x):
    pass

class Attention:
  def __init__(self, name: str, args: ModelArgs):
    pass
  def __call__(self, x):
    pass
  
class FeedForward:
  def __init__(self, name: str):
    pass
  def __call__(self, x):
    pass

class TrnasformerBlock:
  def __init__(self, name: str, layer_id: int, args: ModelArgs):
    self.name = name
    self.attention = Attention(self.name + '.attention', args)
    
  def __call__(self, x):
    h = x + self.attention