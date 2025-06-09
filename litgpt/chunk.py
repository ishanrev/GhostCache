
import math
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from litgpt.config import Config
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble


class ChunkManager(nn.Module):
   

    def __init__(
        self,
        k_shape,
        v_shape,
        device,
        dtype,
        max_chunk_size=10,
        max_tokens = 80
    ) -> None:
        super().__init__()
        self.chunks: List[Chunk] = []
        self.chunk_filled:List[int] = []
        self.max_chunk_size = max_chunk_size
        self.k_shape = k_shape,
        self.v_shape = v_shape
        self.max_tokens = max_tokens
          
    def add_token(self, k, v):
        
        chunk = None
        
        #  Determine the location here
        location = None
        if(self.chunk_filled.sum() > self.max_tokens/2):
          location = "cpu"
        else:
          location = "cuda"
        
        if len(self.chunks) == 0: # Prefill section
          
          T = k.shape[-2]
          num_chunks = (T // self.max_chunk_size) + 1
          offset = 0
          for i in range(num_chunks):
            chunk = Chunk(
              k_shape = self.k_shape,
              v_shape = self.v_shape,
              max_chunk_size=self.max_chunk_size,
              device=self.device,
              dtype = self.dtype,
              location = location
            )
            
            if i < num_chunks - 1:
              self.chunk_filled.append(self.max_chunk_size)
              chunk.add_token(k[:,:, offset: offset + self.max_chunk_size ,:])
              offset += self.max_chunk_size
            else:
              self.chunk_filled.append(self.max_chunk_size)
              chunk.add_token(k[:,:, offset ,:])
          
            self.chunks.append(chunk)
        else:
          # Theoretically must find the next best buffer ocation according to the topographical map networks based routing for chunking information
          if self.chunk_filled[-1] == self.max_chunk_size:
            chunk = Chunk(
              k_shape = self.k_shape,
              v_shape = self.v_shape,
              max_chunk_size=self.max_chunk_size,
              device=self.devic,
              dtype = self.dtype,
              location = location
            )
          
            self.chunks.append(chunk)
            self.chunk_filled.append(1)
          else:
            chunk = self.chunks[-1]
          
          chunk.add_token(k,v)
        
        

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)
        
class Chunk:
  
  def __init__(
    self, 
    k_shape,
    v_shape,
    max_chunk_size,
    device,
    dtype,
    location
               
  ):
      self.token_indicies = torch.tensor(max_chunk_size, device = device, dtype = dtype )
      self.location = location # For now we are startign small but we can definitely impove later on. 
      self.lazy_references = []
      self.filled = torch.tensor(0, dtype=torch.int32)
      B, nh, T, hd = k_shape
      self.k_buffer = torch.tensor((B,nh, max_chunk_size, hd), dtype = dtype, device = location)
      self.v_buffer = torch.tensor((B,nh, max_chunk_size, hd), dtype = dtype, device = location)
  
  
  def batched_index_copy_(self, t, dim, idx, val):
    """Index copy for batched t, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim {dim=}"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)

        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t

 
  def add_token(self, k, v):
    
    k = k.to(self.location)
    v = v.to(self.location)
    bs = k.size(0)
    k = self.batched_index_copy_(self.k[:bs, ...], -2, self.filled, k)
    v = self.batched_index_copy_(self.v[:bs, ...], -2, self.filled, v)
    self.filled = self.filled + 1
    
def chunked_sdpa(q, chunk_manager: ChunkManager, mask: torch.Tensor, config, input_pos: torch.Tensor):
  # raw attention scores need to be in the form of B, nh, T, hd
  
  head_size = config.head_size
  n_head = config.n_head
  n_query_groups = config.n_query_groups
  B, nh_q, T, hs = q.shape
  y = torch.tensor(q.shape, device = q.device, dtype = torch.dtype)
  scale = 1.0 / math.sqrt(chunk_manager.config.attention_scores_scalar or chunk_manager.config.head_size)

  for i in range(len(chunk_manager.chunks)):
    chunk = chunk_manager.chunks[i]
    
    if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
        q_per_kv = n_head // n_query_groups
        k = k.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
        v = v.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
        
        temp_y = F.scaled_dot_product_attention(
            q, chunk.k_buffer, chunk.v_buffer, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )

  return y.transpose(1,2)