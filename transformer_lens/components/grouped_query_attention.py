from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import AbstractAttention
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.attention import complex_attn_linear, simple_attn_linear

from transformers.utils import is_bitsandbytes_available
if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import Params4bit


class GroupedQueryAttention(AbstractAttention):
    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        attn_type: str = "global",
        layer_id: Union[int, None] = None,
    ):
        """Grouped Query Attention Block - see https://arxiv.org/abs/2305.13245 for details.
        Similar to regular attention, W_Q, W_K, and W_V all have shape [head_index, d_model, d_head].
        However, under the hood the key and value weights _W_K and _W_V are stored with shape [n_key_value_heads, d_model, d_head] and are expanded when the corresponding properties' getter is called.
        Similarly, during a forward pass, initially K and V are kept in shapes [batch, pos, n_key_value_heads, d_head] and will only be expanded to shapes [batch, pos, n_heads, d_head]
        using torch.repeat_interleave when the attention pattern and z-scores are calculated.

        Args:
            cfg (Union[Dict, HookedTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistal models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        cfg = HookedTransformerConfig.unwrap(cfg)
        assert cfg.n_key_value_heads is not None
        super().__init__(cfg, attn_type, layer_id)
        self.repeat_kv_heads = cfg.n_heads // cfg.n_key_value_heads
        if self.cfg.load_in_4bit:
            # 4-bit quantization convention
            nq = int((self.cfg.d_model * self.cfg.d_head * self.cfg.n_key_value_heads) / 2)
            self._W_K = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
            self._W_V = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
        else:
            self._W_K = nn.Parameter(
                torch.empty(
                    cfg.n_key_value_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=cfg.dtype,
                )
            )
            self._W_V = nn.Parameter(
                torch.empty(
                    cfg.n_key_value_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=cfg.dtype,
                )
            )
        self._b_K = nn.Parameter(
            torch.zeros(cfg.n_key_value_heads, self.cfg.d_head, dtype=cfg.dtype)
        )
        self._b_V = nn.Parameter(
            torch.zeros(cfg.n_key_value_heads, self.cfg.d_head, dtype=cfg.dtype)
        )

    @property
    def W_K(self):
        return torch.repeat_interleave(self._W_K, dim=0, repeats=self.repeat_kv_heads)

    @W_K.setter
    def W_K(self, value):
        self._W_K = value

    @property
    def W_V(self):
        return torch.repeat_interleave(self._W_V, dim=0, repeats=self.repeat_kv_heads)

    @W_V.setter
    def W_V(self, value):
        self._W_V = value

    @property
    def b_K(self):
        return torch.repeat_interleave(self._b_K, dim=0, repeats=self.repeat_kv_heads)

    @b_K.setter
    def b_K(self, value):
        self._b_K = value

    @property
    def b_V(self):
        return torch.repeat_interleave(self._b_V, dim=0, repeats=self.repeat_kv_heads)

    @b_V.setter
    def b_V(self, value):
        self._b_V = value

    def calculate_qkv_matrices(
        self,
        query_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
        key_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos kv_head_index d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos kv_head_index d_model"],
        ],
    ) -> Tuple[
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch pos kv_head_index d_head"],
        Float[torch.Tensor, "batch pos kv_head_index d_head"],
    ]:
        """Calculate the Q, K, and V matrices for grouped query attention.
        This function uses the unexpanded weights _W_K and _W_V to calculate K and V.

        Args:
        query_input (Union[Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos head_index d_model"]]): The input tensor for the query projection.
        key_input (Union[Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos kv_head_index d_model"]]): The input tensor for the key projection. Note that is has as many head dimensions as the GPA block has key-value heads.
        value_input (Union[Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos kv_head_index d_model"]]): The input tensor for the value projection. Note that is has as many head dimensions as the GPA block has key-value heads.

        Returns:
        Tuple[Float[torch.Tensor, "batch pos head_index d_head"], Float[torch.Tensor, "batch pos kv_head_index d_head"], Float[torch.Tensor, "batch pos kv_head_index d_head"]]:
        A tuple containing the Q, K, and V matrices with the specified shapes.
        """
        attn_fn = (
            complex_attn_linear
            if self.cfg.use_split_qkv_input or self.cfg.use_attn_in
            else simple_attn_linear
        )

        if self.cfg.load_in_4bit:
            q = self.hook_q(
                # call bitsandbytes method to dequantize and multiply
                bnb.matmul_4bit(
                    query_input,
                    self.W_Q.t(),
                    bias=None,
                    quant_state=self.W_Q.quant_state,
                ).reshape(
                    query_input.shape[0],
                    query_input.shape[1],
                    self.cfg.n_heads,
                    self.cfg.d_head,
                )
                + self.b_Q
            )
        else:
            q = self.hook_q(
                attn_fn(query_input, self.W_Q, self.b_Q)
            )  # [batch, pos, head_index, d_head]

        if self.cfg.load_in_4bit:
            if not isinstance(self._W_K, Params4bit):
                raise ValueError("W_K must be a Params4bit object if load_in_4bit is True")
            k = bnb.matmul_4bit(
                    key_input, self._W_K.t(), bias=None, quant_state=self._W_K.quant_state
                ).reshape(
                    key_input.shape[0],
                    key_input.shape[1],
                    self.cfg.n_key_value_heads,
                    self.cfg.d_head,
                ) + self._b_K
        else:
            k = self.hook_k(
                attn_fn(key_input, self.W_K, self.b_K)
                if self.cfg.ungroup_grouped_query_attention
                else attn_fn(key_input, self._W_K, self._b_K)
            )  # [batch, pos, head_index, d_head]
        
        if self.cfg.load_in_4bit:
            if not isinstance(self._W_V, Params4bit):
                raise ValueError("W_V must be a Params4bit object if load_in_4bit is True")
            v = bnb.matmul_4bit(
                    value_input,
                    self._W_V.t(),
                    bias=None,
                    quant_state=self._W_V.quant_state,
                ).reshape(
                    value_input.shape[0],
                    value_input.shape[1],
                    self.cfg.n_key_value_heads,
                    self.cfg.d_head,
                ) + self._b_V
        else:
            v = self.hook_v(
                attn_fn(value_input, self.W_V, self.b_V)
                if self.cfg.ungroup_grouped_query_attention
                else attn_fn(value_input, self._W_V, self._b_V)
            )  # [batch, pos, head_index, d_head]

        return q, k, v

    def calculate_attention_scores(
        self,
        q: Float[torch.Tensor, "batch query_pos head_index d_head"],
        k: Float[torch.Tensor, "batch key_pos kv_head_index d_head"],
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        """Calculate attention scores from Q and the unexpanded K matrix.
        K will be expaned from [batch, pos, n_key_value_head, d_head] to [batch, pos, n_query_heads, d_head] using torch.repeat_interleave.

        Args:
        q (Float[torch.Tensor, "batch query_pos head_index d_head"]): The Q tensor.
        k (Float[torch.Tensor, "batch key_pos kv_head_index d_head"]): The K tensor.

        Returns:
            Float[torch.Tensor, "batch head_index query_pos key_pos"]: The attention scores.
        """
        if not self.cfg.ungroup_grouped_query_attention:
            k = self.hook_k(torch.repeat_interleave(k, dim=2, repeats=self.repeat_kv_heads))
            if self.cfg.dtype not in [torch.float32, torch.float64]:
                k = k.to(torch.float32)
        return super().calculate_attention_scores(q, k)

    def calculate_z_scores(
        self,
        v: Float[torch.Tensor, "batch key_pos kv_head_index d_head"],
        pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    ) -> Float[torch.Tensor, "batch query_pos head_index d_head"]:
        """Calculate z scores from the attention pattern and the unexpanded V matrix.
        V will be expaned from [batch, pos, n_key_value_head, d_head] to [batch, pos, n_query_heads, d_head] using torch.repeat_interleave.

        Args:
        v (Float[torch.Tensor, "batch query_pos head_index d_head"]): The V tensor.
        pattern (Float[torch.Tensor, "batch key_pos kv_head_index d_head"]): The attention pattern.

        Returns:
            Float[torch.Tensor, "batch head_index query_pos key_pos"]: The z scores.
        """
        if not self.cfg.ungroup_grouped_query_attention:
            v = self.hook_v(torch.repeat_interleave(v, dim=2, repeats=self.repeat_kv_heads))
        return super().calculate_z_scores(v, pattern)
