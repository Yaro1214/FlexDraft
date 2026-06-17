# coding=utf-8
"""Core FlexDraft draft model blocks used by the released dual_attn_bias path."""

import time
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    eager_attention_forward,
)

from .utils import build_target_layer_ids, get_flexdraft_config


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


class Qwen3FlexDraftKVAttention(nn.Module):
    """Draft attention weights loaded from the FlexDraft checkpoint."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )


class Qwen3FlexDraftKVDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3FlexDraftKVAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class FlexDraftModel(Qwen3PreTrainedModel):
    """Minimal checkpoint container plus dual-attention layer forward."""

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3FlexDraftKVDecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Qwen3FlexDraftKVDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        flexdraft_config = get_flexdraft_config(config)
        self.target_layer_ids = flexdraft_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_embedding = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.cached_mask_embeds = None
        self.cached_mask_group_embeds = None
        self._rope_cos_cache = None
        self._rope_sin_cache = None
        self._rope_cache_len = 0
        self.post_init()

    @staticmethod
    def _build_fused_qkv(target_layers, draft_layers):
        """Pre-compute fused QKV weights for target/draft layer pairs."""
        fused = []
        for target_layer, draft_layer in zip(target_layers, draft_layers):
            ta = target_layer.self_attn
            da = draft_layer.self_attn
            ta_w = torch.cat([ta.q_proj.weight, ta.k_proj.weight, ta.v_proj.weight])
            da_w = torch.cat([da.q_proj.weight, da.k_proj.weight, da.v_proj.weight])
            ta_b = da_b = None
            if ta.q_proj.bias is not None:
                ta_b = torch.cat([ta.q_proj.bias, ta.k_proj.bias, ta.v_proj.bias])
                da_b = torch.cat([da.q_proj.bias, da.k_proj.bias, da.v_proj.bias])
            fused.append((ta_w, da_w, ta_b, da_b))
        return fused

    def _dual_attn_layer_forward(
        self,
        target_layer,
        draft_layer,
        hidden_states: torch.Tensor,
        candidate_len: int,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value,
        fused_qkv=None,
    ) -> torch.Tensor:
        residual = hidden_states
        x = target_layer.input_layernorm(hidden_states)

        ta = target_layer.self_attn
        da = draft_layer.self_attn

        x_c = x[:, :candidate_len, :]
        x_m = x[:, candidate_len:, :]
        head_dim = ta.head_dim

        if fused_qkv is not None:
            ta_qkv_w, da_qkv_w, ta_qkv_b, da_qkv_b = fused_qkv
            q_dim = ta.config.num_attention_heads * head_dim
            kv_dim = ta.config.num_key_value_heads * head_dim

            qkv_c = torch.nn.functional.linear(x_c, ta_qkv_w, ta_qkv_b)
            q_c, k_c, v_c = qkv_c.split([q_dim, kv_dim, kv_dim], dim=-1)
            hs_c = (*x_c.shape[:-1], -1, head_dim)
            qc = ta.q_norm(q_c.view(hs_c)).transpose(1, 2)
            kc = ta.k_norm(k_c.view(hs_c)).transpose(1, 2)
            vc = v_c.view(hs_c).transpose(1, 2)

            qkv_m = torch.nn.functional.linear(x_m, da_qkv_w, da_qkv_b)
            q_m, k_m, v_m = qkv_m.split([q_dim, kv_dim, kv_dim], dim=-1)
            hs_m = (*x_m.shape[:-1], -1, head_dim)
            qm = da.q_norm(q_m.view(hs_m)).transpose(1, 2)
            km = da.k_norm(k_m.view(hs_m)).transpose(1, 2)
            vm = v_m.view(hs_m).transpose(1, 2)
        else:
            qc, kc, vc = self._qkv(ta, x_c)
            qm, km, vm = self._qkv(da, x_m)

        query_states = torch.cat([qc, qm], dim=2)
        key_states = torch.cat([kc, km], dim=2)
        value_states = torch.cat([vc, vm], dim=2)

        cos, sin = position_embeddings
        query_states, key_states = hf_apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": None}
            k_cached, v_cached = past_key_value.update(
                key_states[:, :, :candidate_len, :],
                value_states[:, :, :candidate_len, :],
                ta.layer_idx,
                cache_kwargs,
            )
            key_states = torch.cat(
                [k_cached, key_states[:, :, candidate_len:, :]], dim=2
            )
            value_states = torch.cat(
                [v_cached, value_states[:, :, candidate_len:, :]], dim=2
            )

        get_interface = getattr(ALL_ATTENTION_FUNCTIONS, "get_interface", None)
        if get_interface is not None:
            attention_interface = get_interface(
                ta.config._attn_implementation, eager_attention_forward
            )
        else:
            attention_interface = (
                eager_attention_forward
                if ta.config._attn_implementation == "eager"
                else ALL_ATTENTION_FUNCTIONS[ta.config._attn_implementation]
            )

        attn_output, _ = attention_interface(
            ta,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not ta.training else ta.attention_dropout,
            scaling=ta.scaling,
            sliding_window=ta.sliding_window,
        )

        input_shape = x.shape[:-1]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        out_c = ta.o_proj(attn_output[:, :candidate_len, :])
        out_m = da.o_proj(attn_output[:, candidate_len:, :])
        hidden_states = residual + torch.cat([out_c, out_m], dim=1)

        residual = hidden_states
        hidden_states = target_layer.post_attention_layernorm(hidden_states)
        ffn_out = target_layer.mlp(hidden_states)
        hidden_states = residual + ffn_out
        return hidden_states

    @staticmethod
    def _qkv(attn, x_seg: torch.Tensor):
        input_shape = x_seg.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x_seg).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x_seg).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(x_seg).view(hidden_shape).transpose(1, 2)
        return q, k, v
