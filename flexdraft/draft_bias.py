# coding=utf-8
"""Released FlexDraft dual_attn_bias model.

Decode anchor for ``get_anchor_bias_logits`` must match training **bonus**:
``candidate_preds[:, acceptance_length]`` — the target's next-token prediction
after the accepted verify prefix — **not** ``Lc[:, 0]`` (block slot 0).

Prefill anchor remains ``first_token`` (next token after context), same idea.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from .draft_model import FlexDraftModel, cuda_time
from .utils import compute_verify_len, get_flexdraft_config, sample


class FlexDraftBiasModel(FlexDraftModel):
    """FlexDraft draft model with anchor-conditioned calibration."""

    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        flexdraft_config = get_flexdraft_config(config)
        self.enable_anchor_bias_logits = flexdraft_config.get(
            "enable_anchor_bias_logits", False
        )
        self.anchor_bias_rank = flexdraft_config.get("anchor_bias_rank", 256)
        if self.enable_anchor_bias_logits:
            self.anchor_bias_mlp = nn.Sequential(
                nn.Linear(config.hidden_size * 2, self.anchor_bias_rank, bias=True),
                nn.SiLU(),
                nn.Linear(self.anchor_bias_rank, config.vocab_size, bias=True),
            )

    def get_anchor_bias_logits(
        self, hidden_states: torch.Tensor, anchor_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute anchor-conditioned logits bias from hidden states and anchor embeddings."""
        if not self.enable_anchor_bias_logits:
            raise ValueError("Anchor bias logits is disabled in the draft config.")
        anchor_bias_input = torch.cat(
            [anchor_embeddings.to(hidden_states.dtype), hidden_states], dim=-1
        )
        return self.anchor_bias_mlp(anchor_bias_input)

    @torch.inference_mode()
    def dual_attn_parallel_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        mask_token_id: int,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        is_debug: bool = False,
        tokenizer=None,
        block_size: Optional[int] = None,
        is_profiling: bool = False,
        draft_confidence_threshold: float = 0.0,
        pruning_strategy: str = "min_confidence",
    ):
        if not self.enable_anchor_bias_logits:
            raise ValueError("dual_attn_bias requires flexdraft_config.enable_anchor_bias_logits=true")

        self.eval()
        batch_size = int(input_ids.shape[0])
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = block_size if block_size is not None else self.block_size
        bs = block_size
        enable_pruning = draft_confidence_threshold > 0.0

        dtype = target.model.embed_tokens.weight.dtype
        mask_emb = self.mask_embedding.to(dtype=dtype)

        self.cached_mask_embeds = (
            mask_emb.view(1, 1, -1).expand(batch_size, bs, -1).contiguous()
        )
        self.cached_mask_group_embeds = (
            mask_emb.view(1, 1, -1).expand(batch_size, bs * bs, -1).contiguous()
        )

        device = target.device
        past_key_values_target = DynamicCache()
        split_layer_idx = self.target_layer_ids[0]

        min_val = torch.finfo(dtype).min

        prefill_start = cuda_time()
        input_embeds = target.model.embed_tokens(input_ids)

        prefill_pos = torch.arange(num_input_tokens, device=device).unsqueeze(0)
        prefill_pos_emb = target.model.rotary_emb(input_embeds, prefill_pos)

        hidden = input_embeds
        for layer in target.model.layers[:split_layer_idx]:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=None,
                position_embeddings=prefill_pos_emb,
                past_key_value=past_key_values_target,
                use_cache=True,
            )

        combined_hidden = torch.cat([hidden, self.cached_mask_embeds], dim=1)

        prefill_phase2_pos = torch.arange(
            num_input_tokens + bs, device=device
        ).unsqueeze(0)
        prefill_phase2_pos_emb = target.model.rotary_emb(
            combined_hidden, prefill_phase2_pos
        )

        prefill_phase2_mask = torch.zeros(
            1, 1, num_input_tokens + bs, num_input_tokens + bs,
            dtype=dtype, device=device,
        )
        prefill_phase2_mask[:, :, :num_input_tokens, :num_input_tokens] = torch.triu(
            torch.full(
                (num_input_tokens, num_input_tokens), min_val,
                dtype=dtype, device=device,
            ),
            diagonal=1,
        )
        prefill_phase2_mask[:, :, :num_input_tokens, num_input_tokens:] = min_val

        _fused_qkv = self._build_fused_qkv(
            target.model.layers[split_layer_idx:], self.layers
        )
        for i, target_layer in enumerate(target.model.layers[split_layer_idx:]):
            draft_layer = self.layers[i]
            combined_hidden = self._dual_attn_layer_forward(
                target_layer=target_layer,
                draft_layer=draft_layer,
                hidden_states=combined_hidden,
                candidate_len=num_input_tokens,
                attention_mask=prefill_phase2_mask,
                position_embeddings=prefill_phase2_pos_emb,
                past_key_value=past_key_values_target,
                fused_qkv=_fused_qkv[i],
            )

        n = num_input_tokens
        relevant_normed = target.model.norm(combined_hidden[:, n - 1 : n + bs, :])
        logits = target.lm_head(relevant_normed)

        output_ids = torch.full(
            (batch_size, max_length + bs * (1 + bs) + bs),
            mask_token_id, dtype=torch.long, device=device,
        )
        output_ids[:, :num_input_tokens] = input_ids

        first_token0 = sample(logits[:1, :1, :], temperature)
        first_token = first_token0.expand(batch_size, -1)
        output_ids[:, num_input_tokens] = first_token.squeeze(-1)

        if self.enable_anchor_bias_logits:
            h = relevant_normed[:, 2:, :]
            anchor_e = target.model.embed_tokens(first_token).to(dtype=h.dtype)
            anchor_e = anchor_e.expand(-1, bs - 1, -1)
            delta = self.get_anchor_bias_logits(h, anchor_e)
            logits[:, 2:, :] = logits[:, 2:, :] + delta.to(logits.dtype)

        all_preds_tail0 = sample(logits[:1, 1:, :], temperature)
        all_preds_tail = all_preds_tail0.expand(batch_size, -1)
        Lc = torch.cat([first_token, all_preds_tail[:, 1:]], dim=1)

        if enable_pruning:
            prefill_draft_logits = logits[:, 2:, :]
            verify_len_init = compute_verify_len(
                prefill_draft_logits,
                threshold=draft_confidence_threshold,
                strategy=pruning_strategy,
                min_len=1,
                max_len=bs - 1,
            )
            current_num_groups = verify_len_init + 1
        else:
            current_num_groups = bs

        past_key_values_target.crop(num_input_tokens)
        time_to_first_token = cuda_time() - prefill_start

        max_pos = max_length + bs * (bs + 1) + bs
        if self._rope_cos_cache is None or self._rope_cache_len < max_pos:
            _all_pos = torch.arange(max_pos, device=device).unsqueeze(0)
            _dummy = torch.empty(1, 1, dtype=dtype, device=device)
            self._rope_cos_cache, self._rope_sin_cache = target.model.rotary_emb(
                _dummy, _all_pos
            )
            self._rope_cache_len = max_pos
            del _dummy, _all_pos
        _rope_cos_table = self._rope_cos_cache
        _rope_sin_table = self._rope_sin_cache

        decode_start = cuda_time()
        acceptance_lengths = []
        num_groups_history = []
        start = num_input_tokens

        self.pos_pattern_phase1 = torch.arange(bs, device=device).unsqueeze(0)

        if enable_pruning:
            self.pos_pattern_phase2_cache = {}
            for g in range(1, bs + 1):
                total_q_g = bs + g * bs
                pattern = torch.empty(1, total_q_g, dtype=torch.long, device=device)
                pattern[0, :bs] = torch.arange(bs, device=device)
                for kk in range(g):
                    off = bs + kk * bs
                    pattern[0, off : off + bs] = torch.arange(
                        kk + 1, kk + 1 + bs, device=device
                    )
                self.pos_pattern_phase2_cache[g] = pattern
        else:
            total_q = bs * (1 + bs)
            self.pos_pattern_phase2 = torch.empty(
                1, total_q, dtype=torch.long, device=device
            )
            self.pos_pattern_phase2[0, :bs] = torch.arange(bs, device=device)
            for k in range(bs):
                off = bs + k * bs
                self.pos_pattern_phase2[0, off : off + bs] = torch.arange(
                    k + 1, k + 1 + bs, device=device
                )

        _triu_bs = torch.triu(
            torch.full((bs, bs), min_val, dtype=dtype, device=device), diagonal=1,
        )
        if enable_pruning:
            _mask_right_cache = {}
            for g in range(1, bs + 1):
                tq = bs + g * bs
                right = torch.full(
                    (1, 1, tq, tq), min_val, dtype=dtype, device=device
                )
                right[:, :, :bs, :bs] = _triu_bs
                for kk in range(g):
                    qs = bs + kk * bs
                    qe = qs + bs
                    right[:, :, qs:qe, :kk + 1] = 0.0
                    right[:, :, qs:qe, qs:qe] = 0.0
                _mask_right_cache[g] = right
        else:
            _mask_right = torch.full(
                (1, 1, total_q, total_q), min_val, dtype=dtype, device=device
            )
            _mask_right[:, :, :bs, :bs] = _triu_bs
            for kk in range(bs):
                qs = bs + kk * bs
                qe = qs + bs
                _mask_right[:, :, qs:qe, :kk + 1] = 0.0
                _mask_right[:, :, qs:qe, qs:qe] = 0.0

        def _stop_check():
            if stop_token_ids is not None:
                for sid in stop_token_ids:
                    if sid in output_ids[0, num_input_tokens:start]:
                        return True
            return False

        while start < max_length:
            ctx_len = past_key_values_target.get_seq_length()

            candidate_embeds = target.model.embed_tokens(Lc)
            cand_pos_ids = self.pos_pattern_phase1 + start
            cand_pos_emb = (
                _rope_cos_table[:, cand_pos_ids[0], :],
                _rope_sin_table[:, cand_pos_ids[0], :],
            )
            cand_mask = torch.zeros(1, 1, bs, ctx_len + bs, dtype=dtype, device=device)
            cand_mask[:, :, :, ctx_len:] = _triu_bs

            hidden = candidate_embeds
            for layer in target.model.layers[:split_layer_idx]:
                hidden = layer(
                    hidden_states=hidden,
                    attention_mask=cand_mask,
                    position_embeddings=cand_pos_emb,
                    past_key_value=past_key_values_target,
                    use_cache=True,
                )

            num_groups = current_num_groups
            num_groups_history.append(num_groups)

            if enable_pruning:
                mask_group_embeds = self.cached_mask_group_embeds[
                    :, : num_groups * bs, :
                ]
                pos_pattern = self.pos_pattern_phase2_cache[num_groups]
            else:
                mask_group_embeds = self.cached_mask_group_embeds
                pos_pattern = self.pos_pattern_phase2

            combined_hidden = torch.cat([hidden, mask_group_embeds], dim=1)

            combined_pos_ids = pos_pattern + start
            combined_pos_emb = (
                _rope_cos_table[:, combined_pos_ids[0], :],
                _rope_sin_table[:, combined_pos_ids[0], :],
            )

            if enable_pruning:
                _right_block = _mask_right_cache[num_groups]
                _total_q = bs + num_groups * bs
            else:
                _right_block = _mask_right
                _total_q = bs * (1 + bs)
            combined_mask = torch.zeros(
                1, 1, _total_q, ctx_len + _total_q, dtype=dtype, device=device
            )
            combined_mask[:, :, :, ctx_len:] = _right_block

            for i, target_layer in enumerate(target.model.layers[split_layer_idx:]):
                draft_layer = self.layers[i]
                combined_hidden = self._dual_attn_layer_forward(
                    target_layer=target_layer,
                    draft_layer=draft_layer,
                    hidden_states=combined_hidden,
                    candidate_len=bs,
                    attention_mask=combined_mask,
                    position_embeddings=combined_pos_emb,
                    past_key_value=past_key_values_target,
                    fused_qkv=_fused_qkv[i],
                )

            cand_normed = target.model.norm(combined_hidden[:, :bs, :])
            candidate_logits = target.lm_head(cand_normed)
            candidate_preds0 = sample(candidate_logits[:1], temperature)
            candidate_preds = candidate_preds0.expand(batch_size, -1)

            if enable_pruning:
                verify_len = num_groups - 1
                if verify_len > 0:
                    acceptance_length = int(
                        (
                            candidate_preds[:, :verify_len] == Lc[:, 1 : 1 + verify_len]
                        )
                        .cumprod(dim=1)
                        .sum(dim=1)[0]
                        .item()
                    )
                else:
                    acceptance_length = 0
            else:
                acceptance_length = int(
                    (candidate_preds[:, : bs - 1] == Lc[:, 1:])
                    .cumprod(dim=1)
                    .sum(dim=1)[0]
                    .item()
                )

            output_ids[:, start : start + acceptance_length + 1] = Lc[
                :, : acceptance_length + 1
            ]

            bonus = candidate_preds[:, acceptance_length : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = bonus.squeeze(-1)

            k = min(acceptance_length, num_groups - 1)

            s = bs + k * bs + 1
            e = bs + (k + 1) * bs
            mask_normed = target.model.norm(combined_hidden[:, s:e, :])
            logits_mask = target.lm_head(mask_normed)
            if self.enable_anchor_bias_logits:
                anchor_e = target.model.embed_tokens(bonus).to(dtype=mask_normed.dtype)
                anchor_e = anchor_e.expand_as(mask_normed)
                delta = self.get_anchor_bias_logits(mask_normed, anchor_e)
                logits_mask = logits_mask + delta.to(logits_mask.dtype)

            mask_preds0 = sample(logits_mask[:1], temperature)
            mask_preds_k = mask_preds0.expand(batch_size, -1)
            Lc = torch.cat([bonus, mask_preds_k], dim=1)

            if enable_pruning:
                verify_len_next = compute_verify_len(
                    logits_mask,
                    threshold=draft_confidence_threshold,
                    strategy=pruning_strategy,
                    min_len=1,
                    max_len=bs - 1,
                )
                current_num_groups = verify_len_next + 1

            if is_debug:
                if tokenizer is not None:
                    draft_text = tokenizer.decode(
                        Lc[0, 1:], skip_special_tokens=True
                    )
                    print(f"[DEBUG dual_attn+bias] Draft Lc: {draft_text!r}")
                if enable_pruning:
                    print(
                        f"[DEBUG dual_attn+bias] accepted {acceptance_length}/{num_groups - 1} "
                        f"(num_groups={num_groups}, next_num_groups={current_num_groups})"
                    )
                else:
                    print(
                        f"[DEBUG dual_attn+bias] accepted {acceptance_length}/{bs - 1}"
                    )

            start += acceptance_length + 1
            past_key_values_target.crop(start)
            acceptance_lengths.append(acceptance_length + 1)

            if _stop_check() or start >= max_length:
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != mask_token_id]
        if stop_token_ids is not None:
            stop_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_ids_t
            ).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_indices[0] + 1
                ]

        total_decode_time = cuda_time() - decode_start
        num_output_tokens = output_ids.shape[1] - num_input_tokens
        time_per_output_token = total_decode_time / max(num_output_tokens * batch_size, 1)

        if is_debug and not is_profiling:
            total_drafted = len(acceptance_lengths) * (bs - 1)
            total_accepted = sum(n - 1 for n in acceptance_lengths)
            acceptance_rate = (
                total_accepted / total_drafted if total_drafted > 0 else 0.0
            )
            return output_ids, acceptance_rate

        if is_profiling:
            from types import SimpleNamespace

            result = SimpleNamespace(
                output_ids=output_ids,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                batch_size=batch_size,
                time_to_first_token=time_to_first_token,
                time_per_output_token=time_per_output_token,
                acceptance_lengths=acceptance_lengths,
            )
            if enable_pruning:
                result.num_groups_history = num_groups_history
            return result
        return output_ids
