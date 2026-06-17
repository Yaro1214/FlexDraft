#!/usr/bin/env python3
"""Minimal FlexDraft dual_attn_bias inference demo.

Measures AR baseline vs. speculative decoding speedup and acceptance length.
This release exposes the dual_attn_bias interface only.
"""
import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, AutoConfig

from flexdraft import FlexDraftBiasModel, load_dataset_prompts, sample


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def ar_generate(
    target,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    """Autoregressive baseline (block_size = 1)."""
    batch_size = int(input_ids.shape[0])
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (batch_size, max_length + 1),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values = DynamicCache()

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    output_ids[:, :num_input_tokens] = input_ids
    first_tok = sample(output.logits[:1], temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_tok.expand(batch_size, -1)
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = num_input_tokens
    while start < max_length:
        block_output_ids = output_ids[:, start : start + 1].clone()
        block_position_ids = position_ids[:, start : start + 1]
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
        )
        posterior = sample(output.logits[:1], temperature).expand(batch_size, -1)
        output_ids[:, start + 1 : start + 2] = posterior
        start += 1
        past_key_values.crop(start)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    if stop_token_ids is not None:
        stop_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_ids_t
        ).nonzero(as_tuple=True)[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens * batch_size, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Target model path")
    parser.add_argument("--draft-name-or-path", type=str, required=True, help="FlexDraft draft checkpoint")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math", "humaneval", "mbpp", "mt-bench", "mtbench", "mtbnech"],
    )
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-confidence-threshold", type=float, default=0.01)
    parser.add_argument("--pruning-strategy", type=str, default="cumulative_product")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    draft_model = FlexDraftBiasModel.from_pretrained(
        args.draft_name_or_path,
        config=draft_config,
        dtype=torch.bfloat16,
    ).to(device).eval()

    print("[FlexDraft] mode = dual_attn_bias")
    print(f"[FlexDraft] draft class = {FlexDraftBiasModel.__name__}")
    print(f"[FlexDraft] target_layer_ids = {draft_model.target_layer_ids}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    dataset = load_dataset_prompts(args.dataset, args.max_samples)

    ar_tpots = []
    flexdraft_tpots = []
    all_acceptance_lengths = []

    for idx in range(len(dataset)):
        instance = dataset[idx]
        messages = [{"role": "user", "content": instance["turns"][0]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # AR baseline
        ar_res = ar_generate(
            target=target,
            input_ids=input_ids,
            mask_token_id=tokenizer.mask_token_id,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
        )
        ar_tpots.append(ar_res.time_per_output_token)

        flexdraft_res = draft_model.dual_attn_parallel_generate(
            target=target,
            input_ids=input_ids,
            mask_token_id=tokenizer.mask_token_id,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
            block_size=args.block_size,
            is_profiling=True,
            draft_confidence_threshold=args.draft_confidence_threshold,
            pruning_strategy=args.pruning_strategy,
        )

        flexdraft_tpots.append(flexdraft_res.time_per_output_token)
        all_acceptance_lengths.extend(flexdraft_res.acceptance_lengths)

        print(
            f"Sample {idx}: AR={ar_res.time_per_output_token*1000:.2f}ms/tok  "
            f"FlexDraft={flexdraft_res.time_per_output_token*1000:.2f}ms/tok  "
            f"AccLen={np.mean(flexdraft_res.acceptance_lengths):.2f}"
        )

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset} | Samples: {len(dataset)} | BlockSize: {args.block_size}")
    print(f"AR  TPOT: {np.mean(ar_tpots)*1000:.2f} ms/tok")
    print(f"FlexDraft TPOT: {np.mean(flexdraft_tpots)*1000:.2f} ms/tok")
    print(f"Speedup:   {np.mean(ar_tpots)/np.mean(flexdraft_tpots):.2f}x")
    print(f"Avg Acceptance Length: {np.mean(all_acceptance_lengths):.2f}")
    print(f"{'='*60}")

    # Print a sample output for sanity check
    print("\n--- Sample 0 FlexDraft Output ---")
    sample_text = tokenizer.decode(
        flexdraft_res.output_ids[0, flexdraft_res.num_input_tokens:],
        skip_special_tokens=True,
    )
    print(sample_text[:800])
    print("..." if len(sample_text) > 800 else "")


if __name__ == "__main__":
    main()
