from typing import Optional

import torch
from datasets import load_dataset


_MATH_FINAL_ANSWER_INSTRUCTION = (
    "\n\nReturn your final response as "
    "'Final Answer: \\boxed{{<answer>}}', where <answer> is the number or "
    "mathematical expression of the solution."
)
_CODE_SUFFIX = (
    "\n\nRespond with only the solution code inside one markdown code block "
    "(```...```)."
)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def get_flexdraft_config(config) -> dict:
    return getattr(config, "flexdraft_config", None) or {}


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def compute_verify_len(
    logits: torch.Tensor,
    threshold: float,
    strategy: str = "min_confidence",
    min_len: int = 1,
    max_len: Optional[int] = None,
) -> int:
    probs = torch.softmax(logits[0], dim=-1)
    max_probs = probs.max(dim=-1).values

    if strategy == "min_confidence":
        verify_len = int((max_probs >= threshold).long().cumprod(dim=0).sum().item())
    elif strategy == "cumulative_product":
        verify_len = int((max_probs.cumprod(dim=0) >= threshold).long().sum().item())
    else:
        raise ValueError(f"Unknown pruning strategy: {strategy}")

    verify_len = max(verify_len, min_len)
    if max_len is not None:
        verify_len = min(verify_len, max_len)
    return verify_len


def _limit_samples(dataset, max_samples: Optional[int] = None):
    if max_samples and len(dataset) > max_samples:
        return dataset.shuffle(seed=0).select(range(max_samples))
    return dataset


def load_dataset_prompts(data_name: str, max_samples: Optional[int] = None):
    name = data_name.lower().replace("_", "-")

    if name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}" + _MATH_FINAL_ANSWER_INSTRUCTION
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif name in ("math", "math500", "math-500"):
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}" + _MATH_FINAL_ANSWER_INSTRUCTION
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif name in ("humaneval", "human-eval"):
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = (
            "Write a solution to the following problem and make sure that it passes "
            "the tests:\n```python\n{prompt}\n```"
        )
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x) + _CODE_SUFFIX]})

    elif name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"] + _CODE_SUFFIX]})

    elif name in ("mt-bench", "mtbench", "mtbnech"):
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    else:
        raise ValueError(
            f"Unsupported dataset: {data_name!r}. "
            "Choose one of: gsm8k, math, humaneval, mbpp, mt-bench."
        )

    return _limit_samples(dataset, max_samples)
