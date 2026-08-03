"""Inference-only preprocessing reproduced from the training notebook."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from typing import Any, Protocol

CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TokenizerProtocol(Protocol):
    cls_token_id: int | None
    sep_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None

    def __call__(self, text: Any, **kwargs: Any) -> dict[str, Any]: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...

    def prepare_for_model(
        self, token_ids: list[int], **kwargs: Any
    ) -> dict[str, Any]: ...


def _single_sequence_special_ids(
    tokenizer: TokenizerProtocol, content_ids: list[int]
) -> list[int]:
    """Build the DeBERTa single-sequence wrapper across tokenizer API versions."""
    prefix_id = getattr(tokenizer, "cls_token_id", None)
    suffix_id = getattr(tokenizer, "sep_token_id", None)
    if prefix_id is None:
        prefix_id = getattr(tokenizer, "bos_token_id", None)
    if suffix_id is None:
        suffix_id = getattr(tokenizer, "eos_token_id", None)
    if prefix_id is None or suffix_id is None:
        raise ValueError("Tokenizer does not expose single-sequence special token IDs.")
    return [int(prefix_id), *content_ids, int(suffix_id)]


def minimal_normalize(text: str) -> str:
    """Apply the notebook's minimal normalization without changing sentiment cues."""
    value = html.unescape(str(text))
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = CONTROL_CHARACTER_PATTERN.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def head_tail_prepare_ids(
    content_ids: Sequence[int],
    tokenizer: TokenizerProtocol,
    max_length: int,
) -> dict[str, list[int]]:
    """Keep equally sized head/tail content segments and re-add special tokens."""
    selected_ids = list(content_ids)
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= 0:
        raise ValueError("max_length is too small for tokenizer special tokens.")
    if len(selected_ids) > budget:
        head_count = (budget + 1) // 2
        tail_count = budget - head_count
        selected_ids = selected_ids[:head_count] + (
            selected_ids[-tail_count:] if tail_count else []
        )
    prepare_for_model = getattr(tokenizer, "prepare_for_model", None)
    if callable(prepare_for_model):
        prepared = prepare_for_model(
            selected_ids,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )
    else:
        input_ids = _single_sequence_special_ids(tokenizer, selected_ids)
        prepared = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
    if len(prepared["input_ids"]) > max_length:
        raise ValueError("Tokenizer produced more tokens than max_length.")
    return {
        "input_ids": list(prepared["input_ids"]),
        "attention_mask": list(prepared["attention_mask"]),
    }


def encode_texts(
    texts: Sequence[str],
    tokenizer: TokenizerProtocol,
    max_length: int,
) -> list[dict[str, list[int]]]:
    """Normalize and head-tail encode a batch with one tokenizer call."""
    normalized = [minimal_normalize(text) for text in texts]
    tokenized = tokenizer(
        normalized,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return [
        head_tail_prepare_ids(content_ids, tokenizer, max_length)
        for content_ids in tokenized["input_ids"]
    ]
