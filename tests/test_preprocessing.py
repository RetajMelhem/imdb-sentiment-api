from __future__ import annotations

import pytest

from sentiment.preprocessing import (
    encode_texts,
    head_tail_prepare_ids,
    minimal_normalize,
)


def test_minimal_normalize_exact_notebook_behavior() -> None:
    text = "  Not GOOD!!!<BR />I wasn't\x00 bored &amp; it\u2019s odd.\n\t"
    assert minimal_normalize(text) == "Not GOOD!!! I wasn't bored & it\u2019s odd."


@pytest.mark.parametrize("break_tag", ["<br>", "<br/>", "<BR />"])
def test_minimal_normalize_handles_break_variants(break_tag: str) -> None:
    assert minimal_normalize(f"one{break_tag}two") == "one two"


def test_head_tail_keeps_short_content(fake_tokenizer) -> None:
    result = head_tail_prepare_ids([10, 11, 12], fake_tokenizer, max_length=8)
    assert result["input_ids"] == [101, 10, 11, 12, 102]


def test_head_tail_gives_odd_budget_extra_token_to_head(fake_tokenizer) -> None:
    result = head_tail_prepare_ids(list(range(10)), fake_tokenizer, max_length=7)
    assert result["input_ids"] == [101, 0, 1, 2, 8, 9, 102]


def test_head_tail_rejects_impossible_max_length(fake_tokenizer) -> None:
    with pytest.raises(ValueError, match="special tokens"):
        head_tail_prepare_ids([1], fake_tokenizer, max_length=2)


def test_head_tail_supports_native_tokenizer_without_legacy_helpers() -> None:
    class NativeTokenizer:
        cls_token_id = 1
        sep_token_id = 2
        bos_token_id = None
        eos_token_id = None

        @staticmethod
        def num_special_tokens_to_add(pair: bool = False) -> int:
            assert pair is False
            return 2

    result = head_tail_prepare_ids([10, 11], NativeTokenizer(), max_length=4)
    assert result == {
        "input_ids": [1, 10, 11, 2],
        "attention_mask": [1, 1, 1, 1],
    }


def test_encode_texts_batches_tokenization_and_preserves_order(fake_tokenizer) -> None:
    encoded = encode_texts(["one two", "three four five"], fake_tokenizer, max_length=8)
    assert len(encoded) == 2
    assert encoded[0]["input_ids"] == [101, 10, 11, 102]
    assert encoded[1]["input_ids"] == [101, 10, 11, 12, 102]
    assert fake_tokenizer.batch_tokenize_calls == 1
