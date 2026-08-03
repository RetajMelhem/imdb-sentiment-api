from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class FakeTokenizer:
    is_fast = False
    cls_token_id = 101
    sep_token_id = 102
    bos_token_id = None
    eos_token_id = None

    def __init__(self) -> None:
        self.pad_calls: list[dict[str, object]] = []
        self.batch_tokenize_calls = 0

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [index + 10 for index, _ in enumerate(text.split())]

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: bool = False,
        **_: object,
    ) -> dict[str, object]:
        assert truncation is False
        assert padding is False
        if isinstance(text, list):
            self.batch_tokenize_calls += 1
            rows = [self._ids(item) for item in text]
            return {"input_ids": rows}
        ids = self._ids(text)
        if add_special_tokens:
            ids = [101, *ids, 102]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def prepare_for_model(
        self, content_ids: list[int], **_: object
    ) -> dict[str, list[int]]:
        ids = [101, *content_ids, 102]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(
        self,
        records: list[dict[str, list[int]]],
        *,
        padding: bool,
        pad_to_multiple_of: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        self.pad_calls.append(
            {
                "records": records,
                "padding": padding,
                "pad_to_multiple_of": pad_to_multiple_of,
                "return_tensors": return_tensors,
            }
        )
        longest = max(len(record["input_ids"]) for record in records)
        width = math.ceil(longest / pad_to_multiple_of) * pad_to_multiple_of
        input_ids = []
        attention_mask = []
        for record in records:
            missing = width - len(record["input_ids"])
            input_ids.append([*record["input_ids"], *([0] * missing)])
            attention_mask.append([*record["attention_mask"], *([0] * missing)])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def save_pretrained(self, destination: Path) -> None:
        (Path(destination) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeModel:
    def __init__(self, logits: list[list[float]] | None = None) -> None:
        self._logits = logits
        self.device_moves: list[str] = []
        self.eval_calls = 0
        self.forward_calls = 0
        self.grad_enabled: list[bool] = []
        self.config = SimpleNamespace(
            num_labels=2,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1},
        )

    def to(self, device: torch.device) -> FakeModel:
        self.device_moves.append(str(device))
        return self

    def eval(self) -> FakeModel:
        self.eval_calls += 1
        return self

    def __call__(self, **batch: torch.Tensor) -> SimpleNamespace:
        self.forward_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        count = batch["input_ids"].shape[0]
        logits = self._logits or [[0.0, float(index)] for index in range(count)]
        return SimpleNamespace(logits=torch.tensor(logits[:count], dtype=torch.float32))

    def save_pretrained(
        self, destination: Path, safe_serialization: bool = True
    ) -> None:
        assert safe_serialization is True
        (Path(destination) / "config.json").write_text("{}", encoding="utf-8")
        (Path(destination) / "model.safetensors").write_bytes(b"weights")


@pytest.fixture
def metadata_path() -> Path:
    return PROJECT_ROOT / "model_metadata.json"


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()
