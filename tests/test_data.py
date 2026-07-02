
import pytest

from distillanything.data.filters import clean_records, dedup_records
from distillanything.data.formats import load_records, save_records
from distillanything.data.tokenize import IGNORE_INDEX, SFTDataset, pad_collate
from distillanything.testing import tiny_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_tokenizer()


def test_load_save_roundtrip(tmp_path):
    records = [{"prompt": "hi", "response": "hello"}, {"text": "raw text"}]
    path = tmp_path / "d.jsonl"
    save_records(records, path)
    assert load_records(path) == records


def test_txt_loading(tmp_path):
    path = tmp_path / "seeds.txt"
    path.write_text("first prompt\n\nsecond prompt\n")
    assert load_records(path) == [{"prompt": "first prompt"}, {"prompt": "second prompt"}]


def test_bad_json_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"prompt": "ok"}\nnot json\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        load_records(path)


def test_dedup_normalized():
    records = [
        {"prompt": "Hello World", "response": "x"},
        {"prompt": "hello   world", "response": "x"},
        {"prompt": "different", "response": "x"},
    ]
    assert len(dedup_records(records)) == 2


def test_clean_drops_empty_responses():
    records = [{"prompt": "a", "response": ""}, {"prompt": "b", "response": "ok"}]
    assert clean_records(records) == [{"prompt": "b", "response": "ok"}]


def test_sft_dataset_masks_prompt(tokenizer):
    ds = SFTDataset([{"prompt": "abc", "response": "xyz"}], tokenizer, max_seq_len=64)
    assert len(ds) == 1
    item = ds[0]
    labels = item["labels"].tolist()
    # Prompt region masked, response region supervised.
    assert labels[0] == IGNORE_INDEX
    assert any(label != IGNORE_INDEX for label in labels)
    assert len(item["input_ids"]) == len(item["labels"])


def test_pad_collate_shapes(tokenizer):
    ds = SFTDataset(
        [{"prompt": "a", "response": "bb"}, {"prompt": "cccc", "response": "ddddddd"}],
        tokenizer,
        max_seq_len=64,
    )
    batch = pad_collate([ds[0], ds[1]], pad_token_id=tokenizer.pad_token_id)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"].sum() < batch["attention_mask"].numel()  # some padding happened
