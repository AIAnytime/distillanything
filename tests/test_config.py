import pytest

from distillanything.config import DistillConfig


def test_yaml_roundtrip(tmp_path):
    cfg = DistillConfig(mode="seqkd")
    cfg.loss.kind = "jsd"
    cfg.train.lr = 3e-4
    path = tmp_path / "recipe.yaml"
    cfg.to_yaml(path)
    loaded = DistillConfig.from_yaml(path)
    assert loaded == cfg


def test_invalid_mode_rejected():
    with pytest.raises(Exception):
        DistillConfig(mode="quantum")


def test_repo_recipes_parse():
    from pathlib import Path

    recipes = Path(__file__).parent.parent / "recipes"
    parsed = [DistillConfig.from_yaml(p) for p in recipes.glob("*.yaml")]
    assert len(parsed) >= 2
