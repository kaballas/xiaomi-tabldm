from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horse_racing_dataset import resolve_race_csvs


def args(**overrides):
    values = {
        "dataset": "Kaballas/races",
        "dataset_revision": None,
        "train_csv": None,
        "validation_csv": None,
        "test_csv": None,
        "evaluate_test": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resolve_race_csvs_downloads_training_and_validation_only(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return f"/cache/{kwargs['filename']}"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )

    train_csv, validation_csv, test_csv, source = resolve_race_csvs(args())

    assert train_csv == Path("/cache/training.csv")
    assert validation_csv == Path("/cache/validation.csv")
    assert test_csv is None
    assert [call["filename"] for call in calls] == ["training.csv", "validation.csv"]
    assert all(call["repo_id"] == "Kaballas/races" for call in calls)
    assert all(call["repo_type"] == "dataset" for call in calls)
    assert source["type"] == "huggingface_dataset"


def test_resolve_race_csvs_downloads_test_when_evaluated(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return f"/cache/{kwargs['filename']}"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )

    _, _, test_csv, source = resolve_race_csvs(
        args(evaluate_test=True, dataset_revision="fixed-revision")
    )

    assert test_csv == Path("/cache/test.csv")
    assert [call["filename"] for call in calls] == [
        "training.csv",
        "validation.csv",
        "test.csv",
    ]
    assert all(call["revision"] == "fixed-revision" for call in calls)
    assert source["files"]["test"] == "test.csv"


def test_resolve_race_csvs_requires_complete_local_overrides():
    with pytest.raises(ValueError, match="--validation-csv"):
        resolve_race_csvs(args(train_csv=Path("training.csv")))


def test_resolve_race_csvs_uses_complete_local_overrides():
    train_csv, validation_csv, test_csv, source = resolve_race_csvs(
        args(
            train_csv=Path("training.csv"),
            validation_csv=Path("validation.csv"),
        )
    )

    assert (train_csv, validation_csv, test_csv) == (
        Path("training.csv"),
        Path("validation.csv"),
        None,
    )
    assert source == {
        "type": "local_csv",
        "train": "training.csv",
        "validation": "validation.csv",
        "test": None,
    }
