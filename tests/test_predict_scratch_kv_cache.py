from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import predict_scratch_model as predictor


class _FakeCache:
    def cache_size_mb(self):
        return 12


class _FakeClassifier:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classes_ = np.array([0, 1])
        self.model_kv_cache_ = None
        self.__class__.instances.append(self)

    def fit(self, context, labels):
        self.fit_context = context
        self.fit_labels = labels
        if self.kwargs["kv_cache"]:
            self.model_kv_cache_ = {"none": _FakeCache()}
        return self

    def predict_proba(self, features):
        self.prediction_features = features
        probability = np.linspace(0.1, 0.9, len(features))
        return np.column_stack((1.0 - probability, probability))


@pytest.mark.parametrize(
    ("mode", "expected"), (("kv", "kv"), ("repr", "repr"), ("off", False))
)
def test_build_classifier_configures_requested_cache(
    monkeypatch, capsys, mode, expected
):
    monkeypatch.setattr(predictor, "TabLDMClassifier", _FakeClassifier)
    args = SimpleNamespace(checkpoint="model.ckpt", device="cpu", kv_cache=mode)
    context_races = [
        SimpleNamespace(
            frame=pd.DataFrame({"speed": [1.0, 2.0]}), y=np.array([0, 1])
        )
    ]

    classifier = predictor.build_classifier(args, context_races, ["speed"])

    assert classifier.kwargs["kv_cache"] == expected
    output = capsys.readouterr().out
    if expected:
        assert f"Built {mode} context cache on cpu (12 MiB)" in output
    else:
        assert output == ""


def test_predict_race_reports_all_missing_feature_cache_bypass(capsys):
    classifier = _FakeClassifier(kv_cache="kv")
    classifier.model_kv_cache_ = {"none": _FakeCache()}
    race = pd.DataFrame(
        {
            "race_id": [42, 42, 42],
            "start_time_iso": ["2026-01-01"] * 3,
            "runner_number": [1, 2, 3],
            "runner_name": ["A", "B", "C"],
            "speed": [1.0, 2.0, 3.0],
            "recent_class": [np.nan, np.nan, np.nan],
        }
    )

    predictor.predict_race(classifier, race, ["speed", "recent_class"])

    assert (
        "Race 42: context cache bypassed because these features are entirely "
        "missing: recent_class" in capsys.readouterr().out
    )
