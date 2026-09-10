"""Predict the top three runners in each race with a scratch-trained TabLDM.

The model receives the same number of complete, labelled context races used
during training. Every race in the prediction CSV is scored independently and
the three highest class-1 probabilities are selected.

By default, all prediction rows are treated as runners to score. This is
intentional because pre-result files can use ``runner_mask=0`` for every row.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from finetune_tabldm import load_races
from tabldm import TabLDMClassifier
from tutorials.horse_racing_top3 import (
    read_feature_columns,
    validate_checkpoint_features,
)


ROOT = Path(__file__).resolve().parent


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise ValueError("value must be at least 1")
    return value


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/tabldm_horse_from_scratch.ckpt",
    )
    parser.add_argument("--context-csv", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--prediction-csv", type=Path, default=ROOT / "data/predict.csv")
    parser.add_argument("--features-json", type=Path, default=ROOT / "a.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/tabldm_predictions.csv",
    )
    parser.add_argument("--context-races", type=positive_integer, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--kv-cache",
        choices=("kv", "repr", "off"),
        default="kv",
        help=(
            "cache the fitted context between prediction races: 'kv' is fastest, "
            "'repr' uses less memory, and 'off' disables caching (default: kv)"
        ),
    )
    parser.add_argument(
        "--use-runner-mask",
        action="store_true",
        help="score only rows where runner_mask equals 1 (off by default for pre-result data)",
    )
    return parser.parse_args()


def load_prediction_frame(path, feature_columns, use_runner_mask):
    frame = pd.read_csv(path, low_memory=False)
    required = {"race_id", "start_time_iso", "runner_number", "runner_name", *feature_columns}
    if use_runner_mask:
        required.add("runner_mask")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if use_runner_mask:
        frame = frame.loc[frame["runner_mask"].eq(1)].copy()
    else:
        frame = frame.copy()
    if frame.empty:
        raise ValueError(f"No runners are available to score in {path}")

    frame["_prediction_time"] = pd.to_datetime(
        frame["start_time_iso"], utc=True, errors="raise"
    )
    times_per_race = frame.groupby("race_id", sort=False)["_prediction_time"].nunique()
    invalid_races = times_per_race.loc[times_per_race.ne(1)].index.astype(str).tolist()
    if invalid_races:
        raise ValueError(f"Prediction races must have one start time: {invalid_races[:10]}")
    return frame


def build_classifier(args, context_races, feature_columns):
    context = pd.concat(
        [race.frame.loc[:, feature_columns] for race in context_races],
        ignore_index=True,
    )
    labels = np.concatenate([race.y for race in context_races])
    kv_cache = False if args.kv_cache == "off" else args.kv_cache
    classifier = TabLDMClassifier(
        model_path=args.checkpoint,
        allow_auto_download=False,
        n_estimators=1,
        norm_methods="none",
        feat_shuffle_method="none",
        class_shuffle_method="none",
        device=args.device,
        use_amp=False,
        use_fa3=False,
        kv_cache=kv_cache,
    )
    classifier.fit(context, labels)
    if kv_cache:
        model_caches = getattr(classifier, "model_kv_cache_", None)
        if not model_caches:
            raise RuntimeError("KV caching was requested but no context cache was built")
        cache_size_mb = sum(cache.cache_size_mb() for cache in model_caches.values())
        print(
            f"Built {args.kv_cache} context cache on {args.device} "
            f"({cache_size_mb:,} MiB); subsequent compatible races reuse it"
        )
    return classifier


def predict_race(classifier, race, feature_columns):
    if len(race) < 3:
        raise ValueError(
            f"Prediction race {race['race_id'].iloc[0]} has fewer than three runners"
        )

    features = race.loc[:, feature_columns]
    all_missing_features = features.columns[features.isna().all()].tolist()
    if (
        all_missing_features
        and getattr(classifier, "model_kv_cache_", None) is not None
    ):
        race_id = race["race_id"].iloc[0]
        print(
            f"Race {race_id}: context cache bypassed because these features are "
            f"entirely missing: {', '.join(all_missing_features)}"
        )

    probabilities = classifier.predict_proba(features)
    positive_columns = np.flatnonzero(classifier.classes_ == 1)
    if len(positive_columns) != 1:
        raise RuntimeError(f"Expected class 1 in classifier classes, got {classifier.classes_}")
    top3_probability = probabilities[:, positive_columns[0]]

    ordering = np.argsort(-top3_probability, kind="stable")
    predicted_top3 = np.zeros(len(race), dtype=np.int64)
    predicted_rank = np.empty(len(race), dtype=np.int64)
    predicted_top3[ordering[:3]] = 1
    predicted_rank[ordering] = np.arange(1, len(race) + 1)

    result = race.loc[
        :, ["race_id", "start_time_iso", "runner_number", "runner_name"]
    ].copy()
    result["top3_probability"] = top3_probability
    result["predicted_top3_mask"] = predicted_top3
    result["predicted_rank"] = predicted_rank
    return result


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {args.checkpoint}")

    context_header = pd.read_csv(args.context_csv, nrows=0).columns
    feature_columns = read_feature_columns(args.features_json, context_header)
    validate_checkpoint_features(args.checkpoint, feature_columns)
    context_races, _ = load_races(args.context_csv, feature_columns)
    predictions = load_prediction_frame(
        args.prediction_csv, feature_columns, args.use_runner_mask
    )

    missing_prediction_features = sorted(set(feature_columns).difference(predictions.columns))
    if missing_prediction_features:
        raise ValueError(
            f"{args.prediction_csv} is missing configured features: "
            f"{missing_prediction_features}"
        )

    # Reuse a fitted classifier when multiple prediction races have the same
    # chronological context window. Simultaneous races therefore cannot leak
    # their unknown outcomes into one another.
    classifier_cache = {}
    outputs = []
    grouped = predictions.groupby("race_id", sort=False)
    for race_id, race in grouped:
        race = race.reset_index(drop=True)
        prediction_time = race["_prediction_time"].iloc[0]
        history = [item for item in context_races if item.start_time < prediction_time]
        selected_context = history[-args.context_races :]
        if len(selected_context) < args.context_races:
            raise ValueError(
                f"Only {len(selected_context)} eligible context races precede race "
                f"{race_id}; requested {args.context_races}"
            )

        context_key = tuple(item.race_id for item in selected_context)
        classifier = classifier_cache.get(context_key)
        if classifier is None:
            print(
                f"Loading model and fitting context for race {race_id}: "
                f"{len(selected_context)} races, "
                f"{sum(len(item.frame) for item in selected_context)} runners"
            )
            classifier = build_classifier(args, selected_context, feature_columns)
            classifier_cache[context_key] = classifier

        result = predict_race(classifier, race, feature_columns)
        selected = result.loc[result["predicted_top3_mask"].eq(1)].sort_values(
            "predicted_rank"
        )
        names = ", ".join(selected["runner_name"].astype(str))
        print(f"Race {race_id}: {names}")
        outputs.append(result)

    predictions_output = pd.concat(outputs, ignore_index=True)
    predictions_output = predictions_output.sort_values(
        ["start_time_iso", "race_id", "predicted_rank"], kind="stable"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.to_csv(args.output, index=False)
    print(f"Saved {len(predictions_output)} runner predictions to {args.output}")


if __name__ == "__main__":
    main()
