"""Compare top-three predictions using one or more complete context races.

``race_id`` is used to keep each race intact while constructing the context.
The ID itself is excluded from the predictors because it is an arbitrary key.

Examples
--------
Use the first 10 eligible races starting at the requested context race::

    python tutorials/horse_racing_top3.py --context-races 10

Compare several nested context sizes against the same prediction race::

    python tutorials/horse_racing_top3.py --context-races 1 2 5 10
"""

from argparse import ArgumentParser
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tabldm import TabLDMClassifier


TARGET = "top3_mask"

# These columns reveal the result, describe label availability, or are row/race
# identifiers. They must not be predictors in an honest pre-race experiment.
EXCLUDED_COLUMNS = {
    "feature_schema_version",
    "race_id",
    "selection_id",
    "winner_index",
    "is_trainable",
    "source_betting_status",
    "finish_place",
    "result_code",
    "status",
    "sp_starting_price",
    "runner_mask",
    "rank_label",
    TARGET,
    "is_winner",
    # This is an exact duplicate of last_six in the supplied CSVs.
    "form_fig",
}


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise ValueError("value must be at least 1")
    return value


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--context-csv", type=Path, default=REPO_ROOT / "data/train.csv")
    parser.add_argument("--prediction-csv", type=Path, default=REPO_ROOT / "data/validation.csv")
    parser.add_argument(
        "--context-races",
        type=positive_integer,
        nargs="+",
        default=[1],
        metavar="N",
        help="one or more context sizes to compare, for example: 1 2 5 10",
    )
    parser.add_argument(
        "--context-race-id",
        default="10868873",
        help="first context race; later eligible races are selected in CSV order",
    )
    parser.add_argument("--prediction-race-id", default="10891607")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=positive_integer, default=2)
    return parser.parse_args()


def read_csv(csv_path):
    frame = pd.read_csv(csv_path, low_memory=False)
    required = {"race_id", "runner_mask", TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return frame


def active_race(frame, race_id):
    race = frame.loc[frame["race_id"].astype(str) == str(race_id)].copy()
    if race.empty:
        available = frame["race_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Race {race_id} is absent. Example race IDs: {available}")

    total_runners = len(race)
    race = race.loc[race["runner_mask"].eq(1)].reset_index(drop=True)
    return race, total_runners - len(race)


def has_complete_top3_labels(race):
    if race.empty or race[TARGET].isna().any():
        return False
    labels = set(race[TARGET].astype(int).unique())
    return labels == {0, 1} and int(race[TARGET].sum()) == 3


def select_context_races(frame, first_race_id, number_of_races):
    race_ids = frame["race_id"].drop_duplicates().tolist()
    race_id_strings = [str(race_id) for race_id in race_ids]
    try:
        start = race_id_strings.index(str(first_race_id))
    except ValueError as error:
        raise ValueError(f"Context race {first_race_id} is absent from the context CSV") from error

    selected_frames = []
    selected_ids = []
    inactive_count = 0
    for race_id in race_ids[start:]:
        race, inactive = active_race(frame, race_id)
        if not has_complete_top3_labels(race):
            continue
        selected_frames.append(race)
        selected_ids.append(race_id)
        inactive_count += inactive
        if len(selected_frames) == number_of_races:
            break

    if len(selected_frames) < number_of_races:
        raise ValueError(
            f"Only {len(selected_frames)} eligible labelled races are available at or after "
            f"race {first_race_id}; requested {number_of_races}"
        )
    return selected_frames, selected_ids, inactive_count


def text_overlap_stats(context, prediction, show_details=False):
    text_columns = list(context.select_dtypes(include=["object", "string", "category"]).columns)
    matched = 0
    observed = 0
    details = []

    for column in text_columns:
        context_values = set(context[column].dropna().astype(str))
        prediction_values = prediction[column].dropna().astype(str)
        known = prediction_values.isin(context_values)
        varies_in_context = context[column].nunique(dropna=False) > 1
        if varies_in_context:
            matched += int(known.sum())
            observed += len(prediction_values)
        details.append(
            {
                "field": column,
                "varies_in_context": varies_in_context,
                "context_unique": len(context_values),
                "prediction_values": len(prediction_values),
                "known_values": int(known.sum()),
            }
        )

    varying_count = sum(item["varies_in_context"] for item in details)
    known_rate = matched / max(observed, 1)
    if show_details:
        detail_frame = pd.DataFrame(details).set_index("field")
        fields_to_show = [
            field
            for field in ("overview", "runner_name", "jockey", "trainer", "sire", "dam", "sex", "track_status")
            if field in detail_frame.index
        ]
        if fields_to_show:
            print("\nSelected text-field overlap for the largest context (exact matches only):")
            print(detail_frame.loc[fields_to_show].to_string())
    return len(text_columns), varying_count, known_rate


def run_prediction(context, prediction, args, show_text_details=False):
    feature_columns = [column for column in context.columns if column not in EXCLUDED_COLUMNS]
    all_missing_columns = [
        column
        for column in feature_columns
        if context[column].isna().all() or prediction[column].isna().all()
    ]
    feature_columns = [column for column in feature_columns if column not in all_missing_columns]

    X_context = context.loc[:, feature_columns].copy()
    y_context = context[TARGET].astype(int).to_numpy()
    X_prediction = prediction.loc[:, feature_columns].copy()
    y_prediction = prediction[TARGET].astype(int).to_numpy()
    text_count, varying_text_count, known_text_rate = text_overlap_stats(
        X_context, X_prediction, show_details=show_text_details
    )

    classifier = TabLDMClassifier(
        n_estimators=args.n_estimators,
        device=args.device,
        model_path=os.environ.get("TABLDM_CLF_CKPT"),
        checkpoint_version="checkpoints/clf_default.ckpt",
    )
    classifier.fit(X_context, y_context)
    probabilities = classifier.predict_proba(X_prediction)

    positive_columns = np.flatnonzero(classifier.classes_ == 1)
    if len(positive_columns) != 1:
        raise RuntimeError(f"Expected class 1 in classifier classes, got {classifier.classes_}")
    top3_probability = probabilities[:, positive_columns[0]]
    raw_prediction = classifier.classes_[probabilities.argmax(axis=1)].astype(int)

    # A binary classifier can return any number of positives. Rank its scores
    # within this race so the race-level output always contains three runners.
    predicted_top3 = np.zeros(len(prediction), dtype=int)
    predicted_top3[np.argsort(-top3_probability, kind="stable")[:3]] = 1

    result = pd.DataFrame(
        {
            "runner_number": prediction["runner_number"].to_numpy(),
            "runner_name": prediction["runner_name"].to_numpy(),
            "top3_probability": top3_probability,
            "predicted_top3_mask": predicted_top3,
            "raw_binary_prediction": raw_prediction,
            "actual_top3_mask": y_prediction,
        }
    ).sort_values("top3_probability", ascending=False)

    correct = int(np.sum((predicted_top3 == 1) & (y_prediction == 1)))
    selected_names = ", ".join(prediction.loc[predicted_top3 == 1, "runner_name"].astype(str))
    summary = {
        "context_races": context["race_id"].nunique(),
        "context_runners": len(context),
        "features": len(feature_columns),
        "text_fields": text_count,
        "varying_text_fields": varying_text_count,
        "known_text_rate": known_text_rate,
        "correct_at_3": correct,
        "raw_positives": int(raw_prediction.sum()),
        "selected_runners": selected_names,
    }
    return result, summary


def main():
    args = parse_args()
    context_counts = sorted(set(args.context_races))
    context_source = read_csv(args.context_csv)
    prediction_source = read_csv(args.prediction_csv)
    if list(context_source.columns) != list(prediction_source.columns):
        raise ValueError("Context and prediction CSV schemas do not match")

    prediction, prediction_inactive = active_race(prediction_source, args.prediction_race_id)
    if len(prediction) < 3:
        raise ValueError("The prediction race must have at least three active runners")
    if not has_complete_top3_labels(prediction):
        raise ValueError("This comparison example needs a prediction race with three known top-three labels")

    context_frames, context_ids, _ = select_context_races(
        context_source, args.context_race_id, max(context_counts)
    )
    if args.context_csv.resolve() == args.prediction_csv.resolve():
        prediction_id = str(args.prediction_race_id)
        if prediction_id in {str(race_id) for race_id in context_ids}:
            raise ValueError("The prediction race cannot also be one of the context races")

    print(
        f"Prediction: {args.prediction_csv.name} race_id={args.prediction_race_id}, "
        f"{len(prediction)} active runners ({prediction_inactive} inactive omitted)"
    )
    print(f"Context starts at {args.context_race_id}; comparing race counts: {context_counts}")
    print("race_id defines whole-race boundaries and is excluded from model features.")

    summaries = []
    largest_result = None
    for context_count in context_counts:
        context = pd.concat(context_frames[:context_count], ignore_index=True)
        used_ids = context_ids[:context_count]
        omitted = sum(
            len(context_source.loc[context_source["race_id"].eq(race_id)])
            - len(context_frames[index])
            for index, race_id in enumerate(used_ids)
        )
        print(
            f"\nRunning with {context_count} context race(s), {len(context)} active runners "
            f"({omitted} inactive omitted): {used_ids}"
        )
        result, summary = run_prediction(
            context,
            prediction,
            args,
            show_text_details=context_count == max(context_counts),
        )
        summaries.append(summary)
        if context_count == max(context_counts):
            largest_result = result

    comparison = pd.DataFrame(summaries)
    comparison["known_text_rate"] = comparison["known_text_rate"].map(lambda value: f"{value:.1%}")
    print("\nContext-size comparison:")
    print(comparison.to_string(index=False))

    print("\nPredictions from the largest context:")
    print(
        largest_result.to_string(
            index=False,
            formatters={"top3_probability": lambda value: f"{value:.4f}"},
        )
    )
    print(
        "\nText note: TabLDM ordinal-encodes exact strings. It does not read prose semantically; "
        "a prediction string absent from all context races is encoded as unknown. More context "
        "races can increase exact category overlap."
    )


if __name__ == "__main__":
    main()
