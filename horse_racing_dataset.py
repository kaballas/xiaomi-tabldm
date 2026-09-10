"""Resolve horse-racing CSV splits from local paths or Hugging Face."""

from pathlib import Path


def resolve_race_csvs(args):
    """Resolve local overrides or download the requested Hub dataset splits."""
    local_paths = (args.train_csv, args.validation_csv, args.test_csv)
    if any(path is not None for path in local_paths):
        missing = []
        if args.train_csv is None:
            missing.append("--train-csv")
        if args.validation_csv is None:
            missing.append("--validation-csv")
        if args.evaluate_test and args.test_csv is None:
            missing.append("--test-csv")
        if missing:
            raise ValueError(
                "Local CSV overrides require matching split paths; missing " + ", ".join(missing)
            )
        return args.train_csv, args.validation_csv, args.test_csv, {
            "type": "local_csv",
            "train": str(args.train_csv),
            "validation": str(args.validation_csv),
            "test": str(args.test_csv) if args.evaluate_test else None,
        }

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Loading --dataset requires huggingface-hub; install the project dependencies first"
        ) from exc

    def download(filename):
        return Path(
            hf_hub_download(
                repo_id=args.dataset,
                filename=filename,
                repo_type="dataset",
                revision=args.dataset_revision,
            )
        )

    train_csv = download("training.csv")
    validation_csv = download("validation.csv")
    test_csv = download("test.csv") if args.evaluate_test else None
    return train_csv, validation_csv, test_csv, {
        "type": "huggingface_dataset",
        "repo_id": args.dataset,
        "revision": args.dataset_revision,
        "files": {
            "train": "training.csv",
            "validation": "validation.csv",
            "test": "test.csv" if args.evaluate_test else None,
        },
    }
