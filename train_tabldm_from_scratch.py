"""Train a new race-aware Xiaomi TabLDM classifier from random initialization.

This is the from-scratch counterpart to ``finetune_tabldm.py``. Each episode
contains K complete earlier races as labelled in-context rows and one later
complete race as query rows. The saved checkpoint can be loaded by
``TabLDMClassifier``; no pretrained checkpoint is read at any point.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import tempfile

import pandas as pd
import torch

from finetune_tabldm import (
    ROOT,
    TARGET,
    build_inference_config,
    checkpoint_score,
    episode_specs,
    evaluate,
    format_metrics,
    load_races,
    nonnegative_float,
    positive_float,
    positive_integer,
    prepare_episodes,
    resolve_device,
    restore_trainable_parameters,
    save_trainable_parameters,
    seed_everything,
    train_epoch,
    validate_chronology,
)
from horse_racing_dataset import resolve_race_csvs
from tabldm._model.attnres_light_rmsnorm_moe import TabLDMSparseMoE
from tutorials.horse_racing_top3 import read_feature_columns


def nonnegative_integer(value):
    value = int(value)
    if value < 0:
        raise ValueError("value must be non-negative")
    return value


def probability(value):
    value = float(value)
    if not 0 <= value < 1:
        raise ValueError("value must be in the range [0, 1)")
    return value


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="Kaballas/races",
        help="Hugging Face dataset repository containing training.csv, validation.csv, and test.csv",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="optional Hugging Face branch, tag, or commit (defaults to the repository's main branch)",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="local training CSV override (requires --validation-csv)",
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=None,
        help="local validation CSV override (requires --train-csv)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="local test CSV override used with --evaluate-test",
    )
    parser.add_argument("--features-json", type=Path, default=ROOT / "a.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/tabldm_horse_from_scratch.ckpt",
    )

    parser.add_argument("--context-races", type=positive_integer, default=10)
    parser.add_argument("--epochs", type=positive_integer, default=50)
    parser.add_argument("--learning-rate", type=positive_float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=nonnegative_float, default=3e-5)
    parser.add_argument("--warmup-epochs", type=nonnegative_integer, default=5)
    parser.add_argument("--weight-decay", type=nonnegative_float, default=1e-4)
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=8,
        help="maximum number of shape-compatible race episodes per GPU batch",
    )
    parser.add_argument("--gradient-accumulation", type=positive_integer, default=4)
    parser.add_argument("--grad-clip", type=nonnegative_float, default=1.0)
    parser.add_argument("--listwise-weight", type=nonnegative_float, default=0.0)
    parser.add_argument("--moe-aux-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--patience", type=positive_integer, default=10)

    # These defaults reproduce the dimensions of the standard TabLDM model,
    # while remaining configurable for experiments and smoke tests.
    parser.add_argument("--embed-dim", type=positive_integer, default=128)
    parser.add_argument("--col-blocks", type=positive_integer, default=3)
    parser.add_argument("--col-heads", type=positive_integer, default=8)
    parser.add_argument("--col-inducing-points", type=positive_integer, default=128)
    parser.add_argument("--row-blocks", type=positive_integer, default=3)
    parser.add_argument("--row-heads", type=positive_integer, default=8)
    parser.add_argument("--row-cls-tokens", type=positive_integer, default=4)
    parser.add_argument("--icl-blocks", type=positive_integer, default=12)
    parser.add_argument("--icl-heads", type=positive_integer, default=8)
    parser.add_argument("--ff-factor", type=positive_integer, default=2)
    parser.add_argument("--dropout", type=probability, default=0.0)
    parser.add_argument("--feature-group-size", type=positive_integer, default=3)
    parser.add_argument("--global-max-span", type=positive_integer, default=32)
    parser.add_argument("--block-size", type=positive_integer, default=4)
    parser.add_argument("--attnres-stride", type=positive_integer, default=2)
    parser.add_argument("--moe-num-experts", type=nonnegative_integer, default=2)
    parser.add_argument("--moe-top-k", type=positive_integer, default=1)
    parser.add_argument("--moe-shared-experts", type=nonnegative_integer, default=1)
    parser.add_argument(
        "--moe-layers",
        default="last_8",
        help="MoE layer selection: none, all, last_half, last_8, every_N, or comma-separated indices",
    )
    parser.add_argument("--recompute", action="store_true", help="use gradient checkpointing")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=positive_integer, default=None)
    parser.add_argument(
        "--preprocessing-workers",
        type=positive_integer,
        default=min(4, os.cpu_count() or 1),
        help="CPU processes used to prepare race episodes",
    )
    parser.add_argument("--max-train-races", type=positive_integer, default=None)
    parser.add_argument("--max-validation-races", type=positive_integer, default=None)
    parser.add_argument("--max-test-races", type=positive_integer, default=None)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the sealed test split after selecting the best checkpoint",
    )
    parser.add_argument("--no-runner-shuffle", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def validate_model_args(args):
    if args.embed_dim % args.col_heads:
        raise ValueError("--embed-dim must be divisible by --col-heads")
    if args.embed_dim % args.row_heads:
        raise ValueError("--embed-dim must be divisible by --row-heads")
    icl_dim = args.embed_dim * args.row_cls_tokens
    if icl_dim % args.icl_heads:
        raise ValueError("--embed-dim * --row-cls-tokens must be divisible by --icl-heads")
    if args.moe_num_experts and args.moe_top_k > args.moe_num_experts:
        raise ValueError("--moe-top-k cannot exceed --moe-num-experts")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("--min-learning-rate cannot exceed --learning-rate")
    if args.warmup_epochs > args.epochs:
        raise ValueError("--warmup-epochs cannot exceed --epochs")

    named_moe_layers = {"none", "all", "last_half", "last_8", "last8"}
    if args.moe_layers not in named_moe_layers:
        if args.moe_layers.startswith("every_"):
            try:
                stride = int(args.moe_layers.split("_", 1)[1])
            except ValueError as exc:
                raise ValueError("--moe-layers every_N requires an integer N") from exc
            if stride < 1:
                raise ValueError("--moe-layers every_N requires N to be positive")
        else:
            try:
                indices = [int(item.strip()) for item in args.moe_layers.split(",")]
            except ValueError as exc:
                raise ValueError("--moe-layers must contain valid integer indices") from exc
            if not indices or any(not -args.icl_blocks <= index < args.icl_blocks for index in indices):
                raise ValueError("--moe-layers contains an index outside the ICL block range")


def build_model_config(args):
    """Return a loader-compatible binary classifier architecture config."""
    return {
        "max_classes": 2,
        "embed_dim": args.embed_dim,
        "col_num_blocks": args.col_blocks,
        "col_nhead": args.col_heads,
        "col_num_inds": args.col_inducing_points,
        "col_affine": False,
        "col_feature_group": "same",
        "col_feature_group_size": args.feature_group_size,
        "col_target_aware": True,
        "col_ssmax": "qassmax-mlp-elementwise",
        "row_num_blocks": args.row_blocks,
        "row_nhead": args.row_heads,
        "row_num_cls": args.row_cls_tokens,
        "row_rope_base": 100000,
        "row_rope_interleaved": True,
        "icl_num_blocks": args.icl_blocks,
        "icl_nhead": args.icl_heads,
        "icl_ssmax": "qassmax-mlp-elementwise",
        "ff_factor": args.ff_factor,
        "dropout": args.dropout,
        "activation": "gelu",
        "norm_first": True,
        # The public classifier constructs its dual-stream embedder this way.
        "bias_free_ln": True,
        # Disable optional zero initialization in components that expose it;
        # all learned weights still originate in the fresh model constructor.
        "zero_init": False,
        "recompute": args.recompute,
        "block_size": args.block_size,
        "attnres_stride": args.attnres_stride,
        "moe_num_experts": args.moe_num_experts,
        "moe_top_k": args.moe_top_k,
        "moe_num_shared_experts": args.moe_shared_experts,
        "moe_layers": args.moe_layers,
        "moe_router_z_loss_coef": 1e-3,
        "moe_load_balance_loss_coef": 1e-2,
        "moe_router_jitter": 0.0,
        "moe_router_weight_mode": "normalized",
        "moe_expert_init_noise": 0.0,
        "moe_auxiliary_free": False,
        "moe_bias_lr": 0.3,
        "moe_expert_init": "warmstart",
        "moe_routed_linear2_scale": 0.05,
        "moe_init_from_dense": True,
        "dual_stream": True,
        "global_dilation": "adaptive",
        "global_max_span": args.global_max_span,
    }


def build_model(args, device):
    config = build_model_config(args)
    model = TabLDMSparseMoE(**config)
    # MoE construction has already copied the randomly initialized dense FFNs
    # into its experts. The dense copies are then dead weights and the public
    # checkpoint loader drops them too.
    model.drop_dense_ffn()
    model.requires_grad_(True)
    return model.to(device), config


def learning_rate_multiplier(epoch, args):
    """Linear warmup followed by cosine decay to min_learning_rate."""
    if args.warmup_epochs and epoch < args.warmup_epochs:
        return (epoch + 1) / args.warmup_epochs
    decay_epochs = args.epochs - args.warmup_epochs
    if decay_epochs <= 1:
        return 1.0
    progress = min((epoch - args.warmup_epochs) / (decay_epochs - 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    minimum = args.min_learning_rate / args.learning_rate
    return minimum + (1.0 - minimum) * cosine


def save_scratch_checkpoint(model, config, output_path, metadata):
    checkpoint = {
        "config": config,
        "dual_stream_config": {
            "global_dilation": config["global_dilation"],
            "global_max_span": config["global_max_span"],
        },
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "scratch_training": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_model_args(args)
    # train_epoch shares this flag with the fine-tuning entry point. "full"
    # enables MoE auxiliary loss because every scratch parameter is trainable.
    args.finetune_mode = "full"
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    train_csv, validation_csv, test_csv, data_source = resolve_race_csvs(args)
    if data_source["type"] == "huggingface_dataset":
        revision = args.dataset_revision or "main"
        print(f"Dataset: https://huggingface.co/datasets/{args.dataset} (revision {revision})")
    else:
        print(f"Dataset: local CSV overrides ({train_csv}, {validation_csv})")

    train_header = pd.read_csv(train_csv, nrows=0).columns
    feature_columns = read_feature_columns(args.features_json, train_header)
    train_races, train_columns = load_races(train_csv, feature_columns)
    validation_races, validation_columns = load_races(validation_csv, feature_columns)
    test_races = None
    if args.evaluate_test:
        test_races, test_columns = load_races(test_csv, feature_columns)
        if list(train_columns) != list(test_columns):
            raise ValueError("Train and test CSV schemas do not match")
    if list(train_columns) != list(validation_columns):
        raise ValueError("Train and validation CSV schemas do not match")
    validate_chronology(train_races, validation_races, test_races)

    train_specs = episode_specs(train_races, [], args.context_races, args.max_train_races)
    validation_specs = episode_specs(
        validation_races, train_races, args.context_races, args.max_validation_races
    )
    test_specs = None
    if test_races is not None:
        test_specs = episode_specs(
            test_races,
            train_races + validation_races,
            args.context_races,
            args.max_test_races,
        )
    if not train_specs or not validation_specs or (test_specs is not None and not test_specs):
        raise ValueError("Every requested split must produce at least one race episode")

    split_summary = f"train={len(train_races)}, validation={len(validation_races)}"
    if test_races is not None:
        split_summary += f", test={len(test_races)}"
    print(f"Chronological races: {split_summary}; context={args.context_races} race(s)")
    print(f"Configured features: {len(feature_columns)} from {args.features_json}")
    print(f"Episode preprocessing workers: {args.preprocessing_workers}")
    train_episodes = prepare_episodes(
        train_specs, feature_columns, "training", args.preprocessing_workers
    )
    validation_episodes = prepare_episodes(
        validation_specs, feature_columns, "validation", args.preprocessing_workers
    )
    test_episodes = (
        prepare_episodes(test_specs, feature_columns, "test", args.preprocessing_workers)
        if test_specs is not None
        else None
    )

    seed_everything(args.seed)
    args.resolved_device = resolve_device(args.device)
    model, model_config = build_model(args, args.resolved_device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(
        f"Device: {args.resolved_device}; random initialization; "
        f"trainable={trainable_count:,}/{total_parameters:,} parameters "
        f"({trainable_count / total_parameters:.2%})"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: learning_rate_multiplier(epoch, args),
    )
    inference_config = build_inference_config(args.resolved_device)
    initial_validation = evaluate(
        model,
        validation_episodes,
        args.resolved_device,
        inference_config,
        batch_size=args.batch_size,
    )
    print(f"Random-init validation: {format_metrics(initial_validation)}")

    history = []
    best_score = None
    best_epoch = None
    stale_epochs = 0
    with tempfile.TemporaryDirectory(prefix="tabldm-scratch-") as temporary_directory:
        best_parameters_path = Path(temporary_directory) / "best-parameters.pt"
        for epoch in range(1, args.epochs + 1):
            current_learning_rate = optimizer.param_groups[0]["lr"]
            train_metrics = train_epoch(
                model, train_episodes, optimizer, trainable_parameters, args, epoch
            )
            validation_metrics = evaluate(
                model,
                validation_episodes,
                args.resolved_device,
                inference_config,
                batch_size=args.batch_size,
            )
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": current_learning_rate,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            print(
                f"Epoch {epoch:03d}: lr={current_learning_rate:.3g} "
                f"train_objective={train_metrics['objective']:.4f} "
                f"ce={train_metrics['cross_entropy']:.4f} "
                f"listwise={train_metrics['listwise']:.4f} "
                f"aux={train_metrics['moe_aux']:.4f}; "
                f"val {format_metrics(validation_metrics)}"
            )

            score = checkpoint_score(validation_metrics)
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                save_trainable_parameters(model, best_parameters_path)
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

        restore_trainable_parameters(model, best_parameters_path)

    final_validation = evaluate(
        model,
        validation_episodes,
        args.resolved_device,
        inference_config,
        batch_size=args.batch_size,
    )
    print(f"Selected checkpoint epoch: {best_epoch}")
    print(f"Best validation: {format_metrics(final_validation)}")
    final_test = None
    if test_episodes is not None:
        final_test = evaluate(
            model,
            test_episodes,
            args.resolved_device,
            inference_config,
            batch_size=args.batch_size,
        )
        print(f"Sealed test:     {format_metrics(final_test)}")
    else:
        print("Sealed test not accessed; pass --evaluate-test only for final evaluation")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "initialization": "random",
        "source_checkpoint": None,
        "data_source": data_source,
        "target": TARGET,
        "feature_config": str(args.features_json),
        "features": feature_columns,
        "preprocessing": "per-episode context-fitted, normalization=none, no feature/class shuffle",
        "model_config": model_config,
        "context_races": args.context_races,
        "batch_size": args.batch_size,
        "preprocessing_workers": args.preprocessing_workers,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "listwise_weight": args.listwise_weight,
        "moe_aux_weight": args.moe_aux_weight,
        "seed": args.seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "checkpoint_selection": ["top3_recall", "exact_top3_rate", "negative_race_log_loss"],
        "best_score": list(best_score),
        "history": history,
        "random_init_validation": initial_validation,
        "validation": final_validation,
        "test": final_test,
    }
    if args.no_save:
        print("Checkpoint saving disabled by --no-save")
    else:
        save_scratch_checkpoint(model, model_config, args.output, metadata)
        print(f"Saved scratch-trained checkpoint: {args.output}")
        print(f"Saved training metadata: {args.output.with_suffix(args.output.suffix + '.json')}")


if __name__ == "__main__":
    main()
